import type { Client } from "@libsql/client";
import type { Config } from "./config.js";
import { platformFilter, queryMessages } from "./db.js";
import { Platform, Role } from "./models.js";
import { POLITENESS_PATTERNS, BACKTRACK_PATTERNS, COMMAND_PATTERN, computeNlpMetrics } from "./nlp.js";
import { classifyPersona } from "./persona.js";
import { computeTrendMetrics } from "./trends.js";

function round(n: number, decimals: number): number {
  const f = 10 ** decimals;
  return Math.round(n * f) / f;
}

function countAllMatches(texts: string[], pattern: RegExp): number {
  let total = 0;
  for (const t of texts) {
    total += (t.match(pattern) ?? []).length;
    pattern.lastIndex = 0;
  }
  return total;
}

export async function computeModelUsage(
  client: Client,
  platform?: Platform,
): Promise<Record<string, any>> {
  const pf = platformFilter(platform);

  const totalHuman = Number((await client.execute({ sql: `SELECT COUNT(*) as cnt FROM messages WHERE role = 'human'${pf.clause}`, args: pf.args })).rows[0].cnt);
  const withMeta = Number((await client.execute({ sql: `SELECT COUNT(*) as cnt FROM messages WHERE role = 'human' AND (model_id IS NOT NULL OR model_provider IS NOT NULL)${pf.clause}`, args: pf.args })).rows[0].cnt);
  const coveragePct = totalHuman ? round((withMeta / totalHuman) * 100, 1) : 0;

  // By model with source breakdown
  const modelSourceRows = (await client.execute({
    sql: `SELECT COALESCE(model_id, 'unknown'), COALESCE(model_provider, 'unknown'), platform, COUNT(*) as cnt, SUM(word_count) as words FROM messages WHERE role = 'human' AND (model_id IS NOT NULL OR model_provider IS NOT NULL)${pf.clause} GROUP BY 1, 2, platform`,
    args: pf.args,
  })).rows;

  const byModelMap = new Map<string, any>();
  for (const r of modelSourceRows) {
    const key = `${r[0]}|${r[1]}`;
    if (!byModelMap.has(key)) byModelMap.set(key, { model_id: String(r[0]), model_provider: String(r[1]), prompts: 0, words: 0, sources: {} });
    const entry = byModelMap.get(key)!;
    entry.prompts += Number(r.cnt ?? r[3]);
    entry.words += Number(r.words ?? r[4]);
    entry.sources[String(r.platform ?? r[2])] = Number(r.cnt ?? r[3]);
  }

  const convRows = (await client.execute({
    sql: `SELECT COALESCE(model_id, 'unknown'), COALESCE(model_provider, 'unknown'), COUNT(DISTINCT conversation_id) as cnt FROM messages WHERE role = 'human' AND (model_id IS NOT NULL OR model_provider IS NOT NULL)${pf.clause} GROUP BY 1, 2`,
    args: pf.args,
  })).rows;
  const convMap = new Map(convRows.map((r) => [`${r[0]}|${r[1]}`, Number(r.cnt ?? r[2])]));

  const byModelList = Array.from(byModelMap.entries())
    .map(([key, stats]) => ({ ...stats, conversations: convMap.get(key) ?? 0 }))
    .sort((a, b) => b.prompts - a.prompts);

  // By provider
  const provRows = (await client.execute({
    sql: `SELECT COALESCE(model_provider, 'unknown'), platform, COUNT(*) as cnt, SUM(word_count) as words FROM messages WHERE role = 'human' AND (model_id IS NOT NULL OR model_provider IS NOT NULL)${pf.clause} GROUP BY 1, platform`,
    args: pf.args,
  })).rows;

  const byProvMap = new Map<string, any>();
  for (const r of provRows) {
    const prov = String(r[0]);
    if (!byProvMap.has(prov)) byProvMap.set(prov, { model_provider: prov, prompts: 0, words: 0, sources: {} });
    const entry = byProvMap.get(prov)!;
    entry.prompts += Number(r.cnt ?? r[2]);
    entry.words += Number(r.words ?? r[3]);
    entry.sources[String(r.platform ?? r[1])] = Number(r.cnt ?? r[2]);
  }

  const provConvRows = (await client.execute({
    sql: `SELECT COALESCE(model_provider, 'unknown'), COUNT(DISTINCT conversation_id) as cnt FROM messages WHERE role = 'human' AND (model_id IS NOT NULL OR model_provider IS NOT NULL)${pf.clause} GROUP BY 1`,
    args: pf.args,
  })).rows;
  const provConvMap = new Map(provConvRows.map((r) => [String(r[0]), Number(r.cnt ?? r[1])]));

  const byProvList = Array.from(byProvMap.entries())
    .map(([prov, stats]) => ({ ...stats, conversations: provConvMap.get(prov) ?? 0 }))
    .sort((a, b) => b.prompts - a.prompts);

  // Time series
  const tsRows = (await client.execute({
    sql: `SELECT platform, local_date, COALESCE(model_id, 'unknown') as mid, COUNT(*) as cnt FROM messages WHERE role = 'human' AND (model_id IS NOT NULL OR model_provider IS NOT NULL)${pf.clause} GROUP BY platform, local_date, 3 ORDER BY platform, local_date`,
    args: pf.args,
  })).rows;

  const bySourceDate = new Map<string, Map<string, Record<string, number>>>();
  for (const r of tsRows) {
    const plat = String(r.platform ?? r[0]);
    const date = String(r.local_date ?? r[1]);
    const mid = String(r.mid ?? r[2]);
    const cnt = Number(r.cnt ?? r[3]);
    if (!bySourceDate.has(plat)) bySourceDate.set(plat, new Map());
    const dayMap = bySourceDate.get(plat)!;
    if (!dayMap.has(date)) dayMap.set(date, {});
    dayMap.get(date)![mid] = cnt;
  }

  const timeSeriesBySource: Record<string, any[]> = {};
  for (const [source, dayMap] of bySourceDate) {
    timeSeriesBySource[source] = Array.from(dayMap.entries())
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([date, models]) => ({
        date,
        total_prompts: Object.values(models).reduce((a, b) => a + b, 0),
        models: Object.fromEntries(Object.entries(models).sort(([, a], [, b]) => b - a)),
      }));
  }

  return {
    coverage: { total_human_prompts: totalHuman, prompts_with_model_metadata: withMeta, metadata_coverage_pct: coveragePct },
    by_model: byModelList,
    by_provider: byProvList,
    time_series_by_source: timeSeriesBySource,
  };
}

export async function computeMetrics(
  client: Client,
  config: Config,
  platform?: Platform,
): Promise<Record<string, any>> {
  const pf = platformFilter(platform);

  // Volume
  const vol = (await client.execute({
    sql: `SELECT COUNT(*) as total, SUM(CASE WHEN role = 'human' THEN 1 ELSE 0 END) as human, SUM(CASE WHEN role = 'assistant' THEN 1 ELSE 0 END) as assistant, SUM(CASE WHEN role = 'human' THEN word_count ELSE 0 END) as words_h, SUM(CASE WHEN role = 'assistant' THEN word_count ELSE 0 END) as words_a, COUNT(DISTINCT conversation_id) as convos FROM messages WHERE 1=1${pf.clause}`,
    args: pf.args,
  })).rows[0];

  const totalMessages = Number(vol.total);
  const totalHuman = Number(vol.human);
  const totalAssistant = Number(vol.assistant);
  const totalWordsHuman = Number(vol.words_h ?? 0);
  const totalWordsAssistant = Number(vol.words_a ?? 0);
  const totalConversations = Number(vol.convos);

  if (!totalHuman) throw new Error("No human messages found in data");
  const avgWordsPerPrompt = totalWordsHuman / totalHuman;

  // Conversation depth
  const depthRows = (await client.execute({
    sql: `SELECT conversation_id, COUNT(*) as cnt FROM messages WHERE 1=1${pf.clause} GROUP BY conversation_id`,
    args: pf.args,
  })).rows;
  const turns = depthRows.map((r) => Number(r.cnt));
  const avgTurns = turns.length ? turns.reduce((a, b) => a + b, 0) / turns.length : 0;
  const maxTurns = turns.length ? Math.max(...turns) : 0;
  const quickAsks = turns.filter((t) => t <= 3).length;
  const workingSessions = turns.filter((t) => t >= 4 && t <= 10).length;
  const deepDives = turns.filter((t) => t > 10).length;
  const responseRatio = totalWordsHuman ? totalWordsAssistant / totalWordsHuman : 0;

  // Temporal
  const heatmapRows = (await client.execute({
    sql: `SELECT local_weekday, local_hour, COUNT(*) as cnt FROM messages WHERE role = 'human'${pf.clause} GROUP BY local_weekday, local_hour`,
    args: pf.args,
  })).rows;
  const heatmap: number[][] = Array.from({ length: 7 }, () => Array(24).fill(0));
  for (const r of heatmapRows) heatmap[Number(r.local_weekday)][Number(r.local_hour)] = Number(r.cnt);

  const nightOwl = Number((await client.execute({
    sql: `SELECT COUNT(*) as cnt FROM messages WHERE role = 'human' AND (local_hour >= 23 OR local_hour < 4)${pf.clause}`,
    args: pf.args,
  })).rows[0].cnt);
  const nightOwlPct = (nightOwl / totalHuman) * 100;

  const hourRows = (await client.execute({
    sql: `SELECT local_hour, COUNT(*) as cnt FROM messages WHERE role = 'human'${pf.clause} GROUP BY local_hour`,
    args: pf.args,
  })).rows;
  const hourCounts: Record<number, number> = {};
  let peakHour = 0, peakHourCount = 0;
  for (const r of hourRows) {
    const h = Number(r.local_hour);
    const c = Number(r.cnt);
    hourCounts[h] = c;
    if (c > peakHourCount) { peakHour = h; peakHourCount = c; }
  }

  const dayRows = (await client.execute({
    sql: `SELECT local_weekday, COUNT(*) as cnt FROM messages WHERE role = 'human'${pf.clause} GROUP BY local_weekday`,
    args: pf.args,
  })).rows;
  let peakDay = 0, peakDayCount = 0;
  for (const r of dayRows) {
    const d = Number(r.local_weekday);
    const c = Number(r.cnt);
    if (c > peakDayCount) { peakDay = d; peakDayCount = c; }
  }
  const dayNames = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"];

  // Style metrics (regex on content)
  const humanMsgs = await queryMessages(client, { role: Role.HUMAN, platform });
  const assistantMsgs = await queryMessages(client, { role: Role.ASSISTANT, platform });
  const humanTexts = humanMsgs.map((m) => m.content);
  const assistantTexts = assistantMsgs.map((m) => m.content);

  const polCounts: Record<string, number> = {};
  for (const [word, pattern] of Object.entries(POLITENESS_PATTERNS)) {
    polCounts[word] = countAllMatches(humanTexts, pattern);
  }
  const totalPoliteness = Object.values(polCounts).reduce((a, b) => a + b, 0);
  const polPer100 = (totalPoliteness / totalHuman) * 100;

  const btCounts: Record<string, number> = {};
  for (const [word, pattern] of Object.entries(BACKTRACK_PATTERNS)) {
    btCounts[word] = countAllMatches(humanTexts, pattern);
  }
  const totalBacktrack = Object.values(btCounts).reduce((a, b) => a + b, 0);
  const btPer100 = (totalBacktrack / totalHuman) * 100;

  const qCount = humanMsgs.filter((m) => m.content.trim().endsWith("?")).length;
  const qRate = (qCount / totalHuman) * 100;
  let cCount = 0;
  for (const m of humanMsgs) {
    if (COMMAND_PATTERN.test(m.content.trim())) cCount++;
    COMMAND_PATTERN.lastIndex = 0;
  }
  const cRate = (cCount / totalHuman) * 100;

  const yrPattern = /you'?re (absolutely )?right/gi;
  const yrCount = countAllMatches(assistantTexts, yrPattern);
  const yrPerConvo = totalConversations ? yrCount / totalConversations : 0;

  // Platform stats
  const platRows = (await client.execute({
    sql: `SELECT platform, COUNT(*) as cnt, SUM(word_count) as words, COUNT(DISTINCT conversation_id) as convos, MIN(timestamp) as first_ts FROM messages WHERE role = 'human'${pf.clause} GROUP BY platform`,
    args: pf.args,
  })).rows;
  const platformStats: Record<string, any> = {};
  for (const r of platRows) {
    platformStats[String(r.platform)] = {
      messages: Number(r.cnt),
      words: Number(r.words),
      conversations: Number(r.convos),
      first_message: String(r.first_ts),
    };
  }

  // Persona
  const persona = classifyPersona(polPer100, btPer100, qRate, cRate, config);

  // Date range
  const dateRange = (await client.execute({
    sql: `SELECT MIN(timestamp) as first_ts, MAX(timestamp) as last_ts FROM messages WHERE role = 'human'${pf.clause}`,
    args: pf.args,
  })).rows[0];

  // Sub-module metrics
  const [modelUsage, trends, nlp] = await Promise.all([
    computeModelUsage(client, platform),
    computeTrendMetrics(client, platform),
    computeNlpMetrics(client, platform),
  ]);

  return {
    generated_at: new Date().toISOString(),
    volume: {
      total_messages: totalMessages,
      total_human: totalHuman,
      total_assistant: totalAssistant,
      total_words_human: totalWordsHuman,
      total_words_assistant: totalWordsAssistant,
      total_conversations: totalConversations,
      avg_words_per_prompt: round(avgWordsPerPrompt, 1),
    },
    conversation_depth: {
      avg_turns: round(avgTurns, 1),
      max_turns: maxTurns,
      quick_asks: quickAsks,
      working_sessions: workingSessions,
      deep_dives: deepDives,
    },
    response_ratio: round(responseRatio, 1),
    temporal: {
      heatmap,
      night_owl_pct: round(nightOwlPct, 1),
      peak_hour: peakHour,
      peak_hour_count: peakHourCount,
      peak_day: dayNames[peakDay],
      peak_day_count: peakDayCount,
      hour_counts: hourCounts,
    },
    politeness: { counts: polCounts, total: totalPoliteness, per_100_prompts: round(polPer100, 1) },
    backtrack: { counts: btCounts, total: totalBacktrack, per_100_prompts: round(btPer100, 1) },
    question: { count: qCount, rate: round(qRate, 1) },
    command: { count: cCount, rate: round(cRate, 1) },
    youre_right: { count: yrCount, per_conversation: round(yrPerConvo, 1) },
    persona,
    platform_stats: platformStats,
    model_usage: modelUsage,
    trends,
    nlp,
    date_range: { first: String(dateRange.first_ts), last: String(dateRange.last_ts) },
  };
}

export async function hasHumanMessages(client: Client, platform?: Platform): Promise<boolean> {
  const pf = platformFilter(platform);
  const r = await client.execute({ sql: `SELECT COUNT(*) as cnt FROM messages WHERE role = 'human'${pf.clause}`, args: pf.args });
  return Number(r.rows[0].cnt) > 0;
}

export async function computeSourceViews(
  client: Client,
  config: Config,
): Promise<{ sourceViews: Record<string, any>; metadata: { default_view: string } }> {
  const allMetrics = await computeMetrics(client, config);
  const ccHas = await hasHumanMessages(client, Platform.CLAUDE_CODE);
  const cxHas = await hasHumanMessages(client, Platform.CODEX);
  const agHas = await hasHumanMessages(client, Platform.AGENT);

  const sourceViews: Record<string, any> = {
    both: allMetrics,
    claude_code: ccHas ? await computeMetrics(client, config, Platform.CLAUDE_CODE) : null,
    codex: cxHas ? await computeMetrics(client, config, Platform.CODEX) : null,
    agent: agHas ? await computeMetrics(client, config, Platform.AGENT) : null,
  };

  let defaultView = "both";
  if (!sourceViews[defaultView]) {
    for (const candidate of ["claude_code", "codex"]) {
      if (sourceViews[candidate]) { defaultView = candidate; break; }
    }
  }

  return { sourceViews, metadata: { default_view: defaultView } };
}
