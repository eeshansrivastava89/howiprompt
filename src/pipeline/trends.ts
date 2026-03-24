import type { Client } from "@libsql/client";
import { queryMessages, platformFilter, type MessageRow } from "./db.js";
import { Platform, Role } from "./models.js";
import { POLITENESS_PATTERNS, BACKTRACK_PATTERNS, COMMAND_PATTERN } from "./nlp.js";

function countPatternMatches(text: string, pattern: RegExp): number {
  const matches = text.match(pattern);
  pattern.lastIndex = 0;
  return matches?.length ?? 0;
}

export function computeStyleSnapshot(messages: MessageRow[]): Record<string, number> {
  const total = messages.length;
  if (total === 0) {
    return { politeness_pct: 0, politeness_per_100: 0, backtrack_per_100: 0, question_rate_pct: 0, command_rate_pct: 0 };
  }

  let polCount = 0;
  let politePrompts = 0;
  const allPatterns = Object.values(POLITENESS_PATTERNS);
  for (const m of messages) {
    let hasPolite = false;
    for (const pattern of allPatterns) {
      const matches = m.content.match(pattern);
      pattern.lastIndex = 0;
      if (matches) {
        polCount += matches.length;
        hasPolite = true;
      }
    }
    if (hasPolite) politePrompts++;
  }

  let btCount = 0;
  for (const pattern of Object.values(BACKTRACK_PATTERNS)) {
    for (const m of messages) btCount += countPatternMatches(m.content, pattern);
  }

  const qCount = messages.filter((m) => m.content.trim().endsWith("?")).length;
  const cCount = messages.filter((m) => COMMAND_PATTERN.test(m.content.trim())).length;
  COMMAND_PATTERN.lastIndex = 0;

  return {
    politeness_pct: round((politePrompts / total) * 100, 1),
    politeness_per_100: round((polCount / total) * 100, 1),
    backtrack_per_100: round((btCount / total) * 100, 1),
    question_rate_pct: round((qCount / total) * 100, 1),
    command_rate_pct: round((cCount / total) * 100, 1),
  };
}

export function buildTrendRollups(
  bucketMap: Map<string, MessageRow[]>,
  bucketKey: string,
): Record<string, any>[] {
  const keys = Array.from(bucketMap.keys()).sort();
  const rollups: Record<string, any>[] = [];

  for (const key of keys) {
    const msgs = bucketMap.get(key)!;
    const totalPrompts = msgs.length;

    const sourceCounts: Record<string, number> = { claude_code: 0, codex: 0 };
    const modelCounts: Record<string, number> = {};

    for (const m of msgs) {
      sourceCounts[m.platform] = (sourceCounts[m.platform] ?? 0) + 1;
      if (m.modelId || m.modelProvider) {
        const mid = m.modelId ?? "unknown";
        modelCounts[mid] = (modelCounts[mid] ?? 0) + 1;
      }
    }

    const sourceSharePct: Record<string, number> = {};
    for (const src of [Platform.CLAUDE_CODE, Platform.CODEX]) {
      sourceSharePct[src] = totalPrompts ? round((sourceCounts[src] ?? 0) / totalPrompts * 100, 1) : 0;
    }

    rollups.push({
      [bucketKey]: key,
      prompts: totalPrompts,
      source_counts: sourceCounts,
      source_share_pct: sourceSharePct,
      style: computeStyleSnapshot(msgs),
      model_prompts: Object.values(modelCounts).reduce((a, b) => a + b, 0),
      models: Object.fromEntries(
        Object.entries(modelCounts).sort(([, a], [, b]) => b - a),
      ),
    });
  }

  return rollups;
}

export function computeTrendDeltas(dailyRollups: Record<string, any>[]): Record<string, any> {
  if (!dailyRollups.length) return {};

  const latestDate = new Date(dailyRollups[dailyRollups.length - 1].date);

  function windowRollups(days: number) {
    const start = new Date(latestDate);
    start.setDate(start.getDate() - (days - 1));
    return dailyRollups.filter((item) => new Date(item.date) >= start);
  }

  function windowAvg(items: Record<string, any>[], selector: (item: any) => number): number | null {
    if (!items.length) return null;
    return round(items.reduce((sum, item) => sum + selector(item), 0) / items.length, 1);
  }

  const recent7 = windowRollups(7);
  const recent30 = windowRollups(30);

  const selectors: Record<string, (item: any) => number> = {
    prompts_per_day: (item) => item.prompts,
    codex_share_pct: (item) => item.source_share_pct[Platform.CODEX],
    politeness_per_100: (item) => item.style.politeness_per_100,
    backtrack_per_100: (item) => item.style.backtrack_per_100,
    question_rate_pct: (item) => item.style.question_rate_pct,
    command_rate_pct: (item) => item.style.command_rate_pct,
    model_coverage_pct: (item) =>
      item.prompts ? round((item.model_prompts / item.prompts) * 100, 1) : 0,
  };

  const deltas: Record<string, any> = {};
  for (const [name, selector] of Object.entries(selectors)) {
    const avg7 = windowAvg(recent7, selector);
    const avg30 = windowAvg(recent30, selector);
    const deltaPct = avg7 != null && avg30 != null && avg30 !== 0
      ? round(((avg7 - avg30) / avg30) * 100, 1)
      : null;
    deltas[name] = { avg_7d: avg7, avg_30d: avg30, delta_pct: deltaPct };
  }

  return deltas;
}

export function detectShiftMarkers(dailyRollups: Record<string, any>[]): Record<string, any>[] {
  if (dailyRollups.length < 2) return [];

  const avgPrompts = dailyRollups.reduce((s, r) => s + r.prompts, 0) / dailyRollups.length;
  const minShift = Math.max(10, Math.floor(avgPrompts * 0.35));
  const markers: Record<string, any>[] = [];

  for (let i = 1; i < dailyRollups.length; i++) {
    const prev = dailyRollups[i - 1];
    const curr = dailyRollups[i];

    const promptDelta = curr.prompts - prev.prompts;
    if (Math.abs(promptDelta) >= minShift) {
      markers.push({
        date: curr.date,
        type: "prompt_shift",
        direction: promptDelta > 0 ? "up" : "down",
        delta_prompts: promptDelta,
        prev_prompts: prev.prompts,
        curr_prompts: curr.prompts,
        magnitude: Math.abs(promptDelta),
      });
    }

    const codexDelta = round(
      curr.source_share_pct[Platform.CODEX] - prev.source_share_pct[Platform.CODEX],
      1,
    );
    if (Math.abs(codexDelta) >= 20) {
      markers.push({
        date: curr.date,
        type: "source_share_shift",
        direction: codexDelta > 0 ? "up" : "down",
        delta_codex_share_pct: codexDelta,
        prev_codex_share_pct: prev.source_share_pct[Platform.CODEX],
        curr_codex_share_pct: curr.source_share_pct[Platform.CODEX],
        magnitude: Math.abs(codexDelta),
      });
    }
  }

  markers.sort((a, b) => b.magnitude - a.magnitude || a.date.localeCompare(b.date));
  return markers.slice(0, 10);
}

export async function computeTrendMetrics(
  client: Client,
  platform?: Platform,
): Promise<Record<string, any>> {
  const humanMsgs = await queryMessages(client, { role: Role.HUMAN, platform });

  const dailyBuckets = new Map<string, MessageRow[]>();
  const weeklyBuckets = new Map<string, MessageRow[]>();

  for (const msg of humanMsgs) {
    const d = new Date(msg.timestamp);
    const dayKey = `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
    const dayOfWeek = (d.getDay() + 6) % 7; // 0=Mon
    const weekStart = new Date(d);
    weekStart.setDate(d.getDate() - dayOfWeek);
    const weekKey = `${weekStart.getFullYear()}-${String(weekStart.getMonth() + 1).padStart(2, "0")}-${String(weekStart.getDate()).padStart(2, "0")}`;

    if (!dailyBuckets.has(dayKey)) dailyBuckets.set(dayKey, []);
    dailyBuckets.get(dayKey)!.push(msg);

    if (!weeklyBuckets.has(weekKey)) weeklyBuckets.set(weekKey, []);
    weeklyBuckets.get(weekKey)!.push(msg);
  }

  const dailyRollups = buildTrendRollups(dailyBuckets, "date");
  const weeklyRollups = buildTrendRollups(weeklyBuckets, "week_start");

  // Enrich weekly rollups with NLP score averages
  await enrichWeeklyNlp(client, weeklyRollups, platform);

  return {
    daily_rollups: dailyRollups,
    weekly_rollups: weeklyRollups,
    deltas_7d_vs_30d: computeTrendDeltas(dailyRollups),
    shift_markers: detectShiftMarkers(dailyRollups),
  };
}

async function enrichWeeklyNlp(
  client: Client,
  weeklyRollups: Record<string, any>[],
  platform?: Platform,
): Promise<void> {
  const pf = platformFilter(platform);
  const rs = await client.execute({
    sql: `SELECT
      strftime('%Y-%m-%d', date(m.local_date, 'weekday 0', '-6 days')) as week_start,
      AVG(e.hitl_score) as hitl,
      AVG(e.vibe_score) as vibe,
      AVG(e.precision_score) as precision,
      AVG(e.curiosity_score) as curiosity,
      AVG(e.tenacity_score) as tenacity,
      AVG(e.trust_score) as trust
    FROM nlp_enrichments e
    JOIN messages m ON e.message_id = m.id
    WHERE m.role = 'human' AND e.hitl_score IS NOT NULL${pf.clause}
    GROUP BY week_start
    ORDER BY week_start`,
    args: pf.args,
  });

  const nlpByWeek = new Map<string, Record<string, number>>();
  for (const row of rs.rows) {
    nlpByWeek.set(String(row.week_start), {
      hitl_score: row.hitl != null ? round(Number(row.hitl), 1) : 0,
      vibe_coder_index: row.vibe != null ? round(Number(row.vibe), 1) : 0,
      precision: row.precision != null ? round(Number(row.precision), 1) : 0,
      curiosity: row.curiosity != null ? round(Number(row.curiosity), 1) : 0,
      tenacity: row.tenacity != null ? round(Number(row.tenacity), 1) : 0,
      trust: row.trust != null ? round(Number(row.trust), 1) : 0,
    });
  }

  for (const rollup of weeklyRollups) {
    rollup.nlp = nlpByWeek.get(rollup.week_start) || null;
  }
}

function round(n: number, decimals: number): number {
  const factor = 10 ** decimals;
  return Math.round(n * factor) / factor;
}
