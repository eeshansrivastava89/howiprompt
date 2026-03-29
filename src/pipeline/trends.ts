import type { Client } from "@libsql/client";
import { queryMessages, platformFilter, type MessageRow } from "./db.js";
import { PLATFORM_VALUES, type Platform, Role } from "./models.js";
import { BACKTRACK_PATTERNS, COMMAND_PATTERN } from "./nlp.js";

function countPatternMatches(text: string, pattern: RegExp): number {
  const matches = text.match(pattern);
  pattern.lastIndex = 0;
  return matches?.length ?? 0;
}

export function computeStyleSnapshot(messages: MessageRow[]): Record<string, number> {
  const total = messages.length;
  if (total === 0) {
    return { backtrack_per_100: 0, question_rate_pct: 0, command_rate_pct: 0 };
  }

  let btCount = 0;
  for (const pattern of Object.values(BACKTRACK_PATTERNS)) {
    for (const m of messages) btCount += countPatternMatches(m.content, pattern);
  }

  const qCount = messages.filter((m) => m.content.trim().endsWith("?")).length;
  const cCount = messages.filter((m) => COMMAND_PATTERN.test(m.content.trim())).length;
  COMMAND_PATTERN.lastIndex = 0;

  return {
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

    const sourceCounts: Record<string, number> = {};
    const modelCounts: Record<string, number> = {};

    for (const m of msgs) {
      sourceCounts[m.platform] = (sourceCounts[m.platform] ?? 0) + 1;
      if (m.modelId || m.modelProvider) {
        const mid = m.modelId ?? "unknown";
        modelCounts[mid] = (modelCounts[mid] ?? 0) + 1;
      }
    }

    const sourceSharePct: Record<string, number> = {};
    for (const src of Object.keys(sourceCounts).sort()) {
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

  function dominantSourceShare(item: any): number {
    return Math.max(0, ...Object.values(item.source_share_pct || {}).map((value) => Number(value) || 0));
  }

  const selectors: Record<string, (item: any) => number> = {
    prompts_per_day: (item) => item.prompts,
    dominant_source_share_pct: dominantSourceShare,
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

    const sources = new Set([
      ...Object.keys(prev.source_share_pct || {}),
      ...Object.keys(curr.source_share_pct || {}),
    ]);
    for (const source of sources) {
      const shareDelta = round(
        Number(curr.source_share_pct?.[source] ?? 0) - Number(prev.source_share_pct?.[source] ?? 0),
        1,
      );
      if (Math.abs(shareDelta) < 20) continue;
      markers.push({
        date: curr.date,
        type: "source_share_shift",
        source,
        direction: shareDelta > 0 ? "up" : "down",
        delta_source_share_pct: shareDelta,
        prev_source_share_pct: Number(prev.source_share_pct?.[source] ?? 0),
        curr_source_share_pct: Number(curr.source_share_pct?.[source] ?? 0),
        magnitude: Math.abs(shareDelta),
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

  const result: Record<string, any> = {
    daily_rollups: dailyRollups,
    weekly_rollups: weeklyRollups,
    deltas_7d_vs_30d: computeTrendDeltas(dailyRollups),
    shift_markers: detectShiftMarkers(dailyRollups),
  };

  // When showing all platforms, add per-platform weekly breakdowns
  if (!platform) {
    const byPlatform: Record<string, Record<string, any>[]> = {};

    for (const plat of PLATFORM_VALUES) {
      const platMsgs = humanMsgs.filter((m) => m.platform === plat);
      if (platMsgs.length === 0) continue;

      const platWeekly = new Map<string, MessageRow[]>();
      for (const msg of platMsgs) {
        const d = new Date(msg.timestamp);
        const dayOfWeek = (d.getDay() + 6) % 7;
        const weekStart = new Date(d);
        weekStart.setDate(d.getDate() - dayOfWeek);
        const weekKey = `${weekStart.getFullYear()}-${String(weekStart.getMonth() + 1).padStart(2, "0")}-${String(weekStart.getDate()).padStart(2, "0")}`;
        if (!platWeekly.has(weekKey)) platWeekly.set(weekKey, []);
        platWeekly.get(weekKey)!.push(msg);
      }

      const rollups = buildTrendRollups(platWeekly, "week_start");
      await enrichWeeklyNlp(client, rollups, plat);
      byPlatform[plat] = rollups;
    }

    result.weekly_by_platform = byPlatform;
  }

  return result;
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
      AVG(e.politeness_score) as politeness
    FROM nlp_enrichments e
    JOIN messages m ON e.message_id = m.id
    WHERE m.role = 'human' AND m.is_excluded = 0 AND e.hitl_score IS NOT NULL${pf.clause}
    GROUP BY week_start
    ORDER BY week_start`,
    args: pf.args,
  });

  const nlpByWeek = new Map<string, Record<string, number | null>>();
  for (const row of rs.rows) {
    nlpByWeek.set(String(row.week_start), {
      hitl_score: row.hitl != null ? round(Number(row.hitl), 1) : 0,
      vibe_coder_index: row.vibe != null ? round(Number(row.vibe), 1) : 0,
      politeness: row.politeness != null ? round(Number(row.politeness), 1) : null,
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
