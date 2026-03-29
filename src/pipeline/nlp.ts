import type { Client } from "@libsql/client";
import { insertNlpEnrichments, platformFilter, type MessageRow } from "./db.js";
import type { Platform } from "./models.js";

export const POLITENESS_PATTERNS: Record<string, RegExp> = {
  please: /\bplease\b/gi,
  thanks: /\b(thanks|thank you|thx)\b/gi,
  sorry: /\b(sorry|apologies|apologize)\b/gi,
};

export const BACKTRACK_PATTERNS: Record<string, RegExp> = {
  actually: /\bactually\b/gi,
  wait: /\bwait\b/gi,
  never_mind: /\b(never\s*mind|nevermind)\b/gi,
  scratch_that: /\b(scratch that|ignore that)\b/gi,
};

export const COMMAND_PATTERN =
  /^(please\s+)?(do|make|create|write|build|add|fix|update|change|remove|delete|show|run|help|can you|could you|would you|tell|explain|find|search|get|set|check|test|debug|implement|refactor)\b/i;

export const INTENT_PATTERNS: Record<string, RegExp[]> = {
  debug_fix: [/\b(debug|fix|error|bug|failing|broken|traceback|stack trace)\b/i],
  build_feature: [/\b(build|create|implement|add|ship|feature|integrate)\b/i],
  analysis_research: [/\b(analyze|audit|review|compare|benchmark|research|investigate)\b/i],
  explanation_learning: [/\b(explain|why|how does|teach|walk me through|help me understand)\b/i],
  planning_strategy: [/\b(plan|roadmap|milestone|phase|next steps|strategy)\b/i],
  ops_commands: [/\b(run|execute|command|script|deploy|release|push|tag)\b/i],
};

const ITERATION_MARKERS: RegExp[] = [
  /\bactually\b/gi,
  /\bwait\b/gi,
  /\binstead\b/gi,
  /\bchange\b/gi,
  /\bupdate\b/gi,
  /\bfix\b/gi,
  /\brevise\b/gi,
  /\bretry\b/gi,
  /\bagain\b/gi,
  /\bdifferent\b/gi,
  /\bscratch that\b/gi,
  /\brework\b/gi,
];

export function classifyIntent(text: string): { intent: string; confidence: number } {
  const scores: Record<string, number> = {};
  for (const [intent, patterns] of Object.entries(INTENT_PATTERNS)) {
    let score = 0;
    for (const pattern of patterns) {
      if (pattern.test(text)) score++;
      pattern.lastIndex = 0;
    }
    scores[intent] = score;
  }

  const entries = Object.entries(scores).sort((a, b) => b[1] - a[1]);
  const topIntent = entries[0][0];
  const topScore = entries[0][1];
  const secondScore = entries.length > 1 ? entries[1][1] : 0;

  if (topScore === 0) return { intent: "other", confidence: 0.5 };

  let confidence = 0.6 + Math.min(topScore, 3) * 0.1;
  if (topScore > secondScore) confidence += 0.1;
  return { intent: topIntent, confidence: round(Math.min(confidence, 0.95), 2) };
}

export function computeComplexity(text: string): { score: number; confidence: number } {
  const content = text.trim();
  const words = content.split(/\s+/).filter(Boolean).length;
  let score = 1.0;

  if (words >= 15) score += 0.7;
  if (words >= 35) score += 0.8;
  if (words >= 70) score += 0.8;

  if (content.includes("\n")) score += 0.5;
  if ((content.match(/,/g)?.length ?? 0) >= 2 || content.includes(";")) score += 0.4;
  if (/[{}`]|--|\.\/|=|->/.test(content)) score += 0.5;

  const constraintHits = (
    content.match(/\b(must|should|without|exactly|at least|at most|step|checklist|constraint)\b/gi) ?? []
  ).length;
  score += Math.min(constraintHits, 3) * 0.2;

  const finalScore = round(Math.min(score, 5.0), 1);
  let confidence = 0.65;
  if (words >= 20) confidence += 0.1;
  if (constraintHits > 0) confidence += 0.1;
  if (content.includes("\n") || /[{}`]|--|\.\/|=|->/.test(content)) confidence += 0.05;

  return { score: finalScore, confidence: round(Math.min(confidence, 0.95), 2) };
}

export function computeIterationStyle(text: string): { score: number; confidence: number } {
  let markerHits = 0;
  for (const pattern of ITERATION_MARKERS) {
    markerHits += (text.match(pattern) ?? []).length;
    pattern.lastIndex = 0;
  }

  let score = markerHits * 20;
  if (text.includes("?")) score += 8;
  if (/\b(again|retry|revise|change|update|different)\b/i.test(text)) score += 8;

  const finalScore = round(Math.min(score, 100), 1);
  let confidence = 0.6 + Math.min(markerHits, 3) * 0.1;
  if (text.includes("?")) confidence += 0.05;

  return { score: finalScore, confidence: round(Math.min(confidence, 0.95), 2) };
}

export async function enrichNlp(client: Client): Promise<number> {
  const result = await client.execute(
    "SELECT m.id, m.content FROM messages m LEFT JOIN nlp_enrichments e ON m.id = e.message_id WHERE m.role = 'human' AND m.is_excluded = 0 AND e.message_id IS NULL",
  );

  const enrichments = result.rows.map((row) => {
    const content = String(row.content);
    const intent = classifyIntent(content);
    const complexity = computeComplexity(content);
    const iteration = computeIterationStyle(content);
    return {
      messageId: Number(row.id),
      intent: intent.intent,
      intentConfidence: intent.confidence,
      complexityScore: complexity.score,
      complexityConfidence: complexity.confidence,
      iterationScore: iteration.score,
      iterationConfidence: iteration.confidence,
    };
  });

  if (enrichments.length > 0) {
    await insertNlpEnrichments(client, enrichments);
  }
  return enrichments.length;
}

async function aggregateClassifier(
  client: Client,
  column: string,
  pf: { clause: string; args: any[] },
): Promise<{ avg_score: number | null; confidence: { mean: number; min: number; max: number } | null }> {
  const result = await client.execute({
    sql: `SELECT AVG(e.${column}) as avg_s, AVG(e.${column.replace("_score", "_confidence")}) as avg_c, MIN(e.${column.replace("_score", "_confidence")}) as min_c, MAX(e.${column.replace("_score", "_confidence")}) as max_c FROM nlp_enrichments e JOIN messages m ON e.message_id = m.id WHERE m.role = 'human' AND m.is_excluded = 0 AND e.${column} IS NOT NULL${pf.clause}`,
    args: pf.args,
  });
  const r = result.rows[0];
  if (r.avg_s == null) return { avg_score: null, confidence: null };
  return {
    avg_score: round(Number(r.avg_s), 1),
    confidence: { mean: round(Number(r.avg_c), 2), min: round(Number(r.min_c), 2), max: round(Number(r.max_c), 2) },
  };
}

export async function computeNlpMetrics(
  client: Client,
  platform?: Platform,
): Promise<Record<string, any>> {
  const pf = platformFilter(platform);

  const totalResult = await client.execute({
    sql: `SELECT COUNT(*) as cnt FROM nlp_enrichments e JOIN messages m ON e.message_id = m.id WHERE m.role = 'human' AND m.is_excluded = 0${pf.clause}`,
    args: pf.args,
  });
  const total = Number(totalResult.rows[0].cnt);

  const emptyClassifier = { method: "embedding_similarity_v1", avg_score: null, confidence: null };

  if (total === 0) {
    return {
      intent: { method: "deterministic_rules_v1", counts: {}, rates_pct: {}, top_intents: [], confidence: { mean: 0, min: 0, max: 0 } },
      hitl_score: emptyClassifier,
      vibe_coder_index: emptyClassifier,
      politeness: emptyClassifier,
    };
  }

  // Intent
  const intentRows = await client.execute({
    sql: `SELECT e.intent, COUNT(*) as cnt FROM nlp_enrichments e JOIN messages m ON e.message_id = m.id WHERE m.role = 'human' AND m.is_excluded = 0${pf.clause} GROUP BY e.intent ORDER BY cnt DESC, e.intent`,
    args: pf.args,
  });
  const intentCounts: Record<string, number> = {};
  const intentRates: Record<string, number> = {};
  const topIntents: any[] = [];
  for (const row of intentRows.rows) {
    const intent = String(row.intent);
    const count = Number(row.cnt);
    intentCounts[intent] = count;
    intentRates[intent] = round((count / total) * 100, 1);
  }
  for (const row of intentRows.rows.slice(0, 3)) {
    const intent = String(row.intent);
    topIntents.push({ intent, count: intentCounts[intent], rate_pct: intentRates[intent] });
  }

  const intentConf = await client.execute({
    sql: `SELECT AVG(e.intent_confidence) as avg_c, MIN(e.intent_confidence) as min_c, MAX(e.intent_confidence) as max_c FROM nlp_enrichments e JOIN messages m ON e.message_id = m.id WHERE m.role = 'human' AND m.is_excluded = 0${pf.clause}`,
    args: pf.args,
  });
  const ic = intentConf.rows[0];

  // Aggregate hero embedding classifiers in parallel
  const [hitl, vibe, politeness] = await Promise.all([
    aggregateClassifier(client, "hitl_score", pf),
    aggregateClassifier(client, "vibe_score", pf),
    aggregateClassifier(client, "politeness_score", pf),
  ]);

  return {
    intent: {
      method: "deterministic_rules_v1",
      counts: intentCounts,
      rates_pct: intentRates,
      top_intents: topIntents,
      confidence: { mean: round(Number(ic.avg_c), 2), min: round(Number(ic.min_c), 2), max: round(Number(ic.max_c), 2) },
    },
    hitl_score: { method: "embedding_similarity_v1", ...hitl },
    vibe_coder_index: { method: "embedding_similarity_v1", ...vibe },
    politeness: { method: "embedding_similarity_v1", ...politeness },
  };
}

function round(n: number, decimals: number): number {
  const factor = 10 ** decimals;
  return Math.round(n * factor) / factor;
}
