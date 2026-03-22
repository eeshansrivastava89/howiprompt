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
    "SELECT m.id, m.content FROM messages m LEFT JOIN nlp_enrichments e ON m.id = e.message_id WHERE m.role = 'human' AND e.message_id IS NULL",
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

export async function computeNlpMetrics(
  client: Client,
  platform?: Platform,
): Promise<Record<string, any>> {
  const pf = platformFilter(platform);

  const totalResult = await client.execute({
    sql: `SELECT COUNT(*) as cnt FROM nlp_enrichments e JOIN messages m ON e.message_id = m.id WHERE m.role = 'human'${pf.clause}`,
    args: pf.args,
  });
  const total = Number(totalResult.rows[0].cnt);

  if (total === 0) {
    return {
      intent: { method: "deterministic_rules_v1", counts: {}, rates_pct: {}, top_intents: [], confidence: { mean: 0, min: 0, max: 0 } },
      complexity: { method: "heuristic_complexity_v1", avg_score: 0, p50_score: 0, p90_score: 0, distribution: { low: 0, medium: 0, high: 0 }, confidence: { mean: 0, min: 0, max: 0 } },
      iteration_style: { method: "iteration_markers_v1", avg_score: 0, distribution: { low: 0, medium: 0, high: 0 }, style: "balanced", confidence: { mean: 0, min: 0, max: 0 } },
    };
  }

  // Intent
  const intentRows = await client.execute({
    sql: `SELECT e.intent, COUNT(*) as cnt FROM nlp_enrichments e JOIN messages m ON e.message_id = m.id WHERE m.role = 'human'${pf.clause} GROUP BY e.intent ORDER BY cnt DESC, e.intent`,
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
    sql: `SELECT AVG(e.intent_confidence) as avg_c, MIN(e.intent_confidence) as min_c, MAX(e.intent_confidence) as max_c FROM nlp_enrichments e JOIN messages m ON e.message_id = m.id WHERE m.role = 'human'${pf.clause}`,
    args: pf.args,
  });

  // Complexity
  const complexityAgg = await client.execute({
    sql: `SELECT AVG(e.complexity_score) as avg_s, SUM(CASE WHEN e.complexity_score < 2.5 THEN 1 ELSE 0 END) as low, SUM(CASE WHEN e.complexity_score >= 2.5 AND e.complexity_score < 3.8 THEN 1 ELSE 0 END) as med, SUM(CASE WHEN e.complexity_score >= 3.8 THEN 1 ELSE 0 END) as high, AVG(e.complexity_confidence) as avg_c, MIN(e.complexity_confidence) as min_c, MAX(e.complexity_confidence) as max_c FROM nlp_enrichments e JOIN messages m ON e.message_id = m.id WHERE m.role = 'human'${pf.clause}`,
    args: pf.args,
  });

  const sortedComplexity = await client.execute({
    sql: `SELECT e.complexity_score FROM nlp_enrichments e JOIN messages m ON e.message_id = m.id WHERE m.role = 'human'${pf.clause} ORDER BY e.complexity_score`,
    args: pf.args,
  });
  const scores = sortedComplexity.rows.map((r) => Number(r.complexity_score));
  const n = scores.length;
  let p50 = 0, p90 = 0;
  if (n > 0) {
    if (n % 2 === 1) {
      p50 = scores[Math.floor(n / 2)];
    } else {
      p50 = (scores[n / 2 - 1] + scores[n / 2]) / 2;
    }
    p90 = scores[Math.max(0, Math.floor(n * 0.9) - 1)];
  }

  // Iteration
  const iterAgg = await client.execute({
    sql: `SELECT AVG(e.iteration_score) as avg_s, SUM(CASE WHEN e.iteration_score < 25 THEN 1 ELSE 0 END) as low, SUM(CASE WHEN e.iteration_score >= 25 AND e.iteration_score < 60 THEN 1 ELSE 0 END) as med, SUM(CASE WHEN e.iteration_score >= 60 THEN 1 ELSE 0 END) as high, AVG(e.iteration_confidence) as avg_c, MIN(e.iteration_confidence) as min_c, MAX(e.iteration_confidence) as max_c FROM nlp_enrichments e JOIN messages m ON e.message_id = m.id WHERE m.role = 'human'${pf.clause}`,
    args: pf.args,
  });

  const avgIteration = round(Number(iterAgg.rows[0].avg_s), 1);
  const iterStyle = avgIteration >= 60 ? "highly_iterative" : avgIteration >= 30 ? "balanced_iterative" : "direct";

  const ca = complexityAgg.rows[0];
  const ic = intentConf.rows[0];
  const ia = iterAgg.rows[0];

  // HITL Score
  const hitlAgg = await client.execute({
    sql: `SELECT AVG(e.hitl_score) as avg_s, SUM(CASE WHEN e.hitl_score < 33 THEN 1 ELSE 0 END) as low, SUM(CASE WHEN e.hitl_score >= 33 AND e.hitl_score < 66 THEN 1 ELSE 0 END) as med, SUM(CASE WHEN e.hitl_score >= 66 THEN 1 ELSE 0 END) as high, AVG(e.hitl_confidence) as avg_c, MIN(e.hitl_confidence) as min_c, MAX(e.hitl_confidence) as max_c FROM nlp_enrichments e JOIN messages m ON e.message_id = m.id WHERE m.role = 'human' AND e.hitl_score IS NOT NULL${pf.clause}`,
    args: pf.args,
  });
  const ha = hitlAgg.rows[0];
  const hasHitl = ha.avg_s != null;

  // Vibe Index
  const vibeAgg = await client.execute({
    sql: `SELECT AVG(e.vibe_score) as avg_s, SUM(CASE WHEN e.vibe_score < 30 THEN 1 ELSE 0 END) as vibe, SUM(CASE WHEN e.vibe_score >= 30 AND e.vibe_score < 70 THEN 1 ELSE 0 END) as balanced, SUM(CASE WHEN e.vibe_score >= 70 THEN 1 ELSE 0 END) as engineer, AVG(e.vibe_confidence) as avg_c, MIN(e.vibe_confidence) as min_c, MAX(e.vibe_confidence) as max_c FROM nlp_enrichments e JOIN messages m ON e.message_id = m.id WHERE m.role = 'human' AND e.vibe_score IS NOT NULL${pf.clause}`,
    args: pf.args,
  });
  const va = vibeAgg.rows[0];
  const hasVibe = va.avg_s != null;

  const avgVibe = hasVibe ? round(Number(va.avg_s), 1) : 0;
  const vibeLabel = avgVibe >= 70 ? "engineer" : avgVibe >= 50 ? "balanced_engineer" : avgVibe >= 30 ? "balanced_vibe" : "vibe_coder";

  return {
    intent: {
      method: "deterministic_rules_v1",
      counts: intentCounts,
      rates_pct: intentRates,
      top_intents: topIntents,
      confidence: { mean: round(Number(ic.avg_c), 2), min: round(Number(ic.min_c), 2), max: round(Number(ic.max_c), 2) },
    },
    complexity: {
      method: "heuristic_complexity_v1",
      avg_score: round(Number(ca.avg_s), 1),
      p50_score: round(p50, 1),
      p90_score: round(p90, 1),
      distribution: { low: Number(ca.low), medium: Number(ca.med), high: Number(ca.high) },
      confidence: { mean: round(Number(ca.avg_c), 2), min: round(Number(ca.min_c), 2), max: round(Number(ca.max_c), 2) },
    },
    iteration_style: {
      method: "iteration_markers_v1",
      avg_score: avgIteration,
      distribution: { low: Number(ia.low), medium: Number(ia.med), high: Number(ia.high) },
      style: iterStyle,
      confidence: { mean: round(Number(ia.avg_c), 2), min: round(Number(ia.min_c), 2), max: round(Number(ia.max_c), 2) },
    },
    hitl_score: {
      method: "embedding_similarity_v1",
      avg_score: hasHitl ? round(Number(ha.avg_s), 1) : null,
      distribution: hasHitl ? { low: Number(ha.low), medium: Number(ha.med), high: Number(ha.high) } : null,
      confidence: hasHitl ? { mean: round(Number(ha.avg_c), 2), min: round(Number(ha.min_c), 2), max: round(Number(ha.max_c), 2) } : null,
    },
    vibe_coder_index: {
      method: "embedding_similarity_v1",
      avg_score: hasVibe ? avgVibe : null,
      label: hasVibe ? vibeLabel : null,
      distribution: hasVibe ? { vibe: Number(va.vibe), balanced: Number(va.balanced), engineer: Number(va.engineer) } : null,
      confidence: hasVibe ? { mean: round(Number(va.avg_c), 2), min: round(Number(va.min_c), 2), max: round(Number(va.max_c), 2) } : null,
    },
  };
}

function round(n: number, decimals: number): number {
  const factor = 10 ** decimals;
  return Math.round(n * factor) / factor;
}
