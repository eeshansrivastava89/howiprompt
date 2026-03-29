// style.ts — Data-driven 2×2 persona scoring.
// Detail Level (brief→detailed) × Communication Style (directive→collaborative)
// Patterns validated on 21k prompts in persona-2-redesign.ipynb

import type { Client } from "@libsql/client";

// --- Interaction-move markers (refined from pattern audit on 1k prompts) ---

const MARKERS: Record<string, RegExp[]> = {
  correction: [
    /\bactually\b/i, /\bwait\b/i, /\binstead\b/i, /\bwrong\b/i,
    /\bstop\b/i, /\bno[,.]?\s+(don't|do not|that's not|not what)/i,
    /\bgo back\b/i, /\bthat's wrong\b/i,
  ],
  inquiry: [
    /\bwhy\b/i, /\bhow (do|can|to|would|does|should|is)\b/i,
    /\bwhat if\b/i, /\bwhat about\b/i, /\bcompare\b/i,
    /\bexplain\b/i, /\bshould (i|we)\b/i,
    /\bwhat's the difference\b/i, /\bis there a (way|better)\b/i,
  ],
  review: [
    /\bcheck\b/i, /\bverify\b/i, /\btest\b/i, /\bvalidate\b/i,
    /\berror\b/i, /\bfix\b/i, /\bdebug\b/i, /\btraceback\b/i, /\breview\b/i,
  ],
  constraint: [
    /\bmust\b/i, /\bdon't\b/i, /\bnever\b/i, /\balways\b/i,
    /\bwithout\b/i, /\bexactly\b/i, /\bonly\b/i, /\bat least\b/i,
  ],
};

const TASK_MARKERS: Record<string, RegExp[]> = {
  task_debug: [
    /\berror\b/i, /\btraceback\b/i, /\bcrash(ed|es|ing)?\b/i,
    /\bfailing\b/i, /\bbroken\b/i, /\bexception\b/i,
    /\bnot working\b/i, /\bdoesn't work\b/i, /\bdoes not work\b/i,
  ],
  task_explain: [
    /\bexplain\b/i, /\bwhat does\b/i, /\bhow does\b/i, /\bwhat is\b/i,
    /\bwalk me through\b/i, /\bhelp me understand\b/i,
    /\bwhy does\b/i, /\bwhy is\b/i,
  ],
  task_generate: [
    /\bcreate\b/i, /\bbuild\b/i, /\bwrite\b/i, /\bimplement\b/i,
    /\bgenerate\b/i, /\bset up\b/i, /\bsetup\b/i,
  ],
  task_refactor: [
    /\brefactor\b/i, /\bclean up\b/i, /\brename\b/i,
    /\brestructure\b/i, /\bsimplify\b/i,
  ],
};

// Directive vs Collaborative — the Y-axis of the 2×2
const STYLE_MARKERS: Record<string, RegExp[]> = {
  directive: [
    /^(yes|no|ok|okay|yep|nope|sure)\b/i,
    /\bjust\b/i, /\bi want\b/i, /\bi need\b/i,
    /\bgive me\b/i, /\btell me\b/i, /\bshow me\b/i,
    /\bdo (this|that|it)\b/i, /\bgo ahead\b/i,
  ],
  collaborative: [
    /\bcan you\b/i, /\bcould you\b/i, /\bwould you\b/i,
    /\bi think\b/i, /\bmaybe\b/i, /\bperhaps\b/i,
    /\blet's\b/i, /\bwe should\b/i, /\bwe could\b/i,
    /\bwhat do you think\b/i, /\bany (suggestions?|ideas?|thoughts?)\b/i,
    /\bhelp\b/i,
  ],
};

const FILE_REF_RE = /[\w\-./]+\.(js|ts|py|go|rs|java|sql|sh|json|yaml|md|html|css)\b/g;
const LIST_MARKER_RE = /^\s*[-*\d]+\.?\s/gm;
const LINE_REF_RE = /line\s+\d+/i;
const THANKS_RE = /\b(thanks|thank you|thx)\b/i;

export interface StyleFeatures {
  word_count: number;
  char_count: number;
  newline_count: number;
  has_code_block: number;
  question_count: number;
  list_marker_count: number;
  file_ref_count: number;
  has_line_ref: number;
  correction_count: number;
  inquiry_count: number;
  review_count: number;
  constraint_count: number;
  task_debug_count: number;
  task_explain_count: number;
  task_generate_count: number;
  task_refactor_count: number;
  directive_count: number;
  collaborative_count: number;
  has_please: number;
  has_thanks: number;
}

function countPatterns(text: string, patterns: RegExp[]): number {
  let total = 0;
  for (const p of patterns) {
    const m = text.match(new RegExp(p.source, p.flags + (p.flags.includes("g") ? "" : "g")));
    total += m ? m.length : 0;
  }
  return total;
}

export function extractFeatures(content: string, wordCount: number): StyleFeatures {
  const t = content.toLowerCase();
  return {
    word_count: wordCount || content.split(/\s+/).length,
    char_count: content.length,
    newline_count: (content.match(/\n/g) || []).length,
    has_code_block: content.includes("```") ? 1 : 0,
    question_count: (content.match(/\?/g) || []).length,
    list_marker_count: (content.match(LIST_MARKER_RE) || []).length,
    file_ref_count: (content.match(FILE_REF_RE) || []).length,
    has_line_ref: LINE_REF_RE.test(t) ? 1 : 0,
    correction_count: countPatterns(t, MARKERS.correction),
    inquiry_count: countPatterns(t, MARKERS.inquiry),
    review_count: countPatterns(t, MARKERS.review),
    constraint_count: countPatterns(t, MARKERS.constraint),
    task_debug_count: countPatterns(t, TASK_MARKERS.task_debug),
    task_explain_count: countPatterns(t, TASK_MARKERS.task_explain),
    task_generate_count: countPatterns(t, TASK_MARKERS.task_generate),
    task_refactor_count: countPatterns(t, TASK_MARKERS.task_refactor),
    directive_count: countPatterns(t, STYLE_MARKERS.directive),
    collaborative_count: countPatterns(t, STYLE_MARKERS.collaborative),
    has_please: t.includes("please") ? 1 : 0,
    has_thanks: THANKS_RE.test(t) ? 1 : 0,
  };
}

// Detail score: sum of structural features, sigmoid-mapped to 0–100.
// Calibrated on 21k prompts: median raw ~= −1.6, 95th pctl ~= 9.
export function computeDetailScore(f: StyleFeatures): number {
  const raw =
    zscore(f.word_count, 41, 74) +
    zscore(f.char_count, 198, 446) +
    zscore(f.newline_count, 2.4, 6.8) +
    zscore(f.file_ref_count, 0.3, 1.2) +
    f.has_code_block +
    f.has_line_ref +
    zscore(f.constraint_count, 0.7, 1.5) +
    zscore(f.list_marker_count, 0.2, 0.9);
  return sigmoid(raw, 0, 4) * 100;
}

// Style score: collaborative minus directive signals, sigmoid-mapped to 0–100.
// Calibrated on 21k prompts: median raw ~= −1.1.
export function computeStyleScore(f: StyleFeatures): number {
  const raw =
    zscore(f.collaborative_count, 0.4, 0.9) +
    (f.collaborative_count > 0 ? 1 : 0) +
    f.has_please +
    f.has_thanks +
    zscore(f.question_count, 0.4, 0.9) +
    zscore(f.inquiry_count, 0.3, 0.8) -
    zscore(f.directive_count, 0.6, 1.1) -
    (f.directive_count > 0 ? 1 : 0);
  return sigmoid(raw, 0, 3) * 100;
}

export type Quadrant =
  | "Brief + Directive"
  | "Brief + Collaborative"
  | "Detailed + Directive"
  | "Detailed + Collaborative";

export interface PersonaDefinition {
  name: string;
  quadrant: Quadrant;
  description: string;
  traits: string[];
}

export const PERSONAS: Record<Quadrant, PersonaDefinition> = {
  "Brief + Directive": {
    name: "The Commander",
    quadrant: "Brief + Directive",
    description: "You give precise orders in short bursts. No wasted words, no hand-holding — you know exactly what you want and expect the AI to deliver it on the first try.",
    traits: ["Direct", "Efficient", "Decisive"],
  },
  "Brief + Collaborative": {
    name: "The Partner",
    quadrant: "Brief + Collaborative",
    description: "You keep it short but keep it human. Quick exchanges, polite nudges, and a conversational flow that treats AI as a co-pilot, not a tool.",
    traits: ["Conversational", "Collaborative", "Adaptive"],
  },
  "Detailed + Directive": {
    name: "The Architect",
    quadrant: "Detailed + Directive",
    description: "You write specs, not prompts. Every request comes with constraints, file paths, and numbered requirements. You leave nothing to chance.",
    traits: ["Precise", "Methodical", "Thorough"],
  },
  "Detailed + Collaborative": {
    name: "The Explorer",
    quadrant: "Detailed + Collaborative",
    description: "You bring context and ask questions. Every conversation is an investigation — you explain what you're thinking, ask for alternatives, and dig into the details together.",
    traits: ["Curious", "Analytical", "Open-minded"],
  },
};

export function classifyQuadrant(detailScore: number, styleScore: number): Quadrant {
  const detailed = detailScore >= 50;
  const collaborative = styleScore >= 50;
  if (detailed && collaborative) return "Detailed + Collaborative";
  if (detailed) return "Detailed + Directive";
  if (collaborative) return "Brief + Collaborative";
  return "Brief + Directive";
}

export function getPersona(detailScore: number, styleScore: number): PersonaDefinition {
  return PERSONAS[classifyQuadrant(detailScore, styleScore)];
}

// --- Pipeline integration ---

export async function enrichStyle(
  client: Client,
  onBatchProgress?: (done: number, total: number) => void,
): Promise<number> {
  const result = await client.execute(
    `SELECT m.id, m.content, m.word_count FROM messages m
     JOIN nlp_enrichments e ON m.id = e.message_id
     WHERE m.role = 'human' AND m.is_excluded = 0
       AND e.detail_score IS NULL`,
  );

  if (result.rows.length === 0) return 0;

  const batchSize = 500;
  let done = 0;

  for (let i = 0; i < result.rows.length; i += batchSize) {
    const batch = result.rows.slice(i, i + batchSize);
    const stmts = batch.map((row) => {
      const features = extractFeatures(String(row.content), Number(row.word_count));
      const detail = round(computeDetailScore(features), 1);
      const style = round(computeStyleScore(features), 1);
      const quadrant = classifyQuadrant(detail, style);
      return {
        sql: `UPDATE nlp_enrichments SET detail_score = ?, style_score = ?, quadrant = ? WHERE message_id = ?`,
        args: [detail, style, quadrant, Number(row.id)],
      };
    });

    await client.batch(stmts, "write");
    done += batch.length;
    onBatchProgress?.(done, result.rows.length);
  }

  return done;
}

// --- Helpers ---

function zscore(value: number, mean: number, std: number): number {
  return std > 0 ? (value - mean) / std : 0;
}

function sigmoid(x: number, center: number, scale: number): number {
  return 1 / (1 + Math.exp(-(x - center) / scale));
}

function round(n: number, decimals: number): number {
  const f = 10 ** decimals;
  return Math.round(n * f) / f;
}
