// scorers.ts — Calibrated vibe and politeness scoring.
// Uses logistic regression coefficients exported from notebook (config/classifiers.json).
// No model download needed — scoring is a dot product on extracted features.

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import type { Client } from "@libsql/client";
import { extractFeatures, type StyleFeatures } from "./style.js";

interface ClassifierConfig {
  features: string[];
  scaler: { mean: number[]; std: number[] };
  vibe: { classes: number[]; coef: number[][]; intercept: number[] };
  politeness: { classes: number[]; coef: number[][]; intercept: number[] };
}

export interface ScoredResult {
  score: number;       // 0-100
  classLabel: number;  // 1-5
  confidence: number;  // 0-1
}

let config: ClassifierConfig | null = null;

function round(n: number, decimals: number): number {
  const f = 10 ** decimals;
  return Math.round(n * f) / f;
}

export function resetScorersForTests(): void {
  config = null;
}

function loadConfig(): ClassifierConfig {
  if (config) return config;

  const __dirname = path.dirname(fileURLToPath(import.meta.url));
  const candidates = [
    path.join(__dirname, "..", "..", "config", "classifiers.json"),
    path.join(__dirname, "..", "config", "classifiers.json"),
  ];

  for (const p of candidates) {
    try {
      config = JSON.parse(fs.readFileSync(p, "utf-8")) as ClassifierConfig;
      return config;
    } catch { /* try next */ }
  }
  throw new Error("classifiers.json not found");
}

function featuresToVector(features: StyleFeatures, featureNames: string[]): number[] {
  return featureNames.map((name) => (features as unknown as Record<string, number>)[name] ?? 0);
}

function predict(
  x: number[],
  mean: number[],
  std: number[],
  coef: number[][],
  intercept: number[],
  classes: number[],
): { classLabel: number; probabilities: number[] } {
  // Standardize
  const xScaled = x.map((v, i) => (std[i] > 0 ? (v - mean[i]) / std[i] : 0));

  // Compute logits for each class
  const logits = coef.map((classWeights, c) => {
    let logit = intercept[c];
    for (let i = 0; i < xScaled.length; i++) {
      logit += xScaled[i] * classWeights[i];
    }
    return logit;
  });

  // Softmax
  const maxLogit = Math.max(...logits);
  const expLogits = logits.map((l) => Math.exp(l - maxLogit));
  const sumExp = expLogits.reduce((a, b) => a + b, 0);
  const probabilities = expLogits.map((e) => e / sumExp);

  // Predicted class
  let bestIdx = 0;
  for (let i = 1; i < probabilities.length; i++) {
    if (probabilities[i] > probabilities[bestIdx]) bestIdx = i;
  }

  return { classLabel: classes[bestIdx], probabilities };
}

function classToScore(classLabel: number, probabilities: number[], classes: number[]): number {
  // Weighted average of class labels by probability → 0-100
  let weightedLabel = 0;
  for (let i = 0; i < classes.length; i++) {
    weightedLabel += classes[i] * probabilities[i];
  }
  // Map 1-5 → 0-100
  return Math.round(((weightedLabel - 1) / 4) * 100 * 10) / 10;
}

export function scoreVibe(features: StyleFeatures): ScoredResult {
  const cfg = loadConfig();
  const x = featuresToVector(features, cfg.features);
  const { classLabel, probabilities } = predict(
    x, cfg.scaler.mean, cfg.scaler.std, cfg.vibe.coef, cfg.vibe.intercept, cfg.vibe.classes,
  );
  const score = classToScore(classLabel, probabilities, cfg.vibe.classes);
  const confidence = Math.round(Math.max(...probabilities) * 100) / 100;
  return { score, classLabel, confidence };
}

export function scorePoliteness(features: StyleFeatures): ScoredResult {
  const cfg = loadConfig();
  const x = featuresToVector(features, cfg.features);
  const { classLabel, probabilities } = predict(
    x, cfg.scaler.mean, cfg.scaler.std, cfg.politeness.coef, cfg.politeness.intercept, cfg.politeness.classes,
  );
  const score = classToScore(classLabel, probabilities, cfg.politeness.classes);
  const confidence = Math.round(Math.max(...probabilities) * 100) / 100;
  return { score, classLabel, confidence };
}

export interface ExplanationEntry {
  feature: string;
  label: string;
  contribution: number; // positive = pushes toward high score, negative = toward low
  stat: string;         // human-readable stat like "34% of prompts"
}

const FEATURE_LABELS: Record<string, string> = {
  word_count: "Prompt length",
  char_count: "Character count",
  newline_count: "Multi-line prompts",
  has_code_block: "Code blocks",
  question_count: "Questions asked",
  list_marker_count: "Bullet/numbered lists",
  file_ref_count: "File references",
  has_line_ref: "Line number references",
  correction_count: "Corrections/redirects",
  inquiry_count: "Inquiry language",
  review_count: "Review/debug language",
  constraint_count: "Constraints/requirements",
  task_debug_count: "Debug tasks",
  task_explain_count: "Explanation requests",
  task_generate_count: "Generation tasks",
  task_refactor_count: "Refactor tasks",
  directive_count: "Directive language",
  collaborative_count: "Collaborative language",
  has_please: "Says \"please\"",
  has_thanks: "Says \"thanks\"",
  code_ratio: "Code-heavy content",
  has_inline_code: "Contains code",
  is_terse: "Very short prompts",
  is_single_token: "Single-word prompts",
  is_jailbreak: "Template/boilerplate",
  is_rewrite_request: "Rewrite delegation",
  has_please_in_instruction: "Polite framing",
  has_thanks_in_instruction: "Gratitude in request",
  has_greeting: "Greeting",
  is_frustrated: "Frustrated tone",
  instruction_ratio: "Instruction vs content ratio",
  is_short_vague: "Short vague prompts",
  has_numbered_list: "Structured requirements",
  has_specific_ref: "Specific code references",
  has_question_about_code: "Questions about code",
  is_bare_directive: "Bare commands",
  is_neutral_tone: "Neutral tone",
  exclamation_count: "Exclamation marks",
};

export function explainScore(
  classifier: "vibe" | "politeness",
  avgFeatures: Record<string, number>,
  totalPrompts: number,
): ExplanationEntry[] {
  const cfg = loadConfig();
  const clsConfig = cfg[classifier];
  const x = cfg.features.map((f) => avgFeatures[f] ?? 0);
  const xScaled = x.map((v, i) => (cfg.scaler.std[i] > 0 ? (v - cfg.scaler.mean[i]) / cfg.scaler.std[i] : 0));

  // For each feature, compute its average contribution across all classes
  // weighted by the predicted class probabilities
  const logits = clsConfig.coef.map((weights: number[], c: number) => {
    let logit = clsConfig.intercept[c];
    for (let i = 0; i < xScaled.length; i++) logit += xScaled[i] * weights[i];
    return logit;
  });
  const maxLogit = Math.max(...logits);
  const expLogits = logits.map((l: number) => Math.exp(l - maxLogit));
  const sumExp = expLogits.reduce((a: number, b: number) => a + b, 0);
  const probs = expLogits.map((e: number) => e / sumExp);

  // Compute signed contribution per feature (positive = pushes toward higher class)
  const contributions: Array<{ feature: string; contribution: number }> = [];
  for (let fi = 0; fi < cfg.features.length; fi++) {
    let weightedContrib = 0;
    for (let c = 0; c < clsConfig.classes.length; c++) {
      // Contribution = scaled_feature * weight * class_label * probability
      weightedContrib += xScaled[fi] * clsConfig.coef[c][fi] * clsConfig.classes[c] * probs[c];
    }
    contributions.push({ feature: cfg.features[fi], contribution: weightedContrib });
  }

  // Sort by absolute contribution, take top 4
  contributions.sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution));

  return contributions.slice(0, 4).map((c) => {
    const val = avgFeatures[c.feature] ?? 0;
    let stat: string;
    // Generate human-readable stat
    if (c.feature.startsWith("is_") || c.feature.startsWith("has_")) {
      const pct = Math.round(val * 100);
      stat = `${pct}% of prompts`;
    } else if (c.feature === "word_count") {
      stat = `avg ${Math.round(val)} words/prompt`;
    } else if (c.feature === "code_ratio") {
      stat = `${Math.round(val * 100)}% code characters`;
    } else if (c.feature === "instruction_ratio") {
      stat = `${Math.round(val * 100)}% instruction text`;
    } else if (c.feature.endsWith("_count")) {
      stat = `avg ${round(val, 1)} per prompt`;
    } else {
      stat = `avg ${round(val, 2)}`;
    }

    return {
      feature: c.feature,
      label: FEATURE_LABELS[c.feature] ?? c.feature,
      contribution: round(c.contribution, 3),
      stat,
    };
  });
}

export async function enrichScores(
  client: Client,
  onBatchProgress?: (done: number, total: number) => void,
): Promise<number> {
  // Find messages with style scores but without vibe/politeness scores
  const result = await client.execute(
    `SELECT m.id, m.content, m.word_count FROM messages m
     JOIN nlp_enrichments e ON m.id = e.message_id
     WHERE m.role = 'human' AND m.is_excluded = 0
       AND e.detail_score IS NOT NULL
       AND (e.vibe_score IS NULL OR e.politeness_score IS NULL)`,
  );

  if (result.rows.length === 0) return 0;

  const batchSize = 500;
  let done = 0;

  for (let i = 0; i < result.rows.length; i += batchSize) {
    const batch = result.rows.slice(i, i + batchSize);
    const stmts = batch.map((row) => {
      const features = extractFeatures(String(row.content), Number(row.word_count));
      const vibe = scoreVibe(features);
      const pol = scorePoliteness(features);
      return {
        sql: `UPDATE nlp_enrichments SET
          vibe_score = ?, vibe_confidence = ?,
          politeness_score = ?, politeness_confidence = ?
          WHERE message_id = ?`,
        args: [vibe.score, vibe.confidence, pol.score, pol.confidence, Number(row.id)],
      };
    });

    await client.batch(stmts, "write");
    done += batch.length;
    onBatchProgress?.(done, result.rows.length);
  }

  return done;
}
