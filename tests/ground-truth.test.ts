/**
 * Ground-truth tests for WS5: Metrics Accuracy & Test Harness.
 *
 * Tests persona classification with known radar scores derived from
 * the synthetic datasets, validates full pipeline structure, and
 * ensures metrics output is consistent across all datasets.
 */
import { describe, it, expect, beforeEach, afterEach } from "vitest";
import fs from "node:fs";
import path from "node:path";
import os from "node:os";
import { createDbClient, insertMessages } from "../src/pipeline/db.js";
import { bootstrapDb } from "../bin/bootstrap-db.mjs";
import { enrichNlp, computeNlpMetrics } from "../src/pipeline/nlp.js";
import { computeMetrics } from "../src/pipeline/metrics.js";
import { computeTrendMetrics } from "../src/pipeline/trends.js";
import { classifyPersona } from "../src/pipeline/persona.js";
import { loadConfig } from "../src/pipeline/config.js";
import { Role } from "../src/pipeline/models.js";
import type { Client } from "@libsql/client";
import { ALL_DATASETS, type TestDataset } from "./fixtures/synthetic-datasets.js";

let client: Client;
let dbPath: string;

beforeEach(async () => {
  dbPath = path.join(os.tmpdir(), `howiprompt-gt-${Date.now()}.db`);
  await bootstrapDb(dbPath);
  client = createDbClient(dbPath);
});

afterEach(() => {
  client.close();
  try { fs.unlinkSync(dbPath); } catch {}
});

// ---------------------------------------------------------------------------
// 1. Persona classification ground truth (known radar → expected persona)
// ---------------------------------------------------------------------------

describe("persona ground truth", () => {
  const GROUND_TRUTH: Array<{
    name: string;
    radar: { precision: number; curiosity: number; tenacity: number; trust: number };
    expectedPersona: string;
  }> = [
    // Architect variants: high precision + low trust
    { name: "architect-classic", radar: { precision: 85, curiosity: 30, tenacity: 50, trust: 25 }, expectedPersona: "architect" },
    { name: "architect-review", radar: { precision: 80, curiosity: 35, tenacity: 45, trust: 30 }, expectedPersona: "architect" },
    { name: "architect-constraint", radar: { precision: 90, curiosity: 20, tenacity: 45, trust: 20 }, expectedPersona: "architect" },

    // Explorer variants: curiosity is top axis
    { name: "explorer-learning", radar: { precision: 40, curiosity: 85, tenacity: 50, trust: 55 }, expectedPersona: "explorer" },
    { name: "explorer-brainstorm", radar: { precision: 30, curiosity: 90, tenacity: 40, trust: 50 }, expectedPersona: "explorer" },
    { name: "explorer-investigation", radar: { precision: 50, curiosity: 80, tenacity: 55, trust: 45 }, expectedPersona: "explorer" },

    // Commander variants: high precision + low tenacity
    { name: "commander-rapid", radar: { precision: 85, curiosity: 20, tenacity: 25, trust: 30 }, expectedPersona: "commander" },
    { name: "commander-technical", radar: { precision: 80, curiosity: 25, tenacity: 30, trust: 25 }, expectedPersona: "commander" },
    { name: "commander-file-targeted", radar: { precision: 90, curiosity: 15, tenacity: 20, trust: 20 }, expectedPersona: "commander" },

    // Partner variants: tenacity is top or strong second axis
    { name: "partner-iterative", radar: { precision: 50, curiosity: 45, tenacity: 85, trust: 50 }, expectedPersona: "partner" },
    { name: "partner-refinement", radar: { precision: 55, curiosity: 35, tenacity: 80, trust: 45 }, expectedPersona: "partner" },
    { name: "partner-collaborative", radar: { precision: 50, curiosity: 50, tenacity: 75, trust: 55 }, expectedPersona: "partner" },
    { name: "partner-polite", radar: { precision: 45, curiosity: 50, tenacity: 80, trust: 50 }, expectedPersona: "partner" },

    // Delegator variants: trust is top axis
    { name: "delegator-outcome", radar: { precision: 25, curiosity: 30, tenacity: 30, trust: 85 }, expectedPersona: "delegator" },
    { name: "delegator-trust-heavy", radar: { precision: 20, curiosity: 25, tenacity: 25, trust: 90 }, expectedPersona: "delegator" },
    { name: "delegator-acceptance", radar: { precision: 30, curiosity: 20, tenacity: 25, trust: 80 }, expectedPersona: "delegator" },
    { name: "delegator-open-ended", radar: { precision: 25, curiosity: 30, tenacity: 20, trust: 85 }, expectedPersona: "delegator" },
  ];

  for (const tc of GROUND_TRUTH) {
    it(`classifies ${tc.name} as ${tc.expectedPersona}`, () => {
      const result = classifyPersona(tc.radar);
      expect(result.type).toBe(tc.expectedPersona);
    });
  }
});

// ---------------------------------------------------------------------------
// 2. Pipeline structure validation per synthetic dataset
// ---------------------------------------------------------------------------

describe("pipeline structure per dataset", () => {
  for (const dataset of ALL_DATASETS) {
    it(`${dataset.name}: inserts, enriches, computes metrics`, async () => {
      const humanMsgs = dataset.messages.filter((m) => m.role === Role.HUMAN);
      const { inserted } = await insertMessages(client, dataset.messages);
      expect(inserted).toBe(dataset.messages.length);

      const enriched = await enrichNlp(client);
      expect(enriched).toBe(humanMsgs.length);

      const config = loadConfig("/tmp/test-howiprompt");
      const metrics = await computeMetrics(client, config);

      // Volume checks
      expect(metrics.volume.total_human).toBe(humanMsgs.length);
      expect(metrics.volume.total_conversations).toBeGreaterThan(0);
      expect(metrics.volume.avg_words_per_prompt).toBeGreaterThan(0);

      // Structure checks — all required top-level keys
      expect(metrics).toHaveProperty("volume");
      expect(metrics).toHaveProperty("conversation_depth");
      expect(metrics).toHaveProperty("temporal");
      expect(metrics).toHaveProperty("politeness");
      expect(metrics).toHaveProperty("nlp");
      expect(metrics).toHaveProperty("trends");
      expect(metrics).toHaveProperty("persona");
      expect(metrics).toHaveProperty("normalized");

      // NLP structure — all 7 embedding classifiers present
      expect(metrics.nlp).toHaveProperty("hitl_score");
      expect(metrics.nlp).toHaveProperty("vibe_coder_index");
      expect(metrics.nlp).toHaveProperty("precision");
      expect(metrics.nlp).toHaveProperty("curiosity");
      expect(metrics.nlp).toHaveProperty("tenacity");
      expect(metrics.nlp).toHaveProperty("trust");
      expect(metrics.nlp).toHaveProperty("politeness");

      // Persona structure
      expect(metrics.persona).toHaveProperty("type");
      expect(metrics.persona).toHaveProperty("name");
      expect(metrics.persona).toHaveProperty("radar");
      expect(metrics.persona.radar).toHaveProperty("precision");
      expect(metrics.persona.radar).toHaveProperty("curiosity");
      expect(metrics.persona.radar).toHaveProperty("tenacity");
      expect(metrics.persona.radar).toHaveProperty("trust");

      // Heatmap structure
      expect(metrics.temporal.heatmap).toHaveLength(7);
      expect(metrics.temporal.heatmap[0]).toHaveLength(24);

      // Trends structure
      expect(metrics.trends).toHaveProperty("daily_rollups");
      expect(metrics.trends).toHaveProperty("weekly_rollups");
      expect(metrics.trends.daily_rollups.length).toBeGreaterThan(0);
    });
  }
});

// ---------------------------------------------------------------------------
// 3. Metrics consistency: all scores in valid ranges
// ---------------------------------------------------------------------------

describe("metrics score ranges", () => {
  it("all normalized scores are 0-100", async () => {
    // Use the largest dataset for more interesting metrics
    const dataset = ALL_DATASETS[0];
    const { inserted } = await insertMessages(client, dataset.messages);
    await enrichNlp(client);

    const config = loadConfig("/tmp/test-howiprompt");
    const metrics = await computeMetrics(client, config);

    for (const [key, value] of Object.entries(metrics.normalized)) {
      expect(value).toBeGreaterThanOrEqual(0);
      expect(value).toBeLessThanOrEqual(100);
    }
  });

  it("persona radar scores are 0-100", async () => {
    const dataset = ALL_DATASETS[0];
    await insertMessages(client, dataset.messages);
    await enrichNlp(client);

    const config = loadConfig("/tmp/test-howiprompt");
    const metrics = await computeMetrics(client, config);

    for (const axis of ["precision", "curiosity", "tenacity", "trust"]) {
      const score = metrics.persona.radar[axis];
      expect(score).toBeGreaterThanOrEqual(0);
      expect(score).toBeLessThanOrEqual(100);
    }
  });

  it("politeness score is 0-100", async () => {
    const dataset = ALL_DATASETS[0];
    await insertMessages(client, dataset.messages);
    await enrichNlp(client);

    const config = loadConfig("/tmp/test-howiprompt");
    const metrics = await computeMetrics(client, config);

    expect(metrics.politeness.score).toBeGreaterThanOrEqual(0);
    expect(metrics.politeness.score).toBeLessThanOrEqual(100);
  });

  it("conversation depth stats are non-negative", async () => {
    const dataset = ALL_DATASETS[0];
    await insertMessages(client, dataset.messages);

    const config = loadConfig("/tmp/test-howiprompt");
    const metrics = await computeMetrics(client, config);

    expect(metrics.conversation_depth.avg_turns).toBeGreaterThan(0);
    expect(metrics.conversation_depth.max_turns).toBeGreaterThan(0);
    expect(metrics.conversation_depth.quick_asks).toBeGreaterThanOrEqual(0);
    expect(metrics.conversation_depth.working_sessions).toBeGreaterThanOrEqual(0);
    expect(metrics.conversation_depth.deep_dives).toBeGreaterThanOrEqual(0);
  });
});

// ---------------------------------------------------------------------------
// 4. Trend rollup consistency
// ---------------------------------------------------------------------------

describe("trend rollups", () => {
  it("weekly rollups have NLP sub-object with correct keys", async () => {
    const dataset = ALL_DATASETS[9]; // partner1 — has many messages across days
    await insertMessages(client, dataset.messages);
    await enrichNlp(client);

    const trends = await computeTrendMetrics(client);
    expect(trends.weekly_rollups.length).toBeGreaterThan(0);

    for (const week of trends.weekly_rollups) {
      expect(week).toHaveProperty("prompts");
      expect(week).toHaveProperty("style");
      expect(week.style).toHaveProperty("backtrack_per_100");
      expect(week.style).toHaveProperty("question_rate_pct");
      expect(week.style).toHaveProperty("command_rate_pct");
      // politeness_pct should NOT be in style (it's embedding-based now)
      expect(week.style).not.toHaveProperty("politeness_pct");
      expect(week.style).not.toHaveProperty("politeness_per_100");
    }
  });

  it("daily rollups sum to total prompts", async () => {
    const dataset = ALL_DATASETS[0];
    const humanCount = dataset.messages.filter((m) => m.role === Role.HUMAN).length;
    await insertMessages(client, dataset.messages);

    const trends = await computeTrendMetrics(client);
    const totalFromRollups = trends.daily_rollups.reduce((sum: number, d: any) => sum + d.prompts, 0);
    expect(totalFromRollups).toBe(humanCount);
  });
});
