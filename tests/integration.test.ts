import { describe, it, expect, beforeEach, afterEach } from "vitest";
import fs from "node:fs";
import path from "node:path";
import os from "node:os";
import { createDbClient, insertMessages } from "../src/pipeline/db.js";
import { bootstrapDb } from "../bin/bootstrap-db.mjs";
import { enrichNlp, computeNlpMetrics } from "../src/pipeline/nlp.js";
import { computeMetrics, computeSourceViews, hasHumanMessages } from "../src/pipeline/metrics.js";
import { computeTrendMetrics } from "../src/pipeline/trends.js";
import { loadConfig } from "../src/pipeline/config.js";
import { Platform, Role, type Message } from "../src/pipeline/models.js";
import type { Client } from "@libsql/client";

let client: Client;
let dbPath: string;

function makeMsg(content: string, overrides: Partial<Message> = {}): Message {
  return {
    timestamp: new Date("2026-02-01T10:00:00Z"),
    platform: Platform.CLAUDE_CODE,
    role: Role.HUMAN,
    content,
    conversationId: "session-1",
    wordCount: content.split(/\s+/).length,
    ...overrides,
  };
}

function generateTestMessages(): Message[] {
  const msgs: Message[] = [];
  // Generate diverse messages across dates and platforms
  for (let d = 1; d <= 10; d++) {
    const date = new Date(2026, 1, d, 10, 0, 0);
    // Claude Code human messages
    msgs.push(makeMsg("please fix the authentication bug", {
      timestamp: date,
      conversationId: `cc-session-${d}`,
      platform: Platform.CLAUDE_CODE,
    }));
    msgs.push(makeMsg("can you explain how the middleware works?", {
      timestamp: new Date(date.getTime() + 60000),
      conversationId: `cc-session-${d}`,
      platform: Platform.CLAUDE_CODE,
    }));
    // Assistant response
    msgs.push(makeMsg("Here is the explanation of the middleware...", {
      timestamp: new Date(date.getTime() + 120000),
      conversationId: `cc-session-${d}`,
      platform: Platform.CLAUDE_CODE,
      role: Role.ASSISTANT,
    }));
    // Codex message
    msgs.push(makeMsg("build a new REST API endpoint", {
      timestamp: new Date(date.getTime() + 180000),
      conversationId: `cx-session-${d}`,
      platform: Platform.CODEX,
      modelId: "o3-mini",
      modelProvider: "openai",
    }));
  }
  return msgs;
}

beforeEach(async () => {
  dbPath = path.join(os.tmpdir(), `howiprompt-integ-${Date.now()}.db`);
  await bootstrapDb(dbPath);
  client = createDbClient(dbPath);
});

afterEach(() => {
  client.close();
  try { fs.unlinkSync(dbPath); } catch {}
});

describe("full pipeline integration", () => {
  it("inserts, enriches, and computes metrics", async () => {
    const msgs = generateTestMessages();
    const { inserted } = await insertMessages(client, msgs);
    expect(inserted).toBe(msgs.length);

    // NLP enrichment
    const enriched = await enrichNlp(client);
    expect(enriched).toBeGreaterThan(0);

    // Second enrichment should skip already-enriched
    const enriched2 = await enrichNlp(client);
    expect(enriched2).toBe(0);

    // Compute metrics
    const config = loadConfig("/tmp/test-howiprompt");
    const metrics = await computeMetrics(client, config);

    expect(metrics.volume.total_human).toBe(30); // 3 human per day * 10 days
    expect(metrics.volume.total_assistant).toBe(10);
    expect(metrics.volume.total_conversations).toBeGreaterThan(0);
    expect(metrics.volume.avg_words_per_prompt).toBeGreaterThan(0);

    expect(metrics.conversation_depth.avg_turns).toBeGreaterThan(0);
    expect(metrics.temporal.heatmap).toHaveLength(7);
    expect(metrics.temporal.heatmap[0]).toHaveLength(24);

    expect(metrics.politeness.score).toBeGreaterThanOrEqual(0);
    expect(metrics.politeness.score).toBeLessThanOrEqual(100);
    expect(metrics.persona).toHaveProperty("type");
    expect(metrics.persona).toHaveProperty("name");
    expect(metrics.persona).toHaveProperty("radar");

    expect(metrics.nlp.intent).toHaveProperty("counts");
    expect(metrics.nlp).toHaveProperty("hitl_score");
    expect(metrics.nlp).toHaveProperty("vibe_coder_index");
    expect(metrics.nlp).toHaveProperty("precision");
    expect(metrics.nlp).toHaveProperty("curiosity");
    expect(metrics.nlp).toHaveProperty("tenacity");
    expect(metrics.nlp).toHaveProperty("trust");
    expect(metrics.nlp).toHaveProperty("politeness");

    expect(metrics.trends).toHaveProperty("daily_rollups");
    expect(metrics.trends).toHaveProperty("weekly_rollups");

    expect(metrics.model_usage.coverage.total_human_prompts).toBe(30);

    // Normalized scores
    expect(metrics.normalized).toBeDefined();
    expect(typeof metrics.normalized.politeness).toBe("number");
    expect(metrics.normalized.politeness).toBeGreaterThanOrEqual(0);
    expect(metrics.normalized.politeness).toBeLessThanOrEqual(100);
  });

  it("computes source views correctly", async () => {
    await insertMessages(client, generateTestMessages());
    await enrichNlp(client);

    const config = loadConfig("/tmp/test-howiprompt");
    const { sourceViews, metadata } = await computeSourceViews(client, config);

    expect(sourceViews.both).not.toBeNull();
    expect(sourceViews.claude_code).not.toBeNull();
    expect(sourceViews.codex).not.toBeNull();
    expect(metadata.default_view).toBe("both");

    // Claude Code view should only have Claude Code messages
    expect(sourceViews.claude_code.platform_stats).toHaveProperty("claude_code");
    expect(sourceViews.claude_code.platform_stats).not.toHaveProperty("codex");
  });

  it("hasHumanMessages works with platform filter", async () => {
    await insertMessages(client, generateTestMessages());

    expect(await hasHumanMessages(client)).toBe(true);
    expect(await hasHumanMessages(client, Platform.CLAUDE_CODE)).toBe(true);
    expect(await hasHumanMessages(client, Platform.CODEX)).toBe(true);
  });

  it("NLP metrics include intent distribution", async () => {
    await insertMessages(client, generateTestMessages());
    await enrichNlp(client);

    const nlp = await computeNlpMetrics(client);
    expect(Object.keys(nlp.intent.counts).length).toBeGreaterThan(0);
    expect(nlp.intent.confidence.mean).toBeGreaterThan(0);
    expect(nlp).toHaveProperty("hitl_score");
    expect(nlp).toHaveProperty("precision");
  });

  it("trend metrics include rollups", async () => {
    await insertMessages(client, generateTestMessages());
    const trends = await computeTrendMetrics(client);

    expect(trends.daily_rollups.length).toBeGreaterThan(0);
    expect(trends.weekly_rollups.length).toBeGreaterThan(0);
    expect(trends.daily_rollups[0]).toHaveProperty("prompts");
    expect(trends.daily_rollups[0]).toHaveProperty("style");
  });
});
