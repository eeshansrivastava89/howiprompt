import { describe, it, expect, beforeEach, afterEach } from "vitest";
import fs from "node:fs";
import path from "node:path";
import os from "node:os";
import { createDbClient, insertMessages } from "../src/pipeline/db.js";
import { bootstrapDb } from "../bin/bootstrap-db.mjs";
import { enrichNlp, computeNlpMetrics } from "../src/pipeline/nlp.js";
import { computeMetrics } from "../src/pipeline/metrics.js";
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

beforeEach(async () => {
  dbPath = path.join(os.tmpdir(), `howiprompt-excl-${Date.now()}.db`);
  await bootstrapDb(dbPath);
  client = createDbClient(dbPath);
});

afterEach(() => {
  client.close();
  try { fs.unlinkSync(dbPath); } catch {}
});

describe("exclusion filtering", () => {
  it("excluded messages do not affect volume metrics", async () => {
    const msgs = [
      makeMsg("real prompt one"),
      makeMsg("real prompt two", { timestamp: new Date("2026-02-01T10:01:00Z") }),
      makeMsg("/commit", { timestamp: new Date("2026-02-01T10:02:00Z") }),
      makeMsg("y", { timestamp: new Date("2026-02-01T10:03:00Z") }),
    ];
    await insertMessages(client, msgs);

    // Mark last two as excluded
    await client.execute("UPDATE messages SET is_excluded = 1 WHERE content IN ('/commit', 'y')");

    await enrichNlp(client);
    const config = loadConfig("/tmp/test-excl");
    const metrics = await computeMetrics(client, config);

    expect(metrics.volume.total_human).toBe(2);
    expect(metrics.volume.total_conversations).toBe(1);
  });

  it("excluded messages do not affect NLP aggregates", async () => {
    const msgs = [
      makeMsg("please help me debug this authentication issue"),
      makeMsg("/commit", { timestamp: new Date("2026-02-01T10:01:00Z") }),
    ];
    await insertMessages(client, msgs);
    await client.execute("UPDATE messages SET is_excluded = 1 WHERE content = '/commit'");

    await enrichNlp(client);

    // Verify excluded message was not enriched
    const enriched = await client.execute(
      "SELECT COUNT(*) as cnt FROM nlp_enrichments e JOIN messages m ON e.message_id = m.id WHERE m.is_excluded = 1",
    );
    expect(Number(enriched.rows[0].cnt)).toBe(0);

    // NLP metrics should only reflect the real prompt
    const nlp = await computeNlpMetrics(client);
    const totalEnriched = await client.execute(
      "SELECT COUNT(*) as cnt FROM nlp_enrichments",
    );
    expect(Number(totalEnriched.rows[0].cnt)).toBe(1);
  });

  it("excluded messages do not affect temporal stats", async () => {
    const msgs = [
      makeMsg("real prompt", { timestamp: new Date("2026-02-01T14:00:00Z") }),
      makeMsg("/commit", { timestamp: new Date("2026-02-01T23:30:00Z") }),
    ];
    await insertMessages(client, msgs);
    await client.execute("UPDATE messages SET is_excluded = 1 WHERE content = '/commit'");
    await enrichNlp(client);

    const config = loadConfig("/tmp/test-excl");
    const metrics = await computeMetrics(client, config);

    // Night owl should be 0% since only the excluded message was at night
    expect(metrics.temporal.night_owl_pct).toBe(0);
  });

  it("excluded messages do not appear in trend rollups", async () => {
    const msgs = [
      makeMsg("real prompt", { timestamp: new Date("2026-02-01T10:00:00Z") }),
      makeMsg("/commit", { timestamp: new Date("2026-02-01T10:01:00Z") }),
      makeMsg("another real prompt", { timestamp: new Date("2026-02-02T10:00:00Z") }),
    ];
    await insertMessages(client, msgs);
    await client.execute("UPDATE messages SET is_excluded = 1 WHERE content = '/commit'");

    const trends = await computeTrendMetrics(client);
    const totalPrompts = trends.daily_rollups.reduce(
      (sum: number, r: any) => sum + r.prompts, 0,
    );
    expect(totalPrompts).toBe(2);
  });
});
