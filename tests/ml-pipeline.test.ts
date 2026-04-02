import { describe, it, expect, beforeEach, afterEach } from "vitest";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { bootstrapDb } from "../bin/bootstrap-db.mjs";
import { createDbClient, insertMessages } from "../src/pipeline/db.js";
import { enrichNlp } from "../src/pipeline/nlp.js";
import { enrichStyle } from "../src/pipeline/style.js";
import { enrichScores, resetScorersForTests } from "../src/pipeline/scorers.js";
import { computeMetrics } from "../src/pipeline/metrics.js";
import { loadConfig } from "../src/pipeline/config.js";
import { Platform, Role, type Message } from "../src/pipeline/models.js";
import type { Client } from "@libsql/client";

describe("Scoring pipeline integration", () => {
  let client: Client;
  let dbPath: string;

  beforeEach(async () => {
    resetScorersForTests();
    dbPath = path.join(os.tmpdir(), `howiprompt-ml-${Date.now()}.db`);
    await bootstrapDb(dbPath);
    client = createDbClient(dbPath);
  });

  afterEach(() => {
    client.close();
    try { fs.unlinkSync(dbPath); } catch {}
  });

  it("extracts features, scores vibe & politeness end-to-end", async () => {
    const prompts = [
      "fix the bug in auth.ts on line 42 — the JWT validation is wrong",
      "can you please help me understand how the caching layer works?",
      "just make it work",
    ];

    const messages: Message[] = prompts.flatMap((content, i) => ([
      {
        timestamp: new Date(2026, 2, i + 1, 10, 0, 0),
        platform: i % 2 === 0 ? Platform.CLAUDE_CODE : Platform.CODEX,
        role: Role.HUMAN,
        content,
        conversationId: `ml-session-${i}`,
        wordCount: content.split(/\s+/).length,
      },
      {
        timestamp: new Date(2026, 2, i + 1, 10, 1, 0),
        platform: i % 2 === 0 ? Platform.CLAUDE_CODE : Platform.CODEX,
        role: Role.ASSISTANT,
        content: "Acknowledged.",
        conversationId: `ml-session-${i}`,
        wordCount: 1,
      },
    ]));

    await insertMessages(client, messages);
    await enrichNlp(client);

    const dataDir = fs.mkdtempSync(path.join(os.tmpdir(), "howiprompt-ml-data-"));

    // Style scoring (persona 2×2)
    const styled = await enrichStyle(client);
    expect(styled).toBe(prompts.length);

    // Vibe + Politeness scoring (logistic regression)
    const scored = await enrichScores(client);
    expect(scored).toBe(prompts.length);

    // Verify scores exist
    const scoreResult = await client.execute(
      "SELECT COUNT(*) as cnt FROM nlp_enrichments WHERE vibe_score IS NOT NULL AND politeness_score IS NOT NULL",
    );
    expect(Number(scoreResult.rows[0].cnt)).toBe(prompts.length);

    // Compute full metrics
    const metrics = await computeMetrics(client, loadConfig(dataDir));

    expect(metrics.nlp.vibe_coder_index.avg_score).not.toBeNull();
    expect(metrics.nlp.politeness.avg_score).not.toBeNull();
    expect(metrics.persona.quadrant).toBeTruthy();

    // "just make it work" should have higher vibe score than the detailed bug fix
    const vibeScores = await client.execute(
      `SELECT m.content, e.vibe_score FROM nlp_enrichments e
       JOIN messages m ON e.message_id = m.id
       WHERE m.role = 'human' ORDER BY m.id`,
    );
    const scores = vibeScores.rows.map((r) => ({
      content: String(r.content).slice(0, 30),
      vibe: Number(r.vibe_score),
    }));
    // The vibe-y prompt should score higher than the engineered one
    const engineered = scores.find((s) => s.content.includes("fix the bug"));
    const vibey = scores.find((s) => s.content.includes("just make it"));
    expect(vibey!.vibe).toBeGreaterThan(engineered!.vibe);

    fs.rmSync(dataDir, { recursive: true, force: true });
  });
});
