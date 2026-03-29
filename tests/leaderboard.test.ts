import { describe, it, expect, beforeEach, afterEach } from "vitest";
import fs from "node:fs";
import path from "node:path";
import os from "node:os";
import { createDbClient, insertMessages } from "../src/pipeline/db.js";
import { bootstrapDb } from "../bin/bootstrap-db.mjs";
import { enrichNlp } from "../src/pipeline/nlp.js";
import { computeMetrics } from "../src/pipeline/metrics.js";
import { loadConfig } from "../src/pipeline/config.js";
import { Platform, Role, type Message } from "../src/pipeline/models.js";
import type { Client } from "@libsql/client";
import {
  CLIENT_ID_STORAGE_KEY,
  USERNAME_STORAGE_KEY,
  createStableClientId,
  findEntryRank,
  getSubmissionPayload,
  sortLeaderboardEntries,
} from "../frontend/src/scripts/leaderboard.js";

function makeStorage(initial: Record<string, string> = {}) {
  const store = new Map(Object.entries(initial));
  return {
    getItem(key: string) {
      return store.get(key) ?? null;
    },
    setItem(key: string, value: string) {
      store.set(key, value);
    },
  };
}

describe("stable leaderboard client id", () => {
  it("creates and reuses a persistent client id", () => {
    const storage = makeStorage();
    const first = createStableClientId(storage as Storage);
    const second = createStableClientId(storage as Storage);
    expect(first).toMatch(/^hip_/);
    expect(second).toBe(first);
    expect(storage.getItem(CLIENT_ID_STORAGE_KEY)).toBe(first);
  });

  it("does not overwrite an existing username key while creating client id", () => {
    const storage = makeStorage({ [USERNAME_STORAGE_KEY]: "Existing User" });
    createStableClientId(storage as Storage);
    expect(storage.getItem(USERNAME_STORAGE_KEY)).toBe("Existing User");
  });
});

describe("leaderboard sorting and rank helpers", () => {
  const entries = [
    { display_name: "Regular Otter", fingerprint: "hip_a", hitl_score: 40, vibe_index: 60, politeness: 30, total_prompts: 200, total_conversations: 30, quadrant: "Brief + Collaborative" },
    { display_name: "Bold Falcon", fingerprint: "hip_b", hitl_score: 90, vibe_index: 50, politeness: 80, total_prompts: 900, total_conversations: 90, quadrant: "Detailed + Directive" },
    { display_name: "Calm Panda", fingerprint: "hip_c", hitl_score: 70, vibe_index: 85, politeness: 60, total_prompts: 500, total_conversations: 40, quadrant: "Detailed + Collaborative" },
  ];

  it("sorts by the selected metric", () => {
    expect(sortLeaderboardEntries(entries, "hitl_score")[0].display_name).toBe("Bold Falcon");
    expect(sortLeaderboardEntries(entries, "vibe_index")[0].display_name).toBe("Calm Panda");
    expect(sortLeaderboardEntries(entries, "total_conversations")[0].display_name).toBe("Bold Falcon");
  });

  it("finds the current user's rank with the stable client id", () => {
    expect(findEntryRank(entries, "hitl_score", "hip_c")).toBe(2);
    expect(findEntryRank(entries, "vibe_index", "hip_c")).toBe(1);
    expect(findEntryRank(entries, "hitl_score", "missing")).toBe(0);
  });
});

describe("frontend submission payload contract", () => {
  let client: Client;
  let dbPath: string;

  beforeEach(async () => {
    dbPath = path.join(os.tmpdir(), `howiprompt-lb-${Date.now()}.db`);
    await bootstrapDb(dbPath);
    client = createDbClient(dbPath);
  });

  afterEach(() => {
    client.close();
    try { fs.unlinkSync(dbPath); } catch {}
  });

  it("builds the same aggregate payload shape the dashboard submits", async () => {
    const msgs: Message[] = [];
    for (let i = 0; i < 20; i++) {
      msgs.push({
        timestamp: new Date(2026, 1, 1 + i, 10, 0, 0),
        platform: Platform.CLAUDE_CODE,
        role: Role.HUMAN,
        content: "please fix the authentication bug in src/auth.ts",
        conversationId: `session-${i}`,
        wordCount: 8,
      });
      msgs.push({
        timestamp: new Date(2026, 1, 1 + i, 10, 1, 0),
        platform: Platform.CLAUDE_CODE,
        role: Role.ASSISTANT,
        content: "Here is the fix...",
        conversationId: `session-${i}`,
        wordCount: 5,
      });
    }

    await insertMessages(client, msgs);
    await enrichNlp(client);

    const metrics = await computeMetrics(client, loadConfig("/tmp/test-howiprompt"));
    const payload = getSubmissionPayload(metrics, "both");

    expect(payload).toEqual({
      hitl_score: Math.round(metrics.nlp.hitl_score?.avg_score ?? 0),
      vibe_index: Math.round(metrics.nlp.vibe_coder_index?.avg_score ?? 0),
      politeness: Math.round(metrics.nlp.politeness?.avg_score ?? 0),
      total_prompts: metrics.volume.total_human,
      total_conversations: metrics.volume.total_conversations,
      quadrant: metrics.persona?.quadrant ?? "unknown",
      platform: "both",
    });
  });
});
