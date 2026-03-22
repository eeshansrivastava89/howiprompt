import { describe, it, expect, beforeEach, afterEach } from "vitest";
import fs from "node:fs";
import path from "node:path";
import os from "node:os";
import { createDbClient, insertMessages, messageHash, queryMessages, logSync, getLastSync } from "../src/pipeline/db.js";
import { bootstrapDb } from "../bin/bootstrap-db.mjs";
import { Platform, Role, type Message } from "../src/pipeline/models.js";
import type { Client } from "@libsql/client";

let client: Client;
let dbPath: string;

function makeMessage(overrides: Partial<Message> = {}): Message {
  return {
    timestamp: new Date("2026-02-01T10:00:00Z"),
    platform: Platform.CLAUDE_CODE,
    role: Role.HUMAN,
    content: "fix the authentication bug",
    conversationId: "session-1",
    wordCount: 4,
    ...overrides,
  };
}

beforeEach(async () => {
  dbPath = path.join(os.tmpdir(), `howiprompt-test-${Date.now()}.db`);
  await bootstrapDb(dbPath);
  client = createDbClient(dbPath);
});

afterEach(() => {
  client.close();
  try { fs.unlinkSync(dbPath); } catch {}
});

describe("insertMessages", () => {
  it("inserts messages into database", async () => {
    const msgs = [makeMessage(), makeMessage({ content: "add new feature", conversationId: "session-2" })];
    const result = await insertMessages(client, msgs);
    expect(result.inserted).toBe(2);
    expect(result.skipped).toBe(0);
  });

  it("deduplicates by hash", async () => {
    const msg = makeMessage();
    await insertMessages(client, [msg]);
    const result = await insertMessages(client, [msg]);
    expect(result.inserted).toBe(0);
    expect(result.skipped).toBe(1);
  });

  it("handles empty array", async () => {
    const result = await insertMessages(client, []);
    expect(result.inserted).toBe(0);
  });
});

describe("messageHash", () => {
  it("produces consistent hashes", () => {
    const msg = makeMessage();
    expect(messageHash(msg)).toBe(messageHash(msg));
  });

  it("different content produces different hashes", () => {
    const a = makeMessage({ content: "hello" });
    const b = makeMessage({ content: "world" });
    expect(messageHash(a)).not.toBe(messageHash(b));
  });
});

describe("queryMessages", () => {
  it("filters by role", async () => {
    await insertMessages(client, [
      makeMessage({ role: Role.HUMAN }),
      makeMessage({ role: Role.ASSISTANT, content: "here is the fix" }),
    ]);
    const humans = await queryMessages(client, { role: Role.HUMAN });
    expect(humans.length).toBe(1);
    expect(humans[0].role).toBe("human");
  });

  it("filters by platform", async () => {
    await insertMessages(client, [
      makeMessage({ platform: Platform.CLAUDE_CODE }),
      makeMessage({ platform: Platform.CODEX, content: "codex prompt", conversationId: "cx-1" }),
    ]);
    const codex = await queryMessages(client, { platform: Platform.CODEX });
    expect(codex.length).toBe(1);
    expect(codex[0].platform).toBe("codex");
  });
});

describe("syncLog", () => {
  it("logs and retrieves sync state", async () => {
    await logSync(client, "claude_code", "file.jsonl", "2026-02-01T10:00:00Z", 100);
    const last = await getLastSync(client, "claude_code");
    expect(last.lastTimestamp).toBe("2026-02-01T10:00:00Z");
    expect(last.lastFile).toBe("file.jsonl");
  });

  it("returns null for unknown source", async () => {
    const last = await getLastSync(client, "nonexistent");
    expect(last.lastTimestamp).toBeNull();
  });
});
