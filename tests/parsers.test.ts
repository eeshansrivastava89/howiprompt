import { describe, it, expect, beforeEach, afterEach } from "vitest";
import fs from "node:fs";
import path from "node:path";
import os from "node:os";
import { parseClaudeCode, parseCodexHistory, parseCodexSessionMetadata } from "../src/pipeline/parsers.js";
import { Platform, Role } from "../src/pipeline/models.js";

let tmpDir: string;

beforeEach(() => {
  tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "hip-test-"));
});

afterEach(() => {
  fs.rmSync(tmpDir, { recursive: true, force: true });
});

describe("parseClaudeCode", () => {
  it("parses valid JSONL messages", async () => {
    const projectDir = path.join(tmpDir, "project-1");
    fs.mkdirSync(projectDir, { recursive: true });
    const lines = [
      JSON.stringify({
        type: "user",
        message: { role: "user", content: "fix the bug" },
        timestamp: "2026-02-01T10:00:00Z",
      }),
      JSON.stringify({
        type: "assistant",
        message: { role: "assistant", content: [{ type: "text", text: "Here is the fix" }] },
        timestamp: "2026-02-01T10:01:00Z",
      }),
    ];
    fs.writeFileSync(path.join(projectDir, "session-1.jsonl"), lines.join("\n"));

    const messages = await parseClaudeCode(tmpDir);
    expect(messages.length).toBe(2);
    expect(messages[0].platform).toBe(Platform.CLAUDE_CODE);
    expect(messages[0].role).toBe(Role.HUMAN);
    expect(messages[0].content).toBe("fix the bug");
    expect(messages[1].role).toBe(Role.ASSISTANT);
    expect(messages[1].content).toBe("Here is the fix");
  });

  it("skips meta messages", async () => {
    const projectDir = path.join(tmpDir, "project-1");
    fs.mkdirSync(projectDir, { recursive: true });
    const lines = [
      JSON.stringify({
        type: "user",
        isMeta: true,
        message: { role: "user", content: "meta message" },
        timestamp: "2026-02-01T10:00:00Z",
      }),
      JSON.stringify({
        type: "user",
        message: { role: "user", content: "real message" },
        timestamp: "2026-02-01T10:01:00Z",
      }),
    ];
    fs.writeFileSync(path.join(projectDir, "session-1.jsonl"), lines.join("\n"));

    const messages = await parseClaudeCode(tmpDir);
    expect(messages.length).toBe(1);
    expect(messages[0].content).toBe("real message");
  });

  it("skips command messages", async () => {
    const projectDir = path.join(tmpDir, "project-1");
    fs.mkdirSync(projectDir, { recursive: true });
    const lines = [
      JSON.stringify({
        type: "user",
        message: { role: "user", content: "<command-result>output</command-result>" },
        timestamp: "2026-02-01T10:00:00Z",
      }),
    ];
    fs.writeFileSync(path.join(projectDir, "session-1.jsonl"), lines.join("\n"));

    const messages = await parseClaudeCode(tmpDir);
    expect(messages.length).toBe(0);
  });

  it("returns empty for nonexistent path", async () => {
    const messages = await parseClaudeCode("/nonexistent/path");
    expect(messages).toEqual([]);
  });

  it("handles array content (multi-part user messages)", async () => {
    const projectDir = path.join(tmpDir, "project-1");
    fs.mkdirSync(projectDir, { recursive: true });
    const lines = [
      JSON.stringify({
        type: "user",
        message: {
          role: "user",
          content: [
            { type: "text", text: "first part" },
            { type: "text", text: "second part" },
          ],
        },
        timestamp: "2026-02-01T10:00:00Z",
      }),
    ];
    fs.writeFileSync(path.join(projectDir, "session-1.jsonl"), lines.join("\n"));

    const messages = await parseClaudeCode(tmpDir);
    expect(messages.length).toBe(1);
    expect(messages[0].content).toBe("first part second part");
  });
});

describe("parseCodexHistory", () => {
  it("parses valid history entries", async () => {
    const historyPath = path.join(tmpDir, "history.jsonl");
    const lines = [
      JSON.stringify({ session_id: "s1", ts: 1706781600, text: "fix the issue" }),
      JSON.stringify({ session_id: "s2", ts: 1706781700, text: "add feature" }),
    ];
    fs.writeFileSync(historyPath, lines.join("\n"));

    const messages = await parseCodexHistory(historyPath);
    expect(messages.length).toBe(2);
    expect(messages[0].platform).toBe(Platform.CODEX);
    expect(messages[0].role).toBe(Role.HUMAN);
    expect(messages[0].content).toBe("fix the issue");
  });

  it("skips entries with invalid timestamps", async () => {
    const historyPath = path.join(tmpDir, "history.jsonl");
    fs.writeFileSync(
      historyPath,
      JSON.stringify({ session_id: "s1", ts: "not-a-number", text: "hello" }),
    );

    const messages = await parseCodexHistory(historyPath);
    expect(messages.length).toBe(0);
  });

  it("skips empty text entries", async () => {
    const historyPath = path.join(tmpDir, "history.jsonl");
    fs.writeFileSync(historyPath, JSON.stringify({ session_id: "s1", ts: 1706781600, text: "" }));

    const messages = await parseCodexHistory(historyPath);
    expect(messages.length).toBe(0);
  });

  it("enriches with session model metadata", async () => {
    const historyPath = path.join(tmpDir, "history.jsonl");
    fs.writeFileSync(
      historyPath,
      JSON.stringify({ session_id: "s1", ts: 1706781600, text: "hello" }),
    );

    const sessionModels = new Map([["s1", { modelId: "gpt-4", modelProvider: "openai" }]]);
    const messages = await parseCodexHistory(historyPath, sessionModels);
    expect(messages.length).toBe(1);
    expect(messages[0].modelId).toBe("gpt-4");
    expect(messages[0].modelProvider).toBe("openai");
  });
});

describe("parseCodexSessionMetadata", () => {
  it("extracts model metadata from session files", async () => {
    const sessionDir = path.join(tmpDir, "sessions", "sess-1");
    fs.mkdirSync(sessionDir, { recursive: true });
    const lines = [
      JSON.stringify({
        type: "session_meta",
        payload: { id: "s1", model: "claude-3-opus", model_provider: "anthropic" },
      }),
    ];
    fs.writeFileSync(path.join(sessionDir, "log.jsonl"), lines.join("\n"));

    const result = await parseCodexSessionMetadata(path.join(tmpDir, "sessions"));
    expect(result.get("s1")).toEqual({ modelId: "claude-3-opus", modelProvider: "anthropic" });
  });

  it("returns empty map for nonexistent path", async () => {
    const result = await parseCodexSessionMetadata("/nonexistent");
    expect(result.size).toBe(0);
  });
});
