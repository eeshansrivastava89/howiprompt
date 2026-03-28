import { describe, it, expect, beforeEach, afterEach } from "vitest";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { createClient } from "@libsql/client";
import { getAllBackends } from "../src/pipeline/backends.js";
import {
  discoverAndSyncRules,
  seedSystemRules,
  loadExclusionRules,
  compileRules,
  flagExcludedMessages,
} from "../src/pipeline/exclusions.js";

// ── Skill discovery ───────────────────────────────────

describe("Skill Discovery", () => {
  it("backends with skills implement discoverSkills()", () => {
    const backends = getAllBackends();
    const withSkills = backends.filter((b) => b.discoverSkills);
    const withoutSkills = backends.filter((b) => !b.discoverSkills);

    expect(withSkills.map((b) => b.id)).toEqual(
      expect.arrayContaining(["claude_code", "codex", "cursor"]),
    );
    expect(withoutSkills.map((b) => b.id)).toEqual(["lmstudio"]);
  });

  it("claude_code skills include known skills", () => {
    const backends = getAllBackends();
    const claudeBackend = backends.find((b) => b.id === "claude_code")!;
    const skills = claudeBackend.discoverSkills!();
    const names = skills.map((s) => s.skillName);

    expect(names).toContain("commit");
    expect(names).toContain("context-save");
    expect(names).toContain("context-restore");
  });

  it("codex skills include known skills", () => {
    const backends = getAllBackends();
    const codexBackend = backends.find((b) => b.id === "codex")!;
    const skills = codexBackend.discoverSkills!();
    const names = skills.map((s) => s.skillName);

    expect(names).toContain("commit");
    expect(names).toContain("context-save");
    expect(names).toContain("context-restore");
  });
});

// ── Pattern matching ──────────────────────────────────

describe("Invocation Pattern Matching", () => {
  it("codex patterns match [$skill-name](...) format", () => {
    const codexBackend = getAllBackends().find((b) => b.id === "codex")!;
    const skills = codexBackend.discoverSkills!();
    const contextSave = skills.find((s) => s.skillName === "context-save")!;

    const regex = new RegExp(contextSave.invocationPattern, "i");
    expect(regex.test("[$context-save](/path/to/SKILL.md)")).toBe(true);
    expect(regex.test("$context-save")).toBe(true);
    expect(regex.test("please save the context for me")).toBe(false);
  });

  it("claude patterns only match /slash invocation", () => {
    const claudeBackend = getAllBackends().find((b) => b.id === "claude_code")!;
    const skills = claudeBackend.discoverSkills!();
    const commit = skills.find((s) => s.skillName === "commit")!;

    const regex = new RegExp(commit.invocationPattern, "i");
    expect(regex.test("/commit")).toBe(true);
    expect(regex.test("/commit now")).toBe(true);
    expect(regex.test("commit changes")).toBe(false);
    expect(regex.test("commit this")).toBe(false);
  });
});

// ── DB operations ─────────────────────────────────────

describe("Exclusion Rules DB", () => {
  let client: ReturnType<typeof createClient>;
  let dbPath: string;

  beforeEach(async () => {
    dbPath = path.join(os.tmpdir(), `exclusions-test-${Date.now()}.db`);
    client = createClient({ url: `file:${dbPath}` });

    await client.execute(`CREATE TABLE IF NOT EXISTS skills (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      platform TEXT NOT NULL,
      skill_name TEXT NOT NULL,
      skill_path TEXT NOT NULL,
      invocation_pattern TEXT,
      template_content TEXT,
      content_hash TEXT,
      source TEXT DEFAULT 'discovered',
      discovered_at TEXT NOT NULL,
      UNIQUE(platform, skill_name)
    )`);
    await client.execute(`CREATE TABLE IF NOT EXISTS exclusion_rules (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      platform TEXT NOT NULL DEFAULT '*',
      rule_type TEXT NOT NULL,
      pattern TEXT NOT NULL,
      match_mode TEXT NOT NULL DEFAULT 'starts_with',
      description TEXT,
      source TEXT NOT NULL DEFAULT 'system',
      template_content TEXT,
      is_active INTEGER NOT NULL DEFAULT 1,
      created_at TEXT NOT NULL DEFAULT (datetime('now')),
      UNIQUE(platform, rule_type, pattern)
    )`);
    await client.execute(`CREATE TABLE IF NOT EXISTS messages (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      hash TEXT UNIQUE NOT NULL,
      timestamp TEXT NOT NULL,
      platform TEXT NOT NULL,
      role TEXT NOT NULL,
      content TEXT NOT NULL,
      conversation_id TEXT NOT NULL,
      word_count INTEGER NOT NULL,
      is_excluded INTEGER DEFAULT 0,
      matched_rule_id INTEGER REFERENCES exclusion_rules(id)
    )`);
  });

  afterEach(() => {
    client.close();
    try { fs.unlinkSync(dbPath); } catch { /* ignore */ }
  });

  it("seeds system rules", async () => {
    const seeded = await seedSystemRules(client);
    expect(seeded).toBeGreaterThan(0);

    const rules = await loadExclusionRules(client);
    const types = rules.map((r) => r.ruleType);
    expect(types).toContain("content_prefix");
    expect(types).toContain("cwd_pattern");
    expect(types).toContain("dir_name");
  });

  it("seeds are idempotent", async () => {
    const first = await seedSystemRules(client);
    const second = await seedSystemRules(client);
    expect(second).toBe(0); // all already exist
  });

  it("flags messages matching content_prefix rules", async () => {
    await client.execute({
      sql: `INSERT INTO exclusion_rules (platform, rule_type, pattern, match_mode, description, source)
            VALUES (?, ?, ?, ?, ?, ?)`,
      args: ["claude_code", "content_prefix", "<task-notification>", "starts_with", "task notification", "system"],
    });

    await client.execute({
      sql: `INSERT INTO messages (hash, timestamp, platform, role, content, conversation_id, word_count)
            VALUES (?, ?, ?, ?, ?, ?, ?)`,
      args: ["h1", "2026-01-01", "claude_code", "human", "<task-notification>\n<task-id>abc</task-id>", "c1", 2],
    });
    await client.execute({
      sql: `INSERT INTO messages (hash, timestamp, platform, role, content, conversation_id, word_count)
            VALUES (?, ?, ?, ?, ?, ?, ?)`,
      args: ["h2", "2026-01-01", "claude_code", "human", "Fix the login bug", "c1", 4],
    });

    const flagged = await flagExcludedMessages(client);
    expect(flagged).toBe(1);

    const result = await client.execute("SELECT hash, is_excluded, matched_rule_id FROM messages ORDER BY hash");
    expect(Number(result.rows[0].is_excluded)).toBe(1); // task notification
    expect(result.rows[0].matched_rule_id).not.toBeNull();
    expect(Number(result.rows[1].is_excluded)).toBe(0); // real message
  });

  it("flags skill invocations but not natural language", async () => {
    await client.execute({
      sql: `INSERT INTO exclusion_rules (platform, rule_type, pattern, match_mode, description, source)
            VALUES (?, ?, ?, ?, ?, ?)`,
      args: ["claude_code", "skill_invocation", "^\\/commit\\b", "regex", "commit skill", "discovered"],
    });

    await client.execute({
      sql: `INSERT INTO messages (hash, timestamp, platform, role, content, conversation_id, word_count)
            VALUES (?, ?, ?, ?, ?, ?, ?)`,
      args: ["h1", "2026-01-01", "claude_code", "human", "/commit", "c1", 1],
    });
    await client.execute({
      sql: `INSERT INTO messages (hash, timestamp, platform, role, content, conversation_id, word_count)
            VALUES (?, ?, ?, ?, ?, ?, ?)`,
      args: ["h2", "2026-01-01", "claude_code", "human", "commit these changes", "c1", 3],
    });

    const flagged = await flagExcludedMessages(client);
    expect(flagged).toBe(1);

    const result = await client.execute("SELECT hash, is_excluded FROM messages ORDER BY hash");
    expect(Number(result.rows[0].is_excluded)).toBe(1); // /commit
    expect(Number(result.rows[1].is_excluded)).toBe(0); // natural language
  });
});
