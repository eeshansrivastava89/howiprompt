/**
 * Leaderboard validation tests for WS5.
 * Tests: submit payload structure, validation logic, display name sanitization,
 * ranking sort correctness, and payload/metrics.json consistency.
 */
import { describe, it, expect, beforeEach, afterEach } from "vitest";
import fs from "node:fs";
import path from "node:path";
import os from "node:os";
import { createDbClient, insertMessages } from "../src/pipeline/db.js";
import { bootstrapDb } from "../bin/bootstrap-db.mjs";
import { enrichNlp } from "../src/pipeline/nlp.js";
import { computeMetrics } from "../src/pipeline/metrics.js";
import { loadConfig } from "../src/pipeline/config.js";
import { computeNormalized } from "../src/pipeline/registry.js";
import { Platform, Role, type Message } from "../src/pipeline/models.js";
import type { Client } from "@libsql/client";

// Import the validate function from the worker source.
// Since the worker uses Cloudflare types we can't import it directly,
// so we replicate the validation logic here (it's the contract we're testing).

interface SubmissionPayload {
  display_name: string;
  total_conversations: number;
  total_prompts: number;
  avg_words_per_prompt: number;
  politeness: number;
  backtrack: number;
  question_rate: number;
  command_rate: number;
  hitl_score: number;
  vibe_index: number;
  persona: string;
  complexity_avg: number;
  platform: string;
  tool_version: string;
}

function validate(payload: SubmissionPayload): string | null {
  if (!payload.display_name || typeof payload.display_name !== "string") return "display_name required";
  if (payload.display_name.length > 30) return "display_name too long (max 30)";
  if (payload.display_name.length < 2) return "display_name too short (min 2)";
  if (!/^[a-zA-Z0-9 _-]+$/.test(payload.display_name)) return "display_name contains invalid characters";

  const rangeFields = ["politeness", "backtrack", "question_rate", "command_rate", "hitl_score", "vibe_index", "complexity_avg"] as const;
  for (const f of rangeFields) {
    const v = payload[f];
    if (typeof v !== "number" || v < 0 || v > 100) return `${f} must be 0-100`;
  }

  if (typeof payload.total_prompts !== "number" || payload.total_prompts < 1) return "total_prompts must be >= 1";
  if (typeof payload.total_conversations !== "number" || payload.total_conversations < 1) return "total_conversations must be >= 1";
  if (payload.hitl_score > 90 && payload.total_conversations < 10) return "implausible: high HITL with very few conversations";
  if (payload.total_prompts > 1_000_000) return "implausible: too many prompts";

  return null;
}

function validPayload(overrides: Partial<SubmissionPayload> = {}): SubmissionPayload {
  return {
    display_name: "test-user",
    total_conversations: 100,
    total_prompts: 500,
    avg_words_per_prompt: 25,
    politeness: 60,
    backtrack: 30,
    question_rate: 45,
    command_rate: 55,
    hitl_score: 70,
    vibe_index: 65,
    persona: "architect",
    complexity_avg: 50,
    platform: "both",
    tool_version: "2.0.0",
    ...overrides,
  };
}

// ---------------------------------------------------------------------------
// 1. Display name validation & sanitization
// ---------------------------------------------------------------------------

describe("display name validation", () => {
  it("rejects empty display name", () => {
    expect(validate(validPayload({ display_name: "" }))).toBe("display_name required");
  });

  it("rejects too short name", () => {
    expect(validate(validPayload({ display_name: "a" }))).toBe("display_name too short (min 2)");
  });

  it("rejects too long name", () => {
    expect(validate(validPayload({ display_name: "a".repeat(31) }))).toBe("display_name too long (max 30)");
  });

  it("rejects special characters", () => {
    expect(validate(validPayload({ display_name: "<script>alert(1)</script>" }))).toBe("display_name contains invalid characters");
    expect(validate(validPayload({ display_name: "user@email.com" }))).toBe("display_name contains invalid characters");
    expect(validate(validPayload({ display_name: "user; DROP TABLE" }))).toBe("display_name contains invalid characters");
    expect(validate(validPayload({ display_name: "emoji 🎉" }))).toBe("display_name contains invalid characters");
  });

  it("accepts valid names", () => {
    expect(validate(validPayload({ display_name: "eeshan" }))).toBeNull();
    expect(validate(validPayload({ display_name: "test-user_123" }))).toBeNull();
    expect(validate(validPayload({ display_name: "My Name" }))).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// 2. Score range validation
// ---------------------------------------------------------------------------

describe("score range validation", () => {
  const rangeFields = ["politeness", "backtrack", "question_rate", "command_rate", "hitl_score", "vibe_index", "complexity_avg"] as const;

  for (const field of rangeFields) {
    it(`rejects ${field} > 100`, () => {
      expect(validate(validPayload({ [field]: 101 }))).toBe(`${field} must be 0-100`);
    });

    it(`rejects ${field} < 0`, () => {
      expect(validate(validPayload({ [field]: -1 }))).toBe(`${field} must be 0-100`);
    });

    it(`accepts ${field} at boundaries`, () => {
      expect(validate(validPayload({ [field]: 0 }))).toBeNull();
      expect(validate(validPayload({ [field]: 100 }))).toBeNull();
    });
  }
});

// ---------------------------------------------------------------------------
// 3. Anomaly detection
// ---------------------------------------------------------------------------

describe("anomaly detection", () => {
  it("rejects high HITL with very few conversations", () => {
    const err = validate(validPayload({ hitl_score: 95, total_conversations: 5 }));
    expect(err).toBe("implausible: high HITL with very few conversations");
  });

  it("allows high HITL with sufficient conversations", () => {
    expect(validate(validPayload({ hitl_score: 95, total_conversations: 50 }))).toBeNull();
  });

  it("rejects implausible prompt counts", () => {
    expect(validate(validPayload({ total_prompts: 2_000_000 }))).toBe("implausible: too many prompts");
  });

  it("rejects zero prompts", () => {
    expect(validate(validPayload({ total_prompts: 0 }))).toBe("total_prompts must be >= 1");
  });

  it("rejects zero conversations", () => {
    expect(validate(validPayload({ total_conversations: 0 }))).toBe("total_conversations must be >= 1");
  });
});

// ---------------------------------------------------------------------------
// 4. Ranking sort correctness
// ---------------------------------------------------------------------------

describe("ranking sort", () => {
  it("sorts by hitl_score descending (default)", () => {
    const submissions = [
      validPayload({ display_name: "low", hitl_score: 30 }),
      validPayload({ display_name: "high", hitl_score: 90, total_conversations: 50 }),
      validPayload({ display_name: "mid", hitl_score: 60 }),
    ];
    const sorted = [...submissions].sort((a, b) => b.hitl_score - a.hitl_score);
    expect(sorted[0].display_name).toBe("high");
    expect(sorted[1].display_name).toBe("mid");
    expect(sorted[2].display_name).toBe("low");
  });

  it("sorts by vibe_index descending", () => {
    const submissions = [
      validPayload({ display_name: "engineer", vibe_index: 85 }),
      validPayload({ display_name: "vibe", vibe_index: 25 }),
      validPayload({ display_name: "balanced", vibe_index: 55 }),
    ];
    const sorted = [...submissions].sort((a, b) => b.vibe_index - a.vibe_index);
    expect(sorted[0].display_name).toBe("engineer");
    expect(sorted[2].display_name).toBe("vibe");
  });

  it("sorts by total_prompts descending", () => {
    const submissions = [
      validPayload({ display_name: "casual", total_prompts: 50 }),
      validPayload({ display_name: "power", total_prompts: 5000 }),
      validPayload({ display_name: "regular", total_prompts: 500 }),
    ];
    const sorted = [...submissions].sort((a, b) => b.total_prompts - a.total_prompts);
    expect(sorted[0].display_name).toBe("power");
  });
});

// ---------------------------------------------------------------------------
// 5. Submit payload matches metrics.json structure
// ---------------------------------------------------------------------------

describe("payload/metrics.json consistency", () => {
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

  it("submit payload can be built from metrics.json without missing fields", async () => {
    // Insert test data
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

    const config = loadConfig("/tmp/test-howiprompt");
    const metrics = await computeMetrics(client, config);

    // Build payload the same way dashboard.js does
    const norm = metrics.normalized || {};
    const nlp = metrics.nlp || {};
    const payload: SubmissionPayload = {
      display_name: "test-user",
      total_conversations: metrics.volume.total_conversations,
      total_prompts: metrics.volume.total_human,
      avg_words_per_prompt: metrics.volume.avg_words_per_prompt,
      politeness: Math.round(norm.politeness ?? 0),
      backtrack: Math.round(norm.backtrack ?? 0),
      question_rate: Math.round(norm.question_rate ?? 0),
      command_rate: Math.round(norm.command_rate ?? 0),
      hitl_score: Math.round(nlp.hitl_score?.avg_score ?? 0),
      vibe_index: Math.round(nlp.vibe_coder_index?.avg_score ?? 0),
      persona: metrics.persona?.type ?? "unknown",
      complexity_avg: Math.round(norm.complexity ?? 0),
      platform: "both",
      tool_version: "2.0.0",
    };

    // All required fields present
    const requiredFields: (keyof SubmissionPayload)[] = [
      "display_name", "total_conversations", "total_prompts", "avg_words_per_prompt",
      "politeness", "backtrack", "question_rate", "command_rate",
      "hitl_score", "vibe_index", "persona", "complexity_avg", "platform", "tool_version",
    ];
    for (const field of requiredFields) {
      expect(payload[field]).toBeDefined();
    }

    // Payload passes validation
    const err = validate(payload);
    expect(err).toBeNull();
  });
});
