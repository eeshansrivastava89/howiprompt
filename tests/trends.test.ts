import { describe, it, expect } from "vitest";
import { computeStyleSnapshot, buildTrendRollups, computeTrendDeltas, detectShiftMarkers } from "../src/pipeline/trends.js";
import type { MessageRow } from "../src/pipeline/db.js";

function makeRow(overrides: Partial<MessageRow> = {}): MessageRow {
  return {
    id: 1,
    timestamp: "2026-02-01T10:00:00Z",
    platform: "claude_code",
    role: "human",
    content: "fix the bug please",
    conversationId: "s1",
    wordCount: 4,
    modelId: null,
    modelProvider: null,
    ...overrides,
  };
}

describe("computeStyleSnapshot", () => {
  it("returns zeros for empty array", () => {
    const result = computeStyleSnapshot([]);
    expect(result.backtrack_per_100).toBe(0);
    expect(result.question_rate_pct).toBe(0);
  });

  it("counts question rate", () => {
    const msgs = [
      makeRow({ content: "how does this work?" }),
      makeRow({ content: "fix the bug" }),
    ];
    const result = computeStyleSnapshot(msgs);
    expect(result.question_rate_pct).toBe(50);
  });
});

describe("buildTrendRollups", () => {
  it("builds rollups from bucket map", () => {
    const buckets = new Map<string, MessageRow[]>();
    buckets.set("2026-02-01", [makeRow(), makeRow({ content: "another prompt" })]);
    buckets.set("2026-02-02", [makeRow({ content: "third prompt" })]);

    const rollups = buildTrendRollups(buckets, "date");
    expect(rollups.length).toBe(2);
    expect(rollups[0].date).toBe("2026-02-01");
    expect(rollups[0].prompts).toBe(2);
    expect(rollups[1].prompts).toBe(1);
  });
});

describe("computeTrendDeltas", () => {
  it("returns empty for no rollups", () => {
    expect(computeTrendDeltas([])).toEqual({});
  });

  it("computes deltas for sufficient data", () => {
    const rollups = Array.from({ length: 30 }, (_, i) => {
      const d = new Date(2026, 0, i + 1);
      return {
        date: d.toISOString().split("T")[0],
        prompts: 10 + (i > 22 ? 5 : 0),
        source_share_pct: { claude_code: 80, codex: 20 },
        style: { backtrack_per_100: 5, question_rate_pct: 30, command_rate_pct: 20 },
        model_prompts: 8,
      };
    });
    const deltas = computeTrendDeltas(rollups);
    expect(deltas).toHaveProperty("prompts_per_day");
    expect(deltas.prompts_per_day).toHaveProperty("avg_7d");
    expect(deltas.prompts_per_day).toHaveProperty("avg_30d");
    expect(deltas.prompts_per_day).toHaveProperty("delta_pct");
  });
});

describe("detectShiftMarkers", () => {
  it("returns empty for too few rollups", () => {
    expect(detectShiftMarkers([])).toEqual([]);
    expect(detectShiftMarkers([{ date: "2026-01-01", prompts: 10, source_share_pct: { codex: 20 } }])).toEqual([]);
  });

  it("detects large prompt shifts", () => {
    const rollups = [
      { date: "2026-01-01", prompts: 10, source_share_pct: { claude_code: 80, codex: 20 } },
      { date: "2026-01-02", prompts: 50, source_share_pct: { claude_code: 80, codex: 20 } },
    ];
    const markers = detectShiftMarkers(rollups);
    expect(markers.length).toBeGreaterThan(0);
    expect(markers[0].type).toBe("prompt_shift");
    expect(markers[0].direction).toBe("up");
  });
});
