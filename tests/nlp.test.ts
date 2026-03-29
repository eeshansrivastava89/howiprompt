import { describe, it, expect } from "vitest";
import { classifyIntent, computeComplexity, computeIterationStyle } from "../src/pipeline/nlp.js";

describe("classifyIntent", () => {
  it("returns debug_fix for error-related text", () => {
    const result = classifyIntent("fix the broken authentication bug");
    expect(result.intent).toBe("debug_fix");
    expect(result.confidence).toBeGreaterThanOrEqual(0.5);
    expect(result.confidence).toBeLessThanOrEqual(0.95);
  });

  it("returns build_feature for creation text", () => {
    const result = classifyIntent("create a new login feature");
    expect(result.intent).toBe("build_feature");
  });

  it("returns explanation_learning for explanation requests", () => {
    const result = classifyIntent("explain how the auth middleware works");
    expect(result.intent).toBe("explanation_learning");
  });

  it("returns other with low confidence for generic text", () => {
    const result = classifyIntent("hello world");
    expect(result.intent).toBe("other");
    expect(result.confidence).toBe(0.5);
  });

  it("confidence never exceeds 0.95", () => {
    const result = classifyIntent("debug fix error bug failing broken traceback stack trace");
    expect(result.confidence).toBeLessThanOrEqual(0.95);
  });
});

describe("computeComplexity", () => {
  it("returns low score for short simple text", () => {
    const result = computeComplexity("fix the bug");
    expect(result.score).toBeLessThan(2.5);
    expect(result.confidence).toBeGreaterThanOrEqual(0.65);
  });

  it("returns higher score for long structured text", () => {
    const result = computeComplexity(
      "Please create a new authentication module that must handle OAuth2 flows.\nIt should support at least Google and GitHub providers.\nThe implementation must be backwards compatible with the existing session store.",
    );
    expect(result.score).toBeGreaterThan(2.5);
    expect(result.confidence).toBeGreaterThan(0.65);
  });

  it("score is capped at 5.0", () => {
    const longText = "must should without exactly at least at most step checklist constraint\n".repeat(20);
    const result = computeComplexity(longText);
    expect(result.score).toBeLessThanOrEqual(5.0);
  });

  it("confidence capped at 0.95", () => {
    const result = computeComplexity(
      "must should without exactly\ncode { } ` -- ./ = ->\n" + "word ".repeat(80),
    );
    expect(result.confidence).toBeLessThanOrEqual(0.95);
  });
});

describe("computeIterationStyle", () => {
  it("returns 0 for text with no iteration markers", () => {
    const result = computeIterationStyle("hello world");
    expect(result.score).toBe(0);
  });

  it("scores higher with multiple markers", () => {
    const result = computeIterationStyle("actually wait, change that to something different, retry again");
    expect(result.score).toBeGreaterThan(40);
  });

  it("adds bonus for question marks", () => {
    const withQ = computeIterationStyle("actually, can you change that?");
    const withoutQ = computeIterationStyle("actually, change that");
    expect(withQ.score).toBeGreaterThan(withoutQ.score);
  });

  it("score capped at 100", () => {
    const text = "actually wait instead change update fix revise retry again different scratch that rework ".repeat(5);
    const result = computeIterationStyle(text);
    expect(result.score).toBeLessThanOrEqual(100);
  });
});
