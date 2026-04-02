import { describe, it, expect, beforeEach } from "vitest";
import { scoreVibe, scorePoliteness, resetScorersForTests } from "../src/pipeline/scorers.js";
import { extractFeatures } from "../src/pipeline/style.js";

describe("scorers", () => {
  beforeEach(() => {
    resetScorersForTests();
  });

  it("scoreVibe returns a score between 0 and 100", () => {
    const features = extractFeatures("fix the auth bug on line 42", 7);
    const result = scoreVibe(features);
    expect(result.score).toBeGreaterThanOrEqual(0);
    expect(result.score).toBeLessThanOrEqual(100);
    expect(result.classLabel).toBeGreaterThanOrEqual(1);
    expect(result.classLabel).toBeLessThanOrEqual(5);
    expect(result.confidence).toBeGreaterThan(0);
    expect(result.confidence).toBeLessThanOrEqual(1);
  });

  it("scorePoliteness returns a score between 0 and 100", () => {
    const features = extractFeatures("could you please help me with this?", 7);
    const result = scorePoliteness(features);
    expect(result.score).toBeGreaterThanOrEqual(0);
    expect(result.score).toBeLessThanOrEqual(100);
    expect(result.classLabel).toBeGreaterThanOrEqual(1);
    expect(result.classLabel).toBeLessThanOrEqual(5);
  });

  it("vibe-y prompts score higher than engineered ones", () => {
    const vibey = scoreVibe(extractFeatures("just do it", 3));
    const engineered = scoreVibe(extractFeatures(
      "1. Fix the JWT validation in auth.ts line 42\n2. Make sure it handles expired tokens\n3. Add unit tests",
      20,
    ));
    expect(vibey.score).toBeGreaterThan(engineered.score);
  });

  it("polite prompts score higher than curt ones", () => {
    const polite = scorePoliteness(extractFeatures(
      "Could you please help me understand how this works? Thanks!",
      10,
    ));
    const curt = scorePoliteness(extractFeatures("fix it", 2));
    expect(polite.score).toBeGreaterThan(curt.score);
  });
});
