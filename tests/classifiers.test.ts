import { describe, it, expect, beforeEach } from "vitest";
import { scoreVibe, scorePoliteness, resetScorersForTests } from "../src/pipeline/scorers.js";
import { extractFeatures } from "../src/pipeline/style.js";
import fs from "node:fs";
import path from "node:path";

describe("classifiers.json", () => {
  const config = JSON.parse(
    fs.readFileSync(path.join(__dirname, "..", "config", "classifiers.json"), "utf-8"),
  );

  it("has required top-level keys", () => {
    expect(config.features).toBeDefined();
    expect(Array.isArray(config.features)).toBe(true);
    expect(config.scaler).toBeDefined();
    expect(config.vibe).toBeDefined();
    expect(config.politeness).toBeDefined();
  });

  it("features list matches scaler dimensions", () => {
    expect(config.scaler.mean.length).toBe(config.features.length);
    expect(config.scaler.std.length).toBe(config.features.length);
  });

  it("vibe coefficients match feature count", () => {
    for (const classCoef of config.vibe.coef) {
      expect(classCoef.length).toBe(config.features.length);
    }
    expect(config.vibe.intercept.length).toBe(config.vibe.classes.length);
  });

  it("politeness coefficients match feature count", () => {
    for (const classCoef of config.politeness.coef) {
      expect(classCoef.length).toBe(config.features.length);
    }
    expect(config.politeness.intercept.length).toBe(config.politeness.classes.length);
  });

  it("all features in config are returned by extractFeatures", () => {
    const features = extractFeatures("test prompt", 2);
    for (const name of config.features) {
      expect(features).toHaveProperty(name);
    }
  });
});

describe("scorer output validation", () => {
  beforeEach(() => {
    resetScorersForTests();
  });

  it("scores are bounded 0-100 for various inputs", () => {
    const inputs = [
      "yes",
      "fix it",
      "can you please help me refactor the auth module?",
      "1. Update the schema\n2. Run migrations\n3. Test endpoint\n4. Deploy",
      "```\nconst x = 42;\nconsole.log(x);\n```\nwhy does this not work?",
    ];

    for (const input of inputs) {
      const features = extractFeatures(input, input.split(/\s+/).length);
      const vibe = scoreVibe(features);
      const pol = scorePoliteness(features);

      expect(vibe.score).toBeGreaterThanOrEqual(0);
      expect(vibe.score).toBeLessThanOrEqual(100);
      expect(pol.score).toBeGreaterThanOrEqual(0);
      expect(pol.score).toBeLessThanOrEqual(100);
    }
  });
});
