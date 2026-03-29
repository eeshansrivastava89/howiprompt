import { describe, it, expect } from "vitest";
import { cosineSimilarity } from "../src/pipeline/embeddings.js";
import { loadMlConfig } from "../src/pipeline/ml-config.js";

describe("cosineSimilarity", () => {
  it("returns 1 for identical vectors", () => {
    const v = new Float32Array([1, 2, 3]);
    expect(cosineSimilarity(v, v)).toBeCloseTo(1.0, 5);
  });

  it("returns 0 for orthogonal vectors", () => {
    const a = new Float32Array([1, 0, 0]);
    const b = new Float32Array([0, 1, 0]);
    expect(cosineSimilarity(a, b)).toBeCloseTo(0.0, 5);
  });

  it("returns -1 for opposite vectors", () => {
    const a = new Float32Array([1, 0, 0]);
    const b = new Float32Array([-1, 0, 0]);
    expect(cosineSimilarity(a, b)).toBeCloseTo(-1.0, 5);
  });

  it("returns correct similarity for non-trivial vectors", () => {
    const a = new Float32Array([1, 2, 3]);
    const b = new Float32Array([4, 5, 6]);
    const sim = cosineSimilarity(a, b);
    expect(sim).toBeGreaterThan(0.9); // highly similar direction
    expect(sim).toBeLessThanOrEqual(1.0);
  });
});

describe("loadMlConfig", () => {
  it("loads default config", () => {
    const config = loadMlConfig("/tmp/nonexistent-dir");
    expect(config.embedding.model).toBe("Xenova/bge-small-en-v1.5");
    expect(config.embedding.dimensions).toBe(384);
    expect(config.embedding.dtype).toBe("int8");
    expect(config.embedding.batchSize).toBe(64);
  });

  it("has HITL classifier weights", () => {
    const config = loadMlConfig("/tmp/nonexistent-dir");
    expect(config.hitl.weights.course_correction).toBe(20);
    expect(config.hitl.weights.passive_delegation).toBe(-10);
    expect(config.hitl.similarityThreshold).toBe(0.55);
  });

  it("has Vibe classifier weights", () => {
    const config = loadMlConfig("/tmp/nonexistent-dir");
    expect(config.vibe.weights.file_reference).toBe(18);
    expect(config.vibe.weights.high_level_delegation).toBe(-15);
    expect(config.vibe.centerPoint).toBe(50);
  });
});
