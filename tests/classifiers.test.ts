import { describe, it, expect } from "vitest";
import { cosineSimilarity } from "../src/pipeline/embeddings.js";
import { loadMlConfig } from "../src/pipeline/ml-config.js";
import fs from "node:fs";
import path from "node:path";

describe("reference_clusters.json", () => {
  const clusters = JSON.parse(
    fs.readFileSync(path.join(__dirname, "..", "data", "reference_clusters.json"), "utf-8"),
  );

  it("has hitl clusters with expected categories", () => {
    expect(clusters.hitl).toBeDefined();
    const expected = [
      "course_correction", "architectural_decision", "constraint_spec",
      "scope_control", "review_qa", "tradeoff_nav", "passive_delegation",
    ];
    for (const key of expected) {
      expect(clusters.hitl[key]).toBeDefined();
      expect(clusters.hitl[key].length).toBeGreaterThanOrEqual(10);
    }
  });

  it("has vibe clusters with expected categories", () => {
    expect(clusters.vibe).toBeDefined();
    const expected = [
      "file_reference", "technical_spec", "code_sharing",
      "iterative_refinement", "high_level_delegation", "outcome_only", "acceptance",
    ];
    for (const key of expected) {
      expect(clusters.vibe[key]).toBeDefined();
      expect(clusters.vibe[key].length).toBeGreaterThanOrEqual(10);
    }
  });

  it("all cluster examples are non-empty strings", () => {
    for (const [classifier, clusterMap] of Object.entries(clusters)) {
      for (const [cluster, examples] of Object.entries(clusterMap as Record<string, string[]>)) {
        for (const example of examples) {
          expect(typeof example).toBe("string");
          expect(example.trim().length).toBeGreaterThan(0);
        }
      }
    }
  });
});

describe("ml config weights match clusters", () => {
  const config = loadMlConfig("/tmp/nonexistent");
  const clusters = JSON.parse(
    fs.readFileSync(path.join(__dirname, "..", "data", "reference_clusters.json"), "utf-8"),
  );

  it("every hitl cluster has a weight", () => {
    for (const cluster of Object.keys(clusters.hitl)) {
      expect(config.hitl.weights[cluster]).toBeDefined();
    }
  });

  it("every vibe cluster has a weight", () => {
    for (const cluster of Object.keys(clusters.vibe)) {
      expect(config.vibe.weights[cluster]).toBeDefined();
    }
  });
});

describe("scoring logic (unit, no model)", () => {
  it("cosine similarity of normalized vectors is bounded [-1, 1]", () => {
    // Simulate what the classifier does: compare embedding to centroid
    const a = new Float32Array(384);
    const b = new Float32Array(384);
    // Fill with random values
    for (let i = 0; i < 384; i++) {
      a[i] = Math.random() - 0.5;
      b[i] = Math.random() - 0.5;
    }
    const sim = cosineSimilarity(a, b);
    expect(sim).toBeGreaterThanOrEqual(-1);
    expect(sim).toBeLessThanOrEqual(1);
  });

  it("similar vectors have high cosine similarity", () => {
    const a = new Float32Array(384);
    for (let i = 0; i < 384; i++) a[i] = Math.random();
    // b = a + small noise
    const b = new Float32Array(384);
    for (let i = 0; i < 384; i++) b[i] = a[i] + (Math.random() - 0.5) * 0.1;
    const sim = cosineSimilarity(a, b);
    expect(sim).toBeGreaterThan(0.9);
  });
});
