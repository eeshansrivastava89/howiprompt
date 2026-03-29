import { describe, it, expect } from "vitest";
import {
  extractFeatures,
  computeDetailScore,
  computeStyleScore,
  classifyQuadrant,
  getPersona,
  PERSONAS,
} from "../src/pipeline/style.js";

describe("extractFeatures", () => {
  it("extracts basic structural features", () => {
    const f = extractFeatures("fix the bug in auth.ts", 5);
    expect(f.word_count).toBe(5);
    expect(f.file_ref_count).toBe(1);
    expect(f.has_code_block).toBe(0);
  });

  it("detects code blocks and newlines", () => {
    const text = "Update this:\n```\nconst x = 1;\n```\nThanks!";
    const f = extractFeatures(text, 6);
    expect(f.has_code_block).toBe(1);
    expect(f.newline_count).toBe(4);
    expect(f.has_thanks).toBe(1);
  });

  it("counts directive vs collaborative markers", () => {
    const directive = "just do it, I need this now";
    const collab = "could you maybe help me think through this?";
    const fd = extractFeatures(directive, 7);
    const fc = extractFeatures(collab, 9);
    expect(fd.directive_count).toBeGreaterThan(0);
    expect(fc.collaborative_count).toBeGreaterThan(0);
  });
});

describe("computeDetailScore", () => {
  it("returns 0-100 range", () => {
    const brief = extractFeatures("fix bug", 2);
    const detailed = extractFeatures(
      "Please refactor the auth middleware in src/auth.ts line 42.\n" +
      "Requirements:\n- Must handle OAuth tokens\n- Must validate expiry\n" +
      "- Must not break existing sessions\n```\nconst token = getToken();\n```",
      30,
    );
    const scoreBrief = computeDetailScore(brief);
    const scoreDetailed = computeDetailScore(detailed);
    expect(scoreBrief).toBeGreaterThanOrEqual(0);
    expect(scoreBrief).toBeLessThanOrEqual(100);
    expect(scoreDetailed).toBeGreaterThanOrEqual(0);
    expect(scoreDetailed).toBeLessThanOrEqual(100);
    expect(scoreDetailed).toBeGreaterThan(scoreBrief);
  });
});

describe("computeStyleScore", () => {
  it("returns 0-100 range", () => {
    const directive = extractFeatures("just fix it now", 4);
    const collab = extractFeatures("could you please help me understand this? maybe we should try a different approach", 14);
    const scoreDir = computeStyleScore(directive);
    const scoreCollab = computeStyleScore(collab);
    expect(scoreDir).toBeGreaterThanOrEqual(0);
    expect(scoreDir).toBeLessThanOrEqual(100);
    expect(scoreCollab).toBeGreaterThanOrEqual(0);
    expect(scoreCollab).toBeLessThanOrEqual(100);
    expect(scoreCollab).toBeGreaterThan(scoreDir);
  });
});

describe("classifyQuadrant", () => {
  it("maps to Brief + Directive when both below 50", () => {
    expect(classifyQuadrant(30, 25)).toBe("Brief + Directive");
  });
  it("maps to Brief + Collaborative when detail < 50, style >= 50", () => {
    expect(classifyQuadrant(30, 75)).toBe("Brief + Collaborative");
  });
  it("maps to Detailed + Directive when detail >= 50, style < 50", () => {
    expect(classifyQuadrant(80, 25)).toBe("Detailed + Directive");
  });
  it("maps to Detailed + Collaborative when both >= 50", () => {
    expect(classifyQuadrant(80, 75)).toBe("Detailed + Collaborative");
  });
  it("uses 50 as the boundary (exactly 50 = detailed/collaborative)", () => {
    expect(classifyQuadrant(50, 50)).toBe("Detailed + Collaborative");
    expect(classifyQuadrant(49.9, 49.9)).toBe("Brief + Directive");
  });
});

describe("getPersona", () => {
  it("returns Commander for Brief + Directive", () => {
    const p = getPersona(20, 20);
    expect(p.name).toBe("The Commander");
    expect(p.quadrant).toBe("Brief + Directive");
  });
  it("returns Partner for Brief + Collaborative", () => {
    const p = getPersona(20, 80);
    expect(p.name).toBe("The Partner");
  });
  it("returns Architect for Detailed + Directive", () => {
    const p = getPersona(80, 20);
    expect(p.name).toBe("The Architect");
  });
  it("returns Explorer for Detailed + Collaborative", () => {
    const p = getPersona(80, 80);
    expect(p.name).toBe("The Explorer");
  });
});

describe("PERSONAS registry", () => {
  it("has exactly 4 entries", () => {
    expect(Object.keys(PERSONAS)).toHaveLength(4);
  });
  it("each persona has name, quadrant, description, traits", () => {
    for (const p of Object.values(PERSONAS)) {
      expect(p.name).toBeTruthy();
      expect(p.quadrant).toBeTruthy();
      expect(p.description).toBeTruthy();
      expect(p.traits.length).toBeGreaterThan(0);
    }
  });
});
