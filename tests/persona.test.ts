import { describe, it, expect } from "vitest";
import { classifyPersona } from "../src/pipeline/persona.js";
import { loadConfig } from "../src/pipeline/config.js";

const config = loadConfig("/tmp/test-howiprompt");

describe("classifyPersona", () => {
  it("classifies Collaborator (high engagement, high politeness)", () => {
    const result = classifyPersona(20.0, 15.0, 25.0, 5.0, config);
    expect(result.type).toBe("collaborator");
    expect(result.name).toBe("The Collaborator");
    expect(result.quadrant.high_engagement).toBe(true);
    expect(result.quadrant.high_politeness).toBe(true);
  });

  it("classifies Explorer (high engagement, low politeness)", () => {
    const result = classifyPersona(2.0, 15.0, 20.0, 10.0, config);
    expect(result.type).toBe("explorer");
    expect(result.quadrant.high_engagement).toBe(true);
    expect(result.quadrant.high_politeness).toBe(false);
  });

  it("classifies Efficient (low engagement, high politeness)", () => {
    const result = classifyPersona(20.0, 2.0, 3.0, 5.0, config);
    expect(result.type).toBe("efficient");
    expect(result.quadrant.high_engagement).toBe(false);
    expect(result.quadrant.high_politeness).toBe(true);
  });

  it("classifies Pragmatist (low engagement, low politeness)", () => {
    const result = classifyPersona(2.0, 2.0, 3.0, 10.0, config);
    expect(result.type).toBe("pragmatist");
    expect(result.quadrant.high_engagement).toBe(false);
    expect(result.quadrant.high_politeness).toBe(false);
  });

  it("includes all expected keys", () => {
    const result = classifyPersona(10.0, 10.0, 10.0, 10.0, config);
    expect(result).toHaveProperty("type");
    expect(result).toHaveProperty("name");
    expect(result).toHaveProperty("description");
    expect(result).toHaveProperty("traits");
    expect(result).toHaveProperty("quadrant");
    expect(result).toHaveProperty("scores");
    expect(result.traits).toBeInstanceOf(Array);
  });
});
