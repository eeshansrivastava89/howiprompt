import { describe, it, expect } from "vitest";
import { classifyPersona } from "../src/pipeline/persona.js";

describe("classifyPersona", () => {
  it("classifies Architect (high precision, low trust)", () => {
    const result = classifyPersona({ precision: 90, curiosity: 30, tenacity: 60, trust: 20 });
    expect(result.type).toBe("architect");
    expect(result.name).toBe("The Architect");
  });

  it("classifies Explorer (high curiosity)", () => {
    const result = classifyPersona({ precision: 35, curiosity: 90, tenacity: 70, trust: 65 });
    expect(result.type).toBe("explorer");
    expect(result.name).toBe("The Explorer");
  });

  it("classifies Commander (high precision, low tenacity)", () => {
    const result = classifyPersona({ precision: 80, curiosity: 15, tenacity: 25, trust: 15 });
    expect(result.type).toBe("commander");
    expect(result.name).toBe("The Commander");
  });

  it("classifies Partner (high tenacity)", () => {
    const result = classifyPersona({ precision: 55, curiosity: 60, tenacity: 85, trust: 70 });
    expect(result.type).toBe("partner");
    expect(result.name).toBe("The Partner");
  });

  it("classifies Delegator (high trust)", () => {
    const result = classifyPersona({ precision: 20, curiosity: 25, tenacity: 30, trust: 90 });
    expect(result.type).toBe("delegator");
    expect(result.name).toBe("The Delegator");
  });

  it("includes all expected keys", () => {
    const result = classifyPersona({ precision: 50, curiosity: 50, tenacity: 50, trust: 50 });
    expect(result).toHaveProperty("type");
    expect(result).toHaveProperty("name");
    expect(result).toHaveProperty("description");
    expect(result).toHaveProperty("traits");
    expect(result).toHaveProperty("radar");
    expect(result.traits).toBeInstanceOf(Array);
    expect(result.radar).toHaveProperty("precision");
  });
});
