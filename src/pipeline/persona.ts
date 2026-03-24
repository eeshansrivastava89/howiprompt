import { PERSONAS, PersonaType } from "./models.js";

export interface RadarScores {
  precision: number;
  curiosity: number;
  tenacity: number;
  trust: number;
}

export function classifyPersona(
  radar: RadarScores,
): Record<string, any> {
  const { precision, curiosity, tenacity, trust } = radar;

  // Find the dominant dimension(s) and derive persona
  const dims = [
    { key: "precision", value: precision },
    { key: "curiosity", value: curiosity },
    { key: "tenacity", value: tenacity },
    { key: "trust", value: trust },
  ].sort((a, b) => b.value - a.value);

  const top = dims[0];
  const second = dims[1];

  let personaType: PersonaType;

  if (top.key === "precision" && tenacity < 40) {
    personaType = PersonaType.COMMANDER;
  } else if (top.key === "precision" && trust < 40) {
    personaType = PersonaType.ARCHITECT;
  } else if (top.key === "curiosity") {
    personaType = PersonaType.EXPLORER;
  } else if (top.key === "tenacity" || (second.key === "tenacity" && tenacity > 60)) {
    personaType = PersonaType.PARTNER;
  } else if (top.key === "trust") {
    personaType = PersonaType.DELEGATOR;
  } else {
    // Fallback: pick based on strongest signal
    const mapping: Record<string, PersonaType> = {
      precision: PersonaType.ARCHITECT,
      curiosity: PersonaType.EXPLORER,
      tenacity: PersonaType.PARTNER,
      trust: PersonaType.DELEGATOR,
    };
    personaType = mapping[top.key] ?? PersonaType.PARTNER;
  }

  const persona = PERSONAS[personaType];

  return {
    type: personaType,
    name: persona.name,
    description: persona.description,
    traits: persona.traits,
    radar,
  };
}
