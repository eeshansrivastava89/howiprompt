import type { Config } from "./config.js";
import { PERSONAS, PersonaType } from "./models.js";

export function classifyPersona(
  politeness: number,
  backtrack: number,
  questionRate: number,
  commandRate: number,
  config: Config,
): Record<string, any> {
  const engagementScore = (questionRate + backtrack) / 2;
  const politenessScore = politeness - commandRate * 0.5;

  const highEngagement = engagementScore >= config.engagementThreshold;
  const highPoliteness = politenessScore >= config.politenessThreshold;

  let personaType: PersonaType;
  if (highEngagement && highPoliteness) personaType = PersonaType.COLLABORATOR;
  else if (highEngagement && !highPoliteness) personaType = PersonaType.EXPLORER;
  else if (!highEngagement && highPoliteness) personaType = PersonaType.EFFICIENT;
  else personaType = PersonaType.PRAGMATIST;

  const persona = PERSONAS[personaType];

  return {
    type: personaType,
    name: persona.name,
    description: persona.description,
    traits: persona.traits,
    quadrant: {
      engagement_score: round(engagementScore, 1),
      politeness_score: round(politenessScore, 1),
      high_engagement: highEngagement,
      high_politeness: highPoliteness,
    },
    scores: {
      politeness: round(politeness, 1),
      backtrack: round(backtrack, 1),
      question_rate: round(questionRate, 1),
      command_rate: round(commandRate, 1),
    },
  };
}

function round(n: number, decimals: number): number {
  const factor = 10 ** decimals;
  return Math.round(n * factor) / factor;
}
