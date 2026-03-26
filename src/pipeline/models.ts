export enum Platform {
  CLAUDE_CODE = "claude_code",
  CODEX = "codex",
  COPILOT_CHAT = "copilot_chat",
  CURSOR = "cursor",
  LMSTUDIO = "lmstudio",
}

export const PLATFORM_VALUES: Platform[] = [
  Platform.CLAUDE_CODE,
  Platform.CODEX,
  Platform.COPILOT_CHAT,
  Platform.CURSOR,
  Platform.LMSTUDIO,
];

export enum Role {
  HUMAN = "human",
  ASSISTANT = "assistant",
}

export interface Message {
  timestamp: Date;
  platform: Platform;
  role: Role;
  content: string;
  conversationId: string;
  wordCount: number;
  modelId?: string;
  modelProvider?: string;
  sourceFile?: string;
}

export enum PersonaType {
  ARCHITECT = "architect",
  EXPLORER = "explorer",
  COMMANDER = "commander",
  PARTNER = "partner",
  DELEGATOR = "delegator",
}

export interface PersonaDefinition {
  name: string;
  description: string;
  traits: string[];
}

export const PERSONAS: Record<PersonaType, PersonaDefinition> = {
  [PersonaType.ARCHITECT]: {
    name: "The Architect",
    description: "You approach AI with precision and skepticism \u2014 every prompt is a spec, every output gets reviewed. You don't delegate blindly; you methodically steer the conversation toward exactly what you need.",
    traits: ["Precise", "Skeptical", "Methodical"],
  },
  [PersonaType.EXPLORER]: {
    name: "The Explorer",
    description: "You use AI to learn, investigate, and think out loud. Every conversation is an expedition \u2014 you ask questions, dig into details, and follow threads wherever they lead.",
    traits: ["Curious", "Analytical", "Open-minded"],
  },
  [PersonaType.COMMANDER]: {
    name: "The Commander",
    description: "You give precise orders in short bursts. No wasted words, no hand-holding \u2014 you know exactly what you want and expect the AI to deliver it on the first try.",
    traits: ["Direct", "Exacting", "Efficient"],
  },
  [PersonaType.PARTNER]: {
    name: "The Partner",
    description: "You build together in long collaborative sessions, iterating until it's right. You treat AI as a co-pilot, not a tool \u2014 offering context, refining together, shipping as a team.",
    traits: ["Collaborative", "Persistent", "Iterative"],
  },
  [PersonaType.DELEGATOR]: {
    name: "The Delegator",
    description: "You paint the vision and let AI handle the details. You trust the output, focus on outcomes over process, and move fast by giving AI maximum autonomy.",
    traits: ["Big-picture", "Trusting", "Outcome-focused"],
  },
};
