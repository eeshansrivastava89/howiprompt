export enum Platform {
  CLAUDE_CODE = "claude_code",
  CODEX = "codex",
  AGENT = "agent",
}

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
  COLLABORATOR = "collaborator",
  EXPLORER = "explorer",
  EFFICIENT = "efficient",
  PRAGMATIST = "pragmatist",
}

export interface PersonaDefinition {
  name: string;
  description: string;
  traits: string[];
}

export const PERSONAS: Record<PersonaType, PersonaDefinition> = {
  [PersonaType.COLLABORATOR]: {
    name: "The Collaborator",
    description: "You ask questions politely. AI is your partner, not your tool.",
    traits: ["Inquisitive", "Courteous", "Partnership-oriented"],
  },
  [PersonaType.EXPLORER]: {
    name: "The Explorer",
    description: "You question, iterate, and dig deeper. Thinking out loud.",
    traits: ["Curious", "Iterative", "Thorough"],
  },
  [PersonaType.EFFICIENT]: {
    name: "The Efficient",
    description: "Polite but focused. You know what you want and ask nicely.",
    traits: ["Respectful", "Direct", "Purposeful"],
  },
  [PersonaType.PRAGMATIST]: {
    name: "The Pragmatist",
    description: "Balanced and practical. No frills, just results.",
    traits: ["Balanced", "Practical", "Focused"],
  },
};
