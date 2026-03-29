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

