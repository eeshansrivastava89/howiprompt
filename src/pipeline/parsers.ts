import fs from "node:fs";
import path from "node:path";
import readline from "node:readline";
import { Message, Platform, Role } from "./models.js";

function findJsonlFiles(dir: string, excludedDirs: Set<string> = new Set()): string[] {
  const results: string[] = [];
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      if (excludedDirs.has(entry.name)) continue;
      results.push(...findJsonlFiles(full, excludedDirs));
    } else if (entry.name.endsWith(".jsonl")) results.push(full);
  }
  return results;
}

export async function parseClaudeCode(
  sourceDir: string,
  agentCwds: string[] = [],
): Promise<Message[]> {
  if (!fs.existsSync(sourceDir)) return [];

  // Convert excluded cwd paths to Claude project directory names
  // e.g. "/path/to/project" -> "-path-to-project"
  const excludedDirs = new Set(agentCwds.map((p) => p.replace(/\//g, "-")));
  const messages: Message[] = [];
  const allFiles = findJsonlFiles(sourceDir, excludedDirs);

  for (const filePath of allFiles) {
      const sessionId = path.basename(filePath, ".jsonl");

      const lines = fs.readFileSync(filePath, "utf-8").split("\n");

      for (const line of lines) {
        if (!line.trim()) continue;
        let entry: any;
        try {
          entry = JSON.parse(line);
        } catch {
          continue;
        }

        const entryType = entry.type;
        if (entryType !== "user" && entryType !== "assistant") continue;
        if (entry.isMeta) continue;

        const msgData = entry.message ?? {};
        let content = "";
        let role: Role;

        if (msgData.role === "user") {
          content = extractContent(msgData.content);
          role = Role.HUMAN;
        } else if (msgData.role === "assistant") {
          content = extractContentBlocks(msgData.content);
          role = Role.ASSISTANT;
        } else {
          continue;
        }

        if (!content || !content.trim()) continue;
        if (content.startsWith("<command-") || content.startsWith("<local-command")) continue;

        const tsStr = entry.timestamp;
        if (!tsStr) continue;
        let timestamp: Date;
        try {
          timestamp = new Date(tsStr);
          if (isNaN(timestamp.getTime())) continue;
        } catch {
          continue;
        }

        messages.push({
          timestamp,
          platform: Platform.CLAUDE_CODE,
          role,
          content,
          conversationId: sessionId,
          wordCount: content.split(/\s+/).filter(Boolean).length,
          sourceFile: filePath,
        });
      }
  }

  return messages;
}

function extractContent(content: unknown): string {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .filter((p: any) => typeof p === "object" && p?.type === "text")
      .map((p: any) => p.text ?? "")
      .join(" ");
  }
  return "";
}

function extractContentBlocks(content: unknown): string {
  if (Array.isArray(content)) {
    return content
      .filter((b: any) => typeof b === "object" && b?.type === "text")
      .map((b: any) => b.text ?? "")
      .join(" ");
  }
  return typeof content === "string" ? content : "";
}

export interface SessionModel {
  modelId?: string;
  modelProvider?: string;
}

export async function parseCodexSessionMetadata(
  sessionsPath: string,
): Promise<Map<string, SessionModel>> {
  const result = new Map<string, SessionModel>();
  if (!fs.existsSync(sessionsPath)) return result;

  const walkDir = (dir: string): string[] => {
    const files: string[] = [];
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
      const full = path.join(dir, entry.name);
      if (entry.isDirectory()) files.push(...walkDir(full));
      else if (entry.name.endsWith(".jsonl")) files.push(full);
    }
    return files;
  };

  for (const filePath of walkDir(sessionsPath)) {
    const lines = fs.readFileSync(filePath, "utf-8").split("\n");
    let sessionId: string | undefined;
    let modelId: string | undefined;
    let modelProvider: string | undefined;

    for (let idx = 0; idx < Math.min(lines.length, 400); idx++) {
      const line = lines[idx];
      if (!line.trim()) continue;
      let entry: any;
      try {
        entry = JSON.parse(line);
      } catch {
        continue;
      }

      const payload = entry.payload;
      if (typeof payload !== "object" || payload === null) continue;

      if (entry.type === "session_meta") {
        if (typeof payload.id === "string" && payload.id.trim()) sessionId = payload.id.trim();
        if (typeof payload.model_provider === "string" && payload.model_provider.trim())
          modelProvider = payload.model_provider.trim();
        if (typeof payload.model === "string" && payload.model.trim())
          modelId = payload.model.trim();
      }

      if (entry.type === "turn_context") {
        if (typeof payload.model === "string" && payload.model.trim())
          modelId = payload.model.trim();
      }

      if (sessionId && modelId && modelProvider) break;
    }

    if (sessionId) {
      const existing = result.get(sessionId) ?? {};
      if (modelId && !existing.modelId) existing.modelId = modelId;
      if (modelProvider && !existing.modelProvider) existing.modelProvider = modelProvider;
      result.set(sessionId, existing);
    }
  }

  return result;
}

export async function parseCodexHistory(
  sourcePath: string,
  sessionModels?: Map<string, SessionModel>,
): Promise<Message[]> {
  if (!fs.existsSync(sourcePath)) return [];

  const messages: Message[] = [];
  const lines = fs.readFileSync(sourcePath, "utf-8").split("\n");

  for (const line of lines) {
    if (!line.trim()) continue;
    let entry: any;
    try {
      entry = JSON.parse(line);
    } catch {
      continue;
    }

    const text = entry.text;
    if (typeof text !== "string" || !text.trim()) continue;

    const ts = entry.ts;
    if (typeof ts !== "number") continue;

    const sessionId = String(entry.session_id ?? "unknown");
    let timestamp: Date;
    try {
      timestamp = new Date(ts * 1000);
      if (isNaN(timestamp.getTime())) continue;
    } catch {
      continue;
    }

    const modelData = sessionModels?.get(sessionId);
    messages.push({
      timestamp,
      platform: Platform.CODEX,
      role: Role.HUMAN,
      content: text,
      conversationId: sessionId,
      wordCount: text.split(/\s+/).filter(Boolean).length,
      modelId: modelData?.modelId,
      modelProvider: modelData?.modelProvider,
      sourceFile: sourcePath,
    });
  }

  return messages;
}
