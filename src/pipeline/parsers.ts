import fs from "node:fs";
import path from "node:path";
import { createClient } from "@libsql/client";
import { Message, Platform, Role } from "./models.js";
import type { CompiledRules } from "./exclusions.js";
import { shouldExcludeContent, shouldExcludeCwd, shouldExcludeDir } from "./exclusions.js";

function findJsonlFiles(
  dir: string,
  excludedDirs: Set<string> = new Set(),
  rules?: CompiledRules,
): string[] {
  const results: string[] = [];
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      // Check DB-driven dir exclusion rules, then legacy excludedDirs
      if (rules) {
        if (shouldExcludeDir(rules, entry.name).excluded) continue;
      } else {
        // Fallback: hardcoded subagents skip when no DB rules available
        if (entry.name === "subagents") continue;
      }
      if (excludedDirs.has(entry.name)) continue;
      results.push(...findJsonlFiles(full, excludedDirs, rules));
    } else if (entry.name.endsWith(".jsonl") || entry.name.includes(".jsonl.backup.")) results.push(full);
  }
  return results;
}

function findFilesRecursive(
  dir: string,
  matcher: (filePath: string, entryName: string) => boolean,
): string[] {
  const results: string[] = [];
  if (!fs.existsSync(dir)) return results;

  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      results.push(...findFilesRecursive(full, matcher));
    } else if (matcher(full, entry.name)) {
      results.push(full);
    }
  }

  return results;
}

function sortMessages(messages: Message[]): Message[] {
  return messages.sort((a, b) => {
    const tsDelta = a.timestamp.getTime() - b.timestamp.getTime();
    if (tsDelta !== 0) return tsDelta;
    if (a.conversationId !== b.conversationId) return a.conversationId.localeCompare(b.conversationId);
    if (a.role !== b.role) return a.role === Role.HUMAN ? -1 : 1;
    return a.content.localeCompare(b.content);
  });
}

export async function parseClaudeCode(
  sourceDir: string,
  agentCwds: string[] = [],
  rules?: CompiledRules,
): Promise<Message[]> {
  if (!fs.existsSync(sourceDir)) return [];

  // Convert excluded cwd paths to Claude project directory names
  // e.g. "/path/to/project" -> "-path-to-project"
  const excludedDirs = new Set(agentCwds.map((p) => p.replace(/\//g, "-")));
  const messages: Message[] = [];
  const allFiles = findJsonlFiles(sourceDir, excludedDirs, rules);

  // Track sessions excluded by CWD rules (skip all messages from that session)
  const excludedSessions = new Set<string>();

  for (const filePath of allFiles) {
    const sessionId = path.basename(filePath, ".jsonl");
    if (excludedSessions.has(sessionId)) continue;

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
      // Structural JSONL checks — these are format-level, not content rules
      if (entry.isMeta) continue;

      const tsStr = entry.timestamp;
      if (!tsStr) continue;
      const timestamp = new Date(tsStr);
      if (isNaN(timestamp.getTime())) continue;

      // CWD-based exclusion: if entry has a cwd matching a rule, skip entire session
      if (rules && entry.cwd) {
        const cwdCheck = shouldExcludeCwd(rules, entry.cwd);
        if (cwdCheck.excluded) {
          excludedSessions.add(sessionId);
          break;
        }
      }

      const msgData = entry.message ?? {};
      if (msgData.role === "user") {
        // Structural JSONL checks
        if (entry.sourceToolAssistantUUID || entry.toolUseResult) continue;
        const content = extractContent(msgData.content).trim();
        if (!content) continue;

        // Content-based exclusion: DB rules when available, fallback for tests
        if (rules) {
          const contentCheck = shouldExcludeContent(rules, content, Platform.CLAUDE_CODE);
          if (contentCheck.excluded) continue;
        } else {
          if (content.startsWith("<command-") || content.startsWith("<local-command")) continue;
        }

        appendClaudeTurn(messages, {
          timestamp,
          platform: Platform.CLAUDE_CODE,
          role: Role.HUMAN,
          content,
          conversationId: sessionId,
          wordCount: content.split(/\s+/).filter(Boolean).length,
          sourceFile: filePath,
        });
        continue;
      }

      if (msgData.role !== "assistant") continue;
      const content = extractContentBlocks(msgData.content).trim();
      if (!content) continue;

      appendClaudeTurn(messages, {
        timestamp,
        platform: Platform.CLAUDE_CODE,
        role: Role.ASSISTANT,
        content,
        conversationId: sessionId,
        wordCount: content.split(/\s+/).filter(Boolean).length,
        sourceFile: filePath,
      });
    }
  }

  return sortMessages(messages);
}

function appendClaudeTurn(messages: Message[], next: Message): void {
  const prev = messages[messages.length - 1];
  if (!prev) {
    messages.push(next);
    return;
  }

  if (
    prev.platform === next.platform
    && prev.conversationId === next.conversationId
    && prev.role === next.role
  ) {
    if (!prev.content.includes(next.content)) {
      prev.content = `${prev.content} ${next.content}`.trim();
      prev.wordCount = prev.content.split(/\s+/).filter(Boolean).length;
    }
    if (next.timestamp < prev.timestamp) prev.timestamp = next.timestamp;
    return;
  }

  messages.push(next);
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

  return sortMessages(messages);
}

function parseDateLike(value: unknown): Date | null {
  if (typeof value !== "number" && typeof value !== "string") return null;
  const asNumber = Number(value);
  const millis = Number.isFinite(asNumber) ? asNumber : NaN;
  const dt = Number.isFinite(millis) && millis > 1e11
    ? new Date(millis)
    : new Date(String(value));
  return Number.isNaN(dt.getTime()) ? null : dt;
}

function safeJsonParse(filePath: string): any | null {
  try {
    return JSON.parse(fs.readFileSync(filePath, "utf-8"));
  } catch {
    return null;
  }
}

function extractVsCodeResponseText(response: unknown): string {
  if (!Array.isArray(response)) return "";
  return response
    .filter((part) => typeof part === "object" && part !== null && typeof (part as any).value === "string")
    .map((part: any) => part.value)
    .join("")
    .trim();
}

function inferModelProvider(modelId?: string): string | undefined {
  if (!modelId) return undefined;
  const trimmed = modelId.trim();
  if (!trimmed) return undefined;
  const [provider] = trimmed.split("/", 1);
  return provider || undefined;
}

export async function parseVsCodeChatSessions(
  sourceDir: string,
  platform: Platform.COPILOT_CHAT | Platform.CURSOR,
): Promise<Message[]> {
  if (!fs.existsSync(sourceDir)) return [];

  const files = findFilesRecursive(sourceDir, (filePath, entryName) =>
    entryName.endsWith(".json") && filePath.includes(`${path.sep}chatSessions${path.sep}`),
  );
  const messages: Message[] = [];

  for (const filePath of files) {
    const session = safeJsonParse(filePath);
    if (!session || !Array.isArray(session.requests)) continue;

    const conversationId = typeof session.sessionId === "string" && session.sessionId.trim()
      ? session.sessionId.trim()
      : path.basename(filePath, ".json");

    for (let index = 0; index < session.requests.length; index++) {
      const request = session.requests[index];
      const promptText = request?.message?.text;
      if (typeof promptText !== "string" || !promptText.trim()) continue;

      const baseTime = parseDateLike(request.timestamp)
        ?? parseDateLike(session.creationDate)
        ?? new Date(0);
      const modelId = typeof request.modelId === "string" && request.modelId.trim()
        ? request.modelId.trim()
        : undefined;
      const modelProvider = inferModelProvider(modelId);

      messages.push({
        timestamp: new Date(baseTime.getTime()),
        platform,
        role: Role.HUMAN,
        content: promptText.trim(),
        conversationId,
        wordCount: promptText.trim().split(/\s+/).filter(Boolean).length,
        modelId,
        modelProvider,
        sourceFile: filePath,
      });

      const responseText = extractVsCodeResponseText(request.response);
      if (!responseText) continue;

      messages.push({
        timestamp: new Date(baseTime.getTime() + 1 + index),
        platform,
        role: Role.ASSISTANT,
        content: responseText,
        conversationId,
        wordCount: responseText.split(/\s+/).filter(Boolean).length,
        modelId,
        modelProvider,
        sourceFile: filePath,
      });
    }
  }

  return sortMessages(messages);
}

function extractLmStudioTextBlocks(content: unknown): string {
  if (!Array.isArray(content)) return "";
  return content
    .filter((block) => typeof block === "object" && block !== null && (block as any).type === "text")
    .map((block: any) => String(block.text ?? ""))
    .join(" ")
    .trim();
}

function extractLmStudioStepText(steps: unknown): string {
  if (!Array.isArray(steps)) return "";
  return steps
    .filter((step) => typeof step === "object" && step !== null)
    .flatMap((step: any) => Array.isArray(step.content) ? step.content : [])
    .filter((block) => typeof block === "object" && block !== null && (block as any).type === "text")
    .map((block: any) => String(block.text ?? ""))
    .join(" ")
    .trim();
}

function extractLmStudioStepTimestamp(steps: unknown): Date | null {
  if (!Array.isArray(steps)) return null;
  for (const step of steps) {
    const stepId = typeof (step as any)?.stepIdentifier === "string" ? (step as any).stepIdentifier : "";
    const match = stepId.match(/^(\d{13})/);
    if (match) {
      const dt = new Date(Number(match[1]));
      if (!Number.isNaN(dt.getTime())) return dt;
    }
  }
  return null;
}

export async function parseLmStudioConversations(sourceDir: string): Promise<Message[]> {
  if (!fs.existsSync(sourceDir)) return [];

  const files = findFilesRecursive(sourceDir, (_filePath, entryName) =>
    entryName.endsWith(".conversation.json"),
  );
  const messages: Message[] = [];

  for (const filePath of files) {
    const conversation = safeJsonParse(filePath);
    if (!conversation || !Array.isArray(conversation.messages)) continue;

    const conversationId = path.basename(filePath, ".conversation.json");
    const createdAt = parseDateLike(conversation.createdAt) ?? new Date(0);

    for (let index = 0; index < conversation.messages.length; index++) {
      const message = conversation.messages[index];
      const selectedIndex = Number.isInteger(message?.currentlySelected) ? message.currentlySelected : 0;
      const version = Array.isArray(message?.versions) ? message.versions[selectedIndex] ?? message.versions[0] : null;
      if (!version || typeof version.role !== "string") continue;

      let content = "";
      let role: Role | null = null;
      let timestamp = new Date(createdAt.getTime() + index);
      let modelId: string | undefined;
      let modelProvider: string | undefined;

      if (version.role === "user") {
        content = extractLmStudioTextBlocks(version.content);
        role = Role.HUMAN;
      } else if (version.role === "assistant") {
        content = extractLmStudioStepText(version.steps);
        role = Role.ASSISTANT;
        timestamp = extractLmStudioStepTimestamp(version.steps) ?? timestamp;
        modelId = typeof version.senderInfo?.senderName === "string" && version.senderInfo.senderName.trim()
          ? version.senderInfo.senderName.trim()
          : undefined;
        modelProvider = inferModelProvider(modelId);
        if (!modelId && Array.isArray(version.steps)) {
          for (const step of version.steps) {
            const identifier = typeof step?.genInfo?.identifier === "string" ? step.genInfo.identifier.trim() : "";
            if (identifier) {
              modelId = identifier;
              modelProvider = inferModelProvider(modelId);
              break;
            }
          }
        }
      }

      if (!role || !content) continue;

      if (role === Role.HUMAN) {
        const nextVersion = conversation.messages[index + 1]?.versions?.[
          Number.isInteger(conversation.messages[index + 1]?.currentlySelected)
            ? conversation.messages[index + 1].currentlySelected
            : 0
        ] ?? conversation.messages[index + 1]?.versions?.[0];
        const nextAssistantTime = nextVersion?.role === "assistant"
          ? extractLmStudioStepTimestamp(nextVersion.steps)
          : null;
        if (nextAssistantTime) {
          timestamp = new Date(nextAssistantTime.getTime() - 1);
        } else {
          timestamp = new Date(createdAt.getTime() + index);
        }
      }

      messages.push({
        timestamp,
        platform: Platform.LMSTUDIO,
        role,
        content,
        conversationId,
        wordCount: content.split(/\s+/).filter(Boolean).length,
        modelId,
        modelProvider,
        sourceFile: filePath,
      });
    }
  }

  return sortMessages(messages);
}

// ── Pi agent ────────────────────────────────────────────

export async function parsePiSessions(sourceDir: string): Promise<Message[]> {
  if (!fs.existsSync(sourceDir)) return [];

  const files = findFilesRecursive(sourceDir, (_fp, name) => name.endsWith(".jsonl"));
  const messages: Message[] = [];

  for (const filePath of files) {
    let lines: string[];
    try {
      lines = fs.readFileSync(filePath, "utf-8").split("\n").filter(Boolean);
    } catch {
      continue;
    }

    let conversationId = path.basename(filePath, ".jsonl");
    let currentModel: string | undefined;
    let currentProvider: string | undefined;

    for (const line of lines) {
      let entry: any;
      try {
        entry = JSON.parse(line);
      } catch {
        continue;
      }

      const entryType = entry.type;

      if (entryType === "session") {
        conversationId = entry.id ?? conversationId;
        continue;
      }

      if (entryType === "model_change") {
        currentProvider = entry.provider;
        currentModel = entry.modelId;
        continue;
      }

      if (entryType !== "message") continue;

      const msg = entry.message;
      if (!msg || typeof msg.role !== "string") continue;

      const tsStr = entry.timestamp;
      if (!tsStr) continue;
      const timestamp = new Date(tsStr);
      if (isNaN(timestamp.getTime())) continue;

      const content = extractContent(msg.content);
      if (!content) continue;

      const role = msg.role === "user" ? Role.HUMAN : msg.role === "assistant" ? Role.ASSISTANT : null;
      if (!role) continue;

      // Assistant messages carry their own model; user messages inherit from last model_change
      const modelId = role === Role.ASSISTANT ? (msg.model ?? currentModel) : currentModel;
      const modelProvider = role === Role.ASSISTANT ? (msg.provider ?? currentProvider) : currentProvider;

      messages.push({
        timestamp,
        platform: Platform.PI,
        role,
        content,
        conversationId,
        wordCount: content.split(/\s+/).filter(Boolean).length,
        modelId,
        modelProvider,
        sourceFile: filePath,
      });
    }
  }

  return sortMessages(messages);
}

// ── OpenCode ───────────────────────────────────────────

export async function parseOpenCodeSessions(sourceDir: string, dbPath?: string): Promise<Message[]> {
  if (dbPath && fs.existsSync(dbPath)) {
    return parseOpenCodeDb(dbPath);
  }

  if (!fs.existsSync(sourceDir)) return [];

  const messageDir = path.join(sourceDir, "message");
  const partDir = path.join(sourceDir, "part");
  if (!fs.existsSync(messageDir)) return [];

  const messages: Message[] = [];

  // Iterate all session subdirectories under message/
  let sessionDirs: string[];
  try {
    sessionDirs = fs.readdirSync(messageDir, { withFileTypes: true })
      .filter((e) => e.isDirectory())
      .map((e) => e.name);
  } catch {
    return [];
  }

  for (const sesDir of sessionDirs) {
    const sesMsgDir = path.join(messageDir, sesDir);
    let msgFiles: string[];
    try {
      msgFiles = fs.readdirSync(sesMsgDir)
        .filter((f) => f.endsWith(".json"))
        .sort();
    } catch {
      continue;
    }

    for (const msgFile of msgFiles) {
      const msgData = safeJsonParse(path.join(sesMsgDir, msgFile));
      if (!msgData || typeof msgData.role !== "string") continue;

      const role = msgData.role === "user" ? Role.HUMAN : msgData.role === "assistant" ? Role.ASSISTANT : null;
      if (!role) continue;

      const ts = msgData.time?.created;
      if (!ts) continue;
      const timestamp = new Date(typeof ts === "number" ? ts : ts);
      if (isNaN(timestamp.getTime())) continue;

      // Reconstruct content from parts
      const msgPartDir = path.join(partDir, msgData.id);
      let content = "";
      if (fs.existsSync(msgPartDir)) {
        try {
          const partFiles = fs.readdirSync(msgPartDir)
            .filter((f) => f.endsWith(".json"))
            .sort();
          const textParts: string[] = [];
          for (const pf of partFiles) {
            const part = safeJsonParse(path.join(msgPartDir, pf));
            if (part?.type === "text" && typeof part.text === "string" && part.text.trim()) {
              textParts.push(part.text);
            }
          }
          content = textParts.join("\n");
        } catch { /* skip */ }
      }

      if (!content) continue;

      const modelId = msgData.model?.modelID;
      const modelProvider = msgData.model?.providerID;
      const conversationId = msgData.sessionID ?? sesDir;

      messages.push({
        timestamp,
        platform: Platform.OPENCODE,
        role,
        content,
        conversationId,
        wordCount: content.split(/\s+/).filter(Boolean).length,
        modelId,
        modelProvider,
        sourceFile: path.join(sesMsgDir, msgFile),
      });
    }
  }

  return sortMessages(messages);
}

async function parseOpenCodeDb(dbPath: string): Promise<Message[]> {
  const client = createClient({ url: `file:${dbPath}` });
  try {
    const messageResult = await client.execute(
      "SELECT id, session_id, time_created, data FROM message ORDER BY time_created",
    );
    const partResult = await client.execute(
      "SELECT message_id, data FROM part ORDER BY time_created, id",
    );

    const partsByMessage = new Map<string, string[]>();
    for (const row of partResult.rows) {
      const messageId = String(row.message_id);
      const data = safeJsonValue(row.data);
      if (data?.type !== "text" || typeof data.text !== "string" || !data.text.trim()) continue;
      const parts = partsByMessage.get(messageId) ?? [];
      parts.push(data.text);
      partsByMessage.set(messageId, parts);
    }

    const messages: Message[] = [];
    for (const row of messageResult.rows) {
      const data = safeJsonValue(row.data);
      if (!data || typeof data.role !== "string") continue;

      const role = data.role === "user" ? Role.HUMAN : data.role === "assistant" ? Role.ASSISTANT : null;
      if (!role) continue;

      const created = Number(row.time_created ?? data.time?.created);
      if (!Number.isFinite(created)) continue;
      const timestamp = new Date(created);
      if (isNaN(timestamp.getTime())) continue;

      const messageId = String(row.id);
      const content = (partsByMessage.get(messageId) ?? []).join("\n").trim();
      if (!content) continue;

      messages.push({
        timestamp,
        platform: Platform.OPENCODE,
        role,
        content,
        conversationId: String(row.session_id),
        wordCount: content.split(/\s+/).filter(Boolean).length,
        modelId: data.modelID ?? data.model?.modelID,
        modelProvider: data.providerID ?? data.model?.providerID,
        sourceFile: `${dbPath}#message/${messageId}`,
      });
    }

    return sortMessages(messages);
  } finally {
    client.close();
  }
}

function safeJsonValue(value: unknown): any {
  if (typeof value !== "string") return null;
  try {
    return JSON.parse(value);
  } catch {
    return null;
  }
}
