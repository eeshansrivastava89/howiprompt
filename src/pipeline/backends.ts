import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import {
  type SyncResult,
  syncClaudeCode,
  syncCodex,
  syncLmStudioConversations,
  syncVsCodeChatSessions,
  syncPiSessions,
  syncOpenCodeStorage,
} from "./sync.js";
import {
  parseClaudeCode,
  parseCodexHistory,
  parseCodexSessionMetadata,
  parseLmStudioConversations,
  parseVsCodeChatSessions,
  parsePiSessions,
  parseOpenCodeSessions,
} from "./parsers.js";
import { Platform, type Message } from "./models.js";
import type { Config } from "./config.js";

export interface BackendInfo {
  id: string;
  name: string;
  supported: boolean;
  detected: boolean;
  sourcePath: string;
  status: "available" | "coming_soon" | "not_found";
  messageCount?: number;
  dateRange?: { first: string; last: string };
}

const DETECTION_CACHE_TTL_MS = 10_000;
let detectionCache:
  | {
    expiresAt: number;
    infos: BackendInfo[];
  }
  | null = null;

// ── Platform helpers ───────────────────────────────────

/**
 * Resolve the user-data directory for VS Code-style apps across platforms.
 *   macOS:   ~/Library/Application Support/{appName}/User/workspaceStorage
 *   Windows: %APPDATA%/{appName}/User/workspaceStorage
 *   Linux:   ~/.config/{appName}/User/workspaceStorage
 */
function vsCodeDataDir(appName: string): string {
  const home = os.homedir();
  switch (os.platform()) {
    case "win32":
      return path.join(
        process.env.APPDATA ?? path.join(home, "AppData", "Roaming"),
        appName,
        "User",
        "workspaceStorage",
      );
    case "linux":
      return path.join(home, ".config", appName, "User", "workspaceStorage");
    default: // darwin
      return path.join(
        home,
        "Library",
        "Application Support",
        appName,
        "User",
        "workspaceStorage",
      );
  }
}

// ── Scan helpers ───────────────────────────────────────

function findJsonlFilesRecursive(dir: string): string[] {
  const results: string[] = [];
  try {
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
      const full = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        results.push(...findJsonlFilesRecursive(full));
      } else if (entry.name.endsWith(".jsonl")) {
        results.push(full);
      }
    }
  } catch { /* permission errors, etc */ }
  return results;
}

function findFilesRecursive(
  dir: string,
  matcher: (filePath: string, entryName: string) => boolean,
): string[] {
  const results: string[] = [];
  try {
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
      const full = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        results.push(...findFilesRecursive(full, matcher));
      } else if (matcher(full, entry.name)) {
        results.push(full);
      }
    }
  } catch { /* permission errors, etc */ }
  return results;
}

function parseIso(ms: number | null): string | undefined {
  if (!ms) return undefined;
  const dt = new Date(ms);
  return Number.isNaN(dt.getTime()) ? undefined : dt.toISOString();
}

/** Count lines and extract first/last timestamps from JSONL files. */
function scanClaudeCodeSource(sourceDir: string): { messageCount: number; dateRange?: { first: string; last: string } } {
  const files = findJsonlFilesRecursive(sourceDir);
  if (files.length === 0) return { messageCount: 0 };

  let totalLines = 0;
  let earliest: Date | null = null;
  let latest: Date | null = null;

  for (const file of files) {
    try {
      const content = fs.readFileSync(file, "utf-8");
      const lines = content.split("\n").filter(Boolean);
      totalLines += lines.length;

      // Check first and last lines for timestamps
      for (const line of [lines[0], lines[lines.length - 1]]) {
        if (!line) continue;
        try {
          const obj = JSON.parse(line);
          const ts = obj.timestamp ?? obj.snapshot?.timestamp;
          if (ts) {
            const d = new Date(ts);
            if (!earliest || d < earliest) earliest = d;
            if (!latest || d > latest) latest = d;
          }
        } catch { /* skip malformed lines */ }
      }
    } catch { /* skip unreadable files */ }
  }

  // Each JSONL line is a message (human + assistant + meta), rough /2 for human-only
  const messageCount = Math.round(totalLines / 2);
  const dateRange = earliest && latest
    ? { first: earliest.toISOString(), last: latest.toISOString() }
    : undefined;
  return { messageCount, dateRange };
}

function scanCodexSource(sourcePath: string): { messageCount: number; dateRange?: { first: string; last: string } } {
  try {
    const content = fs.readFileSync(sourcePath, "utf-8");
    const lines = content.split("\n").filter(Boolean);
    if (lines.length === 0) return { messageCount: 0 };

    let earliest: number | null = null;
    let latest: number | null = null;

    // Read first and last lines for ts
    for (const line of [lines[0], lines[lines.length - 1]]) {
      try {
        const obj = JSON.parse(line);
        if (obj.ts) {
          if (!earliest || obj.ts < earliest) earliest = obj.ts;
          if (!latest || obj.ts > latest) latest = obj.ts;
        }
      } catch { /* skip */ }
    }

    const dateRange = earliest && latest
      ? { first: new Date(earliest * 1000).toISOString(), last: new Date(latest * 1000).toISOString() }
      : undefined;
    return { messageCount: lines.length, dateRange };
  } catch {
    return { messageCount: 0 };
  }
}

function scanVsCodeChatSource(sourceDir: string): { messageCount: number; dateRange?: { first: string; last: string } } {
  const files = findFilesRecursive(sourceDir, (filePath, entryName) =>
    entryName.endsWith(".json") && filePath.includes(`${path.sep}chatSessions${path.sep}`),
  );
  if (files.length === 0) return { messageCount: 0 };

  let messageCount = 0;
  let earliestMs: number | null = null;
  let latestMs: number | null = null;

  for (const file of files) {
    try {
      const session = JSON.parse(fs.readFileSync(file, "utf-8"));
      const requests = Array.isArray(session.requests) ? session.requests : [];
      messageCount += requests.length;

      const timestamps = [
        typeof session.creationDate === "number" ? session.creationDate : null,
        typeof session.lastMessageDate === "number" ? session.lastMessageDate : null,
      ].filter((value): value is number => typeof value === "number");
      for (const request of requests) {
        if (typeof request?.timestamp === "number") timestamps.push(request.timestamp);
      }

      for (const ts of timestamps) {
        if (earliestMs === null || ts < earliestMs) earliestMs = ts;
        if (latestMs === null || ts > latestMs) latestMs = ts;
      }
    } catch { /* skip unreadable files */ }
  }

  const first = parseIso(earliestMs);
  const last = parseIso(latestMs);
  return {
    messageCount,
    dateRange: first && last ? { first, last } : undefined,
  };
}

function scanLmStudioSource(sourceDir: string): { messageCount: number; dateRange?: { first: string; last: string } } {
  const files = findFilesRecursive(sourceDir, (_filePath, entryName) =>
    entryName.endsWith(".conversation.json"),
  );
  if (files.length === 0) return { messageCount: 0 };

  let messageCount = 0;
  let earliestMs: number | null = null;
  let latestMs: number | null = null;

  for (const file of files) {
    try {
      const conversation = JSON.parse(fs.readFileSync(file, "utf-8"));
      const messages = Array.isArray(conversation.messages) ? conversation.messages : [];
      for (const message of messages) {
        const selectedIndex = Number.isInteger(message?.currentlySelected) ? message.currentlySelected : 0;
        const version = Array.isArray(message?.versions) ? message.versions[selectedIndex] ?? message.versions[0] : null;
        if (version?.role === "user") messageCount++;
      }

      const timestamps = [
        typeof conversation.createdAt === "number" ? conversation.createdAt : null,
        typeof conversation.userLastMessagedAt === "number" ? conversation.userLastMessagedAt : null,
        typeof conversation.assistantLastMessagedAt === "number" ? conversation.assistantLastMessagedAt : null,
      ].filter((value): value is number => typeof value === "number");

      for (const ts of timestamps) {
        if (earliestMs === null || ts < earliestMs) earliestMs = ts;
        if (latestMs === null || ts > latestMs) latestMs = ts;
      }
    } catch { /* skip unreadable files */ }
  }

  const first = parseIso(earliestMs);
  const last = parseIso(latestMs);
  return {
    messageCount,
    dateRange: first && last ? { first, last } : undefined,
  };
}

export interface SkillDefinition {
  skillName: string;
  skillPath: string;
  invocationPattern: string; // regex string
  templateContent: string;
  /** 'discovered' = scanned from disk, 'config' = platform built-in */
  source?: "discovered" | "config";
}

export interface Backend {
  readonly id: string;
  readonly name: string;
  detect(): BackendInfo;
  sync(config: Config): SyncResult;
  parse(config: Config): Promise<Message[]>;
  /** Discover skills/agents/hooks installed for this platform. */
  discoverSkills?(): SkillDefinition[];
}

// ── Skill discovery helpers ───────────────────────────────

function readSkillTemplate(skillDir: string): string {
  // Try common skill definition filenames
  for (const name of ["SKILL.md", "skill.md", "README.md", "prompt.md", "index.md"]) {
    const p = path.join(skillDir, name);
    try {
      return fs.readFileSync(p, "utf-8");
    } catch { /* not found, try next */ }
  }
  return "";
}

function scanSkillDirs(
  searchPaths: string[],
  buildPattern: (skillName: string) => string,
): SkillDefinition[] {
  const skills: SkillDefinition[] = [];
  for (const base of searchPaths) {
    try {
      if (!fs.existsSync(base)) continue;
      for (const entry of fs.readdirSync(base, { withFileTypes: true })) {
        if (!entry.isDirectory() || entry.name.startsWith(".")) continue;
        const skillPath = path.join(base, entry.name);
        skills.push({
          skillName: entry.name,
          skillPath,
          invocationPattern: buildPattern(entry.name),
          templateContent: readSkillTemplate(skillPath),
        });
      }
    } catch { /* permission errors, etc */ }
  }
  return skills;
}

// ── Claude Code ────────────────────────────────────────

class ClaudeCodeBackend implements Backend {
  readonly id = "claude_code";
  readonly name = "Claude Code";

  detect(): BackendInfo {
    const sourcePath = path.join(os.homedir(), ".claude", "projects");
    const detected = fs.existsSync(sourcePath);
    return {
      id: this.id,
      name: this.name,
      supported: true,
      detected,
      sourcePath,
      status: detected ? "available" : "not_found",
    };
  }

  sync(config: Config): SyncResult {
    return syncClaudeCode(config.claudeCodeSource);
  }

  async parse(config: Config): Promise<Message[]> {
    const exclusions = config.backends?.[this.id]?.exclusions ?? [];
    return parseClaudeCode(config.claudeCodeSource, exclusions);
  }

  discoverSkills(): SkillDefinition[] {
    const home = os.homedir();
    return scanSkillDirs(
      [path.join(home, ".claude", "skills")],
      // Only match explicit /slash invocation at start of message
      (name) => `^\\/${name}\\b`,
    );
  }
}

// ── Codex ──────────────────────────────────────────────

class CodexBackend implements Backend {
  readonly id = "codex";
  readonly name = "Codex";

  detect(): BackendInfo {
    const sourcePath = path.join(os.homedir(), ".codex", "history.jsonl");
    const detected = fs.existsSync(sourcePath);
    return {
      id: this.id,
      name: this.name,
      supported: true,
      detected,
      sourcePath,
      status: detected ? "available" : "not_found",
    };
  }

  sync(config: Config): SyncResult {
    return syncCodex(config.codexHistorySource);
  }

  async parse(config: Config): Promise<Message[]> {
    const sessionModels = await parseCodexSessionMetadata(config.codexSessionsSource);
    return parseCodexHistory(config.codexHistorySource, sessionModels);
  }

  discoverSkills(): SkillDefinition[] {
    const home = os.homedir();
    return scanSkillDirs(
      [path.join(home, ".codex", "skills")],
      // Match [$skill-name](...) or $skill-name at start of message
      (name) => `^(?:\\[\\$${name}\\]\\(|\\$${name}\\b)`,
    );
  }
}

// ── Copilot Chat ───────────────────────────────────────

class CopilotChatBackend implements Backend {
  readonly id = "copilot_chat";
  readonly name = "Copilot Chat";

  private sourceDir(): string {
    return vsCodeDataDir("Code");
  }

  detect(): BackendInfo {
    const sourcePath = this.sourceDir();
    const detected = fs.existsSync(sourcePath);
    return {
      id: this.id,
      name: this.name,
      supported: true,
      detected,
      sourcePath,
      status: detected ? "available" : "not_found",
    };
  }

  sync(config: Config): SyncResult {
    return syncVsCodeChatSessions(
      this.sourceDir(),
      config.copilotChatSource,
    );
  }

  async parse(config: Config): Promise<Message[]> {
    return parseVsCodeChatSessions(config.copilotChatSource, Platform.COPILOT_CHAT);
  }

  discoverSkills(): SkillDefinition[] {
    return [
      {
        skillName: "agent-command",
        skillPath: "",
        invocationPattern: "^@agent\\b",
        templateContent: "",
        source: "config",
      },
    ];
  }
}

// ── Cursor ─────────────────────────────────────────────

class CursorBackend implements Backend {
  readonly id = "cursor";
  readonly name = "Cursor";

  private sourceDir(): string {
    return vsCodeDataDir("Cursor");
  }

  detect(): BackendInfo {
    const sourcePath = this.sourceDir();
    const detected = fs.existsSync(sourcePath);
    return {
      id: this.id,
      name: this.name,
      supported: true,
      detected,
      sourcePath,
      status: detected ? "available" : "not_found",
    };
  }

  sync(config: Config): SyncResult {
    return syncVsCodeChatSessions(
      this.sourceDir(),
      config.cursorSource,
    );
  }

  async parse(config: Config): Promise<Message[]> {
    return parseVsCodeChatSessions(config.cursorSource, Platform.CURSOR);
  }

  discoverSkills(): SkillDefinition[] {
    const home = os.homedir();
    return scanSkillDirs(
      [path.join(home, ".cursor", "skills-cursor")],
      // Only match explicit /slash invocation at start of message
      (name) => `^\\/${name}\\b`,
    );
  }
}

// ── LM Studio ──────────────────────────────────────────

class LmStudioBackend implements Backend {
  readonly id = "lmstudio";
  readonly name = "LM Studio";

  detect(): BackendInfo {
    const sourcePath = path.join(os.homedir(), ".lmstudio", "conversations");
    const detected = fs.existsSync(sourcePath);
    return {
      id: this.id,
      name: this.name,
      supported: true,
      detected,
      sourcePath,
      status: detected ? "available" : "not_found",
    };
  }

  sync(config: Config): SyncResult {
    return syncLmStudioConversations(config.lmStudioSource);
  }

  async parse(config: Config): Promise<Message[]> {
    return parseLmStudioConversations(config.lmStudioSource);
  }
}

// ── Pi agent ──────────────────────────────────────────

class PiBackend implements Backend {
  readonly id = "pi";
  readonly name = "Pi";

  private sourceDir(): string {
    return path.join(os.homedir(), ".pi", "agent", "sessions");
  }

  detect(): BackendInfo {
    const sourcePath = this.sourceDir();
    const detected = fs.existsSync(sourcePath);
    return {
      id: this.id,
      name: this.name,
      supported: true,
      detected,
      sourcePath,
      status: detected ? "available" : "not_found",
    };
  }

  sync(config: Config): SyncResult {
    return syncPiSessions(this.sourceDir(), config.piSource);
  }

  async parse(config: Config): Promise<Message[]> {
    return parsePiSessions(config.piSource);
  }

  discoverSkills(): SkillDefinition[] {
    const home = os.homedir();
    return scanSkillDirs(
      [path.join(home, ".pi", "agent", "skills")],
      (name) => `^(?:\\/skill:${name}\\b|\\[\\$${name}\\])`,
    );
  }
}

// ── OpenCode ──────────────────────────────────────────

/**
 * Resolve the OpenCode storage directory across platforms.
 *   macOS/Linux: ~/.local/share/opencode/storage
 *   Windows:     %LOCALAPPDATA%/opencode/storage
 */
function openCodeDataDir(): string {
  if (os.platform() === "win32") {
    return path.join(
      process.env.LOCALAPPDATA ?? path.join(os.homedir(), "AppData", "Local"),
      "opencode",
      "storage",
    );
  }
  return path.join(os.homedir(), ".local", "share", "opencode", "storage");
}

class OpenCodeBackend implements Backend {
  readonly id = "opencode";
  readonly name = "OpenCode";

  private sourceDir(): string {
    return openCodeDataDir();
  }

  detect(): BackendInfo {
    const sourcePath = this.sourceDir();
    const detected = fs.existsSync(sourcePath);
    return {
      id: this.id,
      name: this.name,
      supported: true,
      detected,
      sourcePath,
      status: detected ? "available" : "not_found",
    };
  }

  sync(config: Config): SyncResult {
    return syncOpenCodeStorage(this.sourceDir(), config.openCodeSource);
  }

  async parse(config: Config): Promise<Message[]> {
    return parseOpenCodeSessions(config.openCodeSource);
  }
}

// ── Registry ───────────────────────────────────────────

const ALL_BACKENDS: Backend[] = [
  new ClaudeCodeBackend(),
  new CodexBackend(),
  new CopilotChatBackend(),
  new CursorBackend(),
  new LmStudioBackend(),
  new PiBackend(),
  new OpenCodeBackend(),
];

export function getAllBackends(): Backend[] {
  return ALL_BACKENDS;
}

export function getBackend(id: string): Backend | undefined {
  return ALL_BACKENDS.find((b) => b.id === id);
}

export function detectAll(): BackendInfo[] {
  const now = Date.now();
  if (detectionCache && detectionCache.expiresAt > now) {
    return detectionCache.infos;
  }

  const infos = ALL_BACKENDS.map((b) => b.detect());
  detectionCache = {
    expiresAt: now + DETECTION_CACHE_TTL_MS,
    infos,
  };
  return infos;
}

export function resetBackendDetectionCacheForTests(): void {
  detectionCache = null;
}

/** Get backends that are enabled and have working parsers */
export function getEnabledBackends(config: Config): Backend[] {
  const available = new Map(detectAll().map((info) => [info.id, info]));
  return ALL_BACKENDS.filter((b) => {
    const info = available.get(b.id);
    if (!info || info.status !== "available") return false;
    return config.backends?.[b.id]?.enabled !== false;
  });
}
