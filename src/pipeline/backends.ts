import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import readline from "node:readline";
import { type SyncResult, syncClaudeCode, syncCodex } from "./sync.js";
import {
  parseClaudeCode,
  parseCodexHistory,
  parseCodexSessionMetadata,
} from "./parsers.js";
import type { Message } from "./models.js";
import type { Config } from "./config.js";

export interface BackendInfo {
  id: string;
  name: string;
  detected: boolean;
  sourcePath: string;
  status: "available" | "coming_soon" | "not_found";
  messageCount?: number;
  dateRange?: { first: string; last: string };
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

export interface Backend {
  readonly id: string;
  readonly name: string;
  detect(): BackendInfo;
  sync(config: Config): SyncResult;
  parse(config: Config): Promise<Message[]>;
}

// ── Claude Code ────────────────────────────────────────

class ClaudeCodeBackend implements Backend {
  readonly id = "claude_code";
  readonly name = "Claude Code";

  detect(): BackendInfo {
    const sourcePath = path.join(os.homedir(), ".claude", "projects");
    const detected = fs.existsSync(sourcePath);
    const scan = detected ? scanClaudeCodeSource(sourcePath) : undefined;
    return {
      id: this.id,
      name: this.name,
      detected,
      sourcePath,
      status: detected ? "available" : "not_found",
      messageCount: scan?.messageCount,
      dateRange: scan?.dateRange,
    };
  }

  sync(config: Config): SyncResult {
    return syncClaudeCode(config.claudeCodeSource);
  }

  async parse(config: Config): Promise<Message[]> {
    const exclusions = config.backends?.[this.id]?.exclusions ?? [];
    return parseClaudeCode(config.claudeCodeSource, exclusions);
  }
}

// ── Codex ──────────────────────────────────────────────

class CodexBackend implements Backend {
  readonly id = "codex";
  readonly name = "Codex";

  detect(): BackendInfo {
    const sourcePath = path.join(os.homedir(), ".codex", "history.jsonl");
    const detected = fs.existsSync(sourcePath);
    const scan = detected ? scanCodexSource(sourcePath) : undefined;
    return {
      id: this.id,
      name: this.name,
      detected,
      sourcePath,
      status: detected ? "available" : "not_found",
      messageCount: scan?.messageCount,
      dateRange: scan?.dateRange,
    };
  }

  sync(config: Config): SyncResult {
    return syncCodex(config.codexHistorySource);
  }

  async parse(config: Config): Promise<Message[]> {
    const sessionModels = await parseCodexSessionMetadata(config.codexSessionsSource);
    return parseCodexHistory(config.codexHistorySource, sessionModels);
  }
}

// ── Copilot Chat (detection only) ──────────────────────

class CopilotChatBackend implements Backend {
  readonly id = "copilot_chat";
  readonly name = "Copilot Chat";

  detect(): BackendInfo {
    const sourcePath = path.join(
      os.homedir(),
      "Library",
      "Application Support",
      "Code",
      "User",
      "globalStorage",
    );
    const detected = fs.existsSync(sourcePath);
    return {
      id: this.id,
      name: this.name,
      detected,
      sourcePath,
      status: detected ? "coming_soon" : "not_found",
    };
  }

  sync(): SyncResult {
    return { files: 0, source: "copilot_chat" };
  }

  async parse(): Promise<Message[]> {
    return [];
  }
}

// ── Cursor (detection only) ────────────────────────────

class CursorBackend implements Backend {
  readonly id = "cursor";
  readonly name = "Cursor";

  detect(): BackendInfo {
    const sourcePath = path.join(
      os.homedir(),
      "Library",
      "Application Support",
      "Cursor",
      "User",
      "workspaceStorage",
    );
    const detected = fs.existsSync(sourcePath);
    return {
      id: this.id,
      name: this.name,
      detected,
      sourcePath,
      status: detected ? "coming_soon" : "not_found",
    };
  }

  sync(): SyncResult {
    return { files: 0, source: "cursor" };
  }

  async parse(): Promise<Message[]> {
    return [];
  }
}

// ── Registry ───────────────────────────────────────────

const ALL_BACKENDS: Backend[] = [
  new ClaudeCodeBackend(),
  new CodexBackend(),
  new CopilotChatBackend(),
  new CursorBackend(),
];

export function getAllBackends(): Backend[] {
  return ALL_BACKENDS;
}

export function getBackend(id: string): Backend | undefined {
  return ALL_BACKENDS.find((b) => b.id === id);
}

export function detectAll(): BackendInfo[] {
  return ALL_BACKENDS.map((b) => b.detect());
}

/** Get backends that are enabled and have working parsers */
export function getEnabledBackends(config: Config): Backend[] {
  return ALL_BACKENDS.filter((b) => {
    const info = b.detect();
    if (info.status !== "available") return false;
    return config.backends?.[b.id]?.enabled !== false;
  });
}
