import fs from "node:fs";
import os from "node:os";
import path from "node:path";

export interface BackendToggle {
  enabled: boolean;
  exclusions: string[];
}

export interface Config {
  dataDir: string;
  claudeCodeSource: string;
  codexHistorySource: string;
  codexSessionsSource: string;
  copilotChatSource: string;
  cursorSource: string;
  lmStudioSource: string;
  piSource: string;
  openCodeSource: string;
  engagementThreshold: number;
  politenessThreshold: number;
  agentCwds: string[];
  backends: Record<string, BackendToggle>;
  hasCompletedSetup: boolean;
}

function defaultBackends(agentCwds: string[]): Record<string, BackendToggle> {
  return {
    claude_code: { enabled: true, exclusions: agentCwds },
    codex: { enabled: true, exclusions: [] },
    copilot_chat: { enabled: false, exclusions: [] },
    cursor: { enabled: false, exclusions: [] },
    lmstudio: { enabled: false, exclusions: [] },
    pi: { enabled: true, exclusions: [] },
    opencode: { enabled: true, exclusions: [] },
  };
}

function mergeBackends(
  defaults: Record<string, BackendToggle>,
  configured: Record<string, BackendToggle> = {},
): Record<string, BackendToggle> {
  const merged: Record<string, BackendToggle> = {};
  const ids = new Set([...Object.keys(defaults), ...Object.keys(configured)]);

  for (const id of ids) {
    merged[id] = {
      enabled: configured[id]?.enabled ?? defaults[id]?.enabled ?? false,
      exclusions: Array.isArray(configured[id]?.exclusions)
        ? configured[id].exclusions
        : (defaults[id]?.exclusions ?? []),
    };
  }

  return merged;
}

export function loadConfig(dataDir?: string): Config {
  const dd = dataDir ?? path.join(os.homedir(), ".howiprompt");

  // Load user config if it exists
  let userConfig: Record<string, any> = {};
  const configPath = path.join(dd, "config.json");
  try {
    userConfig = JSON.parse(fs.readFileSync(configPath, "utf-8"));
  } catch {
    // No config file or invalid — that's fine
  }

  const agentCwds: string[] = Array.isArray(userConfig.agentCwds) ? userConfig.agentCwds : [];

  // Migrate legacy agentCwds → backends.claude_code.exclusions
  const backends = mergeBackends(defaultBackends(agentCwds), userConfig.backends);

  return {
    dataDir: dd,
    claudeCodeSource: path.join(dd, "raw", "claude_code"),
    codexHistorySource: path.join(dd, "raw", "codex", "history.jsonl"),
    codexSessionsSource: path.join(os.homedir(), ".codex", "sessions"),
    copilotChatSource: path.join(dd, "raw", "copilot_chat"),
    cursorSource: path.join(dd, "raw", "cursor"),
    lmStudioSource: path.join(dd, "raw", "lmstudio"),
    piSource: path.join(dd, "raw", "pi"),
    openCodeSource: path.join(dd, "raw", "opencode"),
    engagementThreshold: 12.0,
    politenessThreshold: 4.5,
    agentCwds,
    backends,
    hasCompletedSetup: userConfig.hasCompletedSetup ?? false,
  };
}

export function saveConfig(dataDir: string, updates: Record<string, any>): void {
  const configPath = path.join(dataDir, "config.json");
  let existing: Record<string, any> = {};
  try {
    existing = JSON.parse(fs.readFileSync(configPath, "utf-8"));
  } catch {
    // Start fresh
  }
  const merged = { ...existing, ...updates };
  const agentCwds: string[] = Array.isArray(merged.agentCwds) ? merged.agentCwds : [];
  merged.backends = mergeBackends(defaultBackends(agentCwds), merged.backends);
  fs.mkdirSync(dataDir, { recursive: true });
  fs.writeFileSync(configPath, JSON.stringify(merged, null, 2));
}

export function loadBranding(projectRoot?: string): Record<string, string> | null {
  const brandingPath = path.join(projectRoot ?? process.cwd(), "branding.json");
  try {
    return JSON.parse(fs.readFileSync(brandingPath, "utf-8"));
  } catch {
    return null;
  }
}
