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
  engagementThreshold: number;
  politenessThreshold: number;
  agentCwds: string[];
  backends: Record<string, BackendToggle>;
  hasCompletedSetup: boolean;
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
  const defaultBackends: Record<string, BackendToggle> = {
    claude_code: { enabled: true, exclusions: agentCwds },
    codex: { enabled: true, exclusions: [] },
  };
  const backends: Record<string, BackendToggle> = userConfig.backends ?? defaultBackends;

  return {
    dataDir: dd,
    claudeCodeSource: path.join(dd, "raw", "claude_code"),
    codexHistorySource: path.join(dd, "raw", "codex", "history.jsonl"),
    codexSessionsSource: path.join(os.homedir(), ".codex", "sessions"),
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
