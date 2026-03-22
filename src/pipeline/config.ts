import fs from "node:fs";
import os from "node:os";
import path from "node:path";

export interface Config {
  dataDir: string;
  claudeCodeSource: string;
  codexHistorySource: string;
  codexSessionsSource: string;
  engagementThreshold: number;
  politenessThreshold: number;
  agentCwds: string[];
}

export function loadConfig(dataDir?: string): Config {
  const dd = dataDir ?? path.join(os.homedir(), ".howiprompt");

  // Load user config if it exists
  let agentCwds: string[] = [];
  const configPath = path.join(dd, "config.json");
  try {
    const userConfig = JSON.parse(fs.readFileSync(configPath, "utf-8"));
    if (Array.isArray(userConfig.agentCwds)) {
      agentCwds = userConfig.agentCwds;
    }
  } catch {
    // No config file or invalid — that's fine
  }

  return {
    dataDir: dd,
    claudeCodeSource: path.join(dd, "raw", "claude_code"),
    codexHistorySource: path.join(dd, "raw", "codex", "history.jsonl"),
    codexSessionsSource: path.join(os.homedir(), ".codex", "sessions"),
    engagementThreshold: 12.0,
    politenessThreshold: 4.5,
    agentCwds,
  };
}

export function loadBranding(projectRoot?: string): Record<string, string> | null {
  const brandingPath = path.join(projectRoot ?? process.cwd(), "branding.json");
  try {
    return JSON.parse(fs.readFileSync(brandingPath, "utf-8"));
  } catch {
    return null;
  }
}
