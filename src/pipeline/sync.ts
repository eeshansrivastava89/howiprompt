import fs from "node:fs";
import path from "node:path";
import os from "node:os";

export interface SyncResult {
  files: number;
  source: string;
}

function copyJsonlRecursive(srcDir: string, destDir: string): number {
  let count = 0;
  for (const entry of fs.readdirSync(srcDir, { withFileTypes: true })) {
    const srcPath = path.join(srcDir, entry.name);
    const destPath = path.join(destDir, entry.name);
    if (entry.isDirectory()) {
      count += copyJsonlRecursive(srcPath, destPath);
    } else if (entry.name.endsWith(".jsonl")) {
      fs.mkdirSync(destDir, { recursive: true });
      fs.copyFileSync(srcPath, destPath);
      count++;
    }
  }
  return count;
}

export function syncClaudeCode(destDir: string): SyncResult {
  const sourceDir = path.join(os.homedir(), ".claude", "projects");
  if (!fs.existsSync(sourceDir)) {
    return { files: 0, source: sourceDir };
  }

  const filesCopied = copyJsonlRecursive(sourceDir, destDir);
  return { files: filesCopied, source: sourceDir };
}

export function syncCodex(destPath: string): SyncResult {
  const sourcePath = path.join(os.homedir(), ".codex", "history.jsonl");
  if (!fs.existsSync(sourcePath)) {
    return { files: 0, source: sourcePath };
  }

  fs.mkdirSync(path.dirname(destPath), { recursive: true });
  fs.copyFileSync(sourcePath, destPath);
  return { files: 1, source: sourcePath };
}
