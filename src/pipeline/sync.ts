import fs from "node:fs";
import path from "node:path";
import os from "node:os";

export interface SyncResult {
  files: number;
  source: string;
}

function copyMatchingRecursive(
  srcDir: string,
  destDir: string,
  matcher: (srcPath: string, entryName: string) => boolean,
): number {
  let count = 0;
  for (const entry of fs.readdirSync(srcDir, { withFileTypes: true })) {
    const srcPath = path.join(srcDir, entry.name);
    const destPath = path.join(destDir, entry.name);
    if (entry.isDirectory()) {
      count += copyMatchingRecursive(srcPath, destPath, matcher);
    } else if (matcher(srcPath, entry.name)) {
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

  const filesCopied = copyMatchingRecursive(sourceDir, destDir, (_srcPath, entryName) =>
    entryName.endsWith(".jsonl"),
  );
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

export function syncVsCodeChatSessions(
  sourceDir: string,
  destDir: string,
): SyncResult {
  if (!fs.existsSync(sourceDir)) {
    return { files: 0, source: sourceDir };
  }

  const filesCopied = copyMatchingRecursive(sourceDir, destDir, (srcPath, entryName) =>
    entryName.endsWith(".json") && srcPath.includes(`${path.sep}chatSessions${path.sep}`),
  );
  return { files: filesCopied, source: sourceDir };
}

export function syncLmStudioConversations(
  destDir: string,
): SyncResult {
  const sourceDir = path.join(os.homedir(), ".lmstudio", "conversations");
  if (!fs.existsSync(sourceDir)) {
    return { files: 0, source: sourceDir };
  }

  const filesCopied = copyMatchingRecursive(sourceDir, destDir, (_srcPath, entryName) =>
    entryName.endsWith(".conversation.json"),
  );
  return { files: filesCopied, source: sourceDir };
}
