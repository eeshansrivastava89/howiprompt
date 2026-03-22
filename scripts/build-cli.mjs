#!/usr/bin/env node

/**
 * Build script for NPX distribution.
 * Compiles TypeScript, copies frontend assets, handles platform-aware native bindings.
 */

import { execSync } from "node:child_process";
import { cpSync, rmSync, chmodSync, existsSync, readdirSync, readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { join, resolve } from "node:path";

const run = (cmd) => execSync(cmd, { stdio: "inherit" });

// 1. Compile TypeScript
console.log("Compiling TypeScript...");
run("npx tsc");

// 2. Ensure frontend/dist exists
if (!existsSync("frontend/dist")) {
  console.error("Error: frontend/dist not found. Run `cd frontend && npm run build` first.");
  process.exit(1);
}
console.log("Frontend dist found.");

// 3. Copy platform-aware @libsql native binding
const platformMap = {
  "darwin-arm64": "@libsql/darwin-arm64",
  "darwin-x64": "@libsql/darwin-x64",
  "linux-x64": "@libsql/linux-x64-gnu",
  "linux-arm64": "@libsql/linux-arm64-gnu",
  "win32-x64": "@libsql/win32-x64-msvc",
};

const key = `${process.platform}-${process.arch}`;
const nativePackage = platformMap[key];

if (nativePackage && existsSync(`node_modules/${nativePackage}`)) {
  console.log(`Native binding: ${nativePackage}`);
} else {
  console.warn(`Warning: No native @libsql binding found for ${key}`);
}

// 4. Strip private files from dist
const stripPatterns = [".env", ".env.local", "*.db", "*.db-journal", "*.db-wal", "*.db-shm"];
for (const entry of readdirSync("dist")) {
  if (stripPatterns.some((p) => entry.endsWith(p.replace("*", "")))) {
    rmSync(join("dist", entry), { force: true });
  }
}

// 5. Ensure bin is executable
chmodSync("bin/cli.mjs", 0o755);

console.log("Build complete.");
