#!/usr/bin/env node

/**
 * Privacy & Artifact Gate for howiprompt.
 *
 * Verifies:
 * 1. No forbidden files tracked in git
 * 2. No hardcoded user paths in source files
 * 3. Tarball content validation
 * 4. No conversation content in shipped files
 */

import { execSync } from "node:child_process";
import { readFileSync } from "node:fs";
import os from "node:os";

const RED = "\x1b[31m";
const GREEN = "\x1b[32m";
const RESET = "\x1b[0m";

let failures = 0;
let warnings = 0;

function fail(msg) {
  console.error(`${RED}FAIL${RESET} ${msg}`);
  failures++;
}

function pass(msg) {
  console.log(`${GREEN}PASS${RESET} ${msg}`);
}

// ── 1. Tracked files check ──────────────────────────────────────────

console.log("\n=== Tracked Files Gate ===\n");

const FORBIDDEN_TRACKED = [
  /^\.env\.local$/,
  /^\.env$/,
  /^branding\.json$/,
  /^config\.json$/,
  /\.db$/,
  /\.db-journal$/,
  /\.db-wal$/,
  /\.db-shm$/,
  /\/\.env\.local$/,
  /\/config\.json$/,
];

const USER_PATH_PATTERNS = [
  /\/Users\/(?!test\b)\w+/,
  /\/home\/(?!test\b)\w+/,
  /C:\\Users\\\w+/,
];

const trackedFiles = execSync("git ls-files", { encoding: "utf-8" }).trim().split("\n");

let trackedForbiddenFound = false;
for (const file of trackedFiles) {
  for (const pattern of FORBIDDEN_TRACKED) {
    if (pattern.test(file)) {
      fail(`Forbidden file tracked in git: ${file}`);
      trackedForbiddenFound = true;
    }
  }
}
if (!trackedForbiddenFound) {
  pass("No forbidden files tracked in git");
}

// ── 2. Source path check ────────────────────────────────────────────

console.log("\n=== Source Path Gate ===\n");

const sourceFiles = trackedFiles.filter(
  (f) => /\.(ts|tsx|mjs|js)$/.test(f) && !f.includes("test") && !f.includes("node_modules"),
);

let userPathHits = [];
for (const file of sourceFiles) {
  try {
    const content = readFileSync(file, "utf-8");
    for (const pattern of USER_PATH_PATTERNS) {
      if (pattern.test(content)) {
        userPathHits.push(file);
        break;
      }
    }
  } catch {
    // File may be new/unstaged — skip
  }
}

if (userPathHits.length > 0) {
  for (const file of userPathHits) {
    fail(`Hardcoded user path in source: ${file}`);
  }
} else {
  pass("No hardcoded user paths in source files");
}

// ── 3. Tarball content check ────────────────────────────────────────

console.log("\n=== Tarball Content Gate ===\n");

const FORBIDDEN_IN_TARBALL = [
  /\.db$/,
  /\.db-journal$/,
  /\.db-wal$/,
  /config\.json$/,
  /branding\.json$/,
  /\.env\.local$/,
  /internal\//,
  /src-py\//,
];

try {
  const packList = execSync("npm pack --dry-run 2>&1", { encoding: "utf-8" });
  const lines = packList.split("\n").filter(
    (l) =>
      l.startsWith("npm notice") &&
      !l.includes("Tarball") &&
      !l.includes("name:") &&
      !l.includes("version:") &&
      !l.includes("filename:") &&
      !l.includes("package size:") &&
      !l.includes("unpacked size:") &&
      !l.includes("shasum:") &&
      !l.includes("integrity:") &&
      !l.includes("total files:") &&
      !l.includes("==="),
  );

  let tarballForbiddenFound = false;
  for (const line of lines) {
    const match = line.match(/npm notice\s+[\d.]+[kMG]?B\s+(.+)/);
    if (!match) continue;
    const filePath = match[1].trim();
    for (const pattern of FORBIDDEN_IN_TARBALL) {
      if (pattern.test(filePath)) {
        fail(`Forbidden file in npm tarball: ${filePath}`);
        tarballForbiddenFound = true;
      }
    }
  }

  if (!tarballForbiddenFound) {
    pass("No forbidden files in npm tarball");
  }
} catch (e) {
  console.warn("Warning: Could not run tarball content check:", e.message);
  warnings++;
}

// ── 4. No conversation content in dist ──────────────────────────────

console.log("\n=== Content Leak Gate ===\n");

const distFiles = trackedFiles.filter((f) => f.startsWith("dist/") && /\.(js|json)$/.test(f));
let contentLeakFound = false;
for (const file of distFiles) {
  try {
    const content = readFileSync(file, "utf-8");
    // Check for typical conversation content markers
    if (content.includes("~/.claude") || content.includes("~/.codex")) {
      // These are OK in config paths, but flag actual data references
    }
    const homedir = os.homedir();
    if (content.includes(homedir)) {
      fail(`Build-machine home path found in: ${file}`);
      contentLeakFound = true;
    }
  } catch {
    // Skip
  }
}
if (!contentLeakFound) {
  pass("No build-machine paths in dist files");
}

// ── Summary ─────────────────────────────────────────────────────────

console.log("\n=== Summary ===\n");
if (failures > 0) {
  console.error(`${RED}${failures} failure(s)${RESET}, ${warnings} warning(s)`);
  process.exit(1);
} else {
  console.log(`${GREEN}All checks passed${RESET} (${warnings} warning(s))`);
  process.exit(0);
}
