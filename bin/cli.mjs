#!/usr/bin/env node

/**
 * CLI entry point for How I Prompt.
 * Usage: npx howiprompt [--port <n>] [--no-open] [--help] [--version]
 */

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { bootstrapDb } from "./bootstrap-db.mjs";
import { resolveDataDir, findFreePort, parseArgs, openBrowser, waitForServer } from "./cli-helpers.mjs";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const pkgPath = path.join(__dirname, "..", "package.json");
const pkg = JSON.parse(fs.readFileSync(pkgPath, "utf-8"));

// ── Color helpers ──────────────────────────────────────
const green = (s) => `\x1b[32m${s}\x1b[0m`;
const yellow = (s) => `\x1b[33m${s}\x1b[0m`;
const red = (s) => `\x1b[31m${s}\x1b[0m`;
const bold = (s) => `\x1b[1m${s}\x1b[0m`;
const dim = (s) => `\x1b[2m${s}\x1b[0m`;

// ── Parse args ─────────────────────────────────────────
const parsed = parseArgs(process.argv.slice(2));

if (parsed.help) {
  console.log(`
${bold("howiprompt")} v${pkg.version}
Local-first analytics dashboard for Claude Code + Codex prompting patterns.

Usage: howiprompt [options]

Options:
  --port <n>   Use a specific port (default: auto)
  --no-open    Don't open the browser automatically
  --help       Show this help message
  --version    Show version number
`);
  process.exit(0);
}

if (parsed.version) {
  console.log(pkg.version);
  process.exit(0);
}

// ── Check Node version ─────────────────────────────────
const [major] = process.versions.node.split(".").map(Number);
if (major < 18) {
  console.error(red(`Node >= 18.0.0 required (found ${process.versions.node})`));
  process.exit(1);
}

// ── Resolve data directory ─────────────────────────────
const dataDir = resolveDataDir();
fs.mkdirSync(dataDir, { recursive: true });
fs.mkdirSync(path.join(dataDir, "raw", "claude_code"), { recursive: true });
fs.mkdirSync(path.join(dataDir, "raw", "codex"), { recursive: true });

console.log(`\n${bold("howiprompt")} v${pkg.version}\n`);

// ── Bootstrap database ─────────────────────────────────
const dbPath = path.join(dataDir, "data.db");
process.stdout.write(`  Checking environment...        ${dim("Node " + process.versions.node)} `);
console.log(green("ok"));

process.stdout.write("  Initializing database...       ");
await bootstrapDb(dbPath);
console.log(green("done"));

// ── Run pipeline ───────────────────────────────────────
process.stdout.write("  Syncing conversations...       ");
try {
  const { runPipeline } = await import("../dist/index.js");
  const stats = await runPipeline({ dbPath, dataDir });
  if (stats.newMessages > 0) {
    console.log(green(`+${stats.newMessages} new messages (${stats.totalMessages} total)`));
  } else {
    console.log(green(`${stats.totalMessages} messages (up to date)`));
  }
} catch (err) {
  if (err.code === "ERR_MODULE_NOT_FOUND") {
    console.log(yellow("pipeline not built yet — run `npm run build` first"));
  } else {
    console.log(red(`error: ${err.message}`));
  }
}

// ── Start server ───────────────────────────────────────
const port = await findFreePort(parsed.port);

process.stdout.write("  Starting server...             ");

try {
  const { startServer } = await import("../dist/server.js");
  const server = await startServer({ port, dataDir, dbPath });
  const serverUrl = `http://localhost:${port}`;
  console.log(green(serverUrl));

  console.log(`\n  Dashboard ready. Press ${bold("Ctrl+C")} to stop.\n`);

  // Non-blocking version check
  (async () => {
    try {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 3000);
      const res = await fetch("https://registry.npmjs.org/howiprompt/latest", {
        signal: controller.signal,
      });
      clearTimeout(timeout);
      if (!res.ok) return;
      const data = await res.json();
      const latest = data.version;
      if (latest && latest !== pkg.version) {
        const l = latest.split(".").map(Number);
        const c = pkg.version.split(".").map(Number);
        const newer =
          l[0] > c[0] ||
          (l[0] === c[0] && l[1] > c[1]) ||
          (l[0] === c[0] && l[1] === c[1] && l[2] > c[2]);
        if (newer) {
          console.log(`  ${yellow("Update available:")} ${pkg.version} → ${green(latest)}`);
          console.log(`  Run: ${bold("npx howiprompt@latest")}\n`);
        }
      }
    } catch {
      // Silent on failure
    }
  })();

  if (!parsed.noOpen) {
    openBrowser(serverUrl);
  }

  // Graceful shutdown
  function shutdown() {
    console.log("\n  Shutting down...");
    server.close(() => process.exit(0));
    setTimeout(() => process.exit(1), 5000);
  }

  process.on("SIGINT", shutdown);
  process.on("SIGTERM", shutdown);
} catch (err) {
  if (err.code === "ERR_MODULE_NOT_FOUND") {
    console.log(yellow("server not built yet — run `npm run build` first"));
    process.exit(1);
  } else {
    console.log(red(`error: ${err.message}`));
    process.exit(1);
  }
}
