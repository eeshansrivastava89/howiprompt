#!/usr/bin/env node

/**
 * CLI entry point for How I Prompt.
 * Usage: npx @eeshans/howiprompt [--port <n>] [--no-open] [--help] [--version]
 */

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { bootstrapDb } from "./bootstrap-db.mjs";
import { resolveDataDir, findFreePort, parseArgs, openBrowser } from "./cli-helpers.mjs";

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
Local-first analytics for your AI coding conversations — Claude Code, Codex, Copilot Chat, Cursor, and LM Studio.

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
fs.mkdirSync(path.join(dataDir, "raw", "copilot_chat"), { recursive: true });
fs.mkdirSync(path.join(dataDir, "raw", "cursor"), { recursive: true });
fs.mkdirSync(path.join(dataDir, "raw", "lmstudio"), { recursive: true });

console.log(`\n${bold("howiprompt")} v${pkg.version}\n`);

// ── Bootstrap database ─────────────────────────────────
const dbPath = path.join(dataDir, "data.db");
process.stdout.write(`  Checking environment...        ${dim("Node " + process.versions.node)} `);
console.log(green("ok"));

process.stdout.write("  Initializing database...       ");
await bootstrapDb(dbPath);
console.log(green("done"));

// ── Ensure wizard shows when there's no data ──────────
const metricsPath = path.join(dataDir, "metrics.json");
if (!fs.existsSync(metricsPath)) {
  const configPath = path.join(dataDir, "config.json");
  try {
    const cfg = JSON.parse(fs.readFileSync(configPath, "utf-8"));
    if (cfg.hasCompletedSetup) {
      cfg.hasCompletedSetup = false;
      fs.writeFileSync(configPath, JSON.stringify(cfg, null, 2));
    }
  } catch { /* no config yet — wizard will show by default */ }
}

const projectRoot = path.join(__dirname, "..");
const serverBundlePath = path.join(projectRoot, "dist", "server.js");
const frontendBundlePath = path.join(projectRoot, "frontend", "dist", "index.html");

if (!fs.existsSync(serverBundlePath) || !fs.existsSync(frontendBundlePath)) {
  console.error(red("Bundled app assets are missing."));
  console.error(dim("For local development, run `npm run build:cli` or `npm run dev:cli` from the repo."));
  process.exit(1);
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
  let versionCheckTimeout = null;
  (async () => {
    try {
      const controller = new AbortController();
      versionCheckTimeout = setTimeout(() => controller.abort(), 3000);
      versionCheckTimeout.unref?.();
      const res = await fetch("https://registry.npmjs.org/@eeshans%2fhowiprompt/latest", {
        signal: controller.signal,
      });
      clearTimeout(versionCheckTimeout);
      versionCheckTimeout = null;
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
          console.log(`  Run: ${bold("npx @eeshans/howiprompt@latest")}\n`);
        }
      }
    } catch {
      // Silent on failure
    } finally {
      if (versionCheckTimeout) {
        clearTimeout(versionCheckTimeout);
        versionCheckTimeout = null;
      }
    }
  })();

  if (!parsed.noOpen) {
    openBrowser(serverUrl);
  }

  let shuttingDown = false;
  function shutdown() {
    if (shuttingDown) return;
    shuttingDown = true;
    console.log("\n  Shutting down...");

    if (versionCheckTimeout) {
      clearTimeout(versionCheckTimeout);
      versionCheckTimeout = null;
    }

    const hardExitTimer = setTimeout(() => {
      process.exit(0);
    }, 2000);
    hardExitTimer.unref?.();

    server.close((closeErr) => {
      clearTimeout(hardExitTimer);
      if (closeErr) {
        console.log(red(`error during shutdown: ${closeErr.message}`));
        process.exit(1);
        return;
      }
      process.exitCode = 0;
    });
  }

  process.once("SIGINT", shutdown);
  process.once("SIGTERM", shutdown);
} catch (err) {
  if (err.code === "ERR_MODULE_NOT_FOUND") {
    console.log(yellow("server not built yet — run `npm run build` first"));
    process.exit(1);
  } else {
    console.log(red(`error: ${err.message}`));
    process.exit(1);
  }
}
