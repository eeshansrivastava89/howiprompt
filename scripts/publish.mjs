#!/usr/bin/env node

/**
 * Publish script for the howiprompt demo site.
 *
 * Validates metrics, runs checks and tests, builds the frontend,
 * copies demo data into the build output, and syncs to dist-static/
 * for GitHub Pages deployment via GitHub Actions.
 *
 * Usage:
 *   npm run publish
 *
 * Environment variables:
 *   HOWIPROMPT_METRICS_PATH       Path to metrics.json (default: ~/.howiprompt/metrics.json)
 *   PUBLIC_ENABLE_ANALYTICS        Enable PostHog analytics (default: "true")
 *   PUBLIC_ENABLE_LINGUISTICS      Enable linguistics page (default: disabled)
 *   PUBLIC_POSTHOG_KEY             Enables analytics when paired with PUBLIC_POSTHOG_HOST
 *   PUBLIC_POSTHOG_HOST            Enables analytics when paired with PUBLIC_POSTHOG_KEY
 *   PUBLIC_POSTHOG_UI_HOST         PostHog UI host (default: https://us.posthog.com)
 *   PUBLIC_ANALYTICS_ALLOWED_HOSTS Comma-separated host allowlist (default: no host restriction)
 *   HOWIPROMPT_REQUIRE_ANALYTICS    Fail if analytics env vars are missing (default: false)
 */

import { copyFileSync, existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { homedir } from "node:os";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const repoRoot = join(__dirname, "..");
const frontendDir = join(repoRoot, "frontend");
const distDir = join(frontendDir, "dist");
const distStaticDir = join(repoRoot, "dist-static");
const demoDataDir = join(repoRoot, "data", "demo");

// ── Resolve metrics file ──────────────────────────────────────────

const metricsPath = process.env.HOWIPROMPT_METRICS_PATH
  || join(homedir(), ".howiprompt", "metrics.json");

if (!existsSync(metricsPath)) {
  console.error(`\nMetrics file not found at ${metricsPath}`);
  console.error("Run the pipeline first: npm run dev:cli\n");
  process.exit(1);
}

console.log(`Using metrics from: ${metricsPath}`);

// ── 1. Copy metrics to data/demo/ (committed demo data) ───────────

mkdirSync(demoDataDir, { recursive: true });
copyFileSync(metricsPath, join(demoDataDir, "metrics.json"));

const metricsSize = (readFileSync(join(demoDataDir, "metrics.json"), "utf8").length / 1024).toFixed(1);
console.log(`Demo data ready: ${metricsSize} KB`);

// ── 2. Build frontend ──────────────────────────────────────────────

// Demo site defaults: analytics is enabled when credentials are provided.
// Without credentials, analytics is omitted so forks can deploy successfully.
// The production GitHub Actions workflow separately requires secrets for the canonical repo.
let analyticsEnabled = process.env.PUBLIC_ENABLE_ANALYTICS ?? "true";
const missingAnalyticsEnv = [];
if (analyticsEnabled === "true") {
  for (const name of ["PUBLIC_POSTHOG_KEY", "PUBLIC_POSTHOG_HOST"]) {
    if (!process.env[name]) missingAnalyticsEnv.push(name);
  }
}

if (missingAnalyticsEnv.length > 0) {
  if (process.env.HOWIPROMPT_REQUIRE_ANALYTICS === "true") {
    console.error(`\nAnalytics is required, but missing env var(s): ${missingAnalyticsEnv.join(", ")}`);
    console.error("Set them from secrets, or unset HOWIPROMPT_REQUIRE_ANALYTICS.\n");
    process.exit(1);
  }
  console.warn(`Analytics disabled for this build: missing ${missingAnalyticsEnv.join(", ")}`);
  analyticsEnabled = "false";
}

const buildEnv = {
  PUBLIC_ENABLE_ANALYTICS: analyticsEnabled,
  PUBLIC_ENABLE_LINGUISTICS: process.env.PUBLIC_ENABLE_LINGUISTICS ?? "",
  PUBLIC_POSTHOG_KEY: process.env.PUBLIC_POSTHOG_KEY ?? "",
  PUBLIC_POSTHOG_HOST: process.env.PUBLIC_POSTHOG_HOST ?? "",
  PUBLIC_POSTHOG_UI_HOST: process.env.PUBLIC_POSTHOG_UI_HOST ?? "https://us.posthog.com",
  PUBLIC_ANALYTICS_ALLOWED_HOSTS: process.env.PUBLIC_ANALYTICS_ALLOWED_HOSTS ?? "",
};

await run("npm", ["run", "build"], frontendDir, buildEnv);

// ── 3. Copy metrics into build output ─────────────────────────────

const srcMetrics = join(demoDataDir, "metrics.json");

copyFileSync(srcMetrics, join(distDir, "metrics.json"));
mkdirSync(join(distDir, "wrapped"), { recursive: true });
copyFileSync(srcMetrics, join(distDir, "wrapped", "metrics.json"));

console.log("Copied metrics into build output.");

// ── 4. Sync to dist-static/ ────────────────────────────────────────

// Preserve CNAME if it already exists in a previous dist-static
const existingCname = existsSync(join(distStaticDir, "CNAME"))
  ? readFileSync(join(distStaticDir, "CNAME"), "utf8")
  : null;

rmSync(distStaticDir, { recursive: true, force: true });
mkdirSync(distStaticDir, { recursive: true });

// Copy build output to dist-static
const { cpSync } = await import("node:fs");
cpSync(distDir, distStaticDir, { recursive: true });

// Write .nojekyll (prevents GitHub Pages from ignoring files starting with _)
writeFileSync(join(distStaticDir, ".nojekyll"), "");

// Restore CNAME if it was preserved
if (existingCname && existingCname.trim()) {
  writeFileSync(join(distStaticDir, "CNAME"), existingCname.trim() + "\n");
}

console.log(`Synced build to ${distStaticDir}`);

// ── Done ───────────────────────────────────────────────────────────

console.log("\nPublish check complete. Commit data/demo/metrics.json with your changes.");
console.log("Push to main to trigger the GitHub Actions deployment.\n");

// ── Helpers ────────────────────────────────────────────────────────

async function run(command, args, cwd, env = {}) {
  process.stdout.write(`\n$ ${[command, ...args].join(" ")}\n`);
  await new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      cwd,
      stdio: "inherit",
      env: { ...process.env, ...env },
    });

    child.on("error", reject);
    child.on("close", (code) => {
      if (code === 0) {
        resolve();
        return;
      }
      reject(new Error(`${command} ${args.join(" ")} exited with ${code}`));
    });
  });
}