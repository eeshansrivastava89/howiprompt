#!/usr/bin/env node

import { execSync } from 'node:child_process';
import { cpSync, existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const frontendRoot = path.resolve(__dirname, '..');
const repoRoot = path.resolve(frontendRoot, '..');
const distDir = path.join(frontendRoot, 'dist');
const docsDir = path.join(repoRoot, 'docs');
const demoDeploy = /^(1|true)$/i.test(process.env.DEMO_DEPLOY || 'false');
const metricsPath = process.env.HOWIPROMPT_METRICS_PATH || path.join(os.homedir(), '.howiprompt', 'metrics.json');
const wrappedMetricsPath = path.join(distDir, 'wrapped', 'metrics.json');

function run(command, cwd) {
  execSync(command, {
    cwd,
    stdio: 'inherit',
    env: process.env,
  });
}

function syncDemoArtifacts() {
  if (!existsSync(metricsPath)) {
    throw new Error(`DEMO_DEPLOY=true but metrics file was not found at ${metricsPath}`);
  }

  const existingCname = existsSync(path.join(docsDir, 'CNAME'))
    ? readFileSync(path.join(docsDir, 'CNAME'), 'utf8')
    : null;

  cpSync(metricsPath, path.join(distDir, 'metrics.json'));
  mkdirSync(path.dirname(wrappedMetricsPath), { recursive: true });
  cpSync(metricsPath, wrappedMetricsPath);

  rmSync(docsDir, { recursive: true, force: true });
  mkdirSync(docsDir, { recursive: true });
  cpSync(distDir, docsDir, { recursive: true });
  writeFileSync(path.join(docsDir, '.nojekyll'), '');
  if (existingCname && existingCname.trim()) {
    writeFileSync(path.join(docsDir, 'CNAME'), `${existingCname.trim()}\n`);
  }

  console.log(`Synced demo build to ${docsDir}`);
}

run('npx astro build', frontendRoot);

if (demoDeploy) {
  syncDemoArtifacts();
}
