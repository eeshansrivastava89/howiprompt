> **Disclaimer:** This is an independent personal project, **not affiliated with, endorsed by, or connected to Anthropic or Claude** in any way. This is a non-commercial, open-source tool created for personal use and educational purposes only.

<div align="center">

# How I Prompt

**Local-first analytics for your AI coding conversations — prompts, personas, trends, and a personal wrapped view.**

[![npm](https://img.shields.io/npm/v/@eeshans/howiprompt)](https://www.npmjs.com/package/@eeshans/howiprompt)
[![license](https://img.shields.io/github/license/eeshansrivastava89/howiprompt)](LICENSE)
[![node](https://img.shields.io/node/v/@eeshans/howiprompt)](package.json)
[![platform](https://img.shields.io/badge/platform-macOS%20%7C%20Linux-blue)]()
[![demo](https://img.shields.io/badge/demo-live-cb9f6a)](https://howiprompt.eeshans.com)

[Live Dashboard](https://howiprompt.eeshans.com) • [Live Wrapped](https://howiprompt.eeshans.com/wrapped) • [Source](https://github.com/eeshansrivastava89/howiprompt)

```bash
npx @eeshans/howiprompt
```

> **Requirements:** [Node.js 18+](https://nodejs.org/). Works with Claude Code, Codex, Copilot Chat, Cursor, and LM Studio logs. The local package ships with analytics disabled.

</div>

<br>

<p align="center">
  <img src="https://github.com/user-attachments/assets/12c6fc86-0d08-45c1-b94b-39c3bfbff93d" alt="How I Prompt dashboard" width="900">
</p>

## Highlights

| | |
|---|---|
| **Local-first pipeline** | Sync, parsing, embeddings, classifier scoring, and metrics run on your machine |
| **Multi-source support** | Claude Code, Codex, Copilot Chat, Cursor, and LM Studio |
| **Two ways to explore** | Standard dashboard plus a scroll-through wrapped experience |
| **Metrics that feel personal** | Vibe Coder Index, Politeness, activity trends, heatmaps, and personas |
| **Private by default** | Raw logs stay local and the npm package ships with analytics disabled |
| **Fast repeat refreshes** | Incremental rebuilds reuse the local DB, caches, and exclusions |

## What it does

How I Prompt syncs local AI conversation logs, computes prompting metrics on-device, and opens a browser dashboard with both an overview page and a scroll-through wrapped experience.

That starts a local server and opens the dashboard. On first run, a setup wizard detects supported backends, lets you confirm sources, and then runs the pipeline.

### Options

```bash
npx @eeshans/howiprompt --no-open     # don't auto-open browser
npx @eeshans/howiprompt --port 4000   # custom port
npx @eeshans/howiprompt --help        # usage info
```

### What Happens

1. Detects supported local backends and writes setup to `~/.howiprompt/config.json`
2. Copies raw conversation data into `~/.howiprompt/raw/`
3. Parses and stores messages in a local SQLite database at `~/.howiprompt/data.db`
4. Runs embeddings and classifier scoring for dashboard metrics
5. Writes `~/.howiprompt/metrics.json` and serves the dashboard at `localhost`

Subsequent refreshes are incremental and reuse the local database, caches, and configured exclusions.

---

## What You Get

| Dashboard | Full Experience |
|-----------|-----------------|
| One-page overview of your stats | Scroll-through "Wrapped" presentation |
| [howiprompt.eeshans.com](https://howiprompt.eeshans.com) | [howiprompt.eeshans.com/wrapped](https://howiprompt.eeshans.com/wrapped) |

**Metrics include:** total prompts, conversation depth, activity heatmap, model usage, Vibe Coder Index, Politeness, persona classification (2×2: Detail Level × Communication Style), and trends.

---

## Data Sources

| Source | Location |
|--------|----------|
| **Claude Code** | `~/.claude/projects/*.jsonl` |
| **Codex** | `~/.codex/history.jsonl` |
| **Copilot Chat** | `~/Library/Application Support/Code/User/workspaceStorage` |
| **Cursor** | `~/Library/Application Support/Cursor/User/workspaceStorage` |
| **LM Studio** | `~/.lmstudio/conversations` |

All supported sources are auto-synced into `~/.howiprompt/raw/` and reused across refreshes.

### Backend Status

- Supported today: `Claude Code`, `Codex`, `Copilot Chat`, `Cursor`, `LM Studio`

---

## Privacy

- **Local by default** — Sync, parsing, embeddings, classifier scoring, and metrics run on your machine
- **Persistent storage** — Raw copies, local DB, config, and metrics live under `~/.howiprompt/`
- **No prompt text leaves your machine** — The app does not upload raw logs or prompt content
- **No analytics in the local app** — The `npx @eeshans/howiprompt` package ships with analytics disabled. PostHog is only enabled on the hosted website
- **Ancillary network requests** — The dashboard loads ApexCharts from a CDN. The CLI checks npm for version updates. These do not transmit prompt data

---

## The 4 Personas

Your persona is derived from two independent axes: **Detail Level** (brief → detailed) and **Communication Style** (directive → collaborative). These form a 2×2 grid validated on 21k prompts.

- **The Commander**: Brief + Directive — short, decisive instructions
- **The Partner**: Brief + Collaborative — quick exchanges, conversational flow
- **The Architect**: Detailed + Directive — specs, constraints, numbered requirements
- **The Explorer**: Detailed + Collaborative — context-rich, question-driven investigation

---

## Development

```bash
# Install deps
npm install

# Build TypeScript
npm run build

# Run tests
npm test

# Build frontend
cd frontend && npm run build

# Build for distribution
npm run build:cli

# Privacy gate (run before publish)
npm run check:privacy
```

### Requirements

- Node.js 18+

---

## License

MIT License — Not affiliated with Anthropic.
