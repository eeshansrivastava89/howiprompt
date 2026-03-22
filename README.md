> **Disclaimer:** This is an independent personal project, **not affiliated with, endorsed by, or connected to Anthropic or Claude** in any way. This is a non-commercial, open-source tool created for personal use and educational purposes only.

# How I Prompt

A personal analytics dashboard for your Claude Code + Codex AI conversations. See your prompting patterns at a glance.

**[View Live Demo →](https://howiprompt.eeshans.com)**

<img width="1179" height="859" alt="image" src="https://github.com/user-attachments/assets/12c6fc86-0d08-45c1-b94b-39c3bfbff93d" />

---

## Quick Start

```bash
npx howiprompt
```

That's it. Syncs your conversations, builds analytics, and opens the dashboard in your browser.

### Options

```bash
npx howiprompt --no-open     # don't auto-open browser
npx howiprompt --port 4000   # custom port
npx howiprompt --help        # usage info
```

### What Happens

1. Copies conversation data from `~/.claude/projects/` and `~/.codex/history.jsonl`
2. Parses and stores messages in a local SQLite database (`~/.howiprompt/data.db`)
3. Runs NLP classifiers (intent, complexity, iteration style)
4. Computes analytics (volume, depth, temporal, style, trends, persona)
5. Serves an interactive dashboard at `localhost`

Subsequent runs are incremental — only new messages are synced.

---

## What You Get

| Dashboard | Full Experience |
|-----------|-----------------|
| One-page overview of all your stats | Scroll-through "Wrapped" style presentation |
| [howiprompt.eeshans.com](https://howiprompt.eeshans.com) | [howiprompt.eeshans.com/wrapped](https://howiprompt.eeshans.com/wrapped) |

**Metrics include:** Total prompts, word counts, conversation depth, activity heatmap, prompt style analysis, trend charts, model usage, and your AI persona classification.

---

## Data Sources

| Source | Location |
|--------|----------|
| **Claude Code** | `~/.claude/projects/*.jsonl` |
| **Codex** | `~/.codex/history.jsonl` |

Both are auto-synced on each run. A backup is kept at `~/.howiprompt/raw/`.

---

## Privacy

- **100% local** — Nothing leaves your machine
- **Persistent storage** — Conversations saved in `~/.howiprompt/data.db` (survives even if source files are deleted)
- **No conversation text in output** — Only aggregate statistics

---

## The 4 Personas

Your style is classified on two axes: **Engagement** (questions + iteration) and **Politeness** (courtesy - commands).

|                    | High Politeness | Low Politeness |
|--------------------|-----------------|----------------|
| **High Engagement** | Collaborator | Explorer |
| **Low Engagement**  | Efficient | Pragmatist |

---

## Development

```bash
# Install deps
npm install

# Build TypeScript
npm run build

# Run tests
npm test

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
