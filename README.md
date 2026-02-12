> **Disclaimer:** This is an independent personal project, **not affiliated with, endorsed by, or connected to Anthropic or Claude** in any way. This is a non-commercial, open-source tool created for personal use and educational purposes only.

# How I Prompt

A personal analytics dashboard for your Claude Code + Codex AI conversations. See your prompting patterns at a glance.

**[View Live Demo →](https://howiprompt.eeshans.com)**

<img width="1179" height="859" alt="image" src="https://github.com/user-attachments/assets/12c6fc86-0d08-45c1-b94b-39c3bfbff93d" />

---

## What You Get

| Dashboard | Full Experience |
|-----------|-----------------|
| One-page overview of all your stats | Scroll-through "Wrapped" style presentation |
| [howiprompt.eeshans.com](https://howiprompt.eeshans.com) | [howiprompt.eeshans.com/wrapped](https://howiprompt.eeshans.com/wrapped) |

**Metrics include:** Total prompts, word counts, conversation depth, activity heatmap, prompt style analysis, and your AI persona classification.

---

## Build Your Own

```bash
# 1. Clone
git clone https://github.com/eeshansrivastava89/howiprompt.git
cd howiprompt

# 2. Build (auto-syncs Claude Code + Codex, then opens dashboard)
python build.py

# Optional: skip browser auto-open
python build.py --no-open
```

### Data Sources

| Source | How to Get It |
|--------|---------------|
| **Claude Code** | Auto-copied from `~/.claude/projects/` on each build |
| **Codex** | Auto-copied from `~/.codex/history.jsonl` on each build |

`v2` note: Claude.ai exports are deprecated and ignored by the active build pipeline.

---

## Privacy

- **100% local** — Nothing leaves your machine
- **You control the data** — Only processes files you explicitly add
- **No conversation text in output** — Only aggregate statistics

---

## The 4 Personas

Your style is classified on two axes: **Engagement** (questions + iteration) and **Politeness** (courtesy - commands).

|                    | High Politeness | Low Politeness |
|--------------------|-----------------|----------------|
| **High Engagement** | Collaborator | Explorer |
| **Low Engagement**  | Efficient | Pragmatist |

---

## Testing

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

---

## Requirements

- Python 3.10+ (no external dependencies)
- macOS or Linux (Windows with minor tweaks)

---

## License

MIT License — Not affiliated with Anthropic.
