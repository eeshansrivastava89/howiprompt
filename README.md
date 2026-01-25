> **Disclaimer:** This is an independent personal project, **not affiliated with, endorsed by, or connected to Anthropic or Claude** in any way. This is a non-commercial, open-source tool created for personal use and educational purposes only.

# How I Prompt

A personal analytics dashboard for your Claude AI conversations. See your prompting patterns at a glance.

**[View Live Demo →](https://howiprompt.eeshans.com)**

<img width="800" alt="Dashboard" src="https://github.com/user-attachments/assets/1defc38a-789b-44f7-93a1-fe2b6130a2fa" />

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

# 2. Add your data
python build.py --copy-claude-code                    # Claude Code logs
cp ~/Downloads/claude-export/conversations.json data/claude_ai/  # Claude.ai export

# 3. Build & open
python build.py
open output/dashboard.html   # Dashboard view
open output/index.html       # Full wrapped experience
```

### Data Sources

| Source | How to Get It |
|--------|---------------|
| **Claude Code** | Auto-copied with `--copy-claude-code` from `~/.claude/projects/` |
| **Claude.ai** | Settings → Export Data → Download → copy `conversations.json` to `data/claude_ai/` |

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

## Requirements

- Python 3.10+ (no external dependencies)
- macOS or Linux (Windows with minor tweaks)

---

## License

MIT License — Not affiliated with Anthropic.
