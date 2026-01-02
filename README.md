> **Disclaimer:** This is an independent personal project, **not affiliated with, endorsed by, or connected to Anthropic or Claude** in any way. This is a non-commercial, open-source tool created for personal use and educational purposes only.

# How I Prompt: Wrapped

I made this little "year in review" style visualization of my Claude AI conversations. You can clone and try it out for your conversations too. Analyze your prompting style, discover your AI persona, and generate a shareable wrapped-style report. If you have more expertise in text analysis, then there a lot of possiblities in building new metrics or personas. Feel free to fork or let me know if you have any ideas. 

Made on a mac but should be configurable for Windows too with a bit of tweaking. Python 3.10+ required. No external dependencies.

---

## ⚠️ Privacy & Data Warning

**This tool processes your personal AI conversation data. Please read carefully:**

| | |
|---|---|
| **Your data stays local** | Nothing is uploaded, transmitted, or shared. All processing happens on your machine. |
| **You control what's included** | Exclude sensitive conversations by not copying them to the `data/` folder. |
| **Review before sharing** | The generated report contains aggregate statistics only (no conversation text), but review it before posting publicly. |
| **Your responsibility** | You are solely responsible for any data you choose to process and any reports you choose to share. |

**By default, this script does NOT read:** `memories.json`, `users.json`, or `projects.json`. It only reads conversation data you explicitly provide.

---

## Quick Start

```bash
# 1. Add your data
python build.py --copy-claude-code                    # Copies from ~/.claude/projects
cp ~/Downloads/claude-export/conversations.json data/claude_ai/

# 2. Build
python build.py

# 3. Open
open output/index.html
```

## Data Sources

| Source | How to Get It | Where to Put It |
|--------|---------------|-----------------|
| **Claude Code** | Run `--copy-claude-code` OR manually copy from `~/.claude/projects/` | `data/claude_code/` |
| **Claude.ai** | Settings → Export Data → Download | `data/claude_ai/conversations.json` |

## CLI Options

```bash
python build.py                    # Full build (reads from data/ folder)
python build.py --copy-claude-code # Copy Claude Code data, then build
python build.py --metrics-only     # Only compute metrics.json
python build.py -o ./dist          # Custom output directory
```

## Output

```
output/
├── metrics.json       # Computed metrics (JSON)
└── index.html         # Your wrapped report
```

## The 4 Personas

Your prompting style is classified using a 2x2 matrix:

|                    | High Politeness | Low Politeness |
|--------------------|-----------------|----------------|
| **High Engagement** | The Collaborator | The Explorer |
| **Low Engagement**  | The Efficient | The Pragmatist |

- **Engagement** = (question rate + backtrack rate) / 2
- **Politeness** = polite phrases - (command rate × 0.5)

## Requirements

- Python 3.10+
- No external dependencies (stdlib only)

---

## License

MIT License - See [LICENSE](LICENSE) file.

*This is a personal, non-commercial project. Not affiliated with Anthropic.*
