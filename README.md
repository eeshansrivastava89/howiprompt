# How I Prompt: Wrapped

A "year in review" visualization of your Claude AI conversations. Analyze your prompting style, discover your AI persona, and generate a shareable wrapped-style report.

I first ran this for my 2025 review, but you can run this at any time. You bring your data and are responsible for privacy concerns. By default, this script does not read memories.json, users.json or projects.json. Only reads conversations.json from Claude AI export. You can choose to not include any sensitive files / conversations in the data folder.

## Quick Start

```bash
# 1. Add your data
cp -r ~/.claude/projects/* data/claude_code/
cp ~/path/to/claude-export/conversations.json data/claude_ai/

# 2. Build
python build.py

# 3. Open
open output/index.html
```

## Data Sources

| Source | How to Get It | Where to Put It |
|--------|---------------|-----------------|
| **Claude Code** | Already on your machine at `~/.claude/projects/` | `data/claude_code/` |
| **Claude.ai** | Settings → Export Data → Download | `data/claude_ai/conversations.json` |

## CLI Options

```bash
python build.py                              # Full build (metrics + HTML)
python build.py --metrics-only               # Only compute metrics.json
python build.py --claude-code ~/.claude/projects  # Use data directly (no copy)
python build.py -o ./dist                    # Custom output directory
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
