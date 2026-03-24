-- Leaderboard D1 schema
CREATE TABLE IF NOT EXISTS submissions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    display_name TEXT NOT NULL,
    github_id TEXT,
    total_conversations INTEGER NOT NULL DEFAULT 0,
    total_prompts INTEGER NOT NULL DEFAULT 0,
    avg_words_per_prompt REAL NOT NULL DEFAULT 0,
    politeness INTEGER NOT NULL DEFAULT 0,
    backtrack INTEGER NOT NULL DEFAULT 0,
    question_rate INTEGER NOT NULL DEFAULT 0,
    command_rate INTEGER NOT NULL DEFAULT 0,
    hitl_score INTEGER NOT NULL DEFAULT 0,
    vibe_index INTEGER NOT NULL DEFAULT 0,
    persona TEXT NOT NULL DEFAULT 'unknown',
    complexity_avg INTEGER NOT NULL DEFAULT 0,
    platform TEXT NOT NULL DEFAULT 'both',
    tool_version TEXT NOT NULL DEFAULT '2.0.0',
    submitted_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- One submission per display_name (upsert on resubmit)
CREATE UNIQUE INDEX IF NOT EXISTS idx_submissions_name ON submissions(display_name);
