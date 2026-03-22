/**
 * Idempotent database schema bootstrap for howiprompt.
 * Uses @libsql/client — safe to run every launch.
 */

import { createClient } from "@libsql/client";

const SCHEMA_SQL = [
  `CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    hash TEXT UNIQUE NOT NULL,
    timestamp TEXT NOT NULL,
    platform TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    word_count INTEGER NOT NULL,
    model_id TEXT,
    model_provider TEXT,
    local_hour INTEGER NOT NULL,
    local_weekday INTEGER NOT NULL,
    local_date TEXT NOT NULL,
    source_file TEXT,
    synced_at TEXT NOT NULL,
    embedding BLOB
  )`,

  `CREATE TABLE IF NOT EXISTS nlp_enrichments (
    message_id INTEGER PRIMARY KEY REFERENCES messages(id),
    intent TEXT,
    intent_confidence REAL,
    complexity_score REAL,
    complexity_confidence REAL,
    iteration_score REAL,
    iteration_confidence REAL
  )`,

  `CREATE TABLE IF NOT EXISTS sync_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    last_file TEXT,
    last_timestamp TEXT,
    message_count INTEGER,
    synced_at TEXT NOT NULL
  )`,

  `CREATE INDEX IF NOT EXISTS idx_messages_platform_role ON messages(platform, role)`,
  `CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id)`,
  `CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp)`,
  `CREATE INDEX IF NOT EXISTS idx_messages_hash ON messages(hash)`,
  `CREATE INDEX IF NOT EXISTS idx_messages_local_date ON messages(local_date)`,
];

/**
 * Bootstrap the database schema at the given path.
 * @param {string} dbPath — absolute path to the SQLite file
 */
export async function bootstrapDb(dbPath) {
  const client = createClient({ url: `file:${dbPath}` });

  for (const sql of SCHEMA_SQL) {
    await client.execute(sql);
  }

  client.close();
}
