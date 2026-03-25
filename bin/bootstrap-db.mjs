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
 * Additive migrations — safe to re-run (catch duplicate column errors).
 */
const MIGRATIONS = [
  // Phase 2: semantic classifiers
  `ALTER TABLE nlp_enrichments ADD COLUMN hitl_score REAL`,
  `ALTER TABLE nlp_enrichments ADD COLUMN hitl_confidence REAL`,
  `ALTER TABLE nlp_enrichments ADD COLUMN vibe_score REAL`,
  `ALTER TABLE nlp_enrichments ADD COLUMN vibe_confidence REAL`,
  // Phase 6: radar axes
  `ALTER TABLE nlp_enrichments ADD COLUMN precision_score REAL`,
  `ALTER TABLE nlp_enrichments ADD COLUMN precision_confidence REAL`,
  `ALTER TABLE nlp_enrichments ADD COLUMN curiosity_score REAL`,
  `ALTER TABLE nlp_enrichments ADD COLUMN curiosity_confidence REAL`,
  `ALTER TABLE nlp_enrichments ADD COLUMN tenacity_score REAL`,
  `ALTER TABLE nlp_enrichments ADD COLUMN tenacity_confidence REAL`,
  `ALTER TABLE nlp_enrichments ADD COLUMN trust_score REAL`,
  `ALTER TABLE nlp_enrichments ADD COLUMN trust_confidence REAL`,
  // Politeness embedding classifier
  `ALTER TABLE nlp_enrichments ADD COLUMN politeness_score REAL`,
  `ALTER TABLE nlp_enrichments ADD COLUMN politeness_confidence REAL`,
];

/**
 * New tables added in later phases.
 */
const NEW_TABLES = [
  `CREATE TABLE IF NOT EXISTS reference_embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    classifier TEXT NOT NULL,
    cluster TEXT NOT NULL,
    prompt TEXT NOT NULL,
    embedding F32_BLOB(384) NOT NULL
  )`,
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

  // New tables
  for (const sql of NEW_TABLES) {
    await client.execute(sql);
  }

  // Additive migrations (ignore duplicate column errors)
  for (const sql of MIGRATIONS) {
    try {
      await client.execute(sql);
    } catch (err) {
      if (!String(err).includes("duplicate column")) throw err;
    }
  }

  // Clean up agent/bot messages (excluded at parse time now)
  // 1. Remove rows tagged as agent platform
  await client.execute("DELETE FROM nlp_enrichments WHERE message_id IN (SELECT id FROM messages WHERE platform = 'agent')");
  await client.execute("DELETE FROM messages WHERE platform = 'agent'");
  // 2. Remove entire conversations containing Goalbot system prompts
  await client.execute("DELETE FROM nlp_enrichments WHERE message_id IN (SELECT id FROM messages WHERE conversation_id IN (SELECT DISTINCT conversation_id FROM messages WHERE role = 'human' AND content LIKE 'Learned patterns%'))");
  await client.execute("DELETE FROM messages WHERE conversation_id IN (SELECT DISTINCT conversation_id FROM messages WHERE role = 'human' AND content LIKE 'Learned patterns%')");
  // 3. Remove agent-* compact sessions
  await client.execute("DELETE FROM nlp_enrichments WHERE message_id IN (SELECT id FROM messages WHERE conversation_id LIKE 'agent-%')");
  await client.execute("DELETE FROM messages WHERE conversation_id LIKE 'agent-%'");

  client.close();
}
