import { createHash } from "node:crypto";
import { createClient, type Client, type InStatement } from "@libsql/client";
import { Message, Platform, Role } from "./models.js";

export function createDbClient(dbPath: string): Client {
  return createClient({ url: `file:${dbPath}` });
}

export function messageHash(m: Message): string {
  const data = `${m.platform}|${m.conversationId}|${m.timestamp.toISOString()}|${m.content.slice(0, 200)}`;
  return createHash("sha256").update(data).digest("hex");
}

export async function insertMessages(
  client: Client,
  messages: Message[],
): Promise<{ inserted: number; skipped: number }> {
  let inserted = 0;
  let skipped = 0;
  const batchSize = 500;

  for (let i = 0; i < messages.length; i += batchSize) {
    const batch = messages.slice(i, i + batchSize);
    const stmts: InStatement[] = batch.map((m) => {
      const local = localTime(m.timestamp);
      return {
        sql: `INSERT OR IGNORE INTO messages
              (hash, timestamp, platform, role, content, conversation_id,
               word_count, model_id, model_provider,
               local_hour, local_weekday, local_date, source_file, synced_at)
              VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))`,
        args: [
          messageHash(m),
          m.timestamp.toISOString(),
          m.platform,
          m.role,
          m.content,
          m.conversationId,
          m.wordCount,
          m.modelId ?? null,
          m.modelProvider ?? null,
          local.hour,
          local.weekday,
          local.dateStr,
          m.sourceFile ?? null,
        ],
      };
    });

    const results = await client.batch(stmts, "write");
    for (const r of results) {
      if (r.rowsAffected > 0) inserted++;
      else skipped++;
    }
  }

  return { inserted, skipped };
}

function localTime(d: Date): { hour: number; weekday: number; dateStr: string } {
  return {
    hour: d.getHours(),
    weekday: ((d.getDay() + 6) % 7), // JS: 0=Sun → Python-like: 0=Mon
    dateStr: `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`,
  };
}

export async function insertNlpEnrichments(
  client: Client,
  enrichments: Array<{
    messageId: number;
    intent: string;
    intentConfidence: number;
    complexityScore: number;
    complexityConfidence: number;
    iterationScore: number;
    iterationConfidence: number;
  }>,
): Promise<void> {
  const batchSize = 500;
  for (let i = 0; i < enrichments.length; i += batchSize) {
    const batch = enrichments.slice(i, i + batchSize);
    await client.batch(
      batch.map((e) => ({
        sql: `INSERT OR IGNORE INTO nlp_enrichments
              (message_id, intent, intent_confidence, complexity_score,
               complexity_confidence, iteration_score, iteration_confidence)
              VALUES (?, ?, ?, ?, ?, ?, ?)`,
        args: [
          e.messageId,
          e.intent,
          e.intentConfidence,
          e.complexityScore,
          e.complexityConfidence,
          e.iterationScore,
          e.iterationConfidence,
        ],
      })),
      "write",
    );
  }
}

export interface MessageRow {
  id: number;
  timestamp: string;
  platform: string;
  role: string;
  content: string;
  conversationId: string;
  wordCount: number;
  modelId: string | null;
  modelProvider: string | null;
}

export async function queryMessages(
  client: Client,
  opts?: { role?: Role; platform?: Platform },
): Promise<MessageRow[]> {
  let sql =
    "SELECT id, timestamp, platform, role, content, conversation_id, word_count, model_id, model_provider FROM messages WHERE 1=1";
  const args: any[] = [];
  if (opts?.role) {
    sql += " AND role = ?";
    args.push(opts.role);
  }
  if (opts?.platform) {
    sql += " AND platform = ?";
    args.push(opts.platform);
  }
  sql += " ORDER BY timestamp";

  const result = await client.execute({ sql, args });
  return result.rows.map((r) => ({
    id: Number(r.id),
    timestamp: String(r.timestamp),
    platform: String(r.platform),
    role: String(r.role),
    content: String(r.content),
    conversationId: String(r.conversation_id),
    wordCount: Number(r.word_count),
    modelId: r.model_id ? String(r.model_id) : null,
    modelProvider: r.model_provider ? String(r.model_provider) : null,
  }));
}

export function platformFilter(platform?: Platform): { clause: string; args: any[] } {
  if (platform) return { clause: " AND platform = ?", args: [platform] };
  return { clause: "", args: [] };
}

export async function getLastSync(
  client: Client,
  source: string,
): Promise<{ lastTimestamp: string | null; lastFile: string | null }> {
  const result = await client.execute({
    sql: "SELECT last_timestamp, last_file FROM sync_log WHERE source = ? ORDER BY synced_at DESC LIMIT 1",
    args: [source],
  });
  if (result.rows.length === 0) return { lastTimestamp: null, lastFile: null };
  const row = result.rows[0];
  return {
    lastTimestamp: row.last_timestamp ? String(row.last_timestamp) : null,
    lastFile: row.last_file ? String(row.last_file) : null,
  };
}

export async function logSync(
  client: Client,
  source: string,
  lastFile: string | null,
  lastTimestamp: string | null,
  messageCount: number,
): Promise<void> {
  await client.execute({
    sql: "INSERT INTO sync_log (source, last_file, last_timestamp, message_count, synced_at) VALUES (?, ?, ?, ?, datetime('now'))",
    args: [source, lastFile, lastTimestamp, messageCount],
  });
}
