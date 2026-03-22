import fs from "node:fs";
import path from "node:path";
import { createDbClient, insertMessages, logSync } from "./pipeline/db.js";
import { loadConfig, loadBranding } from "./pipeline/config.js";
import { syncClaudeCode, syncCodex } from "./pipeline/sync.js";
import { parseClaudeCode, parseCodexHistory, parseCodexSessionMetadata } from "./pipeline/parsers.js";
import { enrichNlp } from "./pipeline/nlp.js";
import { computeSourceViews } from "./pipeline/metrics.js";

export interface PipelineOptions {
  dbPath: string;
  dataDir: string;
  projectRoot?: string;
}

export interface PipelineStats {
  newMessages: number;
  totalMessages: number;
  enriched: number;
}

export async function runPipeline(opts: PipelineOptions): Promise<PipelineStats> {
  const config = loadConfig(opts.dataDir);
  const client = createDbClient(opts.dbPath);

  // Sync
  syncClaudeCode(config.claudeCodeSource);
  syncCodex(config.codexHistorySource);

  // Parse
  const ccMessages = await parseClaudeCode(config.claudeCodeSource, config.agentCwds);
  const sessionModels = await parseCodexSessionMetadata(config.codexSessionsSource);
  const cxMessages = await parseCodexHistory(config.codexHistorySource, sessionModels);

  const allMessages = [...ccMessages, ...cxMessages];

  // Insert (dedup via hash)
  const { inserted, skipped } = await insertMessages(client, allMessages);

  // Log sync
  if (ccMessages.length > 0) {
    const lastTs = ccMessages[ccMessages.length - 1].timestamp.toISOString();
    await logSync(client, "claude_code", null, lastTs, ccMessages.length);
  }
  if (cxMessages.length > 0) {
    const lastTs = cxMessages[cxMessages.length - 1].timestamp.toISOString();
    await logSync(client, "codex", null, lastTs, cxMessages.length);
  }

  // NLP enrichment (only un-enriched messages)
  const enriched = await enrichNlp(client);

  // Compute metrics
  const { sourceViews, metadata } = await computeSourceViews(client, config);

  // Add branding
  const branding = loadBranding(opts.projectRoot);
  const output: Record<string, any> = {
    source_views: sourceViews,
    default_view: metadata.default_view,
  };
  if (branding) output.branding = branding;

  // Write metrics.json
  const metricsPath = path.join(opts.dataDir, "metrics.json");
  fs.writeFileSync(metricsPath, JSON.stringify(output, null, 2));

  // Get total count
  const totalResult = await client.execute("SELECT COUNT(*) as cnt FROM messages");
  const totalMessages = Number(totalResult.rows[0].cnt);

  client.close();

  return { newMessages: inserted, totalMessages, enriched };
}
