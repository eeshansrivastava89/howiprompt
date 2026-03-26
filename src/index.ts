import fs from "node:fs";
import path from "node:path";
import { createDbClient, insertMessages, logSync } from "./pipeline/db.js";
import { loadConfig, loadBranding } from "./pipeline/config.js";
import { loadMlConfig } from "./pipeline/ml-config.js";
import { getEnabledBackends } from "./pipeline/backends.js";
import { enrichNlp } from "./pipeline/nlp.js";
import { enrichEmbeddings } from "./pipeline/embeddings.js";
import { enrichClassifiers } from "./pipeline/classifiers.js";
import { computeSourceViews } from "./pipeline/metrics.js";

export interface PipelineOptions {
  dbPath: string;
  dataDir: string;
  projectRoot?: string;
  onProgress?: (stage: string, detail: string) => void;
}

export interface PipelineStats {
  newMessages: number;
  totalMessages: number;
  enriched: number;
  embedded: number;
}

export async function runPipeline(opts: PipelineOptions): Promise<PipelineStats> {
  const config = loadConfig(opts.dataDir);
  const mlConfig = loadMlConfig(opts.dataDir);
  const client = createDbClient(opts.dbPath);
  const log = opts.onProgress ?? (() => {});

  // Sync + parse via backend registry
  const backends = getEnabledBackends(config);
  const allMessages: import("./pipeline/models.js").Message[] = [];

  for (const backend of backends) {
    log("sync", `Syncing ${backend.name}...`);
    backend.sync(config);
    log("parse", `Parsing ${backend.name}...`);
    const msgs = await backend.parse(config);
    if (msgs.length > 0) {
      const lastTs = msgs[msgs.length - 1].timestamp.toISOString();
      await logSync(client, backend.id, null, lastTs, msgs.length);
    }
    allMessages.push(...msgs);
    log("parse", `${backend.name}: ${msgs.length} messages`);
  }

  // Insert (dedup via hash)
  log("insert", `Inserting ${allMessages.length.toLocaleString()} messages...`);
  const { inserted, skipped } = await insertMessages(client, allMessages);
  log("insert", `${inserted.toLocaleString()} new, ${skipped.toLocaleString()} already synced`);

  // NLP enrichment (only un-enriched messages)
  log("nlp", "Running NLP enrichment...");
  const enriched = await enrichNlp(client);
  log("nlp", `${enriched.toLocaleString()} messages enriched`);

  // Embedding enrichment (only un-embedded human messages)
  log("embedding", "Computing embeddings...");
  const embedded = await enrichEmbeddings(
    client, mlConfig, opts.dataDir,
    (progress) => {
      if (progress.status === "download" && progress.progress !== undefined) {
        log("embedding", `Downloading model: ${Math.round(progress.progress)}%`);
      }
    },
    (done, total) => {
      log("embedding", `Embedded ${done.toLocaleString()} / ${total.toLocaleString()} messages`);
    },
  );
  log("embedding", `${embedded.toLocaleString()} embeddings computed`);

  // Classifier enrichment (HITL + Vibe scores from embeddings)
  log("classifiers", "Scoring personas...");
  const classified = await enrichClassifiers(client, mlConfig, opts.dataDir, (done, total) => {
    log("classifiers", `Classified ${done.toLocaleString()} / ${total.toLocaleString()} messages`);
  });
  log("classifiers", `${classified.toLocaleString()} messages classified`);

  // Compute metrics
  log("metrics", "Aggregating metrics...");
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

  return { newMessages: inserted, totalMessages, enriched, embedded };
}
