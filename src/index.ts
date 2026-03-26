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
  onProgress?: (progress: PipelineProgress) => void;
}

export interface PipelineStats {
  newMessages: number;
  totalMessages: number;
  enriched: number;
  embedded: number;
}

export interface PipelineProgress {
  stage: string;
  detail: string;
  progress?: number;
}

export async function runPipeline(opts: PipelineOptions): Promise<PipelineStats> {
  const config = loadConfig(opts.dataDir);
  const mlConfig = loadMlConfig(opts.dataDir);
  const client = createDbClient(opts.dbPath);
  const log = opts.onProgress ?? (() => {});

  // Sync + parse via backend registry
  const backends = getEnabledBackends(config);
  const allMessages: import("./pipeline/models.js").Message[] = [];
  log({ stage: "boot", detail: "Loading configuration...", progress: 20 });
  log({
    stage: "boot",
    detail: `Found ${backends.length} enabled ${backends.length === 1 ? "source" : "sources"}.`,
    progress: 100,
  });

  for (const [index, backend] of backends.entries()) {
    const completedFraction = backends.length > 0 ? index / backends.length : 1;
    log({
      stage: "sync",
      detail: `Syncing ${backend.name}...`,
      progress: Math.round(completedFraction * 100),
    });
    const syncResult = backend.sync(config);
    log({
      stage: "sync",
      detail: `${backend.name}: ${syncResult.files.toLocaleString()} ${syncResult.files === 1 ? "file" : "files"} copied`,
      progress: Math.round(((index + 1) / Math.max(backends.length, 1)) * 100),
    });

    log({
      stage: "parse",
      detail: `Parsing ${backend.name}...`,
      progress: Math.round(completedFraction * 100),
    });
    const msgs = await backend.parse(config);
    if (msgs.length > 0) {
      const lastTs = msgs[msgs.length - 1].timestamp.toISOString();
      await logSync(client, backend.id, null, lastTs, msgs.length);
    }
    allMessages.push(...msgs);
    log({
      stage: "parse",
      detail: `${backend.name}: ${msgs.length.toLocaleString()} messages`,
      progress: Math.round(((index + 1) / Math.max(backends.length, 1)) * 100),
    });
  }

  // Insert (dedup via hash)
  log({ stage: "insert", detail: `Inserting ${allMessages.length.toLocaleString()} messages...`, progress: 20 });
  const { inserted, skipped } = await insertMessages(client, allMessages);
  log({
    stage: "insert",
    detail: `${inserted.toLocaleString()} new, ${skipped.toLocaleString()} already synced`,
    progress: 100,
  });

  // NLP enrichment (only un-enriched messages)
  log({ stage: "nlp", detail: "Running NLP enrichment...", progress: 20 });
  const enriched = await enrichNlp(client);
  log({ stage: "nlp", detail: `${enriched.toLocaleString()} messages enriched`, progress: 100 });

  // Embedding enrichment (only un-embedded human messages)
  log({ stage: "embedding", detail: "Computing embeddings...", progress: 10 });
  const embedded = await enrichEmbeddings(
    client, mlConfig, opts.dataDir,
    (progress) => {
      if (progress.status === "download" && progress.progress !== undefined) {
        log({
          stage: "embedding",
          detail: `Downloading model: ${Math.round(progress.progress)}%`,
          progress: Math.round(progress.progress),
        });
      }
    },
    (done, total) => {
      log({
        stage: "embedding",
        detail: `Embedded ${done.toLocaleString()} / ${total.toLocaleString()} messages`,
        progress: total > 0 ? Math.round((done / total) * 100) : 100,
      });
    },
  );
  log({ stage: "embedding", detail: `${embedded.toLocaleString()} embeddings computed`, progress: 100 });

  // Classifier enrichment (HITL + Vibe scores from embeddings)
  log({ stage: "classifiers", detail: "Scoring personas...", progress: 10 });
  const classified = await enrichClassifiers(client, mlConfig, opts.dataDir, (done, total) => {
    log({
      stage: "classifiers",
      detail: `Classified ${done.toLocaleString()} / ${total.toLocaleString()} messages`,
      progress: total > 0 ? Math.round((done / total) * 100) : 100,
    });
  });
  log({ stage: "classifiers", detail: `${classified.toLocaleString()} messages classified`, progress: 100 });

  // Compute metrics
  log({ stage: "metrics", detail: "Aggregating metrics...", progress: 30 });
  const { sourceViews, metadata } = await computeSourceViews(client, config);
  log({ stage: "metrics", detail: "Writing dashboard output...", progress: 85 });

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
  log({ stage: "metrics", detail: "Dashboard ready.", progress: 100 });

  client.close();

  return { newMessages: inserted, totalMessages, enriched, embedded };
}
