import fs from "node:fs";
import path from "node:path";
import { createDbClient, insertMessages, logSync } from "./pipeline/db.js";
import { loadConfig, loadBranding } from "./pipeline/config.js";
import { loadMlConfig } from "./pipeline/ml-config.js";
import { getEnabledBackends, getAllBackends } from "./pipeline/backends.js";
import { enrichNlp } from "./pipeline/nlp.js";
import { enrichEmbeddings } from "./pipeline/embeddings.js";
import { enrichClassifiers } from "./pipeline/classifiers.js";
import { enrichStyle } from "./pipeline/style.js";
import { computeSourceViews } from "./pipeline/metrics.js";
import {
  seedSystemRules,
  discoverAndSyncRules,
  loadExclusionRules,
  compileRules,
  flagExcludedMessages,
} from "./pipeline/exclusions.js";
import { parseClaudeCode } from "./pipeline/parsers.js";

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

  // Purge messages from excluded source directories
  // (user expects "reanalyze" to respect exclusion changes)
  const allExclusions = Object.entries(config.backends)
    .flatMap(([, toggle]) => toggle.exclusions)
    .map((cwd) => cwd.replace(/\//g, "-").replace(/^-/, "-"));
  for (const dirName of allExclusions) {
    const pattern = `%/raw/claude_code/${dirName}/%`;
    await client.execute({
      sql: "DELETE FROM nlp_enrichments WHERE message_id IN (SELECT id FROM messages WHERE source_file LIKE ?)",
      args: [pattern],
    });
    const result = await client.execute({
      sql: "DELETE FROM messages WHERE source_file LIKE ?",
      args: [pattern],
    });
    if (result.rowsAffected > 0) {
      log({ stage: "boot", detail: `Removed ${result.rowsAffected} messages from excluded project`, progress: 10 });
    }
  }

  // Purge messages from sources matching CWD exclusion rules
  // (catches npx/node_modules sessions that were ingested before rules existed)
  for (const cwdPattern of ["node_modules", ".npm/_npx"]) {
    const likePattern = `%${cwdPattern}%`;
    await client.execute({
      sql: "DELETE FROM nlp_enrichments WHERE message_id IN (SELECT id FROM messages WHERE source_file LIKE ?)",
      args: [likePattern],
    });
    const result = await client.execute({
      sql: "DELETE FROM messages WHERE source_file LIKE ?",
      args: [likePattern],
    });
    if (result.rowsAffected > 0) {
      log({ stage: "boot", detail: `Removed ${result.rowsAffected} messages from programmatic sessions`, progress: 10 });
    }
  }

  // Exclusion rules: seed system rules, discover skill rules, compile
  log({ stage: "exclusions", detail: "Loading exclusion rules...", progress: 10 });
  const seeded = await seedSystemRules(client);
  const { skillsFound, rulesUpserted } = await discoverAndSyncRules(client, getAllBackends());
  const rules = await loadExclusionRules(client);
  const compiledRules = compileRules(rules);
  log({
    stage: "exclusions",
    detail: `${rules.length} rules active (${skillsFound} skills, ${seeded} new system rules)`,
    progress: 100,
  });

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

    // Pass compiled rules to Claude Code parser for content/CWD filtering
    let msgs: import("./pipeline/models.js").Message[];
    if (backend.id === "claude_code") {
      const exclusions = config.backends?.[backend.id]?.exclusions ?? [];
      msgs = await parseClaudeCode(config.claudeCodeSource, exclusions, compiledRules);
    } else {
      msgs = await backend.parse(config);
    }

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

  // Flag existing messages against exclusion rules
  log({ stage: "exclusions", detail: "Flagging excluded messages...", progress: 50 });
  const flagged = await flagExcludedMessages(client);
  log({
    stage: "exclusions",
    detail: flagged > 0
      ? `${flagged.toLocaleString()} messages flagged`
      : "No new messages to flag",
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

  // Classifier enrichment (HITL, Vibe, Politeness from embeddings)
  log({ stage: "classifiers", detail: "Scoring hero metrics...", progress: 10 });
  const classified = await enrichClassifiers(client, mlConfig, opts.dataDir, (done, total) => {
    log({
      stage: "classifiers",
      detail: `Classified ${done.toLocaleString()} / ${total.toLocaleString()} messages`,
      progress: total > 0 ? Math.round((done / total) * 100) : 100,
    });
  });
  log({ stage: "classifiers", detail: `${classified.toLocaleString()} messages classified`, progress: 100 });

  // Style scoring (2×2: detail level × communication style)
  log({ stage: "style", detail: "Computing style scores...", progress: 10 });
  const styled = await enrichStyle(client, (done, total) => {
    log({
      stage: "style",
      detail: `Scored ${done.toLocaleString()} / ${total.toLocaleString()} messages`,
      progress: total > 0 ? Math.round((done / total) * 100) : 100,
    });
  });
  log({ stage: "style", detail: `${styled.toLocaleString()} messages scored`, progress: 100 });

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
