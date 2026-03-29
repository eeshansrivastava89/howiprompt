import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { bootstrapDb } from "../bin/bootstrap-db.mjs";
import { createDbClient, insertMessages } from "../src/pipeline/db.js";
import { enrichNlp } from "../src/pipeline/nlp.js";
import { enrichEmbeddings, resetEmbedderForTests } from "../src/pipeline/embeddings.js";
import { enrichClassifiers, resetClassifiersForTests } from "../src/pipeline/classifiers.js";
import { computeMetrics } from "../src/pipeline/metrics.js";
import { loadConfig } from "../src/pipeline/config.js";
import { loadMlConfig } from "../src/pipeline/ml-config.js";
import { Platform, Role, type Message } from "../src/pipeline/models.js";
import type { Client } from "@libsql/client";

function fakeVectorForText(text: string, dims = 384): Float32Array {
  const vec = new Float32Array(dims);
  const tokens = text.toLowerCase().split(/[^a-z0-9_]+/).filter(Boolean);
  for (const token of tokens) {
    let hash = 0;
    for (let i = 0; i < token.length; i++) {
      hash = (hash * 31 + token.charCodeAt(i)) % dims;
    }
    vec[hash] += 1;
  }
  let norm = 0;
  for (let i = 0; i < vec.length; i++) norm += vec[i] * vec[i];
  norm = Math.sqrt(norm) || 1;
  for (let i = 0; i < vec.length; i++) vec[i] /= norm;
  return vec;
}

vi.mock("@huggingface/transformers", () => ({
  env: {},
  pipeline: async () => async (inputs: string[] | string) => {
    const batch = Array.isArray(inputs) ? inputs : [inputs];
    return batch.map((text) => ({ data: fakeVectorForText(String(text)) }));
  },
}));

describe("ML pipeline integration", () => {
  let client: Client;
  let dbPath: string;

  beforeEach(async () => {
    resetEmbedderForTests();
    resetClassifiersForTests();
    dbPath = path.join(os.tmpdir(), `howiprompt-ml-${Date.now()}.db`);
    await bootstrapDb(dbPath);
    client = createDbClient(dbPath);
  });

  afterEach(() => {
    client.close();
    try { fs.unlinkSync(dbPath); } catch {}
  });

  it("computes embeddings and classifier outputs end-to-end", async () => {
    const clusters = JSON.parse(
      fs.readFileSync(path.join(process.cwd(), "data", "reference_clusters.json"), "utf-8"),
    );

    const prompts = [
      clusters.hitl.course_correction[0],
      clusters.vibe.file_reference[0],
      clusters.politeness.courteous[0],
    ];

    const messages: Message[] = prompts.flatMap((content, i) => ([
      {
        timestamp: new Date(2026, 2, i + 1, 10, 0, 0),
        platform: i % 2 === 0 ? Platform.CLAUDE_CODE : Platform.CODEX,
        role: Role.HUMAN,
        content,
        conversationId: `ml-session-${i}`,
        wordCount: content.split(/\s+/).length,
      },
      {
        timestamp: new Date(2026, 2, i + 1, 10, 1, 0),
        platform: i % 2 === 0 ? Platform.CLAUDE_CODE : Platform.CODEX,
        role: Role.ASSISTANT,
        content: "Acknowledged.",
        conversationId: `ml-session-${i}`,
        wordCount: 1,
      },
    ]));

    await insertMessages(client, messages);
    await enrichNlp(client);

    const dataDir = fs.mkdtempSync(path.join(os.tmpdir(), "howiprompt-ml-data-"));
    const mlConfig = loadMlConfig(dataDir);

    const embedded = await enrichEmbeddings(client, mlConfig, dataDir);
    const classified = await enrichClassifiers(client, mlConfig, dataDir);
    const metrics = await computeMetrics(client, loadConfig(dataDir));

    expect(embedded).toBe(prompts.length);
    expect(classified).toBe(prompts.length);

    const embeddingsResult = await client.execute(
      "SELECT COUNT(*) as cnt FROM messages WHERE role = 'human' AND embedding IS NOT NULL",
    );
    expect(Number(embeddingsResult.rows[0].cnt)).toBe(prompts.length);

    const classifierResult = await client.execute(
      "SELECT COUNT(*) as cnt FROM nlp_enrichments WHERE hitl_score IS NOT NULL AND vibe_score IS NOT NULL AND politeness_score IS NOT NULL",
    );
    expect(Number(classifierResult.rows[0].cnt)).toBe(prompts.length);

    expect(metrics.nlp.hitl_score.avg_score).not.toBeNull();
    expect(metrics.nlp.vibe_coder_index.avg_score).not.toBeNull();
    expect(metrics.nlp.politeness.avg_score).not.toBeNull();
    expect(metrics.persona.quadrant).toBeTruthy();
    expect(metrics.trends.weekly_rollups[0]?.nlp?.hitl_score).not.toBeUndefined();

    fs.rmSync(dataDir, { recursive: true, force: true });
  });
});
