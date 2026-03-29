import path from "node:path";
import type { Client } from "@libsql/client";
import type { MlConfig } from "./ml-config.js";

let extractor: any = null;

export function resetEmbedderForTests(): void {
  extractor = null;
}

export async function initEmbedder(
  mlConfig: MlConfig,
  cacheDir: string,
  onProgress?: (progress: { status: string; file?: string; progress?: number }) => void,
): Promise<void> {
  if (extractor) return;

  const { env, pipeline: createPipeline } = await import("@huggingface/transformers");

  env.cacheDir = cacheDir;
  // Silence unnecessary logs
  env.allowRemoteModels = true;

  extractor = await createPipeline("feature-extraction", mlConfig.embedding.model, {
    dtype: mlConfig.embedding.dtype as any,
    revision: "main",
    progress_callback: onProgress,
  });
}

export async function embedTexts(
  texts: string[],
  mlConfig: MlConfig,
): Promise<Float32Array[]> {
  if (!extractor) throw new Error("Embedder not initialized — call initEmbedder() first");

  const results: Float32Array[] = [];
  const batchSize = mlConfig.embedding.batchSize;

  for (let i = 0; i < texts.length; i += batchSize) {
    const batch = texts.slice(i, i + batchSize);
    const output = await extractor(batch, { pooling: "cls", normalize: true });

    for (let j = 0; j < batch.length; j++) {
      const vec = output[j].data;
      results.push(new Float32Array(vec));
    }
  }

  return results;
}

export async function embedSingle(
  text: string,
  mlConfig: MlConfig,
): Promise<Float32Array> {
  const results = await embedTexts([text], mlConfig);
  return results[0];
}

export function cosineSimilarity(a: Float32Array, b: Float32Array): number {
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

export async function enrichEmbeddings(
  client: Client,
  mlConfig: MlConfig,
  dataDir: string,
  onProgress?: (progress: { status: string; progress?: number }) => void,
  onBatchProgress?: (embedded: number, total: number) => void,
): Promise<number> {
  // Find human messages without embeddings
  const result = await client.execute(
    "SELECT id, content FROM messages WHERE role = 'human' AND is_excluded = 0 AND embedding IS NULL",
  );

  if (result.rows.length === 0) return 0;

  // Init model (downloads on first run)
  const modelCacheDir = path.join(dataDir, "models");
  await initEmbedder(mlConfig, modelCacheDir, onProgress);

  const ids = result.rows.map((r) => Number(r.id));
  const texts = result.rows.map((r) => String(r.content));

  // Embed in small batches with brief pauses to avoid sustained CPU spike
  const batchSize = Math.min(mlConfig.embedding.batchSize, 32);
  let embedded = 0;
  const total = texts.length;

  for (let i = 0; i < total; i += batchSize) {
    const batchTexts = texts.slice(i, i + batchSize);
    const batchIds = ids.slice(i, i + batchSize);
    const vectors = await embedTexts(batchTexts, mlConfig);

    // Store vectors in DB
    const stmts = batchIds.map((id, j) => ({
      sql: "UPDATE messages SET embedding = vector32(?) WHERE id = ?",
      args: [JSON.stringify(Array.from(vectors[j])), id],
    }));

    await client.batch(stmts, "write");
    embedded += batchTexts.length;
    onBatchProgress?.(embedded, total);

    // Brief yield every batch so the event loop breathes
    if (i + batchSize < total) {
      await new Promise((r) => setTimeout(r, 10));
    }
  }

  return embedded;
}
