import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import type { Client } from "@libsql/client";
import type { MlConfig, ClassifierConfig } from "./ml-config.js";
import { initEmbedder, embedTexts, cosineSimilarity } from "./embeddings.js";

interface ClusterCentroid {
  cluster: string;
  centroid: Float32Array;
  weight: number;
}

interface ClassifierResult {
  score: number;
  confidence: number;
  topSignals: Array<{ signal: string; similarity: number }>;
}

let vibeCentroids: ClusterCentroid[] | null = null;
let politenessCentroids: ClusterCentroid[] | null = null;

export function resetClassifiersForTests(): void {
  vibeCentroids = null;
  politenessCentroids = null;
}

function loadReferenceClusters(): Record<string, Record<string, string[]>> {
  const __dirname = path.dirname(fileURLToPath(import.meta.url));
  const candidates = [
    path.join(__dirname, "..", "..", "data", "reference_clusters.json"),
    path.join(__dirname, "..", "data", "reference_clusters.json"),
  ];
  for (const p of candidates) {
    try {
      return JSON.parse(fs.readFileSync(p, "utf-8"));
    } catch { /* try next */ }
  }
  throw new Error("reference_clusters.json not found");
}

async function buildCentroids(
  client: Client,
  classifier: string,
  clusters: Record<string, string[]>,
  weights: Record<string, number>,
  mlConfig: MlConfig,
  dataDir: string,
): Promise<ClusterCentroid[]> {
  // Check if centroids are cached in DB
  const cached = await client.execute({
    sql: "SELECT cluster, embedding FROM reference_embeddings WHERE classifier = ?",
    args: [classifier],
  });

  if (cached.rows.length > 0) {
    return cached.rows.map((r) => {
      const cluster = String(r.cluster);
      // Embedding stored as F32_BLOB — comes back as ArrayBuffer
      const buf = r.embedding as ArrayBuffer;
      const centroid = new Float32Array(buf);
      return { cluster, centroid, weight: weights[cluster] ?? 0 };
    });
  }

  // Compute centroids: embed all examples per cluster, average them
  const modelCacheDir = path.join(dataDir, "models");
  await initEmbedder(mlConfig, modelCacheDir);

  const centroids: ClusterCentroid[] = [];

  for (const [cluster, examples] of Object.entries(clusters)) {
    const vectors = await embedTexts(examples, mlConfig);

    // Average to get centroid
    const dims = mlConfig.embedding.dimensions;
    const centroid = new Float32Array(dims);
    for (const vec of vectors) {
      for (let i = 0; i < dims; i++) centroid[i] += vec[i];
    }
    for (let i = 0; i < dims; i++) centroid[i] /= vectors.length;

    // Normalize centroid
    let norm = 0;
    for (let i = 0; i < dims; i++) norm += centroid[i] * centroid[i];
    norm = Math.sqrt(norm);
    if (norm > 0) for (let i = 0; i < dims; i++) centroid[i] /= norm;

    centroids.push({ cluster, centroid, weight: weights[cluster] ?? 0 });

    // Cache in DB
    await client.execute({
      sql: "INSERT INTO reference_embeddings (classifier, cluster, prompt, embedding) VALUES (?, ?, ?, vector32(?))",
      args: [classifier, cluster, examples.join("\n---\n"), JSON.stringify(Array.from(centroid))],
    });
  }

  return centroids;
}


function scoreFromCentroids(
  embedding: Float32Array,
  centroids: ClusterCentroid[],
  config: ClassifierConfig,
): ClassifierResult {
  const similarities: Array<{ cluster: string; similarity: number; weight: number }> = [];

  for (const c of centroids) {
    const sim = cosineSimilarity(embedding, c.centroid);
    similarities.push({ cluster: c.cluster, similarity: sim, weight: c.weight });
  }

  // Sort by absolute similarity descending
  similarities.sort((a, b) => b.similarity - a.similarity);

  // Signal strength = how much above baseline a match is.
  // Random text-to-text cosine similarity is typically 0.3-0.5 for normalized embeddings.
  // Only count matches meaningfully above this baseline.
  const baseline = config.similarityThreshold;

  let rawScore = 0;
  let maxPossibleScore = 0;

  for (const s of similarities) {
    const strength = Math.max(0, s.similarity - baseline);
    rawScore += strength * s.weight;
    maxPossibleScore += Math.max(0, 1 - baseline) * Math.abs(s.weight);
  }

  // Normalize to 0-100 range
  const center = config.centerPoint ?? 0;
  let score: number;
  if (maxPossibleScore > 0) {
    // Scale raw score to [-1, 1] range relative to max possible, then map to 0-100
    const normalized = rawScore / maxPossibleScore; // -1 to 1
    score = center + normalized * config.normalizationScale * 10;
  } else {
    score = center;
  }
  score = Math.max(0, Math.min(100, score));

  // Confidence: based on how strongly the prompt matches its best cluster above baseline
  const topStrength = Math.max(0, (similarities[0]?.similarity ?? 0) - baseline);
  const confidence = Math.min(0.95, Math.max(
    config.confidenceFloor ?? 0.5,
    0.5 + topStrength * 1.5,
  ));

  const topSignals = similarities
    .filter((s) => s.similarity > baseline)
    .slice(0, 3)
    .map((s) => ({ signal: s.cluster, similarity: round(s.similarity, 3) }));

  return { score: round(score, 1), confidence: round(confidence, 2), topSignals };
}

export async function initClassifiers(
  client: Client,
  mlConfig: MlConfig,
  dataDir: string,
): Promise<void> {
  const clusters = loadReferenceClusters();

  vibeCentroids = await buildCentroids(
    client, "vibe", clusters.vibe, mlConfig.vibe.weights, mlConfig, dataDir,
  );
  politenessCentroids = await buildCentroids(
    client, "politeness", clusters.politeness, mlConfig.politeness.weights, mlConfig, dataDir,
  );
}

export function computeVibeIndex(
  embedding: Float32Array,
  mlConfig: MlConfig,
): ClassifierResult {
  if (!vibeCentroids) throw new Error("Classifiers not initialized");
  return scoreFromCentroids(embedding, vibeCentroids, mlConfig.vibe);
}

export function computePoliteness(
  embedding: Float32Array,
  mlConfig: MlConfig,
): ClassifierResult {
  if (!politenessCentroids) throw new Error("Classifiers not initialized");
  return scoreFromCentroids(embedding, politenessCentroids, mlConfig.politeness);
}

export async function enrichClassifiers(
  client: Client,
  mlConfig: MlConfig,
  dataDir: string,
  onBatchProgress?: (classified: number, total: number) => void,
): Promise<number> {
  // Find messages with embeddings but without hero classifier scores
  const result = await client.execute(
    `SELECT m.id, m.embedding FROM messages m
     JOIN nlp_enrichments e ON m.id = e.message_id
     WHERE m.role = 'human' AND m.embedding IS NOT NULL
       AND (e.vibe_score IS NULL OR e.politeness_score IS NULL)`,
  );

  if (result.rows.length === 0) return 0;

  // Init classifiers if needed
  await initClassifiers(client, mlConfig, dataDir);

  const batchSize = 200;
  let classified = 0;

  for (let i = 0; i < result.rows.length; i += batchSize) {
    const batch = result.rows.slice(i, i + batchSize);
    const stmts = batch.map((row) => {
      const embedding = new Float32Array(row.embedding as ArrayBuffer);
      const vibe = computeVibeIndex(embedding, mlConfig);
      const pol = computePoliteness(embedding, mlConfig);

      return {
        sql: `UPDATE nlp_enrichments SET
          vibe_score = ?, vibe_confidence = ?,
          politeness_score = ?, politeness_confidence = ?
          WHERE message_id = ?`,
        args: [
          vibe.score, vibe.confidence,
          pol.score, pol.confidence,
          Number(row.id),
        ],
      };
    });

    await client.batch(stmts, "write");
    classified += batch.length;
    onBatchProgress?.(classified, result.rows.length);
  }

  return classified;
}

function round(n: number, decimals: number): number {
  const f = 10 ** decimals;
  return Math.round(n * f) / f;
}
