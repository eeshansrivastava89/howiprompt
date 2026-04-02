import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

export interface EmbeddingConfig {
  model: string;
  dtype: string;
  dimensions: number;
  batchSize: number;
}

export interface ClassifierWeights {
  [cluster: string]: number;
}

export interface ClassifierConfig {
  weights: ClassifierWeights;
  similarityThreshold: number;
  confidenceFloor?: number;
  normalizationScale: number;
  centerPoint?: number;
}

export interface MlConfig {
  embedding: EmbeddingConfig;
  vibe: ClassifierConfig;
  politeness: ClassifierConfig;
}

function deepMerge(target: any, source: any): any {
  const result = { ...target };
  for (const key of Object.keys(source)) {
    if (
      source[key] &&
      typeof source[key] === "object" &&
      !Array.isArray(source[key]) &&
      target[key] &&
      typeof target[key] === "object"
    ) {
      result[key] = deepMerge(target[key], source[key]);
    } else {
      result[key] = source[key];
    }
  }
  return result;
}

export function loadMlConfig(dataDir: string): MlConfig {
  // Load shipped defaults from config/ml.json
  const __dirname = path.dirname(fileURLToPath(import.meta.url));
  const defaultsPath = path.join(__dirname, "..", "..", "config", "ml.json");
  let defaults: any = {};
  try {
    defaults = JSON.parse(fs.readFileSync(defaultsPath, "utf-8"));
  } catch {
    // Fallback if running from dist/
    const altPath = path.join(__dirname, "..", "config", "ml.json");
    try {
      defaults = JSON.parse(fs.readFileSync(altPath, "utf-8"));
    } catch {
      // Use hardcoded minimal defaults
      defaults = {
        embedding: { model: "onnx-community/bge-small-en-v1.5-ONNX", dtype: "int8", dimensions: 384, batchSize: 64 },
        vibe: { weights: {}, similarityThreshold: 0.35, centerPoint: 50, normalizationScale: 5 },
      };
    }
  }

  // Load user overrides
  const overridePath = path.join(dataDir, "ml.json");
  let overrides: any = {};
  try {
    overrides = JSON.parse(fs.readFileSync(overridePath, "utf-8"));
  } catch {
    // No overrides — fine
  }

  return deepMerge(defaults, overrides) as MlConfig;
}
