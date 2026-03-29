// registry.ts — Data-driven metric definitions for hero metrics.

export type ComponentType = "gauge" | "spectrum" | "percentage";

export interface MetricDefinition {
  key: string;
  name: string;
  description: string;
  range: readonly [number, number];
  component: ComponentType;
  tier: "hero";
  order: number;
  format: (v: number) => string;
  labels?: { low: string; high: string };
  source?: (metrics: Record<string, any>) => number;
}

export const METRICS: MetricDefinition[] = [
  // === Hero Metrics (standalone displays) ===
  {
    key: "vibe_index",
    name: "Vibe Coder Index",
    description: "How much you vibe-code vs. engineer",
    range: [0, 100] as const,
    component: "spectrum",
    tier: "hero",
    order: 1,
    format: (v) => `${Math.round(100 - v)}`,
    labels: { low: "Engineer", high: "Vibe Coder" },
    source: (m) => 100 - (m.nlp?.vibe_coder_index?.avg_score ?? 0),
  },
  {
    key: "politeness",
    name: "Politeness",
    description: "How courteous and collaborative is your tone?",
    range: [0, 100] as const,
    component: "gauge",
    tier: "hero",
    order: 2,
    format: (v) => `${Math.round(v)}`,
    labels: { low: "Direct", high: "Courteous" },
    source: (m) => m.nlp?.politeness?.avg_score ?? 0,
  },

];

export function getHeroes(): MetricDefinition[] {
  return METRICS.filter((m) => m.tier === "hero").sort((a, b) => a.order - b.order);
}

export function getMetric(key: string): MetricDefinition | undefined {
  return METRICS.find((m) => m.key === key);
}

/** Compute values for all registry metrics from a full metrics object */
export function computeNormalized(metrics: Record<string, any>): Record<string, number> {
  const result: Record<string, number> = {};
  for (const def of METRICS) {
    if (def.source) {
      result[def.key] = Math.round(def.source(metrics) * 10) / 10;
    }
  }
  return result;
}
