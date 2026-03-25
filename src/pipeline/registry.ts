// registry.ts — Data-driven metric definitions.
// Defines hero metrics and radar axes for the player card.

export type ComponentType = "gauge" | "spectrum" | "percentage" | "radar";

export interface MetricDefinition {
  key: string;
  name: string;
  description: string;
  range: readonly [number, number];
  component: ComponentType;
  tier: "hero" | "radar";
  order: number;
  format: (v: number) => string;
  labels?: { low: string; high: string };
  source?: (metrics: Record<string, any>) => number;
}

export const METRICS: MetricDefinition[] = [
  // === Hero Metrics (standalone displays) ===
  {
    key: "hitl_score",
    name: "HITL Score",
    description: "How much you steer the AI",
    range: [0, 100] as const,
    component: "gauge",
    tier: "hero",
    order: 1,
    format: (v) => `${Math.round(v)}`,
    labels: { low: "Passive", high: "High Impact" },
    source: (m) => m.nlp?.hitl_score?.avg_score ?? 0,
  },
  {
    key: "vibe_index",
    name: "Vibe Coder Index",
    description: "Engineer vs vibe coder spectrum",
    range: [0, 100] as const,
    component: "spectrum",
    tier: "hero",
    order: 2,
    format: (v) => `${Math.round(v)}`,
    labels: { low: "Vibe Coder", high: "Engineer" },
    source: (m) => m.nlp?.vibe_coder_index?.avg_score ?? 0,
  },
  {
    key: "politeness",
    name: "Politeness",
    description: "How courteous and collaborative is your tone?",
    range: [0, 100] as const,
    component: "gauge",
    tier: "hero",
    order: 3,
    format: (v) => `${Math.round(v)}`,
    labels: { low: "Direct", high: "Courteous" },
    source: (m) => m.nlp?.politeness?.avg_score ?? 0,
  },

  // === Radar Axes (4-axis spider chart) ===
  {
    key: "precision",
    name: "Precision",
    description: "How detailed are your instructions?",
    range: [0, 100] as const,
    component: "radar",
    tier: "radar",
    order: 1,
    format: (v) => `${Math.round(v)}`,
    source: (m) => m.nlp?.precision?.avg_score ?? 0,
  },
  {
    key: "curiosity",
    name: "Curiosity",
    description: "Do you explore or execute?",
    range: [0, 100] as const,
    component: "radar",
    tier: "radar",
    order: 2,
    format: (v) => `${Math.round(v)}`,
    source: (m) => m.nlp?.curiosity?.avg_score ?? 0,
  },
  {
    key: "tenacity",
    name: "Tenacity",
    description: "How deep do you go on one problem?",
    range: [0, 100] as const,
    component: "radar",
    tier: "radar",
    order: 3,
    format: (v) => `${Math.round(v)}`,
    source: (m) => m.nlp?.tenacity?.avg_score ?? 0,
  },
  {
    key: "trust",
    name: "Trust",
    description: "How much do you delegate to the AI?",
    range: [0, 100] as const,
    component: "radar",
    tier: "radar",
    order: 4,
    format: (v) => `${Math.round(v)}`,
    source: (m) => m.nlp?.trust?.avg_score ?? 0,
  },
];

export function getHeroes(): MetricDefinition[] {
  return METRICS.filter((m) => m.tier === "hero").sort((a, b) => a.order - b.order);
}

export function getRadarAxes(): MetricDefinition[] {
  return METRICS.filter((m) => m.tier === "radar").sort((a, b) => a.order - b.order);
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
