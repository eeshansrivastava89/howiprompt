/**
 * Export synthetic test datasets to JSON for the validation notebook.
 * Run: node scripts/export-synthetic.mjs
 */
import { ALL_DATASETS } from "../tests/fixtures/synthetic-datasets.js";

const exported = ALL_DATASETS.map(ds => ({
  name: ds.name,
  description: ds.description,
  expectedPersona: ds.expectedPersona,
  expectedRadar: ds.expectedRadar,
  expectedHitlDirection: ds.expectedHitlDirection,
  expectedVibeDirection: ds.expectedVibeDirection,
  expectedPoliteDirection: ds.expectedPoliteDirection,
  humanPrompts: ds.messages
    .filter(m => m.role === "human")
    .map(m => m.content),
}));

const outPath = new URL("../notebooks/synthetic-datasets.json", import.meta.url);
import { writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
writeFileSync(fileURLToPath(outPath), JSON.stringify(exported, null, 2));
console.log(`Exported ${exported.length} datasets, ${exported.reduce((s, d) => s + d.humanPrompts.length, 0)} human prompts`);
