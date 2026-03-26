export const CLIENT_ID_STORAGE_KEY = "howiprompt_client_id";
export const USERNAME_STORAGE_KEY = "howiprompt_username";

export function createStableClientId(storage, key = CLIENT_ID_STORAGE_KEY) {
  const existing = storage?.getItem?.(key);
  if (existing) return existing;

  let nextId = null;
  if (globalThis.crypto?.randomUUID) {
    nextId = `hip_${globalThis.crypto.randomUUID()}`;
  } else if (globalThis.crypto?.getRandomValues) {
    const bytes = new Uint8Array(12);
    globalThis.crypto.getRandomValues(bytes);
    nextId = `hip_${Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("")}`;
  } else {
    nextId = `hip_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 10)}`;
  }

  storage?.setItem?.(key, nextId);
  return nextId;
}

export function getSubmissionPayload(view, sourceKey) {
  if (!view) return null;
  const volume = view.volume || {};
  const nlp = view.nlp || {};
  const persona = view.persona || {};

  return {
    hitl_score: Math.round(nlp.hitl_score?.avg_score ?? 0),
    vibe_index: Math.round(nlp.vibe_coder_index?.avg_score ?? 0),
    politeness: Math.round(nlp.politeness?.avg_score ?? 0),
    total_prompts: volume.total_human || 0,
    total_conversations: volume.total_conversations || 0,
    persona: persona.type || "unknown",
    platform: sourceKey,
  };
}

export function sortLeaderboardEntries(entries, sortKey) {
  return [...entries].sort((a, b) => (b?.[sortKey] ?? 0) - (a?.[sortKey] ?? 0));
}

export function findEntryRank(entries, sortKey, clientId) {
  if (!clientId) return -1;
  const sorted = sortLeaderboardEntries(entries, sortKey);
  return sorted.findIndex((entry) => entry.fingerprint === clientId) + 1;
}
