// Cloudflare Worker — How I Prompt Leaderboard API
// Endpoints:
//   GET  /api/leaderboard           — ranked list of submissions
//   GET  /api/leaderboard/:name     — single user's scores
//   POST /api/submit                — submit or update scores

interface Env {
  DB: D1Database;
  CORS_ORIGIN: string;
}

interface SubmissionPayload {
  display_name: string;
  total_conversations: number;
  total_prompts: number;
  avg_words_per_prompt: number;
  politeness: number;
  backtrack: number;
  question_rate: number;
  command_rate: number;
  hitl_score: number;
  vibe_index: number;
  persona: string;
  complexity_avg: number;
  platform: string;
  tool_version: string;
}

function corsHeaders(origin: string): Record<string, string> {
  return {
    "Access-Control-Allow-Origin": origin || "*",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Max-Age": "86400",
  };
}

function jsonResponse(data: unknown, status = 200, origin = "*"): Response {
  return new Response(JSON.stringify(data), {
    status,
    headers: { "Content-Type": "application/json", ...corsHeaders(origin) },
  });
}

function validate(payload: SubmissionPayload): string | null {
  if (!payload.display_name || typeof payload.display_name !== "string") return "display_name required";
  if (payload.display_name.length > 30) return "display_name too long (max 30)";
  if (payload.display_name.length < 2) return "display_name too short (min 2)";

  // Sanitize: alphanumeric, spaces, hyphens, underscores only
  if (!/^[a-zA-Z0-9 _-]+$/.test(payload.display_name)) return "display_name contains invalid characters";

  // Range checks
  const rangeFields = ["politeness", "backtrack", "question_rate", "command_rate", "hitl_score", "vibe_index", "complexity_avg"] as const;
  for (const f of rangeFields) {
    const v = payload[f];
    if (typeof v !== "number" || v < 0 || v > 100) return `${f} must be 0-100`;
  }

  if (typeof payload.total_prompts !== "number" || payload.total_prompts < 1) return "total_prompts must be >= 1";
  if (typeof payload.total_conversations !== "number" || payload.total_conversations < 1) return "total_conversations must be >= 1";

  // Anomaly detection: suspicious scores with very low volume
  if (payload.hitl_score > 90 && payload.total_conversations < 10) return "implausible: high HITL with very few conversations";
  if (payload.total_prompts > 1_000_000) return "implausible: too many prompts";

  return null;
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);
    const origin = env.CORS_ORIGIN || "*";

    // Handle CORS preflight
    if (request.method === "OPTIONS") {
      return new Response(null, { status: 204, headers: corsHeaders(origin) });
    }

    try {
      // GET /api/leaderboard
      if (url.pathname === "/api/leaderboard" && request.method === "GET") {
        const { results } = await env.DB.prepare(
          `SELECT display_name, total_conversations, total_prompts, avg_words_per_prompt,
                  politeness, backtrack, question_rate, command_rate,
                  hitl_score, vibe_index, persona, complexity_avg, platform,
                  submitted_at
           FROM submissions ORDER BY hitl_score DESC LIMIT 100`
        ).all();
        return jsonResponse(results, 200, origin);
      }

      // GET /api/leaderboard/:name
      if (url.pathname.startsWith("/api/leaderboard/") && request.method === "GET") {
        const name = decodeURIComponent(url.pathname.split("/").pop() || "");
        const result = await env.DB.prepare(
          "SELECT * FROM submissions WHERE display_name = ?"
        ).bind(name).first();
        if (!result) return jsonResponse({ error: "not found" }, 404, origin);
        return jsonResponse(result, 200, origin);
      }

      // POST /api/submit
      if (url.pathname === "/api/submit" && request.method === "POST") {
        const payload = await request.json() as SubmissionPayload;
        const err = validate(payload);
        if (err) return jsonResponse({ error: err }, 400, origin);

        // Upsert: insert or update on display_name conflict
        await env.DB.prepare(
          `INSERT INTO submissions (display_name, total_conversations, total_prompts, avg_words_per_prompt,
              politeness, backtrack, question_rate, command_rate,
              hitl_score, vibe_index, persona, complexity_avg, platform, tool_version)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(display_name) DO UPDATE SET
              total_conversations = excluded.total_conversations,
              total_prompts = excluded.total_prompts,
              avg_words_per_prompt = excluded.avg_words_per_prompt,
              politeness = excluded.politeness,
              backtrack = excluded.backtrack,
              question_rate = excluded.question_rate,
              command_rate = excluded.command_rate,
              hitl_score = excluded.hitl_score,
              vibe_index = excluded.vibe_index,
              persona = excluded.persona,
              complexity_avg = excluded.complexity_avg,
              platform = excluded.platform,
              tool_version = excluded.tool_version,
              updated_at = datetime('now')`
        ).bind(
          payload.display_name, payload.total_conversations, payload.total_prompts,
          payload.avg_words_per_prompt, payload.politeness, payload.backtrack,
          payload.question_rate, payload.command_rate, payload.hitl_score,
          payload.vibe_index, payload.persona, payload.complexity_avg,
          payload.platform, payload.tool_version
        ).run();

        return jsonResponse({ ok: true }, 200, origin);
      }

      return jsonResponse({ error: "not found" }, 404, origin);
    } catch (e: unknown) {
      const message = e instanceof Error ? e.message : "internal error";
      return jsonResponse({ error: message }, 500, origin);
    }
  },
};
