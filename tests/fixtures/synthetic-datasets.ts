/**
 * Synthetic conversation datasets for end-to-end testing.
 * Each dataset is designed to produce a specific persona type
 * based on the prompting style of the messages.
 *
 * Ground truth: human-evaluated expected persona, score directions,
 * and metric ranges.
 */
import { Platform, Role, type Message } from "../../src/pipeline/models.js";

export interface TestDataset {
  name: string;
  description: string;
  expectedPersona: string;
  /** Expected direction for each radar axis: "high" (>60), "mid" (40-60), "low" (<40) */
  expectedRadar: {
    precision: "high" | "mid" | "low";
    curiosity: "high" | "mid" | "low";
    tenacity: "high" | "mid" | "low";
    trust: "high" | "mid" | "low";
  };
  expectedHitlDirection: "high" | "mid" | "low";
  expectedVibeDirection: "engineer" | "balanced" | "vibe";
  expectedPoliteDirection: "high" | "mid" | "low";
  messages: Message[];
}

let id = 0;
function msg(content: string, opts: { day?: number; convoId?: string; role?: Role; platform?: Platform } = {}): Message {
  const day = opts.day ?? 1;
  return {
    timestamp: new Date(2026, 1, day, 10 + (id++ % 12), 0, 0),
    platform: opts.platform ?? Platform.CLAUDE_CODE,
    role: opts.role ?? Role.HUMAN,
    content,
    conversationId: opts.convoId ?? `synth-${Math.floor(id / 4)}`,
    wordCount: content.split(/\s+/).length,
  };
}

// ---------------------------------------------------------------------------
// 1-3: THE ARCHITECT (high precision, low trust)
// ---------------------------------------------------------------------------

export const architect1: TestDataset = {
  name: "architect-spec-heavy",
  description: "Detailed specs, file references, tight control, reviews output",
  expectedPersona: "architect",
  expectedRadar: { precision: "high", curiosity: "low", tenacity: "mid", trust: "low" },
  expectedHitlDirection: "high",
  expectedVibeDirection: "engineer",
  expectedPoliteDirection: "mid",
  messages: [
    msg("in src/auth/middleware.ts, change the validateToken function to accept a JWT string and return a typed Result<TokenPayload, AuthError>. Use the existing jose library, not jsonwebtoken."),
    msg("Here's the implementation...", { role: Role.ASSISTANT }),
    msg("the return type is wrong — it should be Result<TokenPayload, AuthError>, not Promise<TokenPayload>. Also the error case needs to distinguish between expired and invalid tokens."),
    msg("Updated version...", { role: Role.ASSISTANT }),
    msg("use exactly this implementation for the token refresh, don't change anything: `const refreshed = await rotateToken(current, { maxAge: '7d' })`"),
    msg("create a migration that adds a status enum column to the orders table with values 'pending', 'processing', 'shipped', 'delivered', 'cancelled'. Default to 'pending'. Add a composite index on (user_id, status)."),
    msg("Migration created...", { role: Role.ASSISTANT }),
    msg("the index should be on (user_id, status, created_at) not just (user_id, status). And make the migration reversible."),
    msg("follow my spec precisely — no creative interpretation. The rate limiter uses sliding window at 100 req/min per API key, stored in Redis with MULTI/EXEC."),
    msg("do NOT refactor anything outside the function I specified. Only change validateToken."),
    msg("I see a potential race condition on line 42 — what happens if two requests hit the refresh endpoint simultaneously?"),
    msg("Good analysis...", { role: Role.ASSISTANT }),
    msg("that doesn't look right. The error message is misleading — fix it to say 'token expired' not 'invalid token' when the exp claim is past."),
  ],
};

export const architect2: TestDataset = {
  name: "architect-code-review",
  description: "Heavy code review, catches bugs, enforces constraints",
  expectedPersona: "architect",
  expectedRadar: { precision: "high", curiosity: "low", tenacity: "mid", trust: "low" },
  expectedHitlDirection: "high",
  expectedVibeDirection: "engineer",
  expectedPoliteDirection: "low",
  messages: [
    msg("the type should be `Record<string, unknown>` not `any`. Fix all instances in src/api/"),
    msg("Fixed...", { role: Role.ASSISTANT }),
    msg("double check the SQL injection risk in the search endpoint. The user input goes directly into the LIKE clause without parameterization."),
    msg("this logic seems inverted — should it be >= not >? Check the edge case where count equals the limit."),
    msg("verify that the tests actually cover the null input scenario. I don't see a test for it."),
    msg("the variable name is confusing. Rename 'data' to 'validatedPayload' throughout the handler."),
    msg("Renamed...", { role: Role.ASSISTANT }),
    msg("run the benchmarks before we merge this. I want to see p99 latency for the new path vs old."),
    msg("it must be backwards compatible with the v1 API. Don't break existing clients."),
    msg("the response time should be under 200ms for the 95th percentile. Add a timeout if the downstream service doesn't respond in 150ms."),
    msg("looks wrong — off-by-one error on the pagination. When page=1 and limit=10, offset should be 0, not 10."),
    msg("use only the libraries I listed. No substitutions, no additional dependencies."),
  ],
};

export const architect3: TestDataset = {
  name: "architect-constraint-driven",
  description: "Constraint specifications, no exploration, tight scope",
  expectedPersona: "architect",
  expectedRadar: { precision: "high", curiosity: "low", tenacity: "mid", trust: "low" },
  expectedHitlDirection: "high",
  expectedVibeDirection: "engineer",
  expectedPoliteDirection: "low",
  messages: [
    msg("the solution must not require any new dependencies. Use only what's in package.json."),
    msg("keep the bundle size under 100KB. Remove the lodash import and use native array methods."),
    msg("implement it exactly as described in the ticket. No extras."),
    msg("match this design pixel for pixel. The spacing is 16px not 12px."),
    msg("Adjusted...", { role: Role.ASSISTANT }),
    msg("the function must be idempotent. Calling it twice with the same input must produce the same result."),
    msg("this must work on Node 18+ without transpilation. No top-level await."),
    msg("never expose internal error details to the client. Sanitize all error responses."),
    msg("stick to my approach even if you think there's a better way. I have context you don't."),
    msg("the migration must be reversible and must not lock the table."),
  ],
};

// ---------------------------------------------------------------------------
// 4-6: THE EXPLORER (high curiosity)
// ---------------------------------------------------------------------------

export const explorer1: TestDataset = {
  name: "explorer-learning",
  description: "Questions, exploration, learning about the codebase",
  expectedPersona: "explorer",
  expectedRadar: { precision: "low", curiosity: "high", tenacity: "mid", trust: "mid" },
  expectedHitlDirection: "mid",
  expectedVibeDirection: "balanced",
  expectedPoliteDirection: "mid",
  messages: [
    msg("how does the garbage collector handle circular references in V8? I'm trying to understand why we have a memory leak."),
    msg("Detailed explanation...", { role: Role.ASSISTANT }),
    msg("can you explain why we'd choose a B-tree over a hash index for this query pattern? The table has 10M rows and we query by date range + user_id."),
    msg("what's the difference between optimistic and pessimistic locking, and which is better for our checkout flow?"),
    msg("I don't understand the CAP theorem implications for our architecture. We're using Postgres with read replicas — where do we sit?"),
    msg("what would happen if we removed the mutex here? Walk me through the race condition step by step."),
    msg("before we build this, what are the unknowns we should investigate?"),
    msg("what if we approached this differently — instead of polling, what about server-sent events?"),
    msg("play devil's advocate — why might this approach fail in production?"),
    msg("compare the approaches: monorepo vs polyrepo for a team of 8 engineers. What are the tradeoffs?"),
    msg("sketch out three different architectures for the notification system and let me pick."),
    msg("teach me how connection pooling works — why can't we just open a new connection per request?"),
  ],
};

export const explorer2: TestDataset = {
  name: "explorer-brainstorming",
  description: "Brainstorming, what-ifs, design space exploration",
  expectedPersona: "explorer",
  expectedRadar: { precision: "low", curiosity: "high", tenacity: "low", trust: "mid" },
  expectedHitlDirection: "mid",
  expectedVibeDirection: "balanced",
  expectedPoliteDirection: "high",
  messages: [
    msg("I'd love to understand the design space here. What are our options for real-time sync?"),
    msg("Options laid out...", { role: Role.ASSISTANT }),
    msg("thanks for that breakdown! What are the security implications of storing tokens in localStorage vs httpOnly cookies?"),
    msg("that's really helpful. Let me think out loud — we could either batch the requests or use a websocket. Which scales better for our use case?"),
    msg("I appreciate the thorough analysis. What would a senior engineer critique about this design?"),
    msg("great question to ask. How would you design this if we had to support 100x the current traffic?"),
    msg("interesting. What's the worst case scenario if the cache layer fails in production?"),
    msg("thanks for walking me through this. I want to understand the mental model for thinking about database isolation levels."),
    msg("could you explain why eventual consistency is acceptable for the feed but not for the payment flow?"),
    msg("really appreciate the patience here. One more — why is this considered an anti-pattern? What problems does it cause in practice?"),
  ],
};

export const explorer3: TestDataset = {
  name: "explorer-investigation",
  description: "Debugging through questions, investigating a problem",
  expectedPersona: "explorer",
  expectedRadar: { precision: "mid", curiosity: "high", tenacity: "mid", trust: "mid" },
  expectedHitlDirection: "mid",
  expectedVibeDirection: "balanced",
  expectedPoliteDirection: "low",
  messages: [
    msg("walk me through how the event loop processes this async code step by step"),
    msg("why does React re-render when I use useCallback here — shouldn't it memoize?"),
    msg("help me understand why this regex is catastrophically backtracking"),
    msg("what happens under the hood when we call .pipe() on a Node stream?"),
    msg("Explanation...", { role: Role.ASSISTANT }),
    msg("I've never used GraphQL subscriptions — how do they work compared to WebSockets?"),
    msg("can you explain the tradeoffs between server-side rendering and client-side rendering for this specific page?"),
    msg("what's the performance difference between Map and plain objects for 100K entries?"),
    msg("how does the V8 hidden class optimization work and are we breaking it with these dynamic property additions?"),
    msg("why is this Promise.all faster than sequential awaits when the operations are independent?"),
  ],
};

// ---------------------------------------------------------------------------
// 7-9: THE COMMANDER (high precision, low tenacity)
// ---------------------------------------------------------------------------

export const commander1: TestDataset = {
  name: "commander-rapid-fire",
  description: "Short, precise orders, no iteration, expects first-try delivery",
  expectedPersona: "commander",
  expectedRadar: { precision: "high", curiosity: "low", tenacity: "low", trust: "low" },
  expectedHitlDirection: "high",
  expectedVibeDirection: "engineer",
  expectedPoliteDirection: "low",
  messages: [
    msg("add an index on user_id in the orders table"),
    msg("Done...", { role: Role.ASSISTANT }),
    msg("change the return type to Promise<void>"),
    msg("Updated...", { role: Role.ASSISTANT }),
    msg("remove the deprecated validateLegacy function from auth.ts"),
    msg("Removed...", { role: Role.ASSISTANT }),
    msg("set the timeout to 5000ms in the http client config"),
    msg("Done...", { role: Role.ASSISTANT }),
    msg("split processOrder into two functions: validateOrder and executeOrder"),
    msg("fix the import path in src/index.ts line 15"),
    msg("Fixed...", { role: Role.ASSISTANT }),
    msg("add the missing NOT NULL constraint to the email column"),
    msg("deploy to staging"),
    msg("Deployed...", { role: Role.ASSISTANT }),
    msg("update the nginx config to proxy /api to port 8080"),
  ],
};

export const commander2: TestDataset = {
  name: "commander-technical-orders",
  description: "Technical specifications but minimal back-and-forth",
  expectedPersona: "commander",
  expectedRadar: { precision: "high", curiosity: "low", tenacity: "low", trust: "low" },
  expectedHitlDirection: "high",
  expectedVibeDirection: "engineer",
  expectedPoliteDirection: "low",
  messages: [
    msg("implement rate limiting with a token bucket at 100 req/min. Use Redis MULTI/EXEC."),
    msg("Done...", { role: Role.ASSISTANT }),
    msg("the WebSocket should reconnect with exponential backoff starting at 100ms, max 30s, with jitter."),
    msg("Implemented...", { role: Role.ASSISTANT }),
    msg("add a composite index on (user_id, created_at DESC) for the queries table"),
    msg("use connection pooling with a max of 20 connections and 10s idle timeout"),
    msg("Configured...", { role: Role.ASSISTANT }),
    msg("implement circuit breaker with half-open state after 30s"),
    msg("use HMAC-SHA256 for the webhook signature verification, key from env var WEBHOOK_SECRET"),
    msg("the API should return 409 Conflict on concurrent writes to the same resource"),
  ],
};

export const commander3: TestDataset = {
  name: "commander-file-targeted",
  description: "Precise file-targeted commands with no exploration",
  expectedPersona: "commander",
  expectedRadar: { precision: "high", curiosity: "low", tenacity: "low", trust: "low" },
  expectedHitlDirection: "high",
  expectedVibeDirection: "engineer",
  expectedPoliteDirection: "low",
  messages: [
    msg("in src/api/routes/users.ts, add input validation using Zod for the POST body"),
    msg("Added...", { role: Role.ASSISTANT }),
    msg("the bug is in frontend/src/components/Header.tsx line 23 — the onClick handler doesn't prevent default"),
    msg("update the schema in prisma/schema.prisma to add a role field on User"),
    msg("fix the error handler in src/api/middleware/error.ts to return structured JSON errors"),
    msg("the CSS in styles/dashboard.css needs display:grid with 3 columns on desktop, 1 on mobile"),
    msg("modify the Dockerfile to use multi-stage builds — builder stage for npm ci, runner stage for node"),
    msg("the test in tests/auth.test.ts is flaky — the token expiry is set to 1s, make it 60s"),
    msg("add the new endpoint GET /api/health to src/routes/api.ts, return { status: 'ok' }"),
  ],
};

// ---------------------------------------------------------------------------
// 10-13: THE PARTNER (high tenacity)
// ---------------------------------------------------------------------------

export const partner1: TestDataset = {
  name: "partner-iterative-build",
  description: "Long iterative session building a feature step by step",
  expectedPersona: "partner",
  expectedRadar: { precision: "mid", curiosity: "mid", tenacity: "high", trust: "mid" },
  expectedHitlDirection: "mid",
  expectedVibeDirection: "balanced",
  expectedPoliteDirection: "mid",
  messages: [
    msg("let's build a debounced search input component. Start with the basic input and onChange."),
    msg("Basic component...", { role: Role.ASSISTANT }),
    msg("good, now add the debounce logic — 300ms after the user stops typing"),
    msg("Debounce added...", { role: Role.ASSISTANT }),
    msg("that works — next, wire it up to the API endpoint. Use fetch with AbortController to cancel in-flight requests."),
    msg("Connected...", { role: Role.ASSISTANT }),
    msg("nice. Now add a loading spinner that shows while the request is in flight."),
    msg("Spinner added...", { role: Role.ASSISTANT }),
    msg("the happy path is done. Now add error handling — show an error message if the fetch fails, with a retry button."),
    msg("Error handling...", { role: Role.ASSISTANT }),
    msg("ok now let's handle the empty state — show 'No results found' when the array is empty."),
    msg("Empty state...", { role: Role.ASSISTANT }),
    msg("perfect. Now write the tests for all the cases we just implemented."),
    msg("Tests written...", { role: Role.ASSISTANT }),
    msg("the tests pass. Let's iterate — the spacing is off. Try 16px gap instead of 12px."),
    msg("Adjusted...", { role: Role.ASSISTANT }),
    msg("almost — the animation easing should be ease-out not ease-in. Also make the transition 150ms."),
  ],
};

export const partner2: TestDataset = {
  name: "partner-refinement",
  description: "Deep refinement, pixel-tweaking, polish session",
  expectedPersona: "partner",
  expectedRadar: { precision: "mid", curiosity: "low", tenacity: "high", trust: "mid" },
  expectedHitlDirection: "mid",
  expectedVibeDirection: "engineer",
  expectedPoliteDirection: "mid",
  messages: [
    msg("the color is slightly wrong. Use oklch(0.7 0.15 250) instead."),
    msg("Updated...", { role: Role.ASSISTANT }),
    msg("that's better but the font weight should be 500, not 600"),
    msg("tweak the shadow — it's too heavy. Use 0 1px 3px rgba(0,0,0,0.08)."),
    msg("Lighter...", { role: Role.ASSISTANT }),
    msg("close, but the border radius should match the other cards at 12px"),
    msg("nudge the icon 2px up — it's not visually centered with the text"),
    msg("the hover state needs more contrast. Darken the background by 5%."),
    msg("Darkened...", { role: Role.ASSISTANT }),
    msg("hmm, the transition duration feels sluggish. Try 150ms instead of 300ms."),
    msg("make the text truncate with ellipsis instead of wrapping to a second line"),
    msg("Truncated...", { role: Role.ASSISTANT }),
    msg("almost right. The ellipsis should only kick in after 2 lines, not 1. Use -webkit-line-clamp: 2."),
    msg("good. Now refactor the duplicate CSS between the card and the modal into shared variables."),
  ],
};

export const partner3: TestDataset = {
  name: "partner-extend-feature",
  description: "Building on top of existing work, extending iteratively",
  expectedPersona: "partner",
  expectedRadar: { precision: "mid", curiosity: "mid", tenacity: "high", trust: "mid" },
  expectedHitlDirection: "mid",
  expectedVibeDirection: "balanced",
  expectedPoliteDirection: "high",
  messages: [
    msg("great, that's step 1 done. For step 2, let's add the WebSocket listener for real-time updates."),
    msg("WebSocket listener...", { role: Role.ASSISTANT }),
    msg("nice work. Now let's handle reconnection — if the socket drops, reconnect with backoff."),
    msg("Reconnection...", { role: Role.ASSISTANT }),
    msg("that looks solid. Let's add optimistic updates — update the UI immediately, then reconcile when the server confirms."),
    msg("thanks, that's clean. Now make the same change to the other three endpoints: update, delete, and archive."),
    msg("All four done...", { role: Role.ASSISTANT }),
    msg("the basic version works great. Let's iterate — add sorting by any column with a click on the header."),
    msg("Sorting...", { role: Role.ASSISTANT }),
    msg("love it. Now add filtering — a text input at the top that filters rows client-side."),
    msg("Filtering...", { role: Role.ASSISTANT }),
    msg("alright, the logic is solid. Now add proper TypeScript types to everything we've built."),
    msg("Types added...", { role: Role.ASSISTANT }),
    msg("one last thing — connect this to the real API instead of the mock data."),
  ],
};

export const partner4: TestDataset = {
  name: "partner-collaborative-polite",
  description: "Warm collaborative session with lots of encouragement",
  expectedPersona: "partner",
  expectedRadar: { precision: "mid", curiosity: "mid", tenacity: "high", trust: "mid" },
  expectedHitlDirection: "mid",
  expectedVibeDirection: "balanced",
  expectedPoliteDirection: "high",
  messages: [
    msg("thanks for getting that started! Let's keep going — can you add validation for the email field?"),
    msg("Validation added...", { role: Role.ASSISTANT }),
    msg("great work on that. Now let's wire it up to the API endpoint."),
    msg("Connected...", { role: Role.ASSISTANT }),
    msg("brilliant, that's looking really good. One more thing — add a loading state while the request is in flight."),
    msg("Loading state...", { role: Role.ASSISTANT }),
    msg("love that approach. Now let's handle the error case too."),
    msg("Error handling...", { role: Role.ASSISTANT }),
    msg("this is way better than what I had in mind, thanks! Can you also add the keyboard shortcuts next?"),
    msg("Shortcuts...", { role: Role.ASSISTANT }),
    msg("really solid work. Last thing — write tests for all the cases we just implemented."),
    msg("Tests...", { role: Role.ASSISTANT }),
    msg("thanks for being so thorough. One more tweak — the submit button should be disabled while loading."),
  ],
};

// ---------------------------------------------------------------------------
// 14-17: THE DELEGATOR (high trust)
// ---------------------------------------------------------------------------

export const delegator1: TestDataset = {
  name: "delegator-outcome-focused",
  description: "High-level goals, delegates implementation details entirely",
  expectedPersona: "delegator",
  expectedRadar: { precision: "low", curiosity: "low", tenacity: "low", trust: "high" },
  expectedHitlDirection: "low",
  expectedVibeDirection: "vibe",
  expectedPoliteDirection: "mid",
  messages: [
    msg("build me a login page with email and password"),
    msg("Login page...", { role: Role.ASSISTANT }),
    msg("looks good, ship it"),
    msg("create a REST API for managing users — CRUD operations"),
    msg("API created...", { role: Role.ASSISTANT }),
    msg("perfect, exactly what I wanted"),
    msg("make a dashboard that shows analytics — daily active users, signups, revenue"),
    msg("Dashboard...", { role: Role.ASSISTANT }),
    msg("add authentication to the app"),
    msg("Auth added...", { role: Role.ASSISTANT }),
    msg("you decide on the implementation details"),
    msg("add dark mode support"),
    msg("Dark mode...", { role: Role.ASSISTANT }),
    msg("LGTM, approved"),
    msg("set up a CI/CD pipeline"),
    msg("Pipeline...", { role: Role.ASSISTANT }),
    msg("go for it"),
  ],
};

export const delegator2: TestDataset = {
  name: "delegator-trust-heavy",
  description: "Maximum delegation, minimal review, trusts AI judgment",
  expectedPersona: "delegator",
  expectedRadar: { precision: "low", curiosity: "low", tenacity: "low", trust: "high" },
  expectedHitlDirection: "low",
  expectedVibeDirection: "vibe",
  expectedPoliteDirection: "mid",
  messages: [
    msg("redesign the entire authentication system however you think is best"),
    msg("Redesigned...", { role: Role.ASSISTANT }),
    msg("I trust your judgment on the implementation details, just make it work"),
    msg("pick whatever approach you think is cleanest"),
    msg("Approach chosen...", { role: Role.ASSISTANT }),
    msg("use your best judgment for the error handling strategy"),
    msg("organize the code however is most maintainable"),
    msg("Reorganized...", { role: Role.ASSISTANT }),
    msg("do what you think is right, I'll review the result"),
    msg("go ahead and refactor this whole module if you think it needs it"),
    msg("pick the testing strategy — unit, integration, whatever covers this best"),
    msg("Strategy...", { role: Role.ASSISTANT }),
    msg("make the architectural decisions, I'll weigh in if I disagree"),
    msg("style it however looks good to you"),
  ],
};

export const delegator3: TestDataset = {
  name: "delegator-acceptance-heavy",
  description: "Quick approvals, no pushback, fast-moving",
  expectedPersona: "delegator",
  expectedRadar: { precision: "low", curiosity: "low", tenacity: "low", trust: "high" },
  expectedHitlDirection: "low",
  expectedVibeDirection: "vibe",
  expectedPoliteDirection: "mid",
  messages: [
    msg("I want users to be able to reset their passwords"),
    msg("Password reset...", { role: Role.ASSISTANT }),
    msg("that works, merge it"),
    msg("the table should be sortable by any column"),
    msg("Sortable...", { role: Role.ASSISTANT }),
    msg("yes that's fine, move on"),
    msg("add pagination to the list view"),
    msg("Pagination...", { role: Role.ASSISTANT }),
    msg("good enough, let's not overthink it"),
    msg("I want a progress bar for file uploads"),
    msg("Progress bar...", { role: Role.ASSISTANT }),
    msg("nice, that's clean"),
    msg("add a confirmation dialog before deleting"),
    msg("Dialog...", { role: Role.ASSISTANT }),
    msg("solid, merge and deploy"),
    msg("make it mobile responsive"),
    msg("Responsive...", { role: Role.ASSISTANT }),
    msg("works for me, next task"),
  ],
};

export const delegator4: TestDataset = {
  name: "delegator-open-ended",
  description: "Open-ended tasks with full autonomy given",
  expectedPersona: "delegator",
  expectedRadar: { precision: "low", curiosity: "low", tenacity: "low", trust: "high" },
  expectedHitlDirection: "low",
  expectedVibeDirection: "vibe",
  expectedPoliteDirection: "mid",
  messages: [
    msg("take a look at the codebase and suggest improvements"),
    msg("Suggestions...", { role: Role.ASSISTANT }),
    msg("sounds good, go ahead with all of those"),
    msg("review this PR and tell me what you think"),
    msg("Review...", { role: Role.ASSISTANT }),
    msg("approved"),
    msg("audit the security of the authentication flow"),
    msg("Audit...", { role: Role.ASSISTANT }),
    msg("fix whatever you found"),
    msg("suggest a better folder structure for this project"),
    msg("Proposal...", { role: Role.ASSISTANT }),
    msg("do it"),
    msg("look at the performance profile and recommend optimizations"),
    msg("Recommendations...", { role: Role.ASSISTANT }),
    msg("implement them all"),
  ],
};

// ---------------------------------------------------------------------------
// ALL DATASETS
// ---------------------------------------------------------------------------

export const ALL_DATASETS: TestDataset[] = [
  architect1, architect2, architect3,
  explorer1, explorer2, explorer3,
  commander1, commander2, commander3,
  partner1, partner2, partner3, partner4,
  delegator1, delegator2, delegator3, delegator4,
];
