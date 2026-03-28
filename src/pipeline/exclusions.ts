/**
 * Unified exclusion rules engine.
 *
 * All message filtering rules live in the `exclusion_rules` DB table.
 * The parser and flagging system read from this table — no hardcoded filters.
 *
 * Rule types:
 *   - content_prefix:    message content starts with pattern
 *   - cwd_pattern:       JSONL entry cwd field contains pattern
 *   - dir_name:          source file directory name matches pattern
 *   - skill_invocation:  message matches a skill invocation syntax (regex)
 *
 * Match modes:
 *   - starts_with:  content.startsWith(pattern)
 *   - contains:     content.includes(pattern)
 *   - exact:        content === pattern
 *   - regex:        new RegExp(pattern).test(content)
 */

import { createHash } from "node:crypto";
import type { Client } from "@libsql/client";
import type { Backend, SkillDefinition } from "./backends.js";

// ── Types ─────────────────────────────────────────────

export interface ExclusionRule {
  id: number;
  platform: string;
  ruleType: string;
  pattern: string;
  matchMode: string;
  description: string;
  source: string;
  templateContent: string;
  isActive: boolean;
}

// Pre-compiled for runtime performance
export interface CompiledRules {
  contentPrefix: Array<{ rule: ExclusionRule; test: (content: string) => boolean }>;
  cwdPattern: Array<{ rule: ExclusionRule; test: (cwd: string) => boolean }>;
  dirName: Array<{ rule: ExclusionRule; test: (dirName: string) => boolean }>;
  skillInvocation: Array<{ rule: ExclusionRule; test: (content: string, platform: string) => boolean }>;
}

// ── Load & Compile ────────────────────────────────────

export async function loadExclusionRules(client: Client): Promise<ExclusionRule[]> {
  const result = await client.execute(
    "SELECT * FROM exclusion_rules WHERE is_active = 1",
  );
  return result.rows.map((r) => ({
    id: Number(r.id),
    platform: String(r.platform),
    ruleType: String(r.rule_type),
    pattern: String(r.pattern),
    matchMode: String(r.match_mode),
    description: String(r.description ?? ""),
    source: String(r.source),
    templateContent: String(r.template_content ?? ""),
    isActive: Boolean(r.is_active),
  }));
}

function buildMatcher(pattern: string, matchMode: string): (text: string) => boolean {
  switch (matchMode) {
    case "starts_with":
      return (text) => text.startsWith(pattern);
    case "contains":
      return (text) => text.includes(pattern);
    case "exact":
      return (text) => text === pattern;
    case "regex":
      try {
        const regex = new RegExp(pattern, "i");
        return (text) => regex.test(text);
      } catch {
        return () => false;
      }
    default:
      return () => false;
  }
}

export function compileRules(rules: ExclusionRule[]): CompiledRules {
  const compiled: CompiledRules = {
    contentPrefix: [],
    cwdPattern: [],
    dirName: [],
    skillInvocation: [],
  };

  for (const rule of rules) {
    const matcher = buildMatcher(rule.pattern, rule.matchMode);

    switch (rule.ruleType) {
      case "content_prefix":
        compiled.contentPrefix.push({ rule, test: matcher });
        break;
      case "cwd_pattern":
        compiled.cwdPattern.push({ rule, test: matcher });
        break;
      case "dir_name":
        compiled.dirName.push({ rule, test: matcher });
        break;
      case "skill_invocation":
        compiled.skillInvocation.push({
          rule,
          test: (content, platform) => {
            if (rule.platform !== "*" && rule.platform !== platform) return false;
            return matcher(content);
          },
        });
        break;
    }
  }

  return compiled;
}

// ── Parse-time checks ─────────────────────────────────

/** Check if a message content should be excluded (content_prefix rules). */
export function shouldExcludeContent(
  compiled: CompiledRules,
  content: string,
  platform: string,
): { excluded: boolean; ruleId: number | null } {
  for (const { rule, test } of compiled.contentPrefix) {
    if (rule.platform !== "*" && rule.platform !== platform) continue;
    if (test(content)) return { excluded: true, ruleId: rule.id };
  }
  return { excluded: false, ruleId: null };
}

/** Check if a CWD indicates a programmatic (non-human) session. */
export function shouldExcludeCwd(
  compiled: CompiledRules,
  cwd: string,
): { excluded: boolean; ruleId: number | null } {
  for (const { rule, test } of compiled.cwdPattern) {
    if (test(cwd)) return { excluded: true, ruleId: rule.id };
  }
  return { excluded: false, ruleId: null };
}

/** Check if a source directory name should be excluded. */
export function shouldExcludeDir(
  compiled: CompiledRules,
  dirName: string,
): { excluded: boolean; ruleId: number | null } {
  for (const { rule, test } of compiled.dirName) {
    if (test(dirName)) return { excluded: true, ruleId: rule.id };
  }
  return { excluded: false, ruleId: null };
}

// ── Post-insert flagging ──────────────────────────────

/**
 * Flag existing messages that match exclusion rules.
 * Checks skill_invocation and content_prefix rules against unflagged messages.
 * Also checks template_content via hash comparison.
 */
export async function flagExcludedMessages(client: Client): Promise<number> {
  const rules = await loadExclusionRules(client);
  const compiled = compileRules(rules);

  // Build template hash lookup (full template content -> rule id)
  const templateHashToRuleId = new Map<string, number>();
  for (const rule of rules) {
    if (rule.templateContent && rule.templateContent.length >= 20) {
      const prefix = rule.templateContent.slice(0, 200);
      const hash = createHash("sha256").update(prefix).digest("hex");
      templateHashToRuleId.set(hash, rule.id);
    }
  }

  // Load unflagged human messages
  const msgRows = await client.execute(
    `SELECT id, platform, content FROM messages
     WHERE role = 'human' AND is_excluded = 0`,
  );

  const updates: Array<{ msgId: number; ruleId: number }> = [];

  for (const row of msgRows.rows) {
    const msgId = Number(row.id);
    const platform = String(row.platform);
    const content = String(row.content);
    let matchedRuleId: number | null = null;

    // Check skill invocation rules
    for (const { rule, test } of compiled.skillInvocation) {
      if (test(content, platform)) {
        matchedRuleId = rule.id;
        break;
      }
    }

    // Check content prefix rules
    if (!matchedRuleId) {
      const result = shouldExcludeContent(compiled, content, platform);
      if (result.excluded) matchedRuleId = result.ruleId;
    }

    // Check template content hash match
    if (!matchedRuleId && templateHashToRuleId.size > 0) {
      const msgPrefix = content.slice(0, 200);
      const msgHash = createHash("sha256").update(msgPrefix).digest("hex");
      const ruleId = templateHashToRuleId.get(msgHash);
      if (ruleId) matchedRuleId = ruleId;
    }

    if (matchedRuleId !== null) {
      updates.push({ msgId, ruleId: matchedRuleId });
    }
  }

  // Batch update
  const batchSize = 500;
  let flagged = 0;
  for (let i = 0; i < updates.length; i += batchSize) {
    const batch = updates.slice(i, i + batchSize);
    await client.batch(
      batch.map((u) => ({
        sql: `UPDATE messages SET is_excluded = 1, matched_rule_id = ? WHERE id = ?`,
        args: [u.ruleId, u.msgId],
      })),
      "write",
    );
    flagged += batch.length;
  }

  return flagged;
}

// ── Discovery → Rules ─────────────────────────────────

/**
 * Discover skills from all backends and upsert as exclusion rules.
 * Also upserts into skills table for template content storage.
 */
export async function discoverAndSyncRules(
  client: Client,
  backends: Backend[],
): Promise<{ skillsFound: number; rulesUpserted: number }> {
  let skillsFound = 0;
  let rulesUpserted = 0;

  for (const backend of backends) {
    if (!backend.discoverSkills) continue;
    const skills = backend.discoverSkills();

    for (const skill of skills) {
      skillsFound++;
      const contentHash = skill.templateContent
        ? createHash("sha256").update(skill.templateContent).digest("hex")
        : "";

      // Upsert into skills table (template content storage)
      await client.execute({
        sql: `INSERT INTO skills (platform, skill_name, skill_path, invocation_pattern, template_content, content_hash, source, discovered_at)
              VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))
              ON CONFLICT(platform, skill_name) DO UPDATE SET
                skill_path = excluded.skill_path,
                invocation_pattern = excluded.invocation_pattern,
                template_content = excluded.template_content,
                content_hash = excluded.content_hash,
                source = excluded.source,
                discovered_at = excluded.discovered_at`,
        args: [
          backend.id,
          skill.skillName,
          skill.skillPath,
          skill.invocationPattern,
          skill.templateContent,
          contentHash,
          skill.source ?? "discovered",
        ],
      });

      // Upsert into exclusion_rules table
      if (skill.invocationPattern) {
        const result = await client.execute({
          sql: `INSERT INTO exclusion_rules (platform, rule_type, pattern, match_mode, description, source, template_content)
                VALUES (?, 'skill_invocation', ?, 'regex', ?, ?, ?)
                ON CONFLICT(platform, rule_type, pattern) DO UPDATE SET
                  description = excluded.description,
                  template_content = excluded.template_content`,
          args: [
            backend.id,
            skill.invocationPattern,
            `${skill.skillName} skill invocation`,
            skill.source ?? "discovered",
            skill.templateContent || null,
          ],
        });
        if (result.rowsAffected > 0) rulesUpserted++;
      }
    }
  }

  return { skillsFound, rulesUpserted };
}

/**
 * Seed system-level exclusion rules.
 * These are platform-inherent rules that don't come from skill discovery.
 * Uses INSERT OR IGNORE so they're only added once.
 */
export async function seedSystemRules(client: Client): Promise<number> {
  const systemRules: Array<{
    platform: string;
    ruleType: string;
    pattern: string;
    matchMode: string;
    description: string;
  }> = [
    // Claude Code framework messages
    {
      platform: "claude_code",
      ruleType: "content_prefix",
      pattern: "<command-",
      matchMode: "starts_with",
      description: "Claude Code system command",
    },
    {
      platform: "claude_code",
      ruleType: "content_prefix",
      pattern: "<local-command",
      matchMode: "starts_with",
      description: "Claude Code local command",
    },
    {
      platform: "claude_code",
      ruleType: "content_prefix",
      pattern: "<task-notification>",
      matchMode: "starts_with",
      description: "Claude Code background task notification",
    },
    {
      platform: "claude_code",
      ruleType: "content_prefix",
      pattern: "This session is being continued from a previous conversation that ran out of context",
      matchMode: "starts_with",
      description: "Claude Code auto-compaction summary",
    },
    {
      platform: "claude_code",
      ruleType: "content_prefix",
      pattern: "<bash-input>",
      matchMode: "starts_with",
      description: "Tool output leaked into human message",
    },
    // Programmatic CWD detection (all platforms)
    {
      platform: "*",
      ruleType: "cwd_pattern",
      pattern: "node_modules",
      matchMode: "contains",
      description: "Programmatic CLI invocation via npm package",
    },
    {
      platform: "*",
      ruleType: "cwd_pattern",
      pattern: ".npm/_npx",
      matchMode: "contains",
      description: "Programmatic CLI invocation via npx",
    },
    // Subagent directory exclusion
    {
      platform: "claude_code",
      ruleType: "dir_name",
      pattern: "subagents",
      matchMode: "exact",
      description: "Claude Code subagent sidechain directory",
    },
  ];

  let seeded = 0;
  for (const rule of systemRules) {
    const result = await client.execute({
      sql: `INSERT OR IGNORE INTO exclusion_rules (platform, rule_type, pattern, match_mode, description, source)
            VALUES (?, ?, ?, ?, ?, 'system')`,
      args: [rule.platform, rule.ruleType, rule.pattern, rule.matchMode, rule.description],
    });
    if (result.rowsAffected > 0) seeded++;
  }

  return seeded;
}
