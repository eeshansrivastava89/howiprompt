#!/usr/bin/env python3
"""
How I Prompt Wrapped 2025 - Build System

One-click command to:
1. Parse Claude Code + Codex conversations
2. Compute all metrics
3. Classify into 1 of 4 personas (algorithmic 2x2 matrix)
4. Generate index.html

Usage:
    python build.py                    # Full build (reads from data/ folder)
    python build.py --metrics-only     # Only compute metrics.json
    python build.py --skip-copy-claude-code # Skip auto-sync for Claude Code logs
    python build.py --skip-copy-codex  # Skip auto-sync for Codex history
    python build.py --no-open          # Build only; do not open dashboard in browser

Data setup:
    1. Copy your Claude Code data to: data/claude_code/
       (auto-copied from ~/.claude/projects by default)
    2. Copy your Codex history to: data/codex/history.jsonl
       (auto-copied from ~/.codex/history.jsonl by default)

Migration note:
    Claude.ai exports are deprecated in v2 and ignored by this pipeline.
"""

import argparse
import json
import logging
import re
import shutil
import subprocess
import sys
import webbrowser
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from collections import defaultdict
from typing import Iterator

# === Logging Setup ===
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s"
)
logger = logging.getLogger(__name__)

# === Configuration ===
# Project root (where build.py lives)
PROJECT_ROOT = Path(__file__).parent.resolve()


@dataclass
class Config:
    """Build configuration - all paths relative to project root."""

    # Data sources - always read from data/ folder
    claude_code_path: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "claude_code")
    codex_history_path: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "codex" / "history.jsonl")

    # Output directory (relative to project root)
    output_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "output")

    # Persona thresholds (calibrated from typical usage)
    # Engagement = (question_rate + backtrack) / 2
    # Politeness = politeness - (command_rate * 0.5)
    persona_engagement_threshold: float = 12.0
    persona_politeness_threshold: float = 4.5

    # Author info (removed in public version)
    author_domain: str = "eeshans.com"


def load_config() -> Config:
    """Load configuration. Could be extended to read from config.yaml."""
    return Config()


def load_branding() -> dict | None:
    """Load branding.json if it exists. Returns None if not found."""
    branding_path = PROJECT_ROOT / "branding.json"
    if branding_path.exists():
        with open(branding_path, 'r') as f:
            return json.load(f)
    return None


# === Data Models ===
class Platform(str, Enum):
    CLAUDE_CODE = "claude_code"
    CODEX = "codex"


class Role(str, Enum):
    HUMAN = "human"
    ASSISTANT = "assistant"


@dataclass
class Message:
    """Unified message from any source."""
    timestamp: datetime
    platform: Platform
    role: Role
    content: str
    conversation_id: str
    word_count: int


class PersonaType(str, Enum):
    """Exactly 4 personas based on 2x2 matrix."""
    COLLABORATOR = "collaborator"   # High engagement, High politeness
    EXPLORER = "explorer"           # High engagement, Low politeness
    EFFICIENT = "efficient"         # Low engagement, High politeness
    PRAGMATIST = "pragmatist"       # Low engagement, Low politeness


@dataclass
class PersonaDefinition:
    """Definition of a persona type."""
    name: str
    description: str
    traits: list[str]


# === Persona Definitions (Exactly 4) ===
PERSONAS: dict[PersonaType, PersonaDefinition] = {
    PersonaType.COLLABORATOR: PersonaDefinition(
        name="The Collaborator",
        description="You ask questions politely. AI is your partner, not your tool.",
        traits=["Inquisitive", "Courteous", "Partnership-oriented"]
    ),
    PersonaType.EXPLORER: PersonaDefinition(
        name="The Explorer",
        description="You question, iterate, and dig deeper. Thinking out loud.",
        traits=["Curious", "Iterative", "Thorough"]
    ),
    PersonaType.EFFICIENT: PersonaDefinition(
        name="The Efficient",
        description="Polite but focused. You know what you want and ask nicely.",
        traits=["Respectful", "Direct", "Purposeful"]
    ),
    PersonaType.PRAGMATIST: PersonaDefinition(
        name="The Pragmatist",
        description="Balanced and practical. No frills, just results.",
        traits=["Balanced", "Practical", "Focused"]
    ),
}


# === Parsers ===
def parse_claude_code(source_path: Path) -> Iterator[Message]:
    """Parse Claude Code conversation JSONL files."""
    if not source_path.exists():
        logger.warning(f"Claude Code path not found: {source_path}")
        return

    files_parsed = 0
    messages_parsed = 0

    for project_dir in source_path.iterdir():
        if not project_dir.is_dir():
            continue

        for jsonl_file in project_dir.glob("*.jsonl"):
            session_id = jsonl_file.stem
            files_parsed += 1

            with open(jsonl_file, 'r') as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    # Skip non-message entries
                    entry_type = entry.get("type")
                    if entry_type not in ("user", "assistant"):
                        continue

                    # Skip meta messages
                    if entry.get("isMeta"):
                        continue

                    # Extract content
                    msg_data = entry.get("message", {})
                    content = ""

                    if msg_data.get("role") == "user":
                        content = msg_data.get("content", "")
                        if isinstance(content, list):
                            content = " ".join(
                                p.get("text", "") for p in content
                                if isinstance(p, dict) and p.get("type") == "text"
                            )
                        role = Role.HUMAN
                    elif msg_data.get("role") == "assistant":
                        content_blocks = msg_data.get("content", [])
                        if isinstance(content_blocks, list):
                            text_parts = []
                            for block in content_blocks:
                                if isinstance(block, dict) and block.get("type") == "text":
                                    text_parts.append(block.get("text", ""))
                            content = " ".join(text_parts)
                        role = Role.ASSISTANT
                    else:
                        continue

                    # Skip empty or command messages
                    if not content or not content.strip():
                        continue
                    if content.startswith("<command-") or content.startswith("<local-command"):
                        continue

                    # Parse timestamp
                    ts_str = entry.get("timestamp")
                    if not ts_str:
                        continue
                    try:
                        timestamp = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                    except (ValueError, AttributeError):
                        continue

                    messages_parsed += 1
                    yield Message(
                        timestamp=timestamp,
                        platform=Platform.CLAUDE_CODE,
                        role=role,
                        content=content,
                        conversation_id=session_id,
                        word_count=len(content.split())
                    )

    logger.info(f"  Claude Code: {messages_parsed} messages from {files_parsed} files")


def parse_codex_history(source_path: Path) -> Iterator[Message]:
    """Parse Codex history.jsonl (user prompts only)."""
    if not source_path or not source_path.exists():
        logger.warning(f"Codex history not found: {source_path}")
        return

    messages_parsed = 0

    with open(source_path, 'r') as f:
        for line in f:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            text = entry.get("text", "")
            if not isinstance(text, str) or not text.strip():
                continue

            ts = entry.get("ts")
            if not isinstance(ts, (int, float)):
                continue

            session_id = str(entry.get("session_id", "unknown"))

            try:
                timestamp = datetime.fromtimestamp(ts, tz=timezone.utc)
            except (ValueError, OSError):
                continue

            messages_parsed += 1
            yield Message(
                timestamp=timestamp,
                platform=Platform.CODEX,
                role=Role.HUMAN,
                content=text,
                conversation_id=session_id,
                word_count=len(text.split())
            )

    logger.info(f"  Codex: {messages_parsed} messages from history.jsonl")


# === Persona Classification (2x2 Matrix Algorithm) ===
def classify_persona(
    politeness: float,
    backtrack: float,
    question_rate: float,
    command_rate: float,
    config: Config
) -> dict:
    """
    Classify user into exactly 1 of 4 personas using 2x2 matrix.

    Axes:
      X: Engagement Score = (question_rate + backtrack) / 2
         High engagement = asks questions, backtracks/iterates

      Y: Politeness Score = politeness - (command_rate * 0.5)
         High politeness = says please/thanks, fewer direct commands

    Quadrants:
      High engagement + High politeness -> Collaborator
      High engagement + Low politeness  -> Explorer
      Low engagement + High politeness  -> Efficient
      Low engagement + Low politeness   -> Pragmatist
    """
    # Compute composite scores
    engagement_score = (question_rate + backtrack) / 2
    politeness_score = politeness - (command_rate * 0.5)

    # Classify based on thresholds
    high_engagement = engagement_score >= config.persona_engagement_threshold
    high_politeness = politeness_score >= config.persona_politeness_threshold

    if high_engagement and high_politeness:
        persona_type = PersonaType.COLLABORATOR
    elif high_engagement and not high_politeness:
        persona_type = PersonaType.EXPLORER
    elif not high_engagement and high_politeness:
        persona_type = PersonaType.EFFICIENT
    else:
        persona_type = PersonaType.PRAGMATIST

    persona = PERSONAS[persona_type]

    return {
        "type": persona_type.value,
        "name": persona.name,
        "description": persona.description,
        "traits": persona.traits,
        "quadrant": {
            "engagement_score": round(engagement_score, 1),
            "politeness_score": round(politeness_score, 1),
            "high_engagement": high_engagement,
            "high_politeness": high_politeness,
        },
        "scores": {
            "politeness": round(politeness, 1),
            "backtrack": round(backtrack, 1),
            "question_rate": round(question_rate, 1),
            "command_rate": round(command_rate, 1),
        }
    }


# === Metrics Computation ===
def compute_metrics(messages: list[Message], config: Config) -> dict:
    """Compute all metrics from unified message stream."""

    human_msgs = [m for m in messages if m.role == Role.HUMAN]
    assistant_msgs = [m for m in messages if m.role == Role.ASSISTANT]

    if not human_msgs:
        raise ValueError("No human messages found in data")

    # === Volume Metrics ===
    total_messages = len(messages)
    total_human = len(human_msgs)
    total_assistant = len(assistant_msgs)
    total_words_human = sum(m.word_count for m in human_msgs)
    total_words_assistant = sum(m.word_count for m in assistant_msgs)
    total_conversations = len(set(m.conversation_id for m in messages))
    avg_words_per_prompt = total_words_human / total_human

    # === Conversation Depth & Response Ratio ===
    turns_per_convo = defaultdict(int)
    for m in messages:
        turns_per_convo[m.conversation_id] += 1
    avg_turns = sum(turns_per_convo.values()) / len(turns_per_convo) if turns_per_convo else 0
    max_turns = max(turns_per_convo.values()) if turns_per_convo else 0
    response_ratio = total_words_assistant / total_words_human if total_words_human else 0

    # Conversation distribution buckets
    quick_asks = sum(1 for t in turns_per_convo.values() if t <= 3)
    working_sessions = sum(1 for t in turns_per_convo.values() if 4 <= t <= 10)
    deep_dives = sum(1 for t in turns_per_convo.values() if t > 10)

    # === Temporal Metrics ===
    heatmap = defaultdict(lambda: defaultdict(int))
    for m in human_msgs:
        local_time = m.timestamp.astimezone()
        hour = local_time.hour
        dow = local_time.weekday()
        heatmap[dow][hour] += 1

    heatmap_data = [[heatmap[dow][hour] for hour in range(24)] for dow in range(7)]

    # Night Owl Index: % of prompts 11pm-4am
    night_owl_count = sum(
        1 for m in human_msgs
        if m.timestamp.astimezone().hour >= 23 or m.timestamp.astimezone().hour < 4
    )
    night_owl_pct = (night_owl_count / total_human * 100)

    # Peak hour
    hour_counts = defaultdict(int)
    for m in human_msgs:
        hour_counts[m.timestamp.astimezone().hour] += 1
    peak_hour = max(hour_counts.items(), key=lambda x: x[1])[0] if hour_counts else 0
    peak_hour_count = hour_counts[peak_hour]

    # Most active day
    day_counts = defaultdict(int)
    for m in human_msgs:
        day_counts[m.timestamp.astimezone().weekday()] += 1
    peak_day = max(day_counts.items(), key=lambda x: x[1])[0] if day_counts else 0
    peak_day_count = day_counts[peak_day]
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    # === Politeness Index ===
    politeness_patterns = {
        "please": r"\bplease\b",
        "thanks": r"\b(thanks|thank you|thx)\b",
        "sorry": r"\b(sorry|apologies|apologize)\b",
    }
    politeness_counts = {}
    for word, pattern in politeness_patterns.items():
        count = sum(
            len(re.findall(pattern, m.content, re.IGNORECASE))
            for m in human_msgs
        )
        politeness_counts[word] = count

    total_politeness = sum(politeness_counts.values())
    politeness_per_100 = (total_politeness / total_human * 100)

    # === Backtrack Index ===
    backtrack_patterns = {
        "actually": r"\bactually\b",
        "wait": r"\bwait\b",
        "never_mind": r"\b(never\s*mind|nevermind)\b",
        "scratch_that": r"\b(scratch that|ignore that)\b",
    }
    backtrack_counts = {}
    for word, pattern in backtrack_patterns.items():
        count = sum(
            len(re.findall(pattern, m.content, re.IGNORECASE))
            for m in human_msgs
        )
        backtrack_counts[word] = count

    total_backtrack = sum(backtrack_counts.values())
    backtrack_per_100 = (total_backtrack / total_human * 100)

    # === Question Rate ===
    question_count = sum(1 for m in human_msgs if m.content.strip().endswith('?'))
    question_rate = (question_count / total_human * 100)

    # === Command Rate ===
    command_pattern = r"^(please\s+)?(do|make|create|write|build|add|fix|update|change|remove|delete|show|run|help|can you|could you|would you|tell|explain|find|search|get|set|check|test|debug|implement|refactor)\b"
    command_count = sum(
        1 for m in human_msgs
        if re.match(command_pattern, m.content.strip(), re.IGNORECASE)
    )
    command_rate = (command_count / total_human * 100)

    # === "You're absolutely right" count ===
    youre_right_count = sum(
        len(re.findall(r"you'?re (absolutely )?right", m.content, re.IGNORECASE))
        for m in assistant_msgs
    )
    youre_right_per_convo = youre_right_count / total_conversations if total_conversations else 0

    # === Platform breakdown ===
    platform_stats = {}
    for platform in Platform:
        platform_msgs = [m for m in human_msgs if m.platform == platform]
        if platform_msgs:
            first_msg = min(m.timestamp for m in platform_msgs)
            platform_stats[platform.value] = {
                "messages": len(platform_msgs),
                "words": sum(m.word_count for m in platform_msgs),
                "conversations": len(set(m.conversation_id for m in platform_msgs)),
                "first_message": first_msg.isoformat(),
            }

    # === Persona Classification ===
    persona = classify_persona(
        politeness_per_100,
        backtrack_per_100,
        question_rate,
        command_rate,
        config
    )

    # === Date range ===
    first_msg = min(m.timestamp for m in human_msgs)
    last_msg = max(m.timestamp for m in human_msgs)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "volume": {
            "total_messages": total_messages,
            "total_human": total_human,
            "total_assistant": total_assistant,
            "total_words_human": total_words_human,
            "total_words_assistant": total_words_assistant,
            "total_conversations": total_conversations,
            "avg_words_per_prompt": round(avg_words_per_prompt, 1),
        },
        "conversation_depth": {
            "avg_turns": round(avg_turns, 1),
            "max_turns": max_turns,
            "quick_asks": quick_asks,
            "working_sessions": working_sessions,
            "deep_dives": deep_dives,
        },
        "response_ratio": round(response_ratio, 1),
        "temporal": {
            "heatmap": heatmap_data,
            "night_owl_pct": round(night_owl_pct, 1),
            "peak_hour": peak_hour,
            "peak_hour_count": peak_hour_count,
            "peak_day": day_names[peak_day],
            "peak_day_count": peak_day_count,
            "hour_counts": dict(hour_counts),
        },
        "politeness": {
            "counts": politeness_counts,
            "total": total_politeness,
            "per_100_prompts": round(politeness_per_100, 1),
        },
        "backtrack": {
            "counts": backtrack_counts,
            "total": total_backtrack,
            "per_100_prompts": round(backtrack_per_100, 1),
        },
        "question": {
            "count": question_count,
            "rate": round(question_rate, 1),
        },
        "command": {
            "count": command_count,
            "rate": round(command_rate, 1),
        },
        "youre_right": {
            "count": youre_right_count,
            "per_conversation": round(youre_right_per_convo, 1),
        },
        "persona": persona,
        "platform_stats": platform_stats,
        "date_range": {
            "first": first_msg.isoformat(),
            "last": last_msg.isoformat(),
        },
    }


def has_human_messages(messages: list[Message]) -> bool:
    """Return True if the message list contains at least one human message."""
    return any(m.role == Role.HUMAN for m in messages)


def compute_source_views(messages: list[Message], config: Config) -> tuple[dict, dict]:
    """
    Build source-scoped metric views for dashboard filtering.

    Source semantics:
      - both: Claude Code + Codex
      - claude_code: Claude Code only
      - codex: Codex only
    """
    claude_code_messages = [m for m in messages if m.platform == Platform.CLAUDE_CODE]
    codex_messages = [m for m in messages if m.platform == Platform.CODEX]

    all_metrics = compute_metrics(messages, config)
    claude_code_metrics = compute_metrics(claude_code_messages, config) if has_human_messages(claude_code_messages) else None
    codex_metrics = compute_metrics(codex_messages, config) if has_human_messages(codex_messages) else None

    source_views = {
        "both": all_metrics,
        "claude_code": claude_code_metrics,
        "codex": codex_metrics,
    }

    default_view = "both"
    if source_views[default_view] is None:
        for candidate in ("claude_code", "codex"):
            if source_views[candidate] is not None:
                default_view = candidate
                break

    return source_views, {"default_view": default_view}


# === HTML Generation ===
def generate_header(branding: dict, personas: dict) -> str:
    """Generate header with nav and methodology modal."""
    return f'''
    <!-- Header -->
    <header class="fixed top-0 left-0 right-0 z-50 bg-bg/80 backdrop-blur-md border-b border-border">
        <nav class="max-w-5xl mx-auto px-4 sm:px-6 py-2 sm:py-3">
            <!-- Row 1: Branding + Subscribe + Theme -->
            <div class="flex items-center justify-between">
                <a href="{branding['site_url']}" class="text-text hover:text-muted transition-colors font-serif text-base sm:text-lg">
                    <span class="font-normal text-muted">Sidequest by</span> <span class="font-semibold">{branding['site_name']}</span>
                </a>
                <div class="flex items-center gap-2 sm:gap-5">
                    <a href="/" class="dashboard-link hidden sm:inline-flex items-center gap-1.5 px-3 py-1 text-sm font-medium text-accent bg-accent/10 rounded-full hover:bg-accent/20 transition-all" id="dashboardLink">
                        <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zM14 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2zM14 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z"></path></svg>
                        Back to Dashboard
                    </a>
                    <button onclick="openMethodology()" class="text-muted hover:text-text transition-colors text-sm hidden sm:block">Methodology</button>
                    <a href="{branding['github_repo']}" target="_blank" class="text-muted hover:text-text transition-colors text-sm hidden sm:block">Build Your Own</a>
                    <a href="{branding['newsletter_url']}" target="_blank" class="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs sm:text-sm font-medium bg-[#FF6719] text-white rounded-full hover:bg-[#E55A15] transition-colors">
                        <svg class="h-3 w-3 sm:h-3.5 sm:w-3.5" viewBox="0 0 24 24" fill="currentColor"><path d="M22.539 8.242H1.46V5.406h21.08v2.836zM1.46 10.812V24L12 18.11 22.54 24V10.812H1.46zM22.54 0H1.46v2.836h21.08V0z"></path></svg>
                        Subscribe
                    </a>
                    <button id="themeToggle" class="p-1.5 rounded-md border border-border hover:bg-surface transition-all duration-200" aria-label="Toggle theme">
                        <svg class="w-4 h-4 text-muted dark:block hidden" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z"></path>
                        </svg>
                        <svg class="w-4 h-4 text-muted dark:hidden block" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z"></path>
                        </svg>
                    </button>
                </div>
            </div>
            <!-- Row 2: Mobile-only nav links -->
            <div class="flex items-center justify-center gap-6 pt-2 sm:hidden">
                <a href="/" class="dashboard-link text-accent text-sm font-medium" id="dashboardLinkMobile">Back to Dashboard</a>
                <button onclick="openMethodology()" class="text-muted hover:text-text transition-colors text-sm">Methodology</button>
                <a href="{branding['github_repo']}" target="_blank" class="text-muted hover:text-text transition-colors text-sm">Build Your Own</a>
            </div>
        </nav>
    </header>

    <!-- Methodology Modal -->
    <div id="methodologyModal" class="fixed inset-0 z-[100] hidden">
        <div class="absolute inset-0 bg-black/60 backdrop-blur-sm" onclick="closeMethodology()"></div>
        <div class="absolute inset-4 md:inset-6 lg:inset-8 bg-bg border border-border rounded-2xl overflow-hidden flex flex-col">
            <!-- Header -->
            <div class="shrink-0 border-b border-border px-6 py-3 flex justify-between items-center">
                <h2 class="text-lg font-bold">Methodology</h2>
                <button onclick="closeMethodology()" class="p-1.5 hover:bg-surface rounded-full transition-colors">
                    <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path>
                    </svg>
                </button>
            </div>
            <!-- Content: Single card, 3 columns on desktop, stacks on mobile -->
            <div class="flex-1 p-4 md:p-6 overflow-y-auto">
                <div class="grid grid-cols-1 lg:grid-cols-3 lg:h-full gap-4 lg:gap-6">
                    <!-- Column 1: How It Works -->
                    <div class="card p-5 lg:p-6">
                        <h3 class="text-sm font-semibold text-muted uppercase tracking-wider mb-5">How It Works</h3>
                        <div class="space-y-5">
                            <div class="flex gap-3">
                                <div class="w-7 h-7 shrink-0 rounded-full bg-accent/20 text-accent flex items-center justify-center text-sm font-bold">1</div>
                                <div>
                                    <p class="font-semibold text-base">Data Collection</p>
                                    <p class="text-muted text-sm leading-relaxed">Reads JSONL conversation logs from Claude Code (~/.claude/projects/) and Codex history from ~/.codex/history.jsonl.</p>
                                </div>
                            </div>
                            <div class="flex gap-3">
                                <div class="w-7 h-7 shrink-0 rounded-full bg-accent/20 text-accent flex items-center justify-center text-sm font-bold">2</div>
                                <div>
                                    <p class="font-semibold text-base">Message Extraction</p>
                                    <p class="text-muted text-sm leading-relaxed">Filters to human-authored prompts only. Excludes system commands, tool outputs, and empty messages. Preserves timestamps for temporal analysis.</p>
                                </div>
                            </div>
                            <div class="flex gap-3">
                                <div class="w-7 h-7 shrink-0 rounded-full bg-accent/20 text-accent flex items-center justify-center text-sm font-bold">3</div>
                                <div>
                                    <p class="font-semibold text-base">Metric Computation</p>
                                    <p class="text-muted text-sm leading-relaxed">Runs regex pattern matching across all prompts to count politeness markers, backtrack phrases, questions, and command patterns.</p>
                                </div>
                            </div>
                            <div class="flex gap-3">
                                <div class="w-7 h-7 shrink-0 rounded-full bg-accent/20 text-accent flex items-center justify-center text-sm font-bold">4</div>
                                <div>
                                    <p class="font-semibold text-base">Persona Classification</p>
                                    <p class="text-muted text-sm leading-relaxed">Computes two composite scores (Engagement and Politeness) and maps them onto a 2×2 matrix to classify your prompting style.</p>
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- Column 2: All Metrics -->
                    <div class="card p-4 lg:p-5 flex flex-col overflow-y-auto">
                        <h3 class="text-xs font-semibold text-muted uppercase tracking-wider mb-3 shrink-0">All Metrics</h3>
                        <div class="space-y-3 flex-1 text-xs">
                            <!-- Volume -->
                            <div>
                                <p class="font-semibold text-text text-[11px] uppercase tracking-wider mb-1.5">Volume</p>
                                <div class="space-y-1 text-muted">
                                    <p><span class="text-text">Total Prompts</span> — your human messages</p>
                                    <p><span class="text-text">Conversations</span> — unique chat sessions</p>
                                    <p><span class="text-text">Words Typed</span> — total word count</p>
                                    <p><span class="text-text">Avg Words/Prompt</span> — verbosity measure</p>
                                </div>
                            </div>
                            <!-- Conversation -->
                            <div>
                                <p class="font-semibold text-text text-[11px] uppercase tracking-wider mb-1.5">Conversation</p>
                                <div class="space-y-1 text-muted">
                                    <p><span class="text-text">Avg Turns</span> — messages per conversation</p>
                                    <p><span class="text-text">Longest Session</span> — marathon conversation</p>
                                    <p><span class="text-text">Quick Asks</span> — 1-3 turn convos</p>
                                    <p><span class="text-text">Working Sessions</span> — 4-10 turn convos</p>
                                    <p><span class="text-text">Deep Dives</span> — 11+ turn convos</p>
                                    <p><span class="text-text">Response Ratio</span> — Claude words ÷ your words</p>
                                </div>
                            </div>
                            <!-- Temporal -->
                            <div>
                                <p class="font-semibold text-text text-[11px] uppercase tracking-wider mb-1.5">Temporal</p>
                                <div class="space-y-1 text-muted">
                                    <p><span class="text-text">Peak Hour</span> — most active hour</p>
                                    <p><span class="text-text">Peak Day</span> — most active weekday</p>
                                    <p><span class="text-text">Night Owl %</span> — 11pm-4am activity</p>
                                </div>
                            </div>
                            <!-- Style -->
                            <div>
                                <p class="font-semibold text-text text-[11px] uppercase tracking-wider mb-1.5">Style</p>
                                <div class="space-y-1 text-muted">
                                    <p><span class="text-accent">Politeness Index</span> — "please", "thanks", "sorry" per 100</p>
                                    <p><span class="text-purple-400">Backtrack Index</span> — "actually", "wait" per 100</p>
                                    <p><span class="text-blue-400">Question Rate</span> — % ending with "?"</p>
                                    <p><span class="text-green-400">Command Rate</span> — % starting with action verbs</p>
                                </div>
                            </div>
                            <!-- Signature -->
                            <div>
                                <p class="font-semibold text-text text-[11px] uppercase tracking-wider mb-1.5">Signature</p>
                                <div class="text-muted">
                                    <p><span class="text-text">"You're absolutely right"</span> — Claude's agreement count</p>
                                </div>
                            </div>
                        </div>
                        <!-- Composite Scores -->
                        <div class="mt-auto pt-3 border-t border-border text-xs text-muted shrink-0">
                            <p class="font-semibold text-text mb-1">Composite Scores:</p>
                            <p><span class="text-text">Engagement</span> = (Question + Backtrack) ÷ 2</p>
                            <p><span class="text-text">Politeness</span> = Index − (Command × 0.5)</p>
                        </div>
                    </div>

                    <!-- Column 3: The 4 Personas -->
                    <div class="card p-5 lg:p-6 flex flex-col">
                        <h3 class="text-sm font-semibold text-muted uppercase tracking-wider mb-4 shrink-0">The 4 Personas</h3>
                        <p class="text-muted text-sm mb-5">Based on where you fall on the Engagement (x-axis) and Politeness (y-axis) scales:</p>
                        <!-- 2x2 Matrix -->
                        <div class="grid grid-cols-[auto_1fr_1fr] gap-2 text-center">
                            <!-- Header row -->
                            <div></div>
                            <div class="text-muted text-xs font-medium py-1">High Politeness</div>
                            <div class="text-muted text-xs font-medium py-1">Low Politeness</div>
                            <!-- High Engagement row -->
                            <div class="text-muted text-xs font-medium flex items-center justify-end pr-2">High<br/>Engage</div>
                            <div class="bg-accent/10 border border-accent/30 rounded-lg p-3 flex flex-col justify-center">
                                <p class="font-bold text-accent text-base">{personas[PersonaType.COLLABORATOR].name.split()[-1]}</p>
                                <p class="text-muted text-xs mt-1.5 leading-snug">{personas[PersonaType.COLLABORATOR].description}</p>
                            </div>
                            <div class="bg-surface border border-border rounded-lg p-3 flex flex-col justify-center">
                                <p class="font-bold text-base">{personas[PersonaType.EXPLORER].name.split()[-1]}</p>
                                <p class="text-muted text-xs mt-1.5 leading-snug">{personas[PersonaType.EXPLORER].description}</p>
                            </div>
                            <!-- Low Engagement row -->
                            <div class="text-muted text-xs font-medium flex items-center justify-end pr-2">Low<br/>Engage</div>
                            <div class="bg-surface border border-border rounded-lg p-3 flex flex-col justify-center">
                                <p class="font-bold text-base">{personas[PersonaType.EFFICIENT].name.split()[-1]}</p>
                                <p class="text-muted text-xs mt-1.5 leading-snug">{personas[PersonaType.EFFICIENT].description}</p>
                            </div>
                            <div class="bg-surface border border-border rounded-lg p-3 flex flex-col justify-center">
                                <p class="font-bold text-base">{personas[PersonaType.PRAGMATIST].name.split()[-1]}</p>
                                <p class="text-muted text-xs mt-1.5 leading-snug">{personas[PersonaType.PRAGMATIST].description}</p>
                            </div>
                        </div>
                        <!-- Thresholds note -->
                        <div class="mt-auto pt-3 border-t border-border text-xs text-muted">
                            <p>Thresholds: Engagement ≥ 12, Politeness Score ≥ 4.5</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        function openMethodology() {{
            document.getElementById('methodologyModal').classList.remove('hidden');
            document.body.style.overflow = 'hidden';
        }}
        function closeMethodology() {{
            document.getElementById('methodologyModal').classList.add('hidden');
            document.body.style.overflow = '';
        }}
        document.addEventListener('keydown', (e) => {{
            if (e.key === 'Escape') closeMethodology();
        }});
    </script>
    '''


def generate_footer_content(branding: dict) -> str:
    """Generate footer content to embed in Section 7."""
    return f'''
            <!-- Footer (embedded in Section 7 for proper scroll-snap) -->
            <div class="mt-8 pt-4 border-t border-border">
                <div class="flex flex-col sm:flex-row items-center justify-between gap-4 text-xs text-muted">
                    <p>© 2025 Eeshan Srivastava. All rights reserved.</p>
                    <p class="italic">Personal project · MIT License · Non-commercial</p>
                    <div class="flex items-center gap-4">
                        <a href="{branding['github_repo']}" target="_blank" class="hover:text-text transition-colors flex items-center gap-1.5">
                            <svg class="w-3.5 h-3.5" fill="currentColor" viewBox="0 0 24 24"><path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/></svg>
                            Source
                        </a>
                        <a href="https://www.linkedin.com/in/eeshans/" target="_blank" class="hover:text-text transition-colors">
                            <svg class="w-3.5 h-3.5" fill="currentColor" viewBox="0 0 24 24"><path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433a2.06 2.06 0 01-2.063-2.065 2.064 2.064 0 112.063 2.065zm1.782 13.019H3.555V9h3.564zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0z"/></svg>
                        </a>
                    </div>
                </div>
            </div>
    '''


def generate_html(metrics: dict, branding: dict | None = None) -> str:
    """Generate HTML from metrics using template substitution."""

    m = metrics
    v = m["volume"]
    t = m["temporal"]
    pol = m["politeness"]
    back = m["backtrack"]
    q = m["question"]
    cmd = m["command"]
    yr = m["youre_right"]
    p = m["persona"]
    cd = m.get("conversation_depth", {})
    rr = m.get("response_ratio", 0)

    # Format peak hour
    peak_hour_12h = f"{t['peak_hour'] % 12 or 12}{'am' if t['peak_hour'] < 12 else 'pm'}"

    # Politeness bar widths (relative to max)
    pol_max = max(pol["counts"].values()) if pol["counts"] else 1
    pol_please_pct = round(pol["counts"].get("please", 0) / pol_max * 100)
    pol_sorry_pct = round(pol["counts"].get("sorry", 0) / pol_max * 100)
    pol_thanks_pct = round(pol["counts"].get("thanks", 0) / pol_max * 100)

    # Backtrack bar widths
    back_max = max(back["counts"].values()) if back["counts"] else 1
    back_actually_pct = round(back["counts"].get("actually", 0) / back_max * 100)
    back_wait_pct = round(back["counts"].get("wait", 0) / back_max * 100)

    # Date display
    from datetime import datetime as dt
    date_display = f"Generated {dt.now().strftime('%b %d, %Y')}"

    # Date range from data
    dr = m.get("date_range", {})
    if dr:
        first_date = dt.fromisoformat(dr["first"]).strftime("%b %d, %Y")
        last_date = dt.fromisoformat(dr["last"]).strftime("%b %d, %Y")
        date_range_display = f"{first_date} – {last_date}"
    else:
        date_range_display = "2025"
    author_line = '<p class="text-muted text-xs">eeshans.com</p>'

    # Heatmap JSON
    heatmap_json = json.dumps(t["heatmap"])

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="Content-Security-Policy" content="upgrade-insecure-requests">
    <script>if(location.protocol === 'http:' && location.hostname !== 'localhost' && location.hostname !== '127.0.0.1') location.replace('https:' + location.href.substring(location.protocol.length));</script>
    <title>How I Prompt: {date_range_display}</title>

    <!-- Open Graph -->
    <meta property="og:title" content="How I Prompt: {date_range_display}">
    <meta property="og:description" content="{v['total_human']:,} prompts. {yr['count']} times Claude said 'You're absolutely right.' A year in AI conversations.">
    <meta property="og:type" content="website">

    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <script src="https://cdn.tailwindcss.com"></script>
    <script>
        tailwind.config = {{
            darkMode: 'class',
            theme: {{
                extend: {{
                    colors: {{
                        'bg': 'var(--bg)',
                        'surface': 'var(--surface)',
                        'border': 'var(--border)',
                        'text': 'var(--text)',
                        'muted': 'var(--muted)',
                        'accent': '#f97316',
                        'accent-dim': '#c2410c',
                    }},
                    fontFamily: {{
                        'sans': ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
                        'serif': ['Georgia', 'Cambria', 'serif'],
                        'mono': ['SF Mono', 'Menlo', 'monospace'],
                    }},
                    gridTemplateColumns: {{
                        '24': 'repeat(24, minmax(0, 1fr))',
                    }}
                }}
            }}
        }}
    </script>
    <style>
        :root {{
            --bg: #f5f5f5;
            --surface: #ffffff;
            --border: #e5e5e5;
            --text: #171717;
            --muted: #737373;
        }}
        .dark {{
            --bg: #0a0a0a;
            --surface: #1a1a1a;
            --border: #262626;
            --text: #e5e5e5;
            --muted: #737373;
        }}
        html {{ scroll-snap-type: y mandatory; scroll-behavior: smooth; }}
        section {{ scroll-snap-align: start; scroll-snap-stop: always; }}
        @keyframes fadeIn {{ from {{ opacity: 0; }} to {{ opacity: 1; }} }}
        @keyframes fadeInUp {{ from {{ opacity: 0; transform: translateY(30px); }} to {{ opacity: 1; transform: translateY(0); }} }}
        .animate-fade-in {{ animation: fadeIn 1s ease-out forwards; }}
        .animate-fade-in-up {{ animation: fadeInUp 0.8s ease-out forwards; }}
        .delay-1 {{ animation-delay: 0.3s; opacity: 0; }}
        .delay-2 {{ animation-delay: 0.6s; opacity: 0; }}
        .stat-accent {{ color: #f97316; }}
        .gradient-text {{ background: linear-gradient(135deg, #f97316, #fb923c); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }}
        .heatmap-cell {{ transition: all 0.3s ease; }}
        .heatmap-cell:hover {{ transform: scale(1.2); z-index: 10; }}
        .card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 12px; }}
        .scroll-hint {{ animation: bounce 2s infinite; }}
        @keyframes bounce {{ 0%, 100% {{ transform: translateY(0); }} 50% {{ transform: translateY(8px); }} }}
        .tabular-nums {{ font-variant-numeric: tabular-nums; }}
    </style>

    <!-- PostHog Analytics -->
    <script>
        !function(t,e){{var o,n,p,r;e.__SV||(window.posthog=e,e._i=[],e.init=function(i,s,a){{function g(t,e){{var o=e.split(".");2==o.length&&(t=t[o[0]],e=o[1]),t[e]=function(){{t.push([e].concat(Array.prototype.slice.call(arguments,0)))}}}}(p=t.createElement("script")).type="text/javascript",p.async=!0,p.src=s.api_host.replace(/\\/$/, "")+"/static/array.js",(r=t.getElementsByTagName("script")[0]).parentNode.insertBefore(p,r);var u=e;for(void 0!==a?u=e[a]=[]:a="posthog",u.people=u.people||[],u.toString=function(t){{var e="posthog";return"posthog"!==a&&(e+="."+a),t||(e+=" (stub)"),e}},u.people.toString=function(){{return u.toString(1)+".people (stub)"}},o="capture identify alias people.set people.set_once set_config register register_once unregister opt_out_capturing has_opted_out_capturing opt_in_capturing reset isFeatureEnabled onFeatureFlags getFeatureFlag getFeatureFlagPayload reloadFeatureFlags group updateEarlyAccessFeatureEnrollment getEarlyAccessFeatures getActiveMatchingSurveys getSurveys onSessionId".split(" "),n=0;n<o.length;n++)g(u,o[n]);e._i.push([i,s,a])}},e.__SV=1)}}(document,window.posthog||[]);
        posthog.init('phc_zfue5Ca8VaxypRHPCi9j2h2R3Qy1eytEHt3TMPWlOOS',{{api_host:'https://api-v2.eeshans.com', ui_host:'https://us.posthog.com', person_profiles: 'identified_only'}})
    </script>
</head>
<body class="bg-bg text-text font-sans antialiased transition-colors duration-300{' pt-14' if branding else ''}">

    {generate_header(branding, PERSONAS) if branding else '''
    <!-- Theme Toggle (standalone when no branding) -->
    <button id="themeToggle" class="fixed top-4 right-4 z-50 p-3 rounded-full bg-surface border border-border hover:bg-border transition-all duration-200" aria-label="Toggle theme">
        <svg class="w-5 h-5 text-text dark:block hidden" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z"></path>
        </svg>
        <svg class="w-5 h-5 text-text dark:hidden block" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z"></path>
        </svg>
    </button>
    '''}

    <!-- Section 1: Cold Open -->
    <section class="min-h-screen flex flex-col items-center justify-center px-6 relative">
        <div class="text-center max-w-3xl">
            <p class="font-serif text-3xl md:text-5xl italic text-text/90 animate-fade-in">"You're absolutely right."</p>
            <p class="text-muted text-lg mt-6 animate-fade-in delay-1">— Claude, <span class="stat-accent font-semibold">{yr['count']}</span> times this year</p>
            <p class="text-muted/60 text-sm mt-2 animate-fade-in delay-1">That's <span class="text-text/80">{yr['per_conversation']}×</span> per conversation</p>
            <div class="mt-16 animate-fade-in delay-2">
                <p class="text-muted text-sm uppercase tracking-widest mb-2">{date_range_display}</p>
                <h1 class="text-4xl md:text-6xl font-bold">How I <span class="gradient-text">Prompt</span></h1>
            </div>
        </div>
        <div class="absolute bottom-12">
            <p class="text-muted text-sm mb-2">Scroll</p>
            <svg class="w-5 h-5 mx-auto text-muted" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M19 14l-7 7m0 0l-7-7m7 7V3"></path>
            </svg>
        </div>
    </section>

    <!-- Section 2: The Numbers -->
    <section class="min-h-screen flex items-center justify-center px-6 py-20">
        <div class="max-w-3xl w-full">
            <p class="text-muted text-sm uppercase tracking-widest mb-8 text-center">The Numbers</p>
            <div class="text-center mb-12">
                <div class="text-7xl md:text-8xl font-bold gradient-text counter" data-target="{v['total_human']}">{v['total_human']:,}</div>
                <p class="text-2xl text-text/80 mt-2">prompts sent</p>
            </div>
            <div class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
                <div class="card p-4 text-center">
                    <p class="text-2xl font-bold text-text">{v['total_conversations']:,}</p>
                    <p class="text-muted text-xs mt-1">conversations</p>
                </div>
                <div class="card p-4 text-center">
                    <p class="text-2xl font-bold text-text">{v['total_words_human'] // 1000}K</p>
                    <p class="text-muted text-xs mt-1">words typed</p>
                </div>
                <div class="card p-4 text-center">
                    <p class="text-2xl font-bold text-text">{cd.get('avg_turns', 0)}</p>
                    <p class="text-muted text-xs mt-1">avg turns</p>
                </div>
                <div class="card p-4 text-center">
                    <p class="text-2xl font-bold text-text">{rr}x</p>
                    <p class="text-muted text-xs mt-1">response ratio</p>
                </div>
                <div class="card p-4 text-center">
                    <p class="text-2xl font-bold text-text">{cd.get('max_turns', 0)}</p>
                    <p class="text-muted text-xs mt-1">longest session</p>
                </div>
                <div class="card p-4 text-center">
                    <p class="text-2xl font-bold text-text">{cd.get('deep_dives', 0)}</p>
                    <p class="text-muted text-xs mt-1">deep dives</p>
                </div>
            </div>
            <div class="card p-6 mt-8 text-center">
                <p class="text-muted font-serif italic">"That's a lot of 'actually, wait...'"</p>
            </div>
        </div>
    </section>

    <!-- Section 3: Temporal Patterns -->
    <section class="min-h-screen flex items-center justify-center px-4 sm:px-6 py-8 sm:py-20">
        <div class="max-w-4xl w-full">
            <p class="text-muted text-sm uppercase tracking-widest mb-4 sm:mb-8 text-center">When You Prompt</p>
            <div class="card p-3 sm:p-4 md:p-8 mb-4 sm:mb-8 overflow-x-auto">
                <p class="text-sm text-muted mb-6">Activity by Hour × Day</p>
                <div class="min-w-[500px] md:min-w-0 w-full">
                    <div class="flex mb-2 ml-12">
                        <div class="flex-1 grid grid-cols-24 gap-[2px]">
                            <span class="text-[10px] text-muted text-center">0</span>
                            <span class="text-[10px] text-muted text-center"></span>
                            <span class="text-[10px] text-muted text-center"></span>
                            <span class="text-[10px] text-muted text-center">3</span>
                            <span class="text-[10px] text-muted text-center"></span>
                            <span class="text-[10px] text-muted text-center"></span>
                            <span class="text-[10px] text-muted text-center">6</span>
                            <span class="text-[10px] text-muted text-center"></span>
                            <span class="text-[10px] text-muted text-center"></span>
                            <span class="text-[10px] text-muted text-center">9</span>
                            <span class="text-[10px] text-muted text-center"></span>
                            <span class="text-[10px] text-muted text-center"></span>
                            <span class="text-[10px] text-muted text-center">12</span>
                            <span class="text-[10px] text-muted text-center"></span>
                            <span class="text-[10px] text-muted text-center"></span>
                            <span class="text-[10px] text-muted text-center">15</span>
                            <span class="text-[10px] text-muted text-center"></span>
                            <span class="text-[10px] text-muted text-center"></span>
                            <span class="text-[10px] text-muted text-center">18</span>
                            <span class="text-[10px] text-muted text-center"></span>
                            <span class="text-[10px] text-muted text-center"></span>
                            <span class="text-[10px] text-muted text-center">21</span>
                            <span class="text-[10px] text-muted text-center"></span>
                            <span class="text-[10px] text-muted text-center"></span>
                        </div>
                    </div>
                    <div id="heatmap" class="space-y-[2px]"></div>
                </div>
            </div>
            <div class="grid grid-cols-3 gap-2 sm:gap-4">
                <div class="card p-3 sm:p-6 text-center">
                    <p class="text-2xl sm:text-4xl font-bold stat-accent">{peak_hour_12h}</p>
                    <p class="text-muted text-xs sm:text-sm mt-1">Peak hour</p>
                    <p class="text-muted text-[10px] sm:text-xs">{t['peak_hour_count']} prompts</p>
                </div>
                <div class="card p-3 sm:p-6 text-center">
                    <p class="text-2xl sm:text-4xl font-bold text-text">{t['night_owl_pct']}%</p>
                    <p class="text-muted text-xs sm:text-sm mt-1">Night Owl</p>
                    <p class="text-muted text-[10px] sm:text-xs">11pm - 4am</p>
                </div>
                <div class="card p-3 sm:p-6 text-center">
                    <p class="text-2xl sm:text-4xl font-bold text-text">{t['peak_day']}</p>
                    <p class="text-muted text-xs sm:text-sm mt-1">Peak day</p>
                    <p class="text-muted text-[10px] sm:text-xs">{t['peak_day_count']} prompts</p>
                </div>
            </div>
            <div class="card p-4 sm:p-6 mt-4 sm:mt-8 text-center">
                <p class="text-muted font-serif italic text-sm sm:text-base">"Just one more fix..."</p>
            </div>
        </div>
    </section>

    <!-- Section 4: Prompt Style -->
    <section class="min-h-screen flex items-center justify-center px-4 sm:px-6 py-8 sm:py-20">
        <div class="max-w-4xl w-full">
            <p class="text-muted text-sm uppercase tracking-widest mb-4 sm:mb-8 text-center">Prompt Style</p>
            <div class="grid grid-cols-2 gap-2 sm:gap-6">
                <!-- Politeness Index -->
                <div class="card p-3 sm:p-6">
                    <div class="flex justify-between items-center mb-1 sm:mb-2">
                        <h3 class="text-sm sm:text-lg font-semibold">Politeness</h3>
                        <span class="text-xl sm:text-2xl font-bold stat-accent">{pol['per_100_prompts']}</span>
                    </div>
                    <p class="text-muted text-[10px] sm:text-xs mb-2 sm:mb-4">per 100 prompts</p>
                    <div class="space-y-1 sm:space-y-2">
                        <div class="flex items-center gap-2 sm:gap-3">
                            <span class="w-12 sm:w-16 text-muted text-[10px] sm:text-xs">"please"</span>
                            <div class="flex-1 h-1.5 sm:h-2 bg-border rounded-full overflow-hidden">
                                <div class="h-full bg-accent rounded-full" style="width: {pol_please_pct}%"></div>
                            </div>
                            <span class="text-[10px] sm:text-xs font-mono w-6 sm:w-8 text-right">{pol['counts'].get('please', 0)}</span>
                        </div>
                        <div class="flex items-center gap-2 sm:gap-3">
                            <span class="w-12 sm:w-16 text-muted text-[10px] sm:text-xs">"sorry"</span>
                            <div class="flex-1 h-1.5 sm:h-2 bg-border rounded-full overflow-hidden">
                                <div class="h-full bg-accent/60 rounded-full" style="width: {pol_sorry_pct}%"></div>
                            </div>
                            <span class="text-[10px] sm:text-xs font-mono w-6 sm:w-8 text-right">{pol['counts'].get('sorry', 0)}</span>
                        </div>
                        <div class="flex items-center gap-2 sm:gap-3">
                            <span class="w-12 sm:w-16 text-muted text-[10px] sm:text-xs">"thanks"</span>
                            <div class="flex-1 h-1.5 sm:h-2 bg-border rounded-full overflow-hidden">
                                <div class="h-full bg-accent/40 rounded-full" style="width: {pol_thanks_pct}%"></div>
                            </div>
                            <span class="text-[10px] sm:text-xs font-mono w-6 sm:w-8 text-right">{pol['counts'].get('thanks', 0)}</span>
                        </div>
                    </div>
                    <p class="text-muted text-[10px] mt-2 sm:mt-3 italic">Says "please" but rarely "thanks"</p>
                </div>

                <!-- Backtrack Index -->
                <div class="card p-3 sm:p-6">
                    <div class="flex justify-between items-center mb-1 sm:mb-2">
                        <h3 class="text-sm sm:text-lg font-semibold">Backtrack</h3>
                        <span class="text-xl sm:text-2xl font-bold text-purple-400">{back['per_100_prompts']}</span>
                    </div>
                    <p class="text-muted text-[10px] sm:text-xs mb-2 sm:mb-4">per 100 prompts</p>
                    <div class="space-y-1 sm:space-y-2">
                        <div class="flex items-center gap-2 sm:gap-3">
                            <span class="w-12 sm:w-16 text-muted text-[10px] sm:text-xs">"actually"</span>
                            <div class="flex-1 h-1.5 sm:h-2 bg-border rounded-full overflow-hidden">
                                <div class="h-full bg-purple-500 rounded-full" style="width: {back_actually_pct}%"></div>
                            </div>
                            <span class="text-[10px] sm:text-xs font-mono w-6 sm:w-8 text-right">{back['counts'].get('actually', 0)}</span>
                        </div>
                        <div class="flex items-center gap-2 sm:gap-3">
                            <span class="w-12 sm:w-16 text-muted text-[10px] sm:text-xs">"wait"</span>
                            <div class="flex-1 h-1.5 sm:h-2 bg-border rounded-full overflow-hidden">
                                <div class="h-full bg-purple-400/60 rounded-full" style="width: {back_wait_pct}%"></div>
                            </div>
                            <span class="text-[10px] sm:text-xs font-mono w-6 sm:w-8 text-right">{back['counts'].get('wait', 0)}</span>
                        </div>
                    </div>
                    <p class="text-muted text-[10px] mt-2 sm:mt-3 italic">Course-corrects frequently</p>
                </div>

                <!-- Question Rate -->
                <div class="card p-3 sm:p-6">
                    <div class="flex justify-between items-center mb-1 sm:mb-2">
                        <h3 class="text-sm sm:text-lg font-semibold">Questions</h3>
                        <span class="text-xl sm:text-2xl font-bold text-blue-500">{q['rate']}%</span>
                    </div>
                    <p class="text-muted text-[10px] sm:text-xs mb-2 sm:mb-4">of prompts</p>
                    <div class="mt-2 sm:mt-4">
                        <div class="h-2 sm:h-3 bg-border rounded-full overflow-hidden">
                            <div class="h-full bg-blue-500 rounded-full" style="width: {q['rate']}%"></div>
                        </div>
                        <div class="flex justify-between mt-1 sm:mt-2">
                            <span class="text-[10px] sm:text-xs text-muted">{q['count']} questions</span>
                            <span class="text-[10px] sm:text-xs text-muted">of {v['total_human']:,}</span>
                        </div>
                    </div>
                    <p class="text-muted text-[10px] mt-2 sm:mt-3 italic">Sometimes asks, mostly tells</p>
                </div>

                <!-- Command Rate -->
                <div class="card p-3 sm:p-6">
                    <div class="flex justify-between items-center mb-1 sm:mb-2">
                        <h3 class="text-sm sm:text-lg font-semibold">Commands</h3>
                        <span class="text-xl sm:text-2xl font-bold text-green-500">{cmd['rate']}%</span>
                    </div>
                    <p class="text-muted text-[10px] sm:text-xs mb-2 sm:mb-4">action verbs</p>
                    <div class="mt-2 sm:mt-4">
                        <div class="h-2 sm:h-3 bg-border rounded-full overflow-hidden">
                            <div class="h-full bg-green-500 rounded-full" style="width: {cmd['rate']}%"></div>
                        </div>
                        <div class="flex justify-between mt-1 sm:mt-2">
                            <span class="text-[10px] sm:text-xs text-muted">{cmd['count']} commands</span>
                            <span class="text-[10px] sm:text-xs text-muted">fix, add...</span>
                        </div>
                    </div>
                    <p class="text-muted text-[10px] mt-2 sm:mt-3 italic">Prefers context over commands</p>
                </div>
            </div>
            <div class="card p-4 sm:p-6 mt-4 sm:mt-8 text-center">
                <p class="text-muted font-serif italic text-sm sm:text-base">"The robots will remember this."</p>
            </div>
        </div>
    </section>

    <!-- Section 5: Conversation Patterns -->
    <section class="min-h-screen flex items-center justify-center px-6 py-20">
        <div class="max-w-3xl w-full">
            <p class="text-muted text-sm uppercase tracking-widest mb-8 text-center">How You Use AI</p>
            <!-- Distribution breakdown -->
            <div class="grid grid-cols-3 gap-4 mb-8">
                <div class="card p-6 text-center">
                    <p class="text-4xl font-bold text-blue-400">{cd.get('quick_asks', 0)}</p>
                    <p class="text-muted text-sm mt-1">Quick Asks</p>
                    <p class="text-muted text-xs">1-3 turns</p>
                </div>
                <div class="card p-6 text-center">
                    <p class="text-4xl font-bold text-accent">{cd.get('working_sessions', 0)}</p>
                    <p class="text-muted text-sm mt-1">Working Sessions</p>
                    <p class="text-muted text-xs">4-10 turns</p>
                </div>
                <div class="card p-6 text-center">
                    <p class="text-4xl font-bold text-purple-400">{cd.get('deep_dives', 0)}</p>
                    <p class="text-muted text-sm mt-1">Deep Dives</p>
                    <p class="text-muted text-xs">11+ turns</p>
                </div>
            </div>
            <!-- Key stats -->
            <div class="grid md:grid-cols-2 gap-6 mb-8">
                <div class="card p-8">
                    <p class="text-muted text-sm uppercase tracking-wider mb-4">Conversation Depth</p>
                    <div class="flex items-baseline gap-2">
                        <p class="text-5xl font-bold">{cd.get('avg_turns', 0)}</p>
                        <p class="text-xl text-muted">avg turns</p>
                    </div>
                    <p class="text-muted text-sm mt-4">Longest marathon: <span class="text-text font-semibold">{cd.get('max_turns', 0)} turns</span></p>
                </div>
                <div class="card p-8">
                    <p class="text-muted text-sm uppercase tracking-wider mb-4">Response Ratio</p>
                    <div class="flex items-baseline gap-2">
                        <p class="text-5xl font-bold">{rr}x</p>
                        <p class="text-xl text-muted">Claude's output</p>
                    </div>
                    <p class="text-muted text-sm mt-4">For every word you type, Claude writes <span class="text-text font-semibold">{rr}</span></p>
                </div>
            </div>
            <div class="card p-6 mt-8 text-center">
                <p class="text-muted font-serif italic">"More collaborator than commander."</p>
            </div>
        </div>
    </section>

    <!-- Section 6: Your AI Persona -->
    <section class="min-h-screen flex items-center justify-center px-6 py-20">
        <div class="max-w-3xl w-full">
            <p class="text-muted text-sm uppercase tracking-widest mb-8 text-center">Diagnosis</p>
            <div class="card p-10 text-center">
                <p class="text-muted text-sm mb-4">Based on your metrics, you are...</p>
                <h2 class="text-3xl md:text-4xl font-bold gradient-text mb-4">{p['name']}</h2>
                <p class="text-xl text-text/80 max-w-lg mx-auto mb-8">{p['description']}</p>
                <div class="flex flex-wrap justify-center gap-3 mb-10">
                    {''.join(f'<span class="px-4 py-2 bg-border rounded-full text-sm font-medium">{trait}</span>' for trait in p['traits'])}
                </div>
                <div class="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
                    <div>
                        <p class="text-2xl font-bold stat-accent">{p['scores']['politeness']}</p>
                        <p class="text-muted text-xs">Politeness</p>
                    </div>
                    <div>
                        <p class="text-2xl font-bold text-purple-400">{p['scores']['backtrack']}</p>
                        <p class="text-muted text-xs">Backtrack</p>
                    </div>
                    <div>
                        <p class="text-2xl font-bold text-blue-400">{p['scores']['question_rate']}%</p>
                        <p class="text-muted text-xs">Questions</p>
                    </div>
                    <div>
                        <p class="text-2xl font-bold text-green-400">{p['scores']['command_rate']}%</p>
                        <p class="text-muted text-xs">Commands</p>
                    </div>
                </div>
            </div>
            <div class="card p-6 mt-8 text-center">
                <p class="text-muted font-serif italic">"Efficient. Almost suspiciously so."</p>
            </div>
        </div>
    </section>

    <!-- Section 7: Final Diagnostic (includes footer for proper scroll-snap) -->
    <section class="min-h-screen flex flex-col items-center px-4 sm:px-6 py-8 sm:py-16">
        <div class="max-w-md w-full flex flex-col justify-center flex-1">
            <div class="card p-4 sm:p-6 font-mono text-xs sm:text-sm">
                <div class="text-center border-b border-border pb-2 sm:pb-3 mb-2 sm:mb-3">
                    <p class="text-sm sm:text-base font-bold">HOW I PROMPT</p>
                    <p class="text-muted text-[10px] sm:text-xs">{date_range_display.upper()} SUMMARY</p>
                </div>
                <div class="space-y-1 sm:space-y-1.5 border-b border-border pb-2 sm:pb-3 mb-2 sm:mb-3">
                    <div class="flex justify-between"><span class="text-muted">Prompts</span><span>{v['total_human']:,}</span></div>
                    <div class="flex justify-between"><span class="text-muted">Conversations</span><span>{v['total_conversations']}</span></div>
                    <div class="flex justify-between"><span class="text-muted">Words Typed</span><span>{v['total_words_human'] // 1000}K</span></div>
                </div>
                <div class="space-y-1 sm:space-y-1.5 border-b border-border pb-2 sm:pb-3 mb-2 sm:mb-3">
                    <div class="flex justify-between"><span class="text-muted">Avg Turns</span><span>{cd.get('avg_turns', 0)}</span></div>
                    <div class="flex justify-between"><span class="text-muted">Longest</span><span class="stat-accent">{cd.get('max_turns', 0)} turns</span></div>
                    <div class="flex justify-between"><span class="text-muted">Deep Dives</span><span>{cd.get('deep_dives', 0)}</span></div>
                </div>
                <div class="space-y-1 sm:space-y-1.5 border-b border-border pb-2 sm:pb-3 mb-2 sm:mb-3">
                    <div class="flex justify-between"><span class="text-muted">Peak Hour</span><span>{peak_hour_12h}</span></div>
                    <div class="flex justify-between"><span class="text-muted">Peak Day</span><span>{t['peak_day']}</span></div>
                    <div class="flex justify-between"><span class="text-muted">Night Owl</span><span>{t['night_owl_pct']}%</span></div>
                </div>
                <div class="space-y-1 sm:space-y-1.5 border-b border-border pb-2 sm:pb-3 mb-2 sm:mb-3">
                    <div class="flex justify-between"><span class="text-muted">"You're absolutely right"</span><span class="stat-accent">{yr['count']}x</span></div>
                </div>
                <div class="text-center pt-2 sm:pt-3">
                    <p class="font-bold text-xs sm:text-sm mb-1">PERSONA: {p['name'].upper()}</p>
                    <p class="text-muted text-[10px] sm:text-xs">{' • '.join(p['traits'])}</p>
                </div>
                <div class="text-center mt-3 sm:mt-4 pt-2 sm:pt-3 border-t border-border">
                    <p class="text-muted text-[10px] sm:text-xs">{date_display}</p>
                    {author_line}
                </div>
            </div>
        </div>
{generate_footer_content(branding) if branding else ''}
    </section>

    <script>
        // Fix dashboard link for local vs production
        const isLocal = location.protocol === 'file:' || location.hostname === 'localhost' || location.hostname === '127.0.0.1';
        if (isLocal) {{
            document.querySelectorAll('.dashboard-link').forEach(link => {{
                const currentPath = location.pathname;
                const basePath = currentPath.substring(0, currentPath.lastIndexOf('/'));
                // Check if we're in /wrapped subfolder (docs structure) or not (output structure)
                if (basePath.endsWith('/wrapped')) {{
                    // docs folder: go up to parent's index.html
                    link.href = basePath.replace('/wrapped', '') + '/index.html';
                }} else {{
                    // output folder: go to dashboard.html in same folder
                    link.href = basePath + '/dashboard.html';
                }}
            }});
        }}

        const heatmapData = {heatmap_json};
        const days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
        const maxVal = Math.max(...heatmapData.flat());

        function getHeatColor(value) {{
            if (value === 0) return 'bg-bg';
            const intensity = value / maxVal;
            if (intensity < 0.2) return 'bg-accent/20';
            if (intensity < 0.4) return 'bg-accent/40';
            if (intensity < 0.6) return 'bg-accent/60';
            if (intensity < 0.8) return 'bg-accent/80';
            return 'bg-accent';
        }}

        const heatmapContainer = document.getElementById('heatmap');
        heatmapData.forEach((row, dayIndex) => {{
            const rowDiv = document.createElement('div');
            rowDiv.className = 'flex items-center gap-[2px]';
            const dayLabel = document.createElement('span');
            dayLabel.className = 'w-12 text-xs text-muted shrink-0';
            dayLabel.textContent = days[dayIndex];
            rowDiv.appendChild(dayLabel);
            const cellsContainer = document.createElement('div');
            cellsContainer.className = 'flex-1 grid grid-cols-24 gap-[2px]';
            row.forEach((value, hourIndex) => {{
                const cell = document.createElement('div');
                cell.className = `heatmap-cell aspect-square rounded-sm ${{getHeatColor(value)}} cursor-pointer`;
                cell.title = `${{days[dayIndex]}} ${{hourIndex}}:00 - ${{value}} prompts`;
                cellsContainer.appendChild(cell);
            }});
            rowDiv.appendChild(cellsContainer);
            heatmapContainer.appendChild(rowDiv);
        }});

        const themeToggle = document.getElementById('themeToggle');
        const html = document.documentElement;
        if (localStorage.getItem('theme') === 'dark' || (!localStorage.getItem('theme') && window.matchMedia('(prefers-color-scheme: dark)').matches)) {{
            html.classList.add('dark');
        }}
        themeToggle.addEventListener('click', () => {{
            html.classList.toggle('dark');
            localStorage.setItem('theme', html.classList.contains('dark') ? 'dark' : 'light');
        }});

        function animateCounter(element, duration = 1500) {{
            const target = parseInt(element.dataset.target);
            if (isNaN(target)) return;
            const startTime = performance.now();
            function update(currentTime) {{
                const elapsed = currentTime - startTime;
                const progress = Math.min(elapsed / duration, 1);
                const easeOut = 1 - Math.pow(1 - progress, 3);
                const current = Math.floor(target * easeOut);
                element.textContent = current.toLocaleString();
                if (progress < 1) requestAnimationFrame(update);
                else element.textContent = target.toLocaleString();
            }}
            requestAnimationFrame(update);
        }}

        const animatedElements = new Set();
        const observer = new IntersectionObserver((entries) => {{
            entries.forEach(entry => {{
                if (entry.isIntersecting && !animatedElements.has(entry.target)) {{
                    animatedElements.add(entry.target);
                    entry.target.classList.add('animate-fade-in-up');
                    entry.target.querySelectorAll('.counter').forEach(counter => {{
                        if (counter.dataset.target) animateCounter(counter);
                    }});
                }}
            }});
        }}, {{ threshold: 0.3 }});
        document.querySelectorAll('section > div').forEach(el => observer.observe(el));
    </script>
</body>
</html>'''

    return html


# === Dashboard HTML Generation ===
def generate_dashboard_html(metrics: dict, branding: dict | None = None) -> str:
    """Generate single-page dashboard HTML from metrics."""

    source_views = metrics.get("source_views", {"both": metrics, "claude_code": None, "codex": None})
    # Backward-compat for older artifacts that used "claude" as a view key.
    if "claude_code" not in source_views and "claude" in source_views:
        source_views["claude_code"] = source_views["claude"]
    source_views.setdefault("both", metrics)
    source_views.setdefault("claude_code", None)
    source_views.setdefault("codex", None)

    default_source = metrics.get("default_view", "both")
    if default_source == "claude" and source_views.get("claude_code") is not None:
        default_source = "claude_code"
    if source_views.get(default_source) is None:
        for candidate in ("both", "claude_code", "codex"):
            if source_views.get(candidate) is not None:
                default_source = candidate
                break

    m = source_views.get(default_source) or metrics
    v = m["volume"]
    t = m["temporal"]
    pol = m["politeness"]
    back = m["backtrack"]
    q = m["question"]
    cmd = m["command"]
    yr = m["youre_right"]
    p = m["persona"]
    cd = m.get("conversation_depth", {})
    rr = m.get("response_ratio", 0)

    # Date range
    from datetime import datetime as dt
    dr = m.get("date_range", {})
    if dr:
        first_date = dt.fromisoformat(dr["first"]).strftime("%b %d, %Y")
        last_date = dt.fromisoformat(dr["last"]).strftime("%b %d, %Y")
        date_range_display = f"{first_date} – {last_date}"
    else:
        date_range_display = "2025"

    # Format peak hour
    peak_hour = t['peak_hour']
    peak_hour_12h = f"{peak_hour % 12 or 12}{'am' if peak_hour < 12 else 'pm'}"

    # Heatmap data for JS
    heatmap_json = json.dumps(t["heatmap"])
    source_views_json = json.dumps(source_views)
    default_source_json = json.dumps(default_source)

    # Conversation depth percentages
    total_convos = cd.get('quick_asks', 0) + cd.get('working_sessions', 0) + cd.get('deep_dives', 0)
    if total_convos > 0:
        quick_pct = round(cd.get('quick_asks', 0) / total_convos * 100)
        working_pct = round(cd.get('working_sessions', 0) / total_convos * 100)
        deep_pct = round(cd.get('deep_dives', 0) / total_convos * 100)
    else:
        quick_pct = working_pct = deep_pct = 0

    # Branding
    site_url = branding.get('site_url', 'https://eeshans.com') if branding else 'https://eeshans.com'
    site_name = branding.get('site_name', 'eeshans.com') if branding else 'eeshans.com'
    github_repo = branding.get('github_repo', 'https://github.com/eeshansrivastava89/howiprompt') if branding else 'https://github.com/eeshansrivastava89/howiprompt'
    newsletter_url = branding.get('newsletter_url', 'https://0to1datascience.substack.com') if branding else 'https://0to1datascience.substack.com'

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="Content-Security-Policy" content="upgrade-insecure-requests">
    <title>How I Prompt: Dashboard</title>

    <!-- Open Graph -->
    <meta property="og:title" content="How I Prompt: Dashboard">
    <meta property="og:description" content="{v['total_human']:,} prompts analyzed. A condensed view of AI conversation patterns.">
    <meta property="og:type" content="website">

    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <script src="https://cdn.tailwindcss.com"></script>
    <script>
        tailwind.config = {{
            corePlugins: {{
                preflight: false
            }}
        }};
    </script>

    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        :root {{
            --bg: #f5f5f7;
            --card: #ffffff;
            --text: #1d1d1f;
            --text-muted: #86868b;
            --border: #d2d2d7;
            --accent: #f97316;
            --accent-light: #fff7ed;
            --success: #34c759;
            --shadow: 0 1px 3px rgba(0,0,0,0.08), 0 4px 12px rgba(0,0,0,0.05);
            --shadow-hover: 0 4px 12px rgba(0,0,0,0.1), 0 8px 24px rgba(0,0,0,0.08);
            --radius: 16px;
            --radius-sm: 10px;
        }}

        .dark {{
            --bg: #000000;
            --card: #1c1c1e;
            --text: #f5f5f7;
            --text-muted: #86868b;
            --border: #38383a;
            --accent: #f97316;
            --accent-light: #1a1207;
            --shadow: 0 1px 3px rgba(0,0,0,0.3), 0 4px 12px rgba(0,0,0,0.2);
            --shadow-hover: 0 4px 12px rgba(0,0,0,0.4), 0 8px 24px rgba(0,0,0,0.3);
        }}

        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'Segoe UI', Roboto, sans-serif;
            background: var(--bg);
            color: var(--text);
            line-height: 1.5;
            min-height: 100vh;
            -webkit-font-smoothing: antialiased;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 24px 16px 48px;
        }}

        @media (min-width: 640px) {{
            .container {{
                padding: 32px 24px 64px;
            }}
        }}

        /* Header */
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 24px;
            flex-wrap: wrap;
            gap: 12px;
        }}

        .header-left {{
            display: flex;
            align-items: baseline;
            gap: 12px;
            flex-wrap: wrap;
        }}

        .header h1 {{
            font-size: 24px;
            font-weight: 700;
            letter-spacing: -0.02em;
        }}

        .header h1 span {{
            color: var(--accent);
        }}

        .date-range {{
            font-size: 14px;
            color: var(--text-muted);
            font-weight: 500;
        }}

        .header-right {{
            display: flex;
            align-items: center;
            gap: 12px;
        }}

        .theme-toggle {{
            width: 36px;
            height: 36px;
            border-radius: 50%;
            border: 1px solid var(--border);
            background: var(--card);
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.2s;
        }}

        .theme-toggle:hover {{
            border-color: var(--text-muted);
        }}

        .theme-toggle svg {{
            width: 18px;
            height: 18px;
            color: var(--text);
        }}

        .dark .theme-toggle .sun {{ display: none; }}
        .theme-toggle .moon {{ display: none; }}
        .dark .theme-toggle .moon {{ display: block; }}

        /* Grid */
        .grid {{
            display: grid;
            gap: 16px;
            grid-template-columns: 1fr;
        }}

        @media (min-width: 640px) {{
            .grid {{
                grid-template-columns: repeat(2, 1fr);
            }}
        }}

        @media (min-width: 1024px) {{
            .grid {{
                grid-template-columns: repeat(4, 1fr);
            }}
        }}

        /* Cards */
        .card {{
            background: var(--card);
            border-radius: var(--radius);
            padding: 20px;
            box-shadow: var(--shadow);
            transition: all 0.2s ease;
        }}

        .card:hover {{
            box-shadow: var(--shadow-hover);
            transform: translateY(-2px);
        }}

        .card-label {{
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            margin-bottom: 8px;
        }}

        .card-value {{
            font-size: 36px;
            font-weight: 700;
            letter-spacing: -0.02em;
            line-height: 1.1;
        }}

        .card-value.accent {{
            color: var(--accent);
        }}

        .card-subtitle {{
            font-size: 14px;
            color: var(--text-muted);
            margin-top: 4px;
        }}

        /* Stat cards row */
        .stat-cards {{
            display: grid;
            gap: 16px;
            grid-template-columns: repeat(2, 1fr);
            margin-bottom: 16px;
        }}

        @media (min-width: 640px) {{
            .stat-cards {{
                grid-template-columns: repeat(4, 1fr);
            }}
        }}

        /* Wide cards */
        .card-wide {{
            grid-column: span 1;
        }}

        @media (min-width: 640px) {{
            .card-wide {{
                grid-column: span 2;
            }}
        }}

        /* Heatmap */
        .heatmap-container {{
            overflow-x: auto;
            margin: 12px 0;
        }}

        .heatmap {{
            display: grid;
            grid-template-columns: 40px repeat(24, 1fr);
            gap: 3px;
            min-width: 500px;
        }}

        .heatmap-label {{
            font-size: 10px;
            color: var(--text-muted);
            display: flex;
            align-items: center;
            font-weight: 500;
        }}

        .heatmap-cell {{
            aspect-ratio: 1;
            border-radius: 3px;
            background: var(--border);
            min-width: 14px;
        }}

        .heatmap-cell.l1 {{ background: rgba(249, 115, 22, 0.2); }}
        .heatmap-cell.l2 {{ background: rgba(249, 115, 22, 0.4); }}
        .heatmap-cell.l3 {{ background: rgba(249, 115, 22, 0.6); }}
        .heatmap-cell.l4 {{ background: rgba(249, 115, 22, 0.8); }}
        .heatmap-cell.l5 {{ background: rgba(249, 115, 22, 1.0); }}

        .heatmap-hours {{
            display: contents;
        }}

        .heatmap-hours span {{
            font-size: 9px;
            color: var(--text-muted);
            text-align: center;
            font-weight: 500;
        }}

        .peak-stats {{
            display: flex;
            gap: 24px;
            margin-top: 16px;
            flex-wrap: wrap;
        }}

        .peak-stat {{
            display: flex;
            align-items: baseline;
            gap: 6px;
        }}

        .peak-stat-label {{
            font-size: 12px;
            color: var(--text-muted);
            font-weight: 500;
        }}

        .peak-stat-value {{
            font-size: 14px;
            font-weight: 600;
            color: var(--text);
        }}

        /* Progress bars */
        .progress-item {{
            margin-bottom: 12px;
        }}

        .progress-item:last-child {{
            margin-bottom: 0;
        }}

        .progress-header {{
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            margin-bottom: 6px;
        }}

        .progress-label {{
            font-size: 13px;
            font-weight: 500;
            color: var(--text);
        }}

        .progress-value {{
            font-size: 13px;
            font-weight: 600;
            color: var(--text-muted);
        }}

        .progress-bar {{
            height: 8px;
            background: var(--bg);
            border-radius: 4px;
            overflow: hidden;
        }}

        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, var(--accent), #fb923c);
            border-radius: 4px;
            transition: width 0.6s ease;
        }}

        /* Style metrics grid */
        .style-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 12px;
        }}

        .style-item {{
            background: var(--bg);
            border-radius: var(--radius-sm);
            padding: 14px;
            text-align: center;
        }}

        .dark .style-item {{
            background: rgba(255,255,255,0.05);
        }}

        .style-value {{
            font-size: 24px;
            font-weight: 700;
            color: var(--text);
            letter-spacing: -0.02em;
        }}

        .style-label {{
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            margin-top: 4px;
        }}

        .style-detail {{
            font-size: 11px;
            color: var(--text-muted);
            margin-top: 6px;
        }}

        /* Persona card */
        .persona-card {{
            grid-column: 1 / -1;
            display: grid;
            gap: 24px;
        }}

        @media (min-width: 768px) {{
            .persona-card {{
                grid-template-columns: 1fr 1fr;
            }}
        }}

        .persona-main {{
            display: flex;
            flex-direction: column;
        }}

        .persona-name {{
            font-size: 28px;
            font-weight: 700;
            letter-spacing: -0.02em;
            color: var(--accent);
            margin-bottom: 8px;
        }}

        .persona-desc {{
            font-size: 16px;
            color: var(--text-muted);
            margin-bottom: 16px;
            line-height: 1.5;
        }}

        .persona-traits {{
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
        }}

        .trait {{
            font-size: 12px;
            font-weight: 500;
            padding: 6px 12px;
            background: var(--accent-light);
            color: var(--accent);
            border-radius: 20px;
        }}

        .persona-stats {{
            display: flex;
            flex-direction: column;
            gap: 16px;
        }}

        .quadrant-scores {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
        }}

        .quadrant-item {{
            background: var(--bg);
            border-radius: var(--radius-sm);
            padding: 14px;
            text-align: center;
        }}

        .dark .quadrant-item {{
            background: rgba(255,255,255,0.05);
        }}

        .quadrant-value {{
            font-size: 28px;
            font-weight: 700;
            letter-spacing: -0.02em;
        }}

        .quadrant-label {{
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            margin-top: 4px;
        }}

        .quadrant-tag {{
            font-size: 10px;
            font-weight: 600;
            padding: 2px 8px;
            border-radius: 10px;
            margin-top: 6px;
            display: inline-block;
        }}

        .quadrant-tag.high {{
            background: rgba(52, 199, 89, 0.15);
            color: var(--success);
        }}

        .quadrant-tag.low {{
            background: rgba(134, 134, 139, 0.15);
            color: var(--text-muted);
        }}

        /* Signature metric */
        .signature {{
            background: var(--accent-light);
            border-radius: var(--radius-sm);
            padding: 16px;
            text-align: center;
        }}

        .signature-quote {{
            font-size: 16px;
            font-style: italic;
            color: var(--text);
            margin-bottom: 8px;
        }}

        .signature-count {{
            font-size: 32px;
            font-weight: 700;
            color: var(--accent);
        }}

        .signature-label {{
            font-size: 12px;
            color: var(--text-muted);
            margin-top: 4px;
        }}

        /* Footer */
        .footer {{
            margin-top: 32px;
            padding-top: 24px;
            border-top: 1px solid var(--border);
            display: flex;
            flex-direction: column;
            gap: 16px;
        }}

        @media (min-width: 640px) {{
            .footer {{
                flex-direction: row;
                justify-content: space-between;
                align-items: center;
            }}
        }}

        .footer-left {{
            font-size: 13px;
            color: var(--text-muted);
            text-align: center;
        }}

        @media (min-width: 640px) {{
            .footer-left {{
                text-align: left;
            }}
        }}

        .footer-center {{
            font-size: 12px;
            color: var(--text-muted);
            font-style: italic;
            text-align: center;
        }}

        .footer-right {{
            display: flex;
            gap: 16px;
            justify-content: center;
        }}

        .footer-link {{
            font-size: 13px;
            color: var(--text-muted);
            text-decoration: none;
            font-weight: 500;
            transition: color 0.2s;
            display: flex;
            align-items: center;
            gap: 6px;
        }}

        .footer-link:hover {{
            color: var(--accent);
        }}

        .footer-link svg {{
            width: 14px;
            height: 14px;
        }}

        /* Header nav links */
        .nav-link {{
            font-size: 14px;
            color: var(--text-muted);
            text-decoration: none;
            font-weight: 500;
            transition: color 0.2s;
            border: none;
            background: none;
            cursor: pointer;
            padding: 0;
        }}

        .nav-link:hover {{
            color: var(--text);
        }}

        .accent-link {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 6px 14px;
            font-size: 14px;
            font-weight: 500;
            color: var(--accent);
            background: rgba(249, 115, 22, 0.1);
            border-radius: 20px;
            text-decoration: none;
            transition: all 0.2s;
        }}

        .accent-link:hover {{
            background: rgba(249, 115, 22, 0.2);
        }}

        .accent-link svg {{
            width: 14px;
            height: 14px;
        }}

        /* Hide desktop nav on mobile, show mobile nav */
        .desktop-nav {{
            display: none;
        }}

        .mobile-nav {{
            display: flex;
            justify-content: center;
            gap: 24px;
            margin-bottom: 16px;
        }}

        @media (min-width: 640px) {{
            .desktop-nav {{
                display: flex;
            }}
            .mobile-nav {{
                display: none;
            }}
        }}

        .subscribe-btn {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 6px 14px;
            font-size: 13px;
            font-weight: 600;
            background: #FF6719;
            color: white;
            border-radius: 20px;
            text-decoration: none;
            transition: background 0.2s;
        }}

        .subscribe-btn:hover {{
            background: #E55A15;
        }}

        .subscribe-btn svg {{
            width: 14px;
            height: 14px;
        }}

        /* Methodology Modal */
        .modal-overlay {{
            position: fixed;
            inset: 0;
            z-index: 100;
            background: rgba(0, 0, 0, 0.5);
            backdrop-filter: blur(4px);
            display: none;
            align-items: center;
            justify-content: center;
            padding: 16px;
        }}

        .modal-overlay.active {{
            display: flex;
        }}

        .modal-content {{
            background: var(--card);
            border-radius: var(--radius);
            width: calc(100% - 32px);
            height: calc(100% - 32px);
            max-width: 1200px;
            max-height: 800px;
            overflow: hidden;
            display: flex;
            flex-direction: column;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
        }}

        @media (min-width: 768px) {{
            .modal-content {{
                width: calc(100% - 48px);
                height: calc(100% - 48px);
            }}
        }}

        @media (min-width: 1024px) {{
            .modal-content {{
                width: calc(100% - 64px);
                height: calc(100% - 64px);
            }}
        }}

        .modal-header {{
            padding: 16px 20px;
            border-bottom: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .modal-header h2 {{
            font-size: 18px;
            font-weight: 700;
        }}

        .modal-close {{
            width: 32px;
            height: 32px;
            border-radius: 50%;
            border: none;
            background: var(--bg);
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: background 0.2s;
        }}

        .modal-close:hover {{
            background: var(--border);
        }}

        .modal-close svg {{
            width: 18px;
            height: 18px;
            color: var(--text);
        }}

        .modal-body {{
            padding: 24px;
            overflow-y: auto;
            flex: 1;
        }}

        @media (min-width: 768px) {{
            .modal-body {{
                padding: 32px;
            }}
        }}

        .modal-grid {{
            display: grid;
            gap: 24px;
            grid-template-columns: 1fr;
            height: 100%;
        }}

        @media (min-width: 768px) {{
            .modal-grid {{
                grid-template-columns: repeat(3, 1fr);
                gap: 28px;
            }}
        }}

        .modal-section {{
            background: var(--bg);
            border-radius: var(--radius-sm);
            padding: 20px;
            display: flex;
            flex-direction: column;
        }}

        @media (min-width: 768px) {{
            .modal-section {{
                padding: 24px;
            }}
        }}

        .dark .modal-section {{
            background: rgba(255,255,255,0.05);
        }}

        .modal-section h3 {{
            font-size: 14px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            margin-bottom: 24px;
        }}

        .step-item {{
            display: flex;
            gap: 16px;
            margin-bottom: 24px;
        }}

        .step-item:last-child {{
            margin-bottom: 0;
        }}

        .step-num {{
            width: 32px;
            height: 32px;
            border-radius: 50%;
            background: var(--accent);
            color: white;
            font-size: 14px;
            font-weight: 700;
            display: flex;
            align-items: center;
            justify-content: center;
            flex-shrink: 0;
        }}

        .step-content h4 {{
            font-size: 16px;
            font-weight: 600;
            margin-bottom: 6px;
        }}

        .step-content p {{
            font-size: 14px;
            color: var(--text-muted);
            line-height: 1.6;
        }}

        .metric-group {{
            margin-bottom: 20px;
        }}

        .metric-group:last-child {{
            margin-bottom: 0;
        }}

        .metric-group h4 {{
            font-size: 12px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text);
            margin-bottom: 8px;
        }}

        .metric-group p {{
            font-size: 14px;
            color: var(--text-muted);
            line-height: 1.7;
        }}

        .metric-group span {{
            color: var(--text);
        }}

        .persona-matrix {{
            display: grid;
            grid-template-columns: auto 1fr 1fr;
            gap: 12px;
            font-size: 13px;
        }}

        .persona-cell {{
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 16px 12px;
            text-align: center;
        }}

        .persona-cell.highlight {{
            background: rgba(249, 115, 22, 0.1);
            border-color: var(--accent);
        }}

        .persona-cell strong {{
            display: block;
            font-size: 15px;
            font-weight: 700;
            margin-bottom: 6px;
        }}

        .persona-cell.highlight strong {{
            color: var(--accent);
        }}

        .matrix-label {{
            font-size: 12px;
            color: var(--text-muted);
            display: flex;
            align-items: center;
            justify-content: center;
            text-align: center;
        }}

        /* Animations */
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}

        .animate {{
            animation: fadeIn 0.4s ease forwards;
        }}

        .delay-1 {{ animation-delay: 0.05s; opacity: 0; }}
        .delay-2 {{ animation-delay: 0.1s; opacity: 0; }}
        .delay-3 {{ animation-delay: 0.15s; opacity: 0; }}
        .delay-4 {{ animation-delay: 0.2s; opacity: 0; }}
        .delay-5 {{ animation-delay: 0.25s; opacity: 0; }}
        .delay-6 {{ animation-delay: 0.3s; opacity: 0; }}
        .delay-7 {{ animation-delay: 0.35s; opacity: 0; }}
        .delay-8 {{ animation-delay: 0.4s; opacity: 0; }}
    </style>

    <!-- PostHog Analytics -->
    <script>
        !function(t,e){{var o,n,p,r;e.__SV||(window.posthog=e,e._i=[],e.init=function(i,s,a){{function g(t,e){{var o=e.split(".");2==o.length&&(t=t[o[0]],e=o[1]),t[e]=function(){{t.push([e].concat(Array.prototype.slice.call(arguments,0)))}}}}(p=t.createElement("script")).type="text/javascript",p.async=!0,p.src=s.api_host.replace(/\\/$/, "")+"/static/array.js",(r=t.getElementsByTagName("script")[0]).parentNode.insertBefore(p,r);var u=e;for(void 0!==a?u=e[a]=[]:a="posthog",u.people=u.people||[],u.toString=function(t){{var e="posthog";return"posthog"!==a&&(e+="."+a),t||(e+=" (stub)"),e}},u.people.toString=function(){{return u.toString(1)+".people (stub)"}},o="capture identify alias people.set people.set_once set_config register register_once unregister opt_out_capturing has_opted_out_capturing opt_in_capturing reset isFeatureEnabled onFeatureFlags getFeatureFlag getFeatureFlagPayload reloadFeatureFlags group updateEarlyAccessFeatureEnrollment getEarlyAccessFeatures getActiveMatchingSurveys onSessionId".split(" "),n=0;n<o.length;n++)g(u,o[n]);e._i.push([i,s,a])}},e.__SV=1)}}(document,window.posthog||[]);
        posthog.init('phc_zfue5Ca8VaxypRHPCi9j2h2R3Qy1eytEHt3TMPWlOOS',{{api_host:'https://api-v2.eeshans.com', ui_host:'https://us.posthog.com', person_profiles: 'identified_only'}})
    </script>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <header class="header animate">
            <div class="header-left">
                <h1>How I <span>Prompt</span></h1>
                <span class="date-range" id="dateRange">{date_range_display}</span>
            </div>
            <div class="header-right">
                <div class="relative">
                    <label for="sourceFilter" class="sr-only">Data source</label>
                    <select
                        id="sourceFilter"
                        class="appearance-none rounded-full border border-[var(--border)] bg-[var(--card)] px-3 py-1.5 pr-8 text-xs sm:text-sm font-medium text-[var(--text)] shadow-sm focus:outline-none focus:ring-2 focus:ring-orange-500"
                    >
                        <option value="both" {'selected' if default_source == 'both' else ''}>Both</option>
                        <option value="claude_code" {'selected' if default_source == 'claude_code' else ''}>Claude Code</option>
                        <option value="codex" {'selected' if default_source == 'codex' else ''}>Codex</option>
                    </select>
                    <svg class="pointer-events-none absolute right-2 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-[var(--text-muted)]" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
                        <path fill-rule="evenodd" d="M5.23 7.21a.75.75 0 011.06.02L10 11.157l3.71-3.928a.75.75 0 111.08 1.04l-4.25 4.5a.75.75 0 01-1.08 0l-4.25-4.5a.75.75 0 01.02-1.06z" clip-rule="evenodd" />
                    </svg>
                </div>
                <div class="desktop-nav" style="gap: 20px; align-items: center;">
                    <a href="/wrapped" class="accent-link">
                        <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 3v4M3 5h4M6 17v4m-2-2h4m5-16l2.286 6.857L21 12l-5.714 2.143L13 21l-2.286-6.857L5 12l5.714-2.143L13 3z"></path></svg>
                        Wrapped Experience
                    </a>
                    <button onclick="openMethodology()" class="nav-link">Methodology</button>
                    <a href="{github_repo}" target="_blank" class="nav-link">Build Your Own</a>
                </div>
                <a href="{newsletter_url}" target="_blank" class="subscribe-btn">
                    <svg viewBox="0 0 24 24" fill="currentColor"><path d="M22.539 8.242H1.46V5.406h21.08v2.836zM1.46 10.812V24L12 18.11 22.54 24V10.812H1.46zM22.54 0H1.46v2.836h21.08V0z"></path></svg>
                    Subscribe
                </a>
                <button class="theme-toggle" id="themeToggle" aria-label="Toggle theme">
                    <svg class="sun" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z"/>
                    </svg>
                    <svg class="moon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z"/>
                    </svg>
                </button>
            </div>
        </header>
        <!-- Mobile nav links -->
        <div class="mobile-nav animate">
            <div class="relative">
                <select
                    id="sourceFilterMobile"
                    class="appearance-none rounded-full border border-[var(--border)] bg-[var(--card)] px-3 py-1 pr-7 text-xs font-medium text-[var(--text)] shadow-sm focus:outline-none focus:ring-2 focus:ring-orange-500"
                >
                    <option value="both" {'selected' if default_source == 'both' else ''}>Both</option>
                    <option value="claude_code" {'selected' if default_source == 'claude_code' else ''}>Claude Code</option>
                    <option value="codex" {'selected' if default_source == 'codex' else ''}>Codex</option>
                </select>
                <svg class="pointer-events-none absolute right-2 top-1/2 h-3 w-3 -translate-y-1/2 text-[var(--text-muted)]" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
                    <path fill-rule="evenodd" d="M5.23 7.21a.75.75 0 011.06.02L10 11.157l3.71-3.928a.75.75 0 111.08 1.04l-4.25 4.5a.75.75 0 01-1.08 0l-4.25-4.5a.75.75 0 01.02-1.06z" clip-rule="evenodd" />
                </svg>
            </div>
            <a href="/wrapped" class="accent-link" style="padding: 4px 12px; font-size: 13px;">Wrapped</a>
            <button onclick="openMethodology()" class="nav-link">Methodology</button>
            <a href="{github_repo}" target="_blank" class="nav-link">Build Your Own</a>
        </div>

        <!-- Hero Stats -->
        <div class="stat-cards">
            <div class="card animate delay-1">
                <div class="card-label">Prompts</div>
                <div class="card-value" id="promptsValue">{v['total_human']:,}</div>
                <div class="card-subtitle" id="promptsSubtitle">{v['avg_words_per_prompt']} words avg</div>
            </div>
            <div class="card animate delay-2">
                <div class="card-label">Conversations</div>
                <div class="card-value" id="conversationsValue">{v['total_conversations']:,}</div>
                <div class="card-subtitle" id="conversationsSubtitle">{cd.get('avg_turns', 0)} turns avg</div>
            </div>
            <div class="card animate delay-3">
                <div class="card-label">Words Typed</div>
                <div class="card-value" id="wordsTypedValue">{v['total_words_human'] // 1000}K</div>
                <div class="card-subtitle" id="wordsTypedSubtitle">{v['total_words_assistant'] // 1000}K from assistants</div>
            </div>
            <div class="card animate delay-4">
                <div class="card-label">Night Owl</div>
                <div class="card-value accent" id="nightOwlValue">{t['night_owl_pct']}%</div>
                <div class="card-subtitle">prompts 11pm–4am</div>
            </div>
        </div>

        <!-- Main Grid -->
        <div class="grid">
            <!-- Heatmap -->
            <div class="card card-wide animate delay-5">
                <div class="card-label">Activity Heatmap</div>
                <div class="heatmap-container">
                    <div class="heatmap" id="heatmap">
                        <!-- Hours header -->
                        <div></div>
                        <div class="heatmap-hours">
                            <span>0</span><span></span><span></span><span>3</span><span></span><span></span>
                            <span>6</span><span></span><span></span><span>9</span><span></span><span></span>
                            <span>12</span><span></span><span></span><span>15</span><span></span><span></span>
                            <span>18</span><span></span><span></span><span>21</span><span></span><span></span>
                        </div>
                        <!-- Rows will be added by JS -->
                    </div>
                </div>
                <div class="peak-stats">
                    <div class="peak-stat">
                        <span class="peak-stat-label">Peak Hour</span>
                        <span class="peak-stat-value" id="peakHourValue">{peak_hour_12h}</span>
                    </div>
                    <div class="peak-stat">
                        <span class="peak-stat-label">Peak Day</span>
                        <span class="peak-stat-value" id="peakDayValue">{t['peak_day']}</span>
                    </div>
                    <div class="peak-stat">
                        <span class="peak-stat-label">Response Ratio</span>
                        <span class="peak-stat-value" id="responseRatioValue">{rr}x</span>
                    </div>
                </div>
            </div>

            <!-- Conversation Depth -->
            <div class="card card-wide animate delay-6">
                <div class="card-label">Conversation Depth</div>
                <div style="margin-top: 12px;">
                    <div class="progress-item">
                        <div class="progress-header">
                            <span class="progress-label">Quick Asks (1-3 turns)</span>
                            <span class="progress-value" id="quickAsksValue">{cd.get('quick_asks', 0)}</span>
                        </div>
                        <div class="progress-bar">
                            <div class="progress-fill" id="quickAsksFill" style="width: {quick_pct}%"></div>
                        </div>
                    </div>
                    <div class="progress-item">
                        <div class="progress-header">
                            <span class="progress-label">Working Sessions (4-10)</span>
                            <span class="progress-value" id="workingSessionsValue">{cd.get('working_sessions', 0)}</span>
                        </div>
                        <div class="progress-bar">
                            <div class="progress-fill" id="workingSessionsFill" style="width: {working_pct}%"></div>
                        </div>
                    </div>
                    <div class="progress-item">
                        <div class="progress-header">
                            <span class="progress-label">Deep Dives (11+)</span>
                            <span class="progress-value" id="deepDivesValue">{cd.get('deep_dives', 0)}</span>
                        </div>
                        <div class="progress-bar">
                            <div class="progress-fill" id="deepDivesFill" style="width: {deep_pct}%"></div>
                        </div>
                    </div>
                </div>
                <div class="peak-stats" style="margin-top: 16px;">
                    <div class="peak-stat">
                        <span class="peak-stat-label">Longest Session</span>
                        <span class="peak-stat-value" id="longestSessionValue">{cd.get('max_turns', 0)} turns</span>
                    </div>
                </div>
            </div>

            <!-- Prompt Style -->
            <div class="card card-wide animate delay-7">
                <div class="card-label">Prompt Style</div>
                <div class="style-grid" style="margin-top: 12px;">
                    <div class="style-item">
                        <div class="style-value" id="politenessValue">{pol['per_100_prompts']}</div>
                        <div class="style-label">Politeness</div>
                        <div class="style-detail" id="politenessDetail">please: {pol['counts']['please']} · thanks: {pol['counts']['thanks']}</div>
                    </div>
                    <div class="style-item">
                        <div class="style-value" id="backtrackValue">{back['per_100_prompts']}</div>
                        <div class="style-label">Backtrack</div>
                        <div class="style-detail" id="backtrackDetail">actually: {back['counts']['actually']} · wait: {back['counts']['wait']}</div>
                    </div>
                    <div class="style-item">
                        <div class="style-value" id="questionsValue">{q['rate']}%</div>
                        <div class="style-label">Questions</div>
                        <div class="style-detail" id="questionsDetail">{q['count']} total</div>
                    </div>
                    <div class="style-item">
                        <div class="style-value" id="commandsValue">{cmd['rate']}%</div>
                        <div class="style-label">Commands</div>
                        <div class="style-detail" id="commandsDetail">{cmd['count']} total</div>
                    </div>
                </div>
            </div>

            <!-- Persona -->
            <div class="card card-wide animate delay-8">
                <div class="card-label">Your AI Persona</div>
                <div class="persona-card" style="margin-top: 12px;">
                    <div class="persona-main">
                        <div class="persona-name" id="personaName">{p['name']}</div>
                        <div class="persona-desc" id="personaDescription">{p['description']}</div>
                        <div class="persona-traits" id="personaTraits">
                            {''.join(f'<span class="trait">{trait}</span>' for trait in p['traits'])}
                        </div>
                    </div>
                    <div class="persona-stats">
                        <div class="quadrant-scores">
                            <div class="quadrant-item">
                                <div class="quadrant-value" id="engagementScore">{p['quadrant']['engagement_score']}</div>
                                <div class="quadrant-label">Engagement</div>
                                <span class="quadrant-tag {'high' if p['quadrant']['high_engagement'] else 'low'}" id="engagementTag">
                                    {'High' if p['quadrant']['high_engagement'] else 'Low'}
                                </span>
                            </div>
                            <div class="quadrant-item">
                                <div class="quadrant-value" id="politenessScore">{p['quadrant']['politeness_score']}</div>
                                <div class="quadrant-label">Politeness</div>
                                <span class="quadrant-tag {'high' if p['quadrant']['high_politeness'] else 'low'}" id="politenessTag">
                                    {'High' if p['quadrant']['high_politeness'] else 'Low'}
                                </span>
                            </div>
                        </div>
                        <div class="signature">
                            <div class="signature-quote">"You're absolutely right."</div>
                            <div class="signature-count" id="youreRightCount">×{yr['count']}</div>
                            <div class="signature-label" id="youreRightLabel">{yr['per_conversation']}× per conversation</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Footer -->
        <footer class="footer">
            <div class="footer-left">© 2025 Eeshan Srivastava</div>
            <div class="footer-center">Personal project · MIT License · Non-commercial</div>
            <div class="footer-right">
                <a href="{github_repo}" target="_blank" class="footer-link">
                    <svg fill="currentColor" viewBox="0 0 24 24"><path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/></svg>
                    Source
                </a>
                <a href="https://www.linkedin.com/in/eeshans/" target="_blank" class="footer-link">
                    <svg fill="currentColor" viewBox="0 0 24 24"><path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433a2.06 2.06 0 01-2.063-2.065 2.064 2.064 0 112.063 2.065zm1.782 13.019H3.555V9h3.564zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0z"/></svg>
                    LinkedIn
                </a>
            </div>
        </footer>
    </div>

    <!-- Methodology Modal -->
    <div class="modal-overlay" id="methodologyModal">
        <div class="modal-content">
            <div class="modal-header">
                <h2>Methodology</h2>
                <button class="modal-close" onclick="closeMethodology()">
                    <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path>
                    </svg>
                </button>
            </div>
            <div class="modal-body">
                <div class="modal-grid">
                    <!-- How It Works -->
                    <div class="modal-section">
                        <h3>How It Works</h3>
                        <div class="step-item">
                            <div class="step-num">1</div>
                            <div class="step-content">
                                <h4>Data Collection</h4>
                                <p>Reads Claude Code JSONL logs and Codex history.jsonl from local machine paths.</p>
                            </div>
                        </div>
                        <div class="step-item">
                            <div class="step-num">2</div>
                            <div class="step-content">
                                <h4>Message Extraction</h4>
                                <p>Filters to human-authored prompts. Excludes system commands and tool outputs.</p>
                            </div>
                        </div>
                        <div class="step-item">
                            <div class="step-num">3</div>
                            <div class="step-content">
                                <h4>Metric Computation</h4>
                                <p>Regex pattern matching to count politeness markers, questions, and commands.</p>
                            </div>
                        </div>
                        <div class="step-item">
                            <div class="step-num">4</div>
                            <div class="step-content">
                                <h4>Persona Classification</h4>
                                <p>Maps Engagement and Politeness scores onto a 2×2 matrix.</p>
                            </div>
                        </div>
                    </div>

                    <!-- All Metrics -->
                    <div class="modal-section">
                        <h3>All Metrics</h3>
                        <div class="metric-group">
                            <h4>Volume</h4>
                            <p><span>Prompts</span> — your messages · <span>Conversations</span> — chat sessions · <span>Words</span> — total typed</p>
                        </div>
                        <div class="metric-group">
                            <h4>Conversation</h4>
                            <p><span>Avg Turns</span> — per convo · <span>Quick Asks</span> — 1-3 turns · <span>Deep Dives</span> — 11+ turns</p>
                        </div>
                        <div class="metric-group">
                            <h4>Temporal</h4>
                            <p><span>Peak Hour/Day</span> — most active · <span>Night Owl</span> — 11pm-4am %</p>
                        </div>
                        <div class="metric-group">
                            <h4>Style (per 100)</h4>
                            <p><span>Politeness</span> — please/thanks · <span>Backtrack</span> — actually/wait · <span>Questions</span> — ending ? · <span>Commands</span> — action verbs</p>
                        </div>
                        <div class="metric-group">
                            <h4>Composite Scores</h4>
                            <p><span>Engagement</span> = (Question + Backtrack) ÷ 2<br/><span>Politeness</span> = Index − (Command × 0.5)</p>
                        </div>
                    </div>

                    <!-- The 4 Personas -->
                    <div class="modal-section">
                        <h3>The 4 Personas</h3>
                        <p style="font-size: 14px; color: var(--text-muted); margin-bottom: 20px; line-height: 1.6;">Based on where you fall on the Engagement (x-axis) and Politeness (y-axis) scales:</p>
                        <div class="persona-matrix">
                            <div></div>
                            <div class="matrix-label">High Polite</div>
                            <div class="matrix-label">Low Polite</div>
                            <div class="matrix-label">High<br/>Engage</div>
                            <div class="persona-cell highlight">
                                <strong>Collaborator</strong>
                                You ask politely. AI is your partner, not your tool.
                            </div>
                            <div class="persona-cell">
                                <strong>Explorer</strong>
                                You question, iterate, and dig deeper. Thinking out loud.
                            </div>
                            <div class="matrix-label">Low<br/>Engage</div>
                            <div class="persona-cell">
                                <strong>Efficient</strong>
                                Polite but focused. You know what you want and ask nicely.
                            </div>
                            <div class="persona-cell">
                                <strong>Pragmatist</strong>
                                Balanced and practical. No frills, just results.
                            </div>
                        </div>
                        <p style="font-size: 13px; color: var(--text-muted); margin-top: 20px; border-top: 1px solid var(--border); padding-top: 16px;">Thresholds: Engagement ≥ 12, Politeness Score ≥ 4.5</p>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Fix wrapped link for local vs production
        const isLocal = location.protocol === 'file:' || location.hostname === 'localhost' || location.hostname === '127.0.0.1';
        if (isLocal) {{
            document.querySelectorAll('a[href="/wrapped"]').forEach(link => {{
                const currentPath = location.pathname;
                const basePath = currentPath.substring(0, currentPath.lastIndexOf('/'));
                const fileName = currentPath.substring(currentPath.lastIndexOf('/') + 1);
                // Check if we're dashboard.html (output structure) or index.html (docs structure)
                if (fileName === 'dashboard.html') {{
                    // output folder: go to index.html
                    link.href = basePath + '/index.html';
                }} else {{
                    // docs folder: go to wrapped/index.html
                    link.href = basePath + '/wrapped/index.html';
                }}
            }});
        }}

        // Theme toggle
        const html = document.documentElement;
        const themeToggle = document.getElementById('themeToggle');
        const savedTheme = localStorage.getItem('dashboard-theme');
        if (savedTheme === 'dark' || (!savedTheme && window.matchMedia('(prefers-color-scheme: dark)').matches)) {{
            html.classList.add('dark');
        }}
        themeToggle.addEventListener('click', () => {{
            html.classList.toggle('dark');
            localStorage.setItem('dashboard-theme', html.classList.contains('dark') ? 'dark' : 'light');
        }});

        // Methodology modal
        function openMethodology() {{
            document.getElementById('methodologyModal').classList.add('active');
            document.body.style.overflow = 'hidden';
        }}
        function closeMethodology() {{
            document.getElementById('methodologyModal').classList.remove('active');
            document.body.style.overflow = '';
        }}
        document.getElementById('methodologyModal').addEventListener('click', (e) => {{
            if (e.target === e.currentTarget) closeMethodology();
        }});
        document.addEventListener('keydown', (e) => {{
            if (e.key === 'Escape') closeMethodology();
        }});

        const sourceViews = {source_views_json};
        const defaultSource = {default_source_json};
        const sourceFilter = document.getElementById('sourceFilter');
        const sourceFilterMobile = document.getElementById('sourceFilterMobile');
        const sourceStorageKey = 'dashboard-source-filter';
        const fallbackHeatmap = {heatmap_json};

        function formatNumber(value) {{
            return (Number(value) || 0).toLocaleString();
        }}

        function formatCompactK(value) {{
            return `${{Math.round((Number(value) || 0) / 1000)}}K`;
        }}

        function formatHour12(hour) {{
            const h = Number(hour) || 0;
            return `${{h % 12 || 12}}${{h < 12 ? 'am' : 'pm'}}`;
        }}

        function formatDateRange(dateRange) {{
            if (!dateRange || !dateRange.first || !dateRange.last) return '2025';
            const first = new Date(dateRange.first);
            const last = new Date(dateRange.last);
            const opts = {{ month: 'short', day: '2-digit', year: 'numeric' }};
            return `${{first.toLocaleDateString('en-US', opts)}} – ${{last.toLocaleDateString('en-US', opts)}}`;
        }}

        function setTagState(el, isHigh) {{
            el.classList.remove('high', 'low');
            el.classList.add(isHigh ? 'high' : 'low');
            el.textContent = isHigh ? 'High' : 'Low';
        }}

        function renderTraits(traits) {{
            const container = document.getElementById('personaTraits');
            container.innerHTML = '';
            (traits || []).forEach((trait) => {{
                const chip = document.createElement('span');
                chip.className = 'trait';
                chip.textContent = trait;
                container.appendChild(chip);
            }});
        }}

        function renderHeatmap(heatmapData) {{
            const data = Array.isArray(heatmapData) ? heatmapData : fallbackHeatmap;
            const heatmap = document.getElementById('heatmap');
            const days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];

            heatmap.innerHTML = `
                <div></div>
                <div class="heatmap-hours">
                    <span>0</span><span></span><span></span><span>3</span><span></span><span></span>
                    <span>6</span><span></span><span></span><span>9</span><span></span><span></span>
                    <span>12</span><span></span><span></span><span>15</span><span></span><span></span>
                    <span>18</span><span></span><span></span><span>21</span><span></span><span></span>
                </div>
            `;

            const maxVal = Math.max(1, ...data.flat());
            days.forEach((day, dayIndex) => {{
                const label = document.createElement('div');
                label.className = 'heatmap-label';
                label.textContent = day;
                heatmap.appendChild(label);

                (data[dayIndex] || []).forEach((value) => {{
                    const cell = document.createElement('div');
                    cell.className = 'heatmap-cell';
                    if (value > 0) {{
                        const intensity = Math.ceil((value / maxVal) * 5);
                        cell.classList.add('l' + Math.min(intensity, 5));
                    }}
                    heatmap.appendChild(cell);
                }});
            }});
        }}

        function getView(sourceKey) {{
            return sourceViews[sourceKey] || null;
        }}

        function renderView(sourceKey) {{
            const view = getView(sourceKey);
            if (!view) return;

            const volume = view.volume || {{}};
            const temporal = view.temporal || {{}};
            const convo = view.conversation_depth || {{}};
            const politeness = view.politeness || {{}};
            const backtrack = view.backtrack || {{}};
            const question = view.question || {{}};
            const command = view.command || {{}};
            const persona = view.persona || {{}};
            const quadrant = persona.quadrant || {{}};
            const youreRight = view.youre_right || {{}};
            const responseRatio = view.response_ratio || 0;

            document.getElementById('dateRange').textContent = formatDateRange(view.date_range);

            document.getElementById('promptsValue').textContent = formatNumber(volume.total_human);
            document.getElementById('promptsSubtitle').textContent = `${{volume.avg_words_per_prompt ?? 0}} words avg`;

            document.getElementById('conversationsValue').textContent = formatNumber(volume.total_conversations);
            document.getElementById('conversationsSubtitle').textContent = `${{convo.avg_turns ?? 0}} turns avg`;

            document.getElementById('wordsTypedValue').textContent = formatCompactK(volume.total_words_human);
            document.getElementById('wordsTypedSubtitle').textContent = `${{formatCompactK(volume.total_words_assistant)}} from assistants`;

            document.getElementById('nightOwlValue').textContent = `${{temporal.night_owl_pct ?? 0}}%`;
            document.getElementById('peakHourValue').textContent = formatHour12(temporal.peak_hour);
            document.getElementById('peakDayValue').textContent = temporal.peak_day || 'N/A';
            document.getElementById('responseRatioValue').textContent = `${{responseRatio}}x`;

            const quick = convo.quick_asks || 0;
            const working = convo.working_sessions || 0;
            const deep = convo.deep_dives || 0;
            const totalConvos = quick + working + deep;
            const quickPct = totalConvos ? Math.round((quick / totalConvos) * 100) : 0;
            const workingPct = totalConvos ? Math.round((working / totalConvos) * 100) : 0;
            const deepPct = totalConvos ? Math.round((deep / totalConvos) * 100) : 0;

            document.getElementById('quickAsksValue').textContent = quick;
            document.getElementById('workingSessionsValue').textContent = working;
            document.getElementById('deepDivesValue').textContent = deep;
            document.getElementById('quickAsksFill').style.width = `${{quickPct}}%`;
            document.getElementById('workingSessionsFill').style.width = `${{workingPct}}%`;
            document.getElementById('deepDivesFill').style.width = `${{deepPct}}%`;
            document.getElementById('longestSessionValue').textContent = `${{convo.max_turns || 0}} turns`;

            const politenessCounts = politeness.counts || {{}};
            const backtrackCounts = backtrack.counts || {{}};
            document.getElementById('politenessValue').textContent = politeness.per_100_prompts ?? 0;
            document.getElementById('politenessDetail').textContent = `please: ${{politenessCounts.please || 0}} · thanks: ${{politenessCounts.thanks || 0}}`;
            document.getElementById('backtrackValue').textContent = backtrack.per_100_prompts ?? 0;
            document.getElementById('backtrackDetail').textContent = `actually: ${{backtrackCounts.actually || 0}} · wait: ${{backtrackCounts.wait || 0}}`;
            document.getElementById('questionsValue').textContent = `${{question.rate ?? 0}}%`;
            document.getElementById('questionsDetail').textContent = `${{question.count || 0}} total`;
            document.getElementById('commandsValue').textContent = `${{command.rate ?? 0}}%`;
            document.getElementById('commandsDetail').textContent = `${{command.count || 0}} total`;

            document.getElementById('personaName').textContent = persona.name || 'No Persona';
            document.getElementById('personaDescription').textContent = persona.description || 'Not enough data for persona classification.';
            renderTraits(persona.traits || []);
            document.getElementById('engagementScore').textContent = quadrant.engagement_score ?? 0;
            document.getElementById('politenessScore').textContent = quadrant.politeness_score ?? 0;
            setTagState(document.getElementById('engagementTag'), Boolean(quadrant.high_engagement));
            setTagState(document.getElementById('politenessTag'), Boolean(quadrant.high_politeness));

            document.getElementById('youreRightCount').textContent = `×${{youreRight.count || 0}}`;
            document.getElementById('youreRightLabel').textContent = `${{youreRight.per_conversation ?? 0}}× per conversation`;

            renderHeatmap(temporal.heatmap);
        }}

        function syncSourceSelectors(value) {{
            sourceFilter.value = value;
            sourceFilterMobile.value = value;
        }}

        function initSourceFilter() {{
            const available = ['both', 'claude_code', 'codex'].filter((key) => Boolean(getView(key)));
            [sourceFilter, sourceFilterMobile].forEach((select) => {{
                Array.from(select.options).forEach((option) => {{
                    option.disabled = !available.includes(option.value);
                }});
            }});

            let selected = localStorage.getItem(sourceStorageKey) || defaultSource;
            if (selected === 'claude' && available.includes('claude_code')) {{
                selected = 'claude_code';
            }}
            if (!available.includes(selected)) {{
                selected = available[0] || 'both';
            }}

            syncSourceSelectors(selected);
            renderView(selected);

            const handleChange = (event) => {{
                const next = event.target.value;
                if (!available.includes(next)) return;
                syncSourceSelectors(next);
                localStorage.setItem(sourceStorageKey, next);
                renderView(next);
            }};

            sourceFilter.addEventListener('change', handleChange);
            sourceFilterMobile.addEventListener('change', handleChange);
        }}

        initSourceFilter();
    </script>
</body>
</html>'''

    return html


# === Main Build Pipeline ===
def copy_claude_code_data(dest_path: Path) -> bool:
    """Copy Claude Code data from ~/.claude/projects to data/claude_code/."""
    source_path = Path.home() / ".claude" / "projects"

    if not source_path.exists():
        print(f"  Error: Claude Code data not found at {source_path}")
        return False

    # Count files to copy
    files_to_copy = list(source_path.rglob("*.jsonl"))
    if not files_to_copy:
        print(f"  Error: No .jsonl files found in {source_path}")
        return False

    print(f"  Found {len(files_to_copy)} conversation files")

    # Copy directory structure
    for jsonl_file in files_to_copy:
        rel_path = jsonl_file.relative_to(source_path)
        dest_file = dest_path / rel_path
        dest_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(jsonl_file, dest_file)

    print(f"  Copied to {dest_path}")
    return True


def copy_codex_data(dest_path: Path) -> bool:
    """Copy Codex history from ~/.codex/history.jsonl to data/codex/history.jsonl."""
    source_path = Path.home() / ".codex" / "history.jsonl"

    if not source_path.exists():
        print(f"  Error: Codex history not found at {source_path}")
        return False

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, dest_path)
    print(f"  Copied to {dest_path}")
    return True


def open_in_browser(path: Path) -> bool:
    """Open a local HTML file in the default browser."""
    resolved = path.resolve()
    try:
        # Prefer native open command on macOS.
        if sys.platform == "darwin" and shutil.which("open"):
            result = subprocess.run(
                ["open", str(resolved)],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return result.returncode == 0

        # Fallback to Python's browser integration on other platforms.
        return webbrowser.open(resolved.as_uri())
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(
        description="How I Prompt Wrapped 2025 - Build System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python build.py                    # Full build
  python build.py --metrics-only     # Only metrics.json
  python build.py --no-open          # Build without opening dashboard in browser
  python build.py --skip-copy-codex  # Skip Codex auto-sync for this run
        """
    )
    parser.add_argument("--metrics-only", action="store_true", help="Only compute metrics, skip HTML")
    parser.add_argument("--output", "-o", type=Path, help="Output directory")
    parser.add_argument("--copy-claude-code", action="store_true", help=argparse.SUPPRESS)  # legacy alias
    parser.add_argument("--copy-codex", action="store_true", help=argparse.SUPPRESS)  # legacy alias
    parser.add_argument("--skip-copy-claude-code", action="store_true", help="Skip Claude Code auto-sync from ~/.claude/projects")
    parser.add_argument("--skip-copy-codex", action="store_true", help="Skip Codex auto-sync from ~/.codex/history.jsonl")
    parser.add_argument("--no-open", action="store_true", help="Do not auto-open dashboard HTML in browser")
    args = parser.parse_args()

    config = load_config()
    output_dir = args.output or config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    auto_sync_claude = (not args.skip_copy_claude_code) or args.copy_claude_code
    auto_sync_codex = (not args.skip_copy_codex) or args.copy_codex

    # Step 0: Sync local source data by default.
    print("\n[0/3] Syncing local data sources...")
    if auto_sync_claude:
        print("  Claude Code...")
        if not copy_claude_code_data(config.claude_code_path):
            print("  Warning: Claude Code sync failed; continuing with existing data.")
    else:
        print("  Skipped Claude Code sync")

    if auto_sync_codex:
        print("  Codex...")
        if not copy_codex_data(config.codex_history_path):
            print("  Warning: Codex sync failed; continuing with existing data.")
    else:
        print("  Skipped Codex sync")

    print("=" * 50)
    print("How I Prompt Wrapped 2025 - Build")
    print("=" * 50)

    # Step 1: Parse data
    print("\n[1/3] Parsing data sources...")
    messages = []
    messages.extend(parse_claude_code(config.claude_code_path))
    messages.extend(parse_codex_history(config.codex_history_path))
    messages.sort(key=lambda m: m.timestamp)
    print(f"  Total: {len(messages)} messages")

    if not messages:
        print("\nError: No messages found. Check your data paths.")
        sys.exit(1)

    # Step 2: Compute metrics
    print("\n[2/3] Computing metrics...")
    source_views, source_defaults = compute_source_views(messages, config)
    dashboard_default_view = source_defaults["default_view"]
    wrapped_base_view = "claude_code" if source_views.get("claude_code") is not None else dashboard_default_view

    metrics = dict(source_views[wrapped_base_view])
    metrics["source_views"] = source_views
    metrics["default_view"] = dashboard_default_view

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"  Saved: {metrics_path}")

    if args.metrics_only:
        print("\n[Done] Metrics-only build complete.")
        return

    # Step 3: Generate HTML
    print("\n[3/3] Generating HTML...")
    branding = load_branding()
    if branding:
        print(f"  Branding: {branding['site_name']}")

    # Main experience
    html = generate_html(metrics, branding)
    html_path = output_dir / "index.html"
    with open(html_path, 'w') as f:
        f.write(html)
    print(f"  Saved: {html_path}")

    # Dashboard (single-page view)
    dashboard_html = generate_dashboard_html(metrics, branding)
    dashboard_path = output_dir / "dashboard.html"
    with open(dashboard_path, 'w') as f:
        f.write(dashboard_html)
    print(f"  Saved: {dashboard_path}")

    # Summary
    print("\n" + "=" * 50)
    print("BUILD COMPLETE")
    print("=" * 50)
    print(f"\nPersona: {metrics['persona']['name']}")
    print(f"  {metrics['persona']['description']}")
    print(f"\nQuadrant scores:")
    print(f"  Engagement: {metrics['persona']['quadrant']['engagement_score']} ({'High' if metrics['persona']['quadrant']['high_engagement'] else 'Low'})")
    print(f"  Politeness: {metrics['persona']['quadrant']['politeness_score']} ({'High' if metrics['persona']['quadrant']['high_politeness'] else 'Low'})")
    print(f"\nOutput:")
    print(f"  {metrics_path}")
    print(f"  {html_path}")
    print(f"  {dashboard_path}")

    # Copy to docs/ for GitHub Pages
    # Dashboard → main page (howiprompt.eeshans.com)
    # Full experience → /wrapped (howiprompt.eeshans.com/wrapped)
    docs_dir = PROJECT_ROOT / "docs"
    wrapped_dir = docs_dir / "wrapped"
    wrapped_dir.mkdir(parents=True, exist_ok=True)

    if docs_dir.exists():
        # Dashboard becomes the main index
        shutil.copy2(dashboard_path, docs_dir / "index.html")
        print(f"  {docs_dir / 'index.html'} (Dashboard → howiprompt.eeshans.com)")

        # Full experience goes to /wrapped
        shutil.copy2(html_path, wrapped_dir / "index.html")
        print(f"  {wrapped_dir / 'index.html'} (Full → /wrapped)")

    if args.no_open:
        print("\n[4/4] Skipped opening browser (--no-open).")
    else:
        print("\n[4/4] Opening dashboard...")
        if open_in_browser(dashboard_path):
            print(f"  Opened: {dashboard_path}")
        else:
            print(f"  Could not open automatically: {dashboard_path}")


if __name__ == "__main__":
    main()
