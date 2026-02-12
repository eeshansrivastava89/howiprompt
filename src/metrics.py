"""Metric computation: volume, depth, temporal, style, and source views."""

import re
from collections import Counter, defaultdict
from datetime import datetime, timezone

from .config import Config
from .models import Message, Platform, Role, SOURCE_VIEW_LABELS
from .nlp import BACKTRACK_PATTERNS, COMMAND_PATTERN, POLITENESS_PATTERNS, compute_nlp_metrics
from .persona import classify_persona
from .trends import compute_trend_metrics


def compute_model_usage(human_msgs: list[Message]) -> dict:
    """Aggregate model usage across source and time for human prompts."""
    total_human = len(human_msgs)
    model_msgs = [m for m in human_msgs if m.model_id or m.model_provider]
    with_metadata = len(model_msgs)
    coverage_pct = round((with_metadata / total_human * 100), 1) if total_human else 0.0

    by_model: dict[tuple[str, str], dict] = {}
    model_conversations: dict[tuple[str, str], set[str]] = defaultdict(set)
    by_provider: dict[str, dict] = {}
    provider_conversations: dict[str, set[str]] = defaultdict(set)
    by_source_date: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(int))
    )

    for msg in model_msgs:
        model_id = msg.model_id or "unknown"
        provider = msg.model_provider or "unknown"
        source = msg.platform.value
        date_key = msg.timestamp.astimezone().date().isoformat()

        model_key = (model_id, provider)
        model_stats = by_model.setdefault(
            model_key,
            {
                "model_id": model_id,
                "model_provider": provider,
                "prompts": 0,
                "words": 0,
                "sources": defaultdict(int),
            },
        )
        model_stats["prompts"] += 1
        model_stats["words"] += msg.word_count
        model_stats["sources"][source] += 1
        model_conversations[model_key].add(msg.conversation_id)

        provider_stats = by_provider.setdefault(
            provider,
            {"model_provider": provider, "prompts": 0, "words": 0, "sources": defaultdict(int)},
        )
        provider_stats["prompts"] += 1
        provider_stats["words"] += msg.word_count
        provider_stats["sources"][source] += 1
        provider_conversations[provider].add(msg.conversation_id)

        by_source_date[source][date_key][model_id] += 1

    by_model_list = []
    for model_key, stats in by_model.items():
        by_model_list.append(
            {
                "model_id": stats["model_id"],
                "model_provider": stats["model_provider"],
                "prompts": stats["prompts"],
                "words": stats["words"],
                "conversations": len(model_conversations[model_key]),
                "sources": dict(sorted(stats["sources"].items())),
            }
        )
    by_model_list.sort(key=lambda item: (-item["prompts"], item["model_id"], item["model_provider"]))

    by_provider_list = []
    for provider, stats in by_provider.items():
        by_provider_list.append(
            {
                "model_provider": provider,
                "prompts": stats["prompts"],
                "words": stats["words"],
                "conversations": len(provider_conversations[provider]),
                "sources": dict(sorted(stats["sources"].items())),
            }
        )
    by_provider_list.sort(key=lambda item: (-item["prompts"], item["model_provider"]))

    time_series_by_source = {}
    for source, day_map in by_source_date.items():
        source_series = []
        for date_key in sorted(day_map.keys()):
            model_counts = day_map[date_key]
            source_series.append(
                {
                    "date": date_key,
                    "total_prompts": sum(model_counts.values()),
                    "models": dict(sorted(model_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
                }
            )
        time_series_by_source[source] = source_series

    return {
        "coverage": {
            "total_human_prompts": total_human,
            "prompts_with_model_metadata": with_metadata,
            "metadata_coverage_pct": coverage_pct,
        },
        "by_model": by_model_list,
        "by_provider": by_provider_list,
        "time_series_by_source": time_series_by_source,
    }


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
    politeness_counts = {}
    for word, pattern in POLITENESS_PATTERNS.items():
        count = sum(
            len(re.findall(pattern, m.content, re.IGNORECASE))
            for m in human_msgs
        )
        politeness_counts[word] = count

    total_politeness = sum(politeness_counts.values())
    politeness_per_100 = (total_politeness / total_human * 100)

    # === Backtrack Index ===
    backtrack_counts = {}
    for word, pattern in BACKTRACK_PATTERNS.items():
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
    command_count = sum(
        1 for m in human_msgs
        if re.match(COMMAND_PATTERN, m.content.strip(), re.IGNORECASE)
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
    model_usage = compute_model_usage(human_msgs)
    trends = compute_trend_metrics(human_msgs)
    nlp = compute_nlp_metrics(human_msgs)

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
        "model_usage": model_usage,
        "trends": trends,
        "nlp": nlp,
        "date_range": {
            "first": first_msg.isoformat(),
            "last": last_msg.isoformat(),
        },
    }


def format_date_range_display(date_range: dict | None) -> str:
    """Return a short date range string for display."""
    if not date_range or not date_range.get("first") or not date_range.get("last"):
        return "n/a"

    first_date = datetime.fromisoformat(date_range["first"]).strftime("%b %d, %Y")
    last_date = datetime.fromisoformat(date_range["last"]).strftime("%b %d, %Y")
    return f"{first_date} - {last_date}"


def build_launch_packet(view: dict, source_key: str, github_repo: str, site_url: str) -> dict:
    """Create copy-ready launch text for summary, release notes, HN, and LinkedIn."""
    source_label = SOURCE_VIEW_LABELS.get(source_key, source_key)
    volume = view.get("volume", {})
    persona = view.get("persona", {})
    model_usage = view.get("model_usage", {})
    platform_stats = view.get("platform_stats", {})
    nlp = view.get("nlp", {})
    date_range = format_date_range_display(view.get("date_range"))

    total_prompts = int(volume.get("total_human", 0) or 0)
    total_conversations = int(volume.get("total_conversations", 0) or 0)
    avg_words = float(volume.get("avg_words_per_prompt", 0) or 0)

    codex_prompts = int((platform_stats.get("codex") or {}).get("messages", 0) or 0)
    codex_share_pct = round((codex_prompts / total_prompts) * 100, 1) if total_prompts else 0.0

    top_model = None
    by_model = model_usage.get("by_model", [])
    if by_model:
        top_model = by_model[0].get("model_id")
    if not top_model:
        top_model = "n/a"

    complexity_avg = float((nlp.get("complexity") or {}).get("avg_score", 0.0) or 0.0)
    iteration_style = (nlp.get("iteration_style") or {}).get("style", "n/a")
    persona_name = persona.get("name", "n/a")

    summary = "\n".join(
        [
            f"How I Prompt v2 snapshot ({source_label})",
            f"Date range: {date_range}",
            f"Prompts: {total_prompts:,} across {total_conversations:,} conversations",
            f"Persona: {persona_name}",
            f"Avg prompt length: {avg_words:.1f} words | Codex share: {codex_share_pct}%",
            f"Top model: {top_model} | Iteration style: {iteration_style} | Complexity: {complexity_avg:.1f}/5",
            f"Source + build: {github_repo}",
        ]
    )

    release_notes = "\n".join(
        [
            "How I Prompt v2 - Release Notes",
            "",
            f"Scope view: {source_label}",
            f"- Date range covered: {date_range}",
            "- Active data sources: Claude Code + Codex",
            "- Added model attribution from Codex session metadata",
            "- Added trend analytics (daily/weekly rollups, 7d vs 30d deltas, shift markers)",
            "- Added deterministic NLP metrics with confidence (intent, complexity, iteration style)",
            "- Added dashboard launch-kit sharing actions (summary + HN + LinkedIn copy)",
            "",
            "Migration note: Claude.ai exports are deprecated and ignored by the v2 pipeline.",
            f"GitHub: {github_repo}",
            f"Live: {site_url}",
        ]
    )

    hn_post = "\n".join(
        [
            "Show HN: How I Prompt v2 (local analytics for Claude Code + Codex)",
            "",
            f"I built a local-first dashboard to analyze my prompting patterns ({source_label}).",
            f"It currently covers {total_prompts:,} prompts across {total_conversations:,} conversations ({date_range}).",
            "v2 adds model attribution, trend analytics, and deterministic NLP metrics.",
            "Migration note: Claude.ai exports are deprecated in v2; active pipeline is Claude Code + Codex.",
            f"Repo: {github_repo}",
            f"Live demo: {site_url}",
            "",
            "Would love feedback on which metrics are most useful for daily AI workflows.",
        ]
    )

    linkedin_post = "\n".join(
        [
            f"Shipped How I Prompt v2 for {source_label}.",
            f"Analyzed {total_prompts:,} prompts across {total_conversations:,} conversations ({date_range}).",
            f"Highlights: model attribution, trend analytics, deterministic NLP metrics, and a new launch sharing kit.",
            "Migration note: Claude.ai exports are deprecated in v2; active sources are Claude Code + Codex.",
            f"Open source: {github_repo}",
            f"Live demo: {site_url}",
            "",
            "#buildinpublic #ai #developertools #analytics",
        ]
    )

    return {
        "source_label": source_label,
        "summary": summary,
        "release_notes": release_notes,
        "hn_post": hn_post,
        "linkedin_post": linkedin_post,
        "attribution_url": github_repo,
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
