"""Metric computation: volume, depth, temporal, style, and source views."""

import re
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone

from .config import Config
from .db import platform_filter, query_messages
from .models import Message, Platform, Role
from .nlp import BACKTRACK_PATTERNS, COMMAND_PATTERN, POLITENESS_PATTERNS, compute_nlp_metrics
from .persona import classify_persona
from .trends import compute_trend_metrics


def compute_model_usage(conn: sqlite3.Connection, platform: Platform | None = None) -> dict:
    """Aggregate model usage across source and time for human prompts."""
    pf, pp = platform_filter(platform)

    # Coverage
    total_human = conn.execute(
        f"SELECT COUNT(*) FROM messages WHERE role = 'human'{pf}", pp,
    ).fetchone()[0]
    with_metadata = conn.execute(
        f"SELECT COUNT(*) FROM messages"
        f" WHERE role = 'human' AND (model_id IS NOT NULL OR model_provider IS NOT NULL){pf}",
        pp,
    ).fetchone()[0]
    coverage_pct = round((with_metadata / total_human * 100), 1) if total_human else 0.0

    # --- By model (with per-platform source breakdown) ---
    model_source_rows = conn.execute(
        f"SELECT COALESCE(model_id, 'unknown'), COALESCE(model_provider, 'unknown'),"
        f" platform, COUNT(*), SUM(word_count)"
        f" FROM messages"
        f" WHERE role = 'human' AND (model_id IS NOT NULL OR model_provider IS NOT NULL){pf}"
        f" GROUP BY 1, 2, platform",
        pp,
    ).fetchall()

    by_model_map: dict[tuple, dict] = {}
    for mid, mprov, plat, prompts, words in model_source_rows:
        key = (mid, mprov)
        if key not in by_model_map:
            by_model_map[key] = {
                "model_id": mid, "model_provider": mprov,
                "prompts": 0, "words": 0, "sources": {},
            }
        by_model_map[key]["prompts"] += prompts
        by_model_map[key]["words"] += words
        by_model_map[key]["sources"][plat] = prompts

    # Conversation counts per model
    conv_rows = conn.execute(
        f"SELECT COALESCE(model_id, 'unknown'), COALESCE(model_provider, 'unknown'),"
        f" COUNT(DISTINCT conversation_id)"
        f" FROM messages"
        f" WHERE role = 'human' AND (model_id IS NOT NULL OR model_provider IS NOT NULL){pf}"
        f" GROUP BY 1, 2",
        pp,
    ).fetchall()
    conv_map = {(r[0], r[1]): r[2] for r in conv_rows}

    by_model_list = [
        {
            "model_id": stats["model_id"],
            "model_provider": stats["model_provider"],
            "prompts": stats["prompts"],
            "words": stats["words"],
            "conversations": conv_map.get(key, 0),
            "sources": dict(sorted(stats["sources"].items())),
        }
        for key, stats in by_model_map.items()
    ]
    by_model_list.sort(key=lambda x: (-x["prompts"], x["model_id"], x["model_provider"]))

    # --- By provider ---
    prov_source_rows = conn.execute(
        f"SELECT COALESCE(model_provider, 'unknown'), platform,"
        f" COUNT(*), SUM(word_count)"
        f" FROM messages"
        f" WHERE role = 'human' AND (model_id IS NOT NULL OR model_provider IS NOT NULL){pf}"
        f" GROUP BY 1, platform",
        pp,
    ).fetchall()

    by_provider_map: dict[str, dict] = {}
    for mprov, plat, prompts, words in prov_source_rows:
        if mprov not in by_provider_map:
            by_provider_map[mprov] = {"model_provider": mprov, "prompts": 0, "words": 0, "sources": {}}
        by_provider_map[mprov]["prompts"] += prompts
        by_provider_map[mprov]["words"] += words
        by_provider_map[mprov]["sources"][plat] = prompts

    prov_conv_rows = conn.execute(
        f"SELECT COALESCE(model_provider, 'unknown'), COUNT(DISTINCT conversation_id)"
        f" FROM messages"
        f" WHERE role = 'human' AND (model_id IS NOT NULL OR model_provider IS NOT NULL){pf}"
        f" GROUP BY 1",
        pp,
    ).fetchall()
    prov_conv_map = {r[0]: r[1] for r in prov_conv_rows}

    by_provider_list = [
        {
            "model_provider": stats["model_provider"],
            "prompts": stats["prompts"],
            "words": stats["words"],
            "conversations": prov_conv_map.get(mprov, 0),
            "sources": dict(sorted(stats["sources"].items())),
        }
        for mprov, stats in by_provider_map.items()
    ]
    by_provider_list.sort(key=lambda x: (-x["prompts"], x["model_provider"]))

    # --- Time series by source ---
    ts_rows = conn.execute(
        f"SELECT platform, local_date, COALESCE(model_id, 'unknown'), COUNT(*)"
        f" FROM messages"
        f" WHERE role = 'human' AND (model_id IS NOT NULL OR model_provider IS NOT NULL){pf}"
        f" GROUP BY platform, local_date, 3"
        f" ORDER BY platform, local_date",
        pp,
    ).fetchall()

    by_source_date: dict[str, dict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(dict))
    for plat, date, mid, cnt in ts_rows:
        by_source_date[plat][date][mid] = cnt

    time_series_by_source = {}
    for source, day_map in by_source_date.items():
        source_series = []
        for date_key in sorted(day_map.keys()):
            model_counts = day_map[date_key]
            source_series.append({
                "date": date_key,
                "total_prompts": sum(model_counts.values()),
                "models": dict(sorted(model_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
            })
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


def compute_metrics(conn: sqlite3.Connection, config: Config, platform: Platform | None = None) -> dict:
    """Compute all metrics from the database."""
    pf, pp = platform_filter(platform)

    # === Volume Metrics (SQL) ===
    vol = conn.execute(
        f"SELECT"
        f" COUNT(*),"
        f" SUM(CASE WHEN role = 'human' THEN 1 ELSE 0 END),"
        f" SUM(CASE WHEN role = 'assistant' THEN 1 ELSE 0 END),"
        f" SUM(CASE WHEN role = 'human' THEN word_count ELSE 0 END),"
        f" SUM(CASE WHEN role = 'assistant' THEN word_count ELSE 0 END),"
        f" COUNT(DISTINCT conversation_id)"
        f" FROM messages WHERE 1=1{pf}",
        pp,
    ).fetchone()

    total_messages = vol[0]
    total_human = vol[1]
    total_assistant = vol[2]
    total_words_human = vol[3] or 0
    total_words_assistant = vol[4] or 0
    total_conversations = vol[5]

    if not total_human:
        raise ValueError("No human messages found in data")

    avg_words_per_prompt = total_words_human / total_human

    # === Conversation Depth (SQL) ===
    depth_rows = conn.execute(
        f"SELECT conversation_id, COUNT(*) FROM messages WHERE 1=1{pf} GROUP BY conversation_id",
        pp,
    ).fetchall()

    turns_values = [r[1] for r in depth_rows]
    avg_turns = sum(turns_values) / len(turns_values) if turns_values else 0
    max_turns = max(turns_values) if turns_values else 0
    quick_asks = sum(1 for t in turns_values if t <= 3)
    working_sessions = sum(1 for t in turns_values if 4 <= t <= 10)
    deep_dives = sum(1 for t in turns_values if t > 10)
    response_ratio = total_words_assistant / total_words_human if total_words_human else 0

    # === Temporal Metrics (SQL via pre-computed local_hour/local_weekday) ===
    heatmap_rows = conn.execute(
        f"SELECT local_weekday, local_hour, COUNT(*)"
        f" FROM messages WHERE role = 'human'{pf}"
        f" GROUP BY local_weekday, local_hour",
        pp,
    ).fetchall()

    heatmap = defaultdict(lambda: defaultdict(int))
    for dow, hour, count in heatmap_rows:
        heatmap[dow][hour] = count
    heatmap_data = [[heatmap[dow][hour] for hour in range(24)] for dow in range(7)]

    night_owl_count = conn.execute(
        f"SELECT COUNT(*) FROM messages"
        f" WHERE role = 'human' AND (local_hour >= 23 OR local_hour < 4){pf}",
        pp,
    ).fetchone()[0]
    night_owl_pct = (night_owl_count / total_human * 100)

    hour_rows = conn.execute(
        f"SELECT local_hour, COUNT(*) as cnt"
        f" FROM messages WHERE role = 'human'{pf}"
        f" GROUP BY local_hour",
        pp,
    ).fetchall()
    hour_counts = {r[0]: r[1] for r in hour_rows}
    peak_hour_row = max(hour_rows, key=lambda x: x[1]) if hour_rows else (0, 0)
    peak_hour = peak_hour_row[0]
    peak_hour_count = peak_hour_row[1]

    day_rows = conn.execute(
        f"SELECT local_weekday, COUNT(*) as cnt"
        f" FROM messages WHERE role = 'human'{pf}"
        f" GROUP BY local_weekday",
        pp,
    ).fetchall()
    peak_day_row = max(day_rows, key=lambda x: x[1]) if day_rows else (0, 0)
    peak_day = peak_day_row[0]
    peak_day_count = peak_day_row[1]
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    # === Style Metrics (Python regex on DB content) ===
    human_msgs = query_messages(conn, role=Role.HUMAN, platform=platform)
    assistant_msgs = query_messages(conn, role=Role.ASSISTANT, platform=platform)

    # Politeness
    politeness_counts = {}
    for word, pattern in POLITENESS_PATTERNS.items():
        count = sum(len(re.findall(pattern, m.content, re.IGNORECASE)) for m in human_msgs)
        politeness_counts[word] = count
    total_politeness = sum(politeness_counts.values())
    politeness_per_100 = (total_politeness / total_human * 100)

    # Backtrack
    backtrack_counts = {}
    for word, pattern in BACKTRACK_PATTERNS.items():
        count = sum(len(re.findall(pattern, m.content, re.IGNORECASE)) for m in human_msgs)
        backtrack_counts[word] = count
    total_backtrack = sum(backtrack_counts.values())
    backtrack_per_100 = (total_backtrack / total_human * 100)

    # Question / Command rates
    question_count = sum(1 for m in human_msgs if m.content.strip().endswith('?'))
    question_rate = (question_count / total_human * 100)
    command_count = sum(
        1 for m in human_msgs if re.match(COMMAND_PATTERN, m.content.strip(), re.IGNORECASE)
    )
    command_rate = (command_count / total_human * 100)

    # "You're absolutely right" count
    youre_right_count = sum(
        len(re.findall(r"you'?re (absolutely )?right", m.content, re.IGNORECASE))
        for m in assistant_msgs
    )
    youre_right_per_convo = youre_right_count / total_conversations if total_conversations else 0

    # === Platform Stats (SQL) ===
    plat_rows = conn.execute(
        f"SELECT platform, COUNT(*), SUM(word_count),"
        f" COUNT(DISTINCT conversation_id), MIN(timestamp)"
        f" FROM messages WHERE role = 'human'{pf}"
        f" GROUP BY platform",
        pp,
    ).fetchall()

    platform_stats = {}
    for plat, msgs, words, convos, first_ts in plat_rows:
        platform_stats[plat] = {
            "messages": msgs,
            "words": words,
            "conversations": convos,
            "first_message": first_ts,
        }

    # === Persona Classification ===
    persona = classify_persona(politeness_per_100, backtrack_per_100, question_rate, command_rate, config)

    # === Date Range (SQL) ===
    date_range_row = conn.execute(
        f"SELECT MIN(timestamp), MAX(timestamp) FROM messages WHERE role = 'human'{pf}",
        pp,
    ).fetchone()

    # === Sub-module metrics ===
    model_usage = compute_model_usage(conn, platform)
    trends = compute_trend_metrics(conn, platform)
    nlp = compute_nlp_metrics(conn, platform)

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
            "first": date_range_row[0],
            "last": date_range_row[1],
        },
    }


def has_human_messages_db(conn: sqlite3.Connection, platform: Platform | None = None) -> bool:
    """Return True if the DB contains at least one human message (optionally filtered by platform)."""
    pf, pp = platform_filter(platform)
    count = conn.execute(
        f"SELECT COUNT(*) FROM messages WHERE role = 'human'{pf}", pp,
    ).fetchone()[0]
    return count > 0


def compute_source_views(conn: sqlite3.Connection, config: Config) -> tuple[dict, dict]:
    """
    Build source-scoped metric views for dashboard filtering.

    Source semantics:
      - both: Claude Code + Codex
      - claude_code: Claude Code only
      - codex: Codex only
    """
    all_metrics = compute_metrics(conn, config)
    claude_code_metrics = (
        compute_metrics(conn, config, Platform.CLAUDE_CODE)
        if has_human_messages_db(conn, Platform.CLAUDE_CODE)
        else None
    )
    codex_metrics = (
        compute_metrics(conn, config, Platform.CODEX)
        if has_human_messages_db(conn, Platform.CODEX)
        else None
    )

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
