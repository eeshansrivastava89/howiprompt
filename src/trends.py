"""Trend analytics: daily/weekly rollups, deltas, shift markers, style snapshots."""

import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta

from .models import Message, Platform
from .nlp import BACKTRACK_PATTERNS, COMMAND_PATTERN, POLITENESS_PATTERNS


def compute_style_snapshot(messages: list[Message]) -> dict:
    """Compute style rates for a message bucket."""
    total = len(messages)
    if total == 0:
        return {
            "politeness_per_100": 0.0,
            "backtrack_per_100": 0.0,
            "question_rate_pct": 0.0,
            "command_rate_pct": 0.0,
        }

    politeness_count = 0
    for pattern in POLITENESS_PATTERNS.values():
        politeness_count += sum(
            len(re.findall(pattern, msg.content, re.IGNORECASE))
            for msg in messages
        )

    backtrack_count = 0
    for pattern in BACKTRACK_PATTERNS.values():
        backtrack_count += sum(
            len(re.findall(pattern, msg.content, re.IGNORECASE))
            for msg in messages
        )

    question_count = sum(1 for msg in messages if msg.content.strip().endswith("?"))
    command_count = sum(
        1 for msg in messages if re.match(COMMAND_PATTERN, msg.content.strip(), re.IGNORECASE)
    )

    return {
        "politeness_per_100": round((politeness_count / total) * 100, 1),
        "backtrack_per_100": round((backtrack_count / total) * 100, 1),
        "question_rate_pct": round((question_count / total) * 100, 1),
        "command_rate_pct": round((command_count / total) * 100, 1),
    }


def build_trend_rollups(bucket_map: dict[str, list[Message]], bucket_key: str) -> list[dict]:
    """Build ordered trend rollups for daily or weekly buckets."""
    rollups = []
    for key in sorted(bucket_map.keys()):
        bucket_msgs = bucket_map[key]
        total_prompts = len(bucket_msgs)
        source_counts = Counter(msg.platform.value for msg in bucket_msgs)
        model_counts = Counter(
            msg.model_id or "unknown"
            for msg in bucket_msgs
            if msg.model_id or msg.model_provider
        )

        source_share_pct = {}
        for source in (Platform.CLAUDE_CODE.value, Platform.CODEX.value):
            count = source_counts.get(source, 0)
            source_share_pct[source] = round((count / total_prompts) * 100, 1) if total_prompts else 0.0

        rollups.append(
            {
                bucket_key: key,
                "prompts": total_prompts,
                "source_counts": {
                    Platform.CLAUDE_CODE.value: source_counts.get(Platform.CLAUDE_CODE.value, 0),
                    Platform.CODEX.value: source_counts.get(Platform.CODEX.value, 0),
                },
                "source_share_pct": source_share_pct,
                "style": compute_style_snapshot(bucket_msgs),
                "model_prompts": sum(model_counts.values()),
                "models": dict(sorted(model_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
            }
        )

    return rollups


def compute_trend_deltas(daily_rollups: list[dict]) -> dict:
    """Compare 7-day and 30-day windows for key trend metrics."""
    if not daily_rollups:
        return {}

    latest_date = datetime.fromisoformat(daily_rollups[-1]["date"]).date()

    def window_rollups(days: int) -> list[dict]:
        start = latest_date - timedelta(days=days - 1)
        return [
            item
            for item in daily_rollups
            if datetime.fromisoformat(item["date"]).date() >= start
        ]

    def window_avg(items: list[dict], selector) -> float | None:
        if not items:
            return None
        values = [selector(item) for item in items]
        return round(sum(values) / len(values), 1)

    recent_7 = window_rollups(7)
    recent_30 = window_rollups(30)

    metric_selectors = {
        "prompts_per_day": lambda item: item["prompts"],
        "codex_share_pct": lambda item: item["source_share_pct"][Platform.CODEX.value],
        "politeness_per_100": lambda item: item["style"]["politeness_per_100"],
        "backtrack_per_100": lambda item: item["style"]["backtrack_per_100"],
        "question_rate_pct": lambda item: item["style"]["question_rate_pct"],
        "command_rate_pct": lambda item: item["style"]["command_rate_pct"],
        "model_coverage_pct": lambda item: round(
            (item["model_prompts"] / item["prompts"] * 100), 1
        ) if item["prompts"] else 0.0,
    }

    deltas = {}
    for name, selector in metric_selectors.items():
        avg_7 = window_avg(recent_7, selector)
        avg_30 = window_avg(recent_30, selector)
        if avg_7 is None or avg_30 is None or avg_30 == 0:
            delta_pct = None
        else:
            delta_pct = round(((avg_7 - avg_30) / avg_30) * 100, 1)

        deltas[name] = {
            "avg_7d": avg_7,
            "avg_30d": avg_30,
            "delta_pct": delta_pct,
        }

    return deltas


def detect_shift_markers(daily_rollups: list[dict]) -> list[dict]:
    """Identify major day-over-day changes in prompting behavior."""
    if len(daily_rollups) < 2:
        return []

    avg_prompts = sum(item["prompts"] for item in daily_rollups) / len(daily_rollups)
    min_prompt_shift = max(10, int(avg_prompts * 0.35))
    markers = []

    for i in range(1, len(daily_rollups)):
        prev_day = daily_rollups[i - 1]
        curr_day = daily_rollups[i]

        prompt_delta = curr_day["prompts"] - prev_day["prompts"]
        if abs(prompt_delta) >= min_prompt_shift:
            markers.append(
                {
                    "date": curr_day["date"],
                    "type": "prompt_shift",
                    "direction": "up" if prompt_delta > 0 else "down",
                    "delta_prompts": prompt_delta,
                    "prev_prompts": prev_day["prompts"],
                    "curr_prompts": curr_day["prompts"],
                    "magnitude": abs(prompt_delta),
                }
            )

        codex_share_delta = round(
            curr_day["source_share_pct"][Platform.CODEX.value]
            - prev_day["source_share_pct"][Platform.CODEX.value],
            1,
        )
        if abs(codex_share_delta) >= 20:
            markers.append(
                {
                    "date": curr_day["date"],
                    "type": "source_share_shift",
                    "direction": "up" if codex_share_delta > 0 else "down",
                    "delta_codex_share_pct": codex_share_delta,
                    "prev_codex_share_pct": prev_day["source_share_pct"][Platform.CODEX.value],
                    "curr_codex_share_pct": curr_day["source_share_pct"][Platform.CODEX.value],
                    "magnitude": abs(codex_share_delta),
                }
            )

    markers.sort(key=lambda item: (-item["magnitude"], item["date"]))
    return markers[:10]


def compute_trend_metrics(human_msgs: list[Message]) -> dict:
    """Build daily/weekly trend data plus 7d/30d deltas and shift markers."""
    daily_buckets: dict[str, list[Message]] = defaultdict(list)
    weekly_buckets: dict[str, list[Message]] = defaultdict(list)

    for msg in human_msgs:
        local_date = msg.timestamp.astimezone().date()
        day_key = local_date.isoformat()
        week_start = (local_date - timedelta(days=local_date.weekday())).isoformat()
        daily_buckets[day_key].append(msg)
        weekly_buckets[week_start].append(msg)

    daily_rollups = build_trend_rollups(daily_buckets, "date")
    weekly_rollups = build_trend_rollups(weekly_buckets, "week_start")

    return {
        "daily_rollups": daily_rollups,
        "weekly_rollups": weekly_rollups,
        "deltas_7d_vs_30d": compute_trend_deltas(daily_rollups),
        "shift_markers": detect_shift_markers(daily_rollups),
    }
