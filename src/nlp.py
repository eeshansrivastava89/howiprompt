"""NLP classifiers: intent, complexity, iteration style with confidence metadata."""

import re
from collections import Counter

from .models import Message

# === Regex Pattern Definitions ===
POLITENESS_PATTERNS = {
    "please": r"\bplease\b",
    "thanks": r"\b(thanks|thank you|thx)\b",
    "sorry": r"\b(sorry|apologies|apologize)\b",
}
BACKTRACK_PATTERNS = {
    "actually": r"\bactually\b",
    "wait": r"\bwait\b",
    "never_mind": r"\b(never\s*mind|nevermind)\b",
    "scratch_that": r"\b(scratch that|ignore that)\b",
}
COMMAND_PATTERN = r"^(please\s+)?(do|make|create|write|build|add|fix|update|change|remove|delete|show|run|help|can you|could you|would you|tell|explain|find|search|get|set|check|test|debug|implement|refactor)\b"
INTENT_PATTERNS = {
    "debug_fix": [
        r"\b(debug|fix|error|bug|failing|broken|traceback|stack trace)\b",
    ],
    "build_feature": [
        r"\b(build|create|implement|add|ship|feature|integrate)\b",
    ],
    "analysis_research": [
        r"\b(analyze|audit|review|compare|benchmark|research|investigate)\b",
    ],
    "explanation_learning": [
        r"\b(explain|why|how does|teach|walk me through|help me understand)\b",
    ],
    "planning_strategy": [
        r"\b(plan|roadmap|milestone|phase|next steps|strategy)\b",
    ],
    "ops_commands": [
        r"\b(run|execute|command|script|deploy|release|push|tag)\b",
    ],
}
ITERATION_MARKERS = [
    r"\bactually\b",
    r"\bwait\b",
    r"\binstead\b",
    r"\bchange\b",
    r"\bupdate\b",
    r"\bfix\b",
    r"\brevise\b",
    r"\bretry\b",
    r"\bagain\b",
    r"\bdifferent\b",
    r"\bscratch that\b",
    r"\brework\b",
]


def classify_intent(text: str) -> tuple[str, float]:
    """Classify a prompt into deterministic intent categories with confidence."""
    scores = {}
    for intent, patterns in INTENT_PATTERNS.items():
        score = 0
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                score += 1
        scores[intent] = score

    top_intent = max(scores.items(), key=lambda item: item[1])[0]
    top_score = scores[top_intent]
    second_score = sorted(scores.values(), reverse=True)[1] if len(scores) > 1 else 0

    if top_score == 0:
        return "other", 0.5

    confidence = 0.6 + min(top_score, 3) * 0.1
    if top_score > second_score:
        confidence += 0.1
    return top_intent, round(min(confidence, 0.95), 2)


def compute_complexity_for_prompt(text: str) -> tuple[float, float]:
    """Compute deterministic complexity score and confidence for a prompt."""
    content = text.strip()
    words = len(content.split())
    score = 1.0

    # Word count contribution
    if words >= 15:
        score += 0.7
    if words >= 35:
        score += 0.8
    if words >= 70:
        score += 0.8

    # Structural and constraint signals
    if "\n" in content:
        score += 0.5
    if content.count(",") >= 2 or ";" in content:
        score += 0.4
    if re.search(r"[{{}}`]|--|\.\/|=|->", content):
        score += 0.5

    constraint_hits = len(
        re.findall(
            r"\b(must|should|without|exactly|at least|at most|step|checklist|constraint)\b",
            content,
            re.IGNORECASE,
        )
    )
    score += min(constraint_hits, 3) * 0.2

    final_score = round(min(score, 5.0), 1)
    confidence = 0.65
    if words >= 20:
        confidence += 0.1
    if constraint_hits > 0:
        confidence += 0.1
    if "\n" in content or re.search(r"[{{}}`]|--|\.\/|=|->", content):
        confidence += 0.05

    return final_score, round(min(confidence, 0.95), 2)


def compute_iteration_style_for_prompt(text: str) -> tuple[float, float]:
    """Compute deterministic iteration-style score and confidence."""
    marker_hits = 0
    for pattern in ITERATION_MARKERS:
        marker_hits += len(re.findall(pattern, text, re.IGNORECASE))

    score = marker_hits * 20
    if "?" in text:
        score += 8
    if re.search(r"\b(again|retry|revise|change|update|different)\b", text, re.IGNORECASE):
        score += 8

    final_score = round(min(score, 100), 1)
    confidence = 0.6 + min(marker_hits, 3) * 0.1
    if "?" in text:
        confidence += 0.05
    return final_score, round(min(confidence, 0.95), 2)


def compute_nlp_metrics(human_msgs: list[Message]) -> dict:
    """Compute deterministic NLP enrichments with confidence metadata."""
    total = len(human_msgs)
    if total == 0:
        return {
            "intent": {
                "method": "deterministic_rules_v1",
                "counts": {},
                "rates_pct": {},
                "top_intents": [],
                "confidence": {"mean": 0.0, "min": 0.0, "max": 0.0},
            },
            "complexity": {
                "method": "heuristic_complexity_v1",
                "avg_score": 0.0,
                "p50_score": 0.0,
                "p90_score": 0.0,
                "distribution": {"low": 0, "medium": 0, "high": 0},
                "confidence": {"mean": 0.0, "min": 0.0, "max": 0.0},
            },
            "iteration_style": {
                "method": "iteration_markers_v1",
                "avg_score": 0.0,
                "distribution": {"low": 0, "medium": 0, "high": 0},
                "style": "balanced",
                "confidence": {"mean": 0.0, "min": 0.0, "max": 0.0},
            },
        }

    intent_counts = Counter()
    intent_confidences = []
    complexity_scores = []
    complexity_confidences = []
    iteration_scores = []
    iteration_confidences = []

    for msg in human_msgs:
        intent, intent_conf = classify_intent(msg.content)
        intent_counts[intent] += 1
        intent_confidences.append(intent_conf)

        complexity, complexity_conf = compute_complexity_for_prompt(msg.content)
        complexity_scores.append(complexity)
        complexity_confidences.append(complexity_conf)

        iteration_score, iteration_conf = compute_iteration_style_for_prompt(msg.content)
        iteration_scores.append(iteration_score)
        iteration_confidences.append(iteration_conf)

    intent_rates = {
        intent: round((count / total) * 100, 1)
        for intent, count in sorted(intent_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    }
    top_intents = [
        {"intent": intent, "count": count, "rate_pct": intent_rates[intent]}
        for intent, count in sorted(intent_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:3]
    ]

    sorted_complexity = sorted(complexity_scores)
    p50_idx = max(0, len(sorted_complexity) // 2 - (1 if len(sorted_complexity) % 2 == 0 else 0))
    p90_idx = max(0, int(len(sorted_complexity) * 0.9) - 1)
    complexity_distribution = {
        "low": sum(1 for value in complexity_scores if value < 2.5),
        "medium": sum(1 for value in complexity_scores if 2.5 <= value < 3.8),
        "high": sum(1 for value in complexity_scores if value >= 3.8),
    }
    iteration_distribution = {
        "low": sum(1 for value in iteration_scores if value < 25),
        "medium": sum(1 for value in iteration_scores if 25 <= value < 60),
        "high": sum(1 for value in iteration_scores if value >= 60),
    }

    avg_iteration = round(sum(iteration_scores) / len(iteration_scores), 1)
    if avg_iteration >= 60:
        iteration_style = "highly_iterative"
    elif avg_iteration >= 30:
        iteration_style = "balanced_iterative"
    else:
        iteration_style = "direct"

    return {
        "intent": {
            "method": "deterministic_rules_v1",
            "counts": dict(sorted(intent_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
            "rates_pct": intent_rates,
            "top_intents": top_intents,
            "confidence": {
                "mean": round(sum(intent_confidences) / len(intent_confidences), 2),
                "min": round(min(intent_confidences), 2),
                "max": round(max(intent_confidences), 2),
            },
        },
        "complexity": {
            "method": "heuristic_complexity_v1",
            "avg_score": round(sum(complexity_scores) / len(complexity_scores), 1),
            "p50_score": round(sorted_complexity[p50_idx], 1),
            "p90_score": round(sorted_complexity[p90_idx], 1),
            "distribution": complexity_distribution,
            "confidence": {
                "mean": round(sum(complexity_confidences) / len(complexity_confidences), 2),
                "min": round(min(complexity_confidences), 2),
                "max": round(max(complexity_confidences), 2),
            },
        },
        "iteration_style": {
            "method": "iteration_markers_v1",
            "avg_score": avg_iteration,
            "distribution": iteration_distribution,
            "style": iteration_style,
            "confidence": {
                "mean": round(sum(iteration_confidences) / len(iteration_confidences), 2),
                "min": round(min(iteration_confidences), 2),
                "max": round(max(iteration_confidences), 2),
            },
        },
    }
