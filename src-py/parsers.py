"""JSONL parsers for Claude Code and Codex data sources."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from .config import logger
from .models import Message, Platform, Role


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


def parse_codex_session_metadata(sessions_path: Path) -> dict[str, dict[str, str | None]]:
    """
    Parse Codex session logs and map session_id to model metadata.

    Metadata sources:
      - session_meta.payload.id
      - session_meta.payload.model_provider
      - session_meta/turn_context payload.model
    """
    if not sessions_path.exists():
        logger.warning(f"Codex sessions path not found: {sessions_path}")
        return {}

    session_models: dict[str, dict[str, str | None]] = {}
    files_parsed = 0

    for jsonl_file in sessions_path.rglob("*.jsonl"):
        files_parsed += 1
        session_id: str | None = None
        model_id: str | None = None
        model_provider: str | None = None

        with open(jsonl_file, 'r') as f:
            for idx, line in enumerate(f):
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                entry_type = entry.get("type")
                payload = entry.get("payload", {})
                if not isinstance(payload, dict):
                    continue

                if entry_type == "session_meta":
                    session_meta_id = payload.get("id")
                    if isinstance(session_meta_id, str) and session_meta_id.strip():
                        session_id = session_meta_id.strip()

                    provider = payload.get("model_provider")
                    if isinstance(provider, str) and provider.strip():
                        model_provider = provider.strip()

                    model = payload.get("model")
                    if isinstance(model, str) and model.strip():
                        model_id = model.strip()

                if entry_type == "turn_context":
                    model = payload.get("model")
                    if isinstance(model, str) and model.strip():
                        model_id = model.strip()

                if session_id and model_id and model_provider:
                    break

                # Session metadata is near the top; avoid scanning huge files.
                if idx > 400:
                    break

        if session_id:
            existing = session_models.get(
                session_id,
                {"model_id": None, "model_provider": None},
            )
            if model_id and not existing.get("model_id"):
                existing["model_id"] = model_id
            if model_provider and not existing.get("model_provider"):
                existing["model_provider"] = model_provider
            session_models[session_id] = existing

    logger.info(
        f"  Codex sessions: model metadata for {len(session_models)} sessions from {files_parsed} files"
    )
    return session_models


def parse_codex_history(
    source_path: Path,
    session_models: dict[str, dict[str, str | None]] | None = None,
) -> Iterator[Message]:
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
            model_data = session_models.get(session_id, {}) if session_models else {}
            model_id = model_data.get("model_id")
            model_provider = model_data.get("model_provider")
            yield Message(
                timestamp=timestamp,
                platform=Platform.CODEX,
                role=Role.HUMAN,
                content=text,
                conversation_id=session_id,
                word_count=len(text.split()),
                model_id=model_id if isinstance(model_id, str) else None,
                model_provider=model_provider if isinstance(model_provider, str) else None,
            )

    logger.info(f"  Codex: {messages_parsed} messages from history.jsonl")
