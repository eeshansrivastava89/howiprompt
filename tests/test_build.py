import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import build


def make_message(
    *,
    ts: datetime,
    platform: build.Platform,
    role: build.Role,
    content: str,
    conversation_id: str,
) -> build.Message:
    return build.Message(
        timestamp=ts,
        platform=platform,
        role=role,
        content=content,
        conversation_id=conversation_id,
        word_count=len(content.split()),
    )


class ParseCodexHistoryTests(unittest.TestCase):
    def test_parse_codex_history_skips_invalid_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            history_path = Path(tmpdir) / "history.jsonl"
            lines = [
                "{not-json}\n",
                json.dumps({"session_id": "s1", "ts": "bad-ts", "text": "hello"}) + "\n",
                json.dumps({"session_id": "s2", "ts": 1700000000, "text": "   "}) + "\n",
                json.dumps({"session_id": "s3", "ts": 1700000000, "text": "hello world"}) + "\n",
                json.dumps({"ts": 1700000001.0, "text": "fallback session"}) + "\n",
            ]
            history_path.write_text("".join(lines), encoding="utf-8")

            messages = list(build.parse_codex_history(history_path))

        self.assertEqual(len(messages), 2)

        first, second = messages
        self.assertEqual(first.platform, build.Platform.CODEX)
        self.assertEqual(first.role, build.Role.HUMAN)
        self.assertEqual(first.conversation_id, "s3")
        self.assertEqual(first.word_count, 2)
        self.assertEqual(
            first.timestamp,
            datetime.fromtimestamp(1700000000, tz=timezone.utc),
        )

        self.assertEqual(second.conversation_id, "unknown")
        self.assertEqual(second.word_count, 2)


class SourceViewsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = build.load_config()
        self.t0 = datetime(2026, 2, 1, 12, 0, tzinfo=timezone.utc)

    def test_compute_source_views_with_both_sources(self) -> None:
        messages = [
            make_message(
                ts=self.t0,
                platform=build.Platform.CLAUDE_CODE,
                role=build.Role.HUMAN,
                content="please explain this?",
                conversation_id="cc-1",
            ),
            make_message(
                ts=self.t0,
                platform=build.Platform.CLAUDE_CODE,
                role=build.Role.ASSISTANT,
                content="you're absolutely right",
                conversation_id="cc-1",
            ),
            make_message(
                ts=self.t0,
                platform=build.Platform.CODEX,
                role=build.Role.HUMAN,
                content="run tests",
                conversation_id="cx-1",
            ),
        ]

        source_views, defaults = build.compute_source_views(messages, self.config)

        self.assertEqual(defaults["default_view"], "both")
        self.assertEqual(set(source_views.keys()), {"both", "claude_code", "codex"})
        self.assertIsNotNone(source_views["both"])
        self.assertIsNotNone(source_views["claude_code"])
        self.assertIsNotNone(source_views["codex"])
        self.assertEqual(source_views["both"]["volume"]["total_human"], 2)
        self.assertEqual(source_views["claude_code"]["volume"]["total_human"], 1)
        self.assertEqual(source_views["codex"]["volume"]["total_human"], 1)

    def test_compute_source_views_with_claude_code_only(self) -> None:
        messages = [
            make_message(
                ts=self.t0,
                platform=build.Platform.CLAUDE_CODE,
                role=build.Role.HUMAN,
                content="please do this",
                conversation_id="cc-1",
            ),
            make_message(
                ts=self.t0,
                platform=build.Platform.CLAUDE_CODE,
                role=build.Role.ASSISTANT,
                content="done",
                conversation_id="cc-1",
            ),
        ]

        source_views, defaults = build.compute_source_views(messages, self.config)

        self.assertEqual(defaults["default_view"], "both")
        self.assertIsNotNone(source_views["both"])
        self.assertIsNotNone(source_views["claude_code"])
        self.assertIsNone(source_views["codex"])

    def test_compute_source_views_with_codex_only(self) -> None:
        messages = [
            make_message(
                ts=self.t0,
                platform=build.Platform.CODEX,
                role=build.Role.HUMAN,
                content="fix this",
                conversation_id="cx-1",
            )
        ]

        source_views, defaults = build.compute_source_views(messages, self.config)

        self.assertEqual(defaults["default_view"], "both")
        self.assertIsNotNone(source_views["both"])
        self.assertIsNone(source_views["claude_code"])
        self.assertIsNotNone(source_views["codex"])


class DashboardCompatibilityTests(unittest.TestCase):
    def test_generate_dashboard_html_handles_legacy_claude_view_key(self) -> None:
        config = build.load_config()
        t0 = datetime(2026, 2, 1, 12, 0, tzinfo=timezone.utc)
        messages = [
            make_message(
                ts=t0,
                platform=build.Platform.CLAUDE_CODE,
                role=build.Role.HUMAN,
                content="please explain this?",
                conversation_id="cc-1",
            ),
            make_message(
                ts=t0,
                platform=build.Platform.CLAUDE_CODE,
                role=build.Role.ASSISTANT,
                content="you're right",
                conversation_id="cc-1",
            ),
            make_message(
                ts=t0,
                platform=build.Platform.CODEX,
                role=build.Role.HUMAN,
                content="run tests",
                conversation_id="cx-1",
            ),
        ]
        source_views, _ = build.compute_source_views(messages, config)

        metrics = dict(source_views["both"])
        metrics["source_views"] = {
            "both": source_views["both"],
            "claude": source_views["claude_code"],
            "codex": source_views["codex"],
        }
        metrics["default_view"] = "claude"

        html = build.generate_dashboard_html(metrics)

        self.assertIn('option value="claude_code" selected', html)
        self.assertIn('const defaultSource = "claude_code";', html)
        self.assertIn("selected === 'claude'", html)


if __name__ == "__main__":
    unittest.main()
