import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from src.config import load_config
from src.db import init_db, insert_messages, query_messages
from src.metrics import compute_metrics, compute_source_views
from src.models import Message, Platform, Role
from src.nlp import enrich_nlp
from src.parsers import parse_claude_code, parse_codex_history, parse_codex_session_metadata


def make_message(
    *,
    ts: datetime,
    platform: Platform,
    role: Role,
    content: str,
    conversation_id: str,
    model_id: str | None = None,
    model_provider: str | None = None,
) -> Message:
    return Message(
        timestamp=ts,
        platform=platform,
        role=role,
        content=content,
        conversation_id=conversation_id,
        word_count=len(content.split()),
        model_id=model_id,
        model_provider=model_provider,
    )


def setup_db(messages: list[Message]):
    """Insert messages into an in-memory DB and run NLP enrichment."""
    conn = init_db()
    insert_messages(conn, messages)
    enrich_nlp(conn)
    return conn


class ParseCodexSessionMetadataTests(unittest.TestCase):
    def test_parse_codex_session_metadata_extracts_model_and_provider(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            sessions_root = Path(tmpdir)
            session_file = sessions_root / "2026" / "02" / "12" / "rollout-test.jsonl"
            session_file.parent.mkdir(parents=True, exist_ok=True)
            session_file.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "type": "session_meta",
                                "payload": {"id": "s1", "model_provider": "openai"},
                            }
                        ),
                        json.dumps(
                            {
                                "type": "turn_context",
                                "payload": {"model": "gpt-5.3-codex"},
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            result = parse_codex_session_metadata(sessions_root)

        self.assertIn("s1", result)
        self.assertEqual(result["s1"]["model_provider"], "openai")
        self.assertEqual(result["s1"]["model_id"], "gpt-5.3-codex")


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

            session_models = {
                "s3": {"model_id": "gpt-5.2-codex", "model_provider": "openai"}
            }
            messages = list(parse_codex_history(history_path, session_models))

        self.assertEqual(len(messages), 2)

        first, second = messages
        self.assertEqual(first.platform, Platform.CODEX)
        self.assertEqual(first.role, Role.HUMAN)
        self.assertEqual(first.conversation_id, "s3")
        self.assertEqual(first.word_count, 2)
        self.assertEqual(
            first.timestamp,
            datetime.fromtimestamp(1700000000, tz=timezone.utc),
        )
        self.assertEqual(first.model_id, "gpt-5.2-codex")
        self.assertEqual(first.model_provider, "openai")

        self.assertEqual(second.conversation_id, "unknown")
        self.assertEqual(second.word_count, 2)
        self.assertIsNone(second.model_id)
        self.assertIsNone(second.model_provider)


class ModelUsageTests(unittest.TestCase):
    def test_compute_metrics_includes_model_usage_aggregates(self) -> None:
        config = load_config()
        t0 = datetime(2026, 2, 1, 12, 0, tzinfo=timezone.utc)
        messages = [
            make_message(
                ts=t0,
                platform=Platform.CODEX,
                role=Role.HUMAN,
                content="run tests",
                conversation_id="cx-1",
                model_id="gpt-5.3-codex",
                model_provider="openai",
            ),
            make_message(
                ts=t0,
                platform=Platform.CODEX,
                role=Role.HUMAN,
                content="fix this",
                conversation_id="cx-1",
                model_id="gpt-5.3-codex",
                model_provider="openai",
            ),
            make_message(
                ts=t0,
                platform=Platform.CLAUDE_CODE,
                role=Role.HUMAN,
                content="please explain",
                conversation_id="cc-1",
            ),
        ]

        conn = setup_db(messages)
        metrics = compute_metrics(conn, config)
        model_usage = metrics["model_usage"]

        self.assertEqual(model_usage["coverage"]["total_human_prompts"], 3)
        self.assertEqual(model_usage["coverage"]["prompts_with_model_metadata"], 2)
        self.assertAlmostEqual(model_usage["coverage"]["metadata_coverage_pct"], 66.7, places=1)

        by_model = model_usage["by_model"]
        self.assertEqual(len(by_model), 1)
        self.assertEqual(by_model[0]["model_id"], "gpt-5.3-codex")
        self.assertEqual(by_model[0]["model_provider"], "openai")
        self.assertEqual(by_model[0]["prompts"], 2)
        self.assertEqual(by_model[0]["conversations"], 1)

        codex_series = model_usage["time_series_by_source"]["codex"]
        self.assertEqual(len(codex_series), 1)
        self.assertEqual(codex_series[0]["total_prompts"], 2)
        self.assertEqual(codex_series[0]["models"]["gpt-5.3-codex"], 2)

    def test_compute_metrics_includes_trend_rollups_and_deltas(self) -> None:
        config = load_config()
        t0 = datetime(2026, 2, 1, 12, 0, tzinfo=timezone.utc)
        t1 = datetime(2026, 2, 2, 12, 0, tzinfo=timezone.utc)
        t2 = datetime(2026, 2, 3, 12, 0, tzinfo=timezone.utc)
        messages = [
            make_message(
                ts=t0,
                platform=Platform.CLAUDE_CODE,
                role=Role.HUMAN,
                content="please explain this?",
                conversation_id="cc-1",
            ),
            make_message(
                ts=t1,
                platform=Platform.CODEX,
                role=Role.HUMAN,
                content="run tests",
                conversation_id="cx-1",
                model_id="gpt-5.2-codex",
                model_provider="openai",
            ),
            make_message(
                ts=t2,
                platform=Platform.CODEX,
                role=Role.HUMAN,
                content="fix this",
                conversation_id="cx-2",
                model_id="gpt-5.3-codex",
                model_provider="openai",
            ),
        ]

        conn = setup_db(messages)
        metrics = compute_metrics(conn, config)
        trends = metrics["trends"]

        self.assertEqual(len(trends["daily_rollups"]), 3)
        self.assertEqual(len(trends["weekly_rollups"]), 2)
        self.assertIn("prompts_per_day", trends["deltas_7d_vs_30d"])
        self.assertIn("codex_share_pct", trends["deltas_7d_vs_30d"])
        self.assertIsInstance(trends["shift_markers"], list)


class NlpMetricsTests(unittest.TestCase):
    def test_compute_metrics_includes_nlp_outputs_with_confidence(self) -> None:
        config = load_config()
        t0 = datetime(2026, 2, 1, 12, 0, tzinfo=timezone.utc)
        messages = [
            make_message(
                ts=t0,
                platform=Platform.CLAUDE_CODE,
                role=Role.HUMAN,
                content="can you debug this failing test and fix the error?",
                conversation_id="cc-1",
            ),
            make_message(
                ts=t0,
                platform=Platform.CLAUDE_CODE,
                role=Role.HUMAN,
                content="plan a release strategy with milestones and next steps",
                conversation_id="cc-1",
            ),
            make_message(
                ts=t0,
                platform=Platform.CLAUDE_CODE,
                role=Role.HUMAN,
                content="please explain why this works and how to improve it",
                conversation_id="cc-2",
            ),
        ]

        conn = setup_db(messages)
        metrics = compute_metrics(conn, config)
        nlp = metrics["nlp"]

        self.assertIn("intent", nlp)
        self.assertIn("complexity", nlp)
        self.assertIn("iteration_style", nlp)

        intent_counts = nlp["intent"]["counts"]
        self.assertEqual(sum(intent_counts.values()), 3)
        self.assertGreaterEqual(nlp["intent"]["confidence"]["mean"], 0.5)
        self.assertLessEqual(nlp["intent"]["confidence"]["max"], 0.95)

        self.assertIn("avg_score", nlp["complexity"])
        self.assertIn("p90_score", nlp["complexity"])
        self.assertIn("confidence", nlp["complexity"])

        self.assertIn(nlp["iteration_style"]["style"], {"direct", "balanced_iterative", "highly_iterative"})
        self.assertIn("confidence", nlp["iteration_style"])


class SourceViewsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config()
        self.t0 = datetime(2026, 2, 1, 12, 0, tzinfo=timezone.utc)

    def test_compute_source_views_with_both_sources(self) -> None:
        messages = [
            make_message(
                ts=self.t0,
                platform=Platform.CLAUDE_CODE,
                role=Role.HUMAN,
                content="please explain this?",
                conversation_id="cc-1",
            ),
            make_message(
                ts=self.t0,
                platform=Platform.CLAUDE_CODE,
                role=Role.ASSISTANT,
                content="you're absolutely right",
                conversation_id="cc-1",
            ),
            make_message(
                ts=self.t0,
                platform=Platform.CODEX,
                role=Role.HUMAN,
                content="run tests",
                conversation_id="cx-1",
            ),
        ]

        conn = setup_db(messages)
        source_views, defaults = compute_source_views(conn, self.config)

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
                platform=Platform.CLAUDE_CODE,
                role=Role.HUMAN,
                content="please do this",
                conversation_id="cc-1",
            ),
            make_message(
                ts=self.t0,
                platform=Platform.CLAUDE_CODE,
                role=Role.ASSISTANT,
                content="done",
                conversation_id="cc-1",
            ),
        ]

        conn = setup_db(messages)
        source_views, defaults = compute_source_views(conn, self.config)

        self.assertEqual(defaults["default_view"], "both")
        self.assertIsNotNone(source_views["both"])
        self.assertIsNotNone(source_views["claude_code"])
        self.assertIsNone(source_views["codex"])

    def test_compute_source_views_with_codex_only(self) -> None:
        messages = [
            make_message(
                ts=self.t0,
                platform=Platform.CODEX,
                role=Role.HUMAN,
                content="fix this",
                conversation_id="cx-1",
            )
        ]

        conn = setup_db(messages)
        source_views, defaults = compute_source_views(conn, self.config)

        self.assertEqual(defaults["default_view"], "both")
        self.assertIsNotNone(source_views["both"])
        self.assertIsNone(source_views["claude_code"])
        self.assertIsNotNone(source_views["codex"])


class DatabaseLayerTests(unittest.TestCase):
    """Tests for the SQLite database layer."""

    def test_init_db_creates_tables(self) -> None:
        conn = init_db()
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        table_names = [t[0] for t in tables]
        self.assertIn("messages", table_names)
        self.assertIn("nlp_enrichments", table_names)

    def test_insert_and_query_roundtrip(self) -> None:
        t0 = datetime(2026, 2, 1, 12, 0, tzinfo=timezone.utc)
        messages = [
            make_message(
                ts=t0,
                platform=Platform.CLAUDE_CODE,
                role=Role.HUMAN,
                content="hello world",
                conversation_id="cc-1",
            ),
            make_message(
                ts=t0,
                platform=Platform.CODEX,
                role=Role.HUMAN,
                content="fix this bug",
                conversation_id="cx-1",
                model_id="gpt-5.3-codex",
                model_provider="openai",
            ),
        ]
        conn = init_db()
        count = insert_messages(conn, messages)
        self.assertEqual(count, 2)

        retrieved = query_messages(conn, role=Role.HUMAN)
        self.assertEqual(len(retrieved), 2)
        self.assertEqual(retrieved[0].content, "hello world")
        self.assertEqual(retrieved[1].model_id, "gpt-5.3-codex")

    def test_query_messages_filters_by_platform(self) -> None:
        t0 = datetime(2026, 2, 1, 12, 0, tzinfo=timezone.utc)
        messages = [
            make_message(ts=t0, platform=Platform.CLAUDE_CODE, role=Role.HUMAN,
                         content="cc msg", conversation_id="cc-1"),
            make_message(ts=t0, platform=Platform.CODEX, role=Role.HUMAN,
                         content="codex msg", conversation_id="cx-1"),
        ]
        conn = init_db()
        insert_messages(conn, messages)

        cc_msgs = query_messages(conn, platform=Platform.CLAUDE_CODE)
        self.assertEqual(len(cc_msgs), 1)
        self.assertEqual(cc_msgs[0].content, "cc msg")

    def test_nlp_enrichment_roundtrip(self) -> None:
        t0 = datetime(2026, 2, 1, 12, 0, tzinfo=timezone.utc)
        messages = [
            make_message(ts=t0, platform=Platform.CLAUDE_CODE, role=Role.HUMAN,
                         content="debug this failing test", conversation_id="cc-1"),
        ]
        conn = init_db()
        insert_messages(conn, messages)
        enrich_nlp(conn)

        row = conn.execute("SELECT intent, complexity_score FROM nlp_enrichments").fetchone()
        self.assertIsNotNone(row)
        self.assertIsInstance(row[0], str)
        self.assertGreater(row[1], 0)


if __name__ == "__main__":
    unittest.main()
