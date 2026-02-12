"""SQLite database layer for the analytics pipeline.

In-memory SQLite (:memory:) rebuilt each run. The DB is a compute engine,
not a data store — keeps the pipeline stateless.
"""

import sqlite3
from datetime import datetime
from typing import Iterable

from .models import Message, Platform, Role

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    platform TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    word_count INTEGER NOT NULL,
    model_id TEXT,
    model_provider TEXT,
    local_hour INTEGER NOT NULL,
    local_weekday INTEGER NOT NULL,
    local_date TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS nlp_enrichments (
    message_id INTEGER NOT NULL REFERENCES messages(id),
    intent TEXT NOT NULL,
    intent_confidence REAL NOT NULL,
    complexity_score REAL NOT NULL,
    complexity_confidence REAL NOT NULL,
    iteration_score REAL NOT NULL,
    iteration_confidence REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_platform_role ON messages(platform, role);
CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp);
CREATE INDEX IF NOT EXISTS idx_nlp_enrichments_message_id ON nlp_enrichments(message_id);
"""


def init_db() -> sqlite3.Connection:
    """Create an in-memory SQLite database with schema."""
    conn = sqlite3.connect(":memory:")
    conn.executescript(SCHEMA_SQL)
    return conn


def insert_messages(conn: sqlite3.Connection, messages: Iterable[Message]) -> int:
    """Batch-insert messages into the database. Returns count inserted."""
    sql = """
        INSERT INTO messages (timestamp, platform, role, content, conversation_id,
                              word_count, model_id, model_provider,
                              local_hour, local_weekday, local_date)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    batch_size = 1000
    count = 0
    batch: list[tuple] = []
    for m in messages:
        local_time = m.timestamp.astimezone()
        batch.append((
            m.timestamp.isoformat(),
            m.platform.value,
            m.role.value,
            m.content,
            m.conversation_id,
            m.word_count,
            m.model_id,
            m.model_provider,
            local_time.hour,
            local_time.weekday(),
            local_time.date().isoformat(),
        ))
        if len(batch) >= batch_size:
            conn.executemany(sql, batch)
            count += len(batch)
            batch.clear()
    if batch:
        conn.executemany(sql, batch)
        count += len(batch)
    conn.commit()
    return count


def insert_nlp_enrichments(conn: sqlite3.Connection, enrichments: list[tuple]) -> None:
    """Batch-insert NLP enrichment results."""
    conn.executemany(
        """INSERT INTO nlp_enrichments
           (message_id, intent, intent_confidence, complexity_score, complexity_confidence,
            iteration_score, iteration_confidence)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        enrichments,
    )
    conn.commit()


def query_messages(
    conn: sqlite3.Connection,
    role: Role | None = None,
    platform: Platform | None = None,
) -> list[Message]:
    """Query messages from DB as Message objects."""
    sql = "SELECT timestamp, platform, role, content, conversation_id, word_count, model_id, model_provider FROM messages WHERE 1=1"
    params: list = []
    if role:
        sql += " AND role = ?"
        params.append(role.value)
    if platform:
        sql += " AND platform = ?"
        params.append(platform.value)
    sql += " ORDER BY timestamp"

    return [
        Message(
            timestamp=datetime.fromisoformat(r[0]),
            platform=Platform(r[1]),
            role=Role(r[2]),
            content=r[3],
            conversation_id=r[4],
            word_count=r[5],
            model_id=r[6],
            model_provider=r[7],
        )
        for r in conn.execute(sql, params).fetchall()
    ]


def platform_filter(platform: Platform | None) -> tuple[str, list]:
    """Return (sql_clause, params) for optional platform filter.

    The clause starts with ' AND' so it can be appended to an existing WHERE.
    """
    if platform:
        return " AND platform = ?", [platform.value]
    return "", []
