from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Protocol

from news_ingestion.schemas import NewsItem, SourceConfig

MOSCOW_TZ = timezone(timedelta(hours=3))
MSK_FORMAT = "%Y-%m-%d %H:%M:%S"


@dataclass(frozen=True)
class SaveResult:
    news_id: str
    created: bool


@dataclass(frozen=True)
class SourceWatermark:
    last_seen_external_id: str | None = None
    last_seen_published_at: datetime | None = None
    last_polled_at: datetime | None = None


class TickerMatchLike(Protocol):
    ticker: str
    matched_by: str
    matched_terms: tuple[str, ...]


def initialize_database(database_path: str | Path) -> None:
    path = Path(database_path)
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)
    with connect(path) as conn:
        conn.executescript(
            """
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS sources (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                type TEXT NOT NULL,
                method TEXT NOT NULL,
                url TEXT,
                interval_seconds INTEGER NOT NULL,
                trust_score REAL NOT NULL,
                enabled INTEGER NOT NULL,
                raw_config TEXT NOT NULL,
                last_seen_external_id TEXT,
                last_seen_published_at TEXT,
                last_polled_at TEXT
            );
            """
        )
        _ensure_news_table(conn)
        _ensure_ticker_tables(conn)
        _ensure_story_tables(conn)
        conn.executescript(
            """
            CREATE INDEX IF NOT EXISTS ix_news_source_received_at
                ON news(source, received_at_msk DESC);
            CREATE INDEX IF NOT EXISTS ix_news_published_at_msk
                ON news(published_at_msk DESC);
            """
        )
        _normalize_datetime_storage(conn)


def connect(database_path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(Path(database_path), timeout=30)
    conn.execute("PRAGMA busy_timeout=30000")
    conn.row_factory = sqlite3.Row
    return conn


def sync_sources(database_path: str | Path, sources: list[SourceConfig]) -> int:
    initialize_database(database_path)
    with connect(database_path) as conn:
        for source in sources:
            payload = source.model_dump(mode="json")
            conn.execute(
                """
                INSERT INTO sources (
                    id, name, type, method, url, interval_seconds, trust_score,
                    enabled, raw_config
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name = excluded.name,
                    type = excluded.type,
                    method = excluded.method,
                    url = excluded.url,
                    interval_seconds = excluded.interval_seconds,
                    trust_score = excluded.trust_score,
                    enabled = excluded.enabled,
                    raw_config = excluded.raw_config
                """,
                (
                    source.id,
                    source.name,
                    source.type,
                    source.method,
                    str(source.url) if source.url else None,
                    source.interval_seconds,
                    source.trust_score,
                    1 if source.enabled else 0,
                    json.dumps(payload, ensure_ascii=False, sort_keys=True),
                ),
            )
    return len(sources)


def known_external_ids(database_path: str | Path, source_id: str) -> set[str]:
    initialize_database(database_path)
    prefix = f"{source_id}:"
    with connect(database_path) as conn:
        rows = conn.execute(
            """
            SELECT news_id
            FROM news
            WHERE source = ?
            """,
            (source_id,),
        )
    return {
        str(row["news_id"])[len(prefix) :]
        for row in rows
        if str(row["news_id"]).startswith(prefix)
    }


def get_source_watermark(database_path: str | Path, source_id: str) -> SourceWatermark:
    initialize_database(database_path)
    with connect(database_path) as conn:
        row = conn.execute(
            """
            SELECT last_seen_external_id, last_seen_published_at, last_polled_at
            FROM sources
            WHERE id = ?
            """,
            (source_id,),
        ).fetchone()
    if row is None:
        return SourceWatermark()
    return SourceWatermark(
        last_seen_external_id=row["last_seen_external_id"],
        last_seen_published_at=_parse_datetime(row["last_seen_published_at"]),
        last_polled_at=_parse_datetime(row["last_polled_at"]),
    )


def update_source_watermark(
    database_path: str | Path,
    source_id: str,
    *,
    external_id: str | None,
    published_at: datetime | None,
    polled_at: datetime,
) -> None:
    initialize_database(database_path)
    current = get_source_watermark(database_path, source_id)
    next_external_id = current.last_seen_external_id
    next_published_at = current.last_seen_published_at

    if published_at is not None and (
        next_published_at is None or published_at >= next_published_at
    ):
        next_published_at = published_at
        next_external_id = external_id or next_external_id
    elif external_id:
        next_external_id = external_id

    with connect(database_path) as conn:
        conn.execute(
            """
            UPDATE sources
            SET last_seen_external_id = ?,
                last_seen_published_at = ?,
                last_polled_at = ?
            WHERE id = ?
            """,
            (
                next_external_id,
                _format_datetime(next_published_at),
                _format_datetime(polled_at),
                source_id,
            ),
        )


def save_news_item(
    database_path: str | Path,
    item: NewsItem,
    *,
    ticker_matches: Sequence[TickerMatchLike] | None = None,
) -> SaveResult:
    initialize_database(database_path)
    item.saved_at = item.saved_at or datetime.now(UTC)
    raw_payload_hash = _raw_payload_hash(item)
    news_id = _news_id(item, raw_payload_hash)
    ticker_matches = ticker_matches or ()
    tickers_json = json.dumps(
        [match.ticker for match in ticker_matches],
        ensure_ascii=False,
    )

    with connect(database_path) as conn:
        cursor = conn.execute(
            """
            INSERT OR IGNORE INTO news (
                news_id, source, published_at_msk, received_at_msk,
                title, text, url, raw_payload_hash, tickers_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                news_id,
                item.source_id,
                _format_datetime(item.published_at),
                _format_datetime(item.saved_at),
                item.title,
                item.text,
                item.url,
                raw_payload_hash,
                tickers_json,
            ),
        )
        if cursor.rowcount > 0:
            replace_news_tickers(conn, news_id, ticker_matches)
    return SaveResult(news_id=news_id, created=cursor.rowcount > 0)


def count_news(database_path: str | Path) -> int:
    initialize_database(database_path)
    with connect(database_path) as conn:
        row = conn.execute("SELECT COUNT(*) AS count FROM news").fetchone()
    return int(row["count"])


def _ensure_news_table(conn: sqlite3.Connection) -> None:
    columns = _table_columns(conn, "news")
    if columns and _is_minimal_news_schema(columns):
        if "tickers_json" not in columns:
            conn.execute("ALTER TABLE news ADD COLUMN tickers_json TEXT NOT NULL DEFAULT '[]'")
        return

    legacy_table: str | None = None
    if columns:
        conn.execute("DROP INDEX IF EXISTS ix_news_source_saved_at")
        conn.execute("DROP INDEX IF EXISTS ix_news_published_at")
        conn.execute("DROP INDEX IF EXISTS ix_news_source_received_at")
        conn.execute("DROP INDEX IF EXISTS ix_news_published_at_msk")
        legacy_table = f"news_legacy_{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}"
        conn.execute(f"ALTER TABLE news RENAME TO {legacy_table}")

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS news (
            news_id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            published_at_msk TEXT,
            received_at_msk TEXT NOT NULL,
            title TEXT,
            text TEXT NOT NULL,
            url TEXT,
            raw_payload_hash TEXT NOT NULL,
            tickers_json TEXT NOT NULL DEFAULT '[]'
        )
        """
    )

    if legacy_table:
        _copy_legacy_news(conn, legacy_table)


def _ensure_ticker_tables(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS news_tickers (
            news_id TEXT NOT NULL,
            ticker TEXT NOT NULL,
            matched_by TEXT NOT NULL,
            matched_terms_json TEXT NOT NULL,
            created_at_msk TEXT NOT NULL,
            PRIMARY KEY(news_id, ticker),
            FOREIGN KEY(news_id) REFERENCES news(news_id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS ix_news_tickers_ticker
            ON news_tickers(ticker);
        """
    )


def _ensure_story_tables(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS story_clusters (
            story_id TEXT PRIMARY KEY,
            canonical_title TEXT,
            first_published_at_msk TEXT,
            last_published_at_msk TEXT,
            item_count INTEGER NOT NULL,
            source_count INTEGER NOT NULL,
            sources_json TEXT NOT NULL,
            created_at_msk TEXT NOT NULL,
            updated_at_msk TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS story_cluster_items (
            story_id TEXT NOT NULL,
            news_id TEXT NOT NULL PRIMARY KEY,
            source TEXT NOT NULL,
            score REAL NOT NULL,
            matched_news_id TEXT,
            created_at_msk TEXT NOT NULL,
            FOREIGN KEY(story_id) REFERENCES story_clusters(story_id) ON DELETE CASCADE,
            FOREIGN KEY(news_id) REFERENCES news(news_id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS ix_story_cluster_items_story_id
            ON story_cluster_items(story_id);
        CREATE INDEX IF NOT EXISTS ix_story_clusters_updated_at
            ON story_clusters(updated_at_msk DESC);
        """
    )


def _copy_legacy_news(conn: sqlite3.Connection, legacy_table: str) -> None:
    legacy_columns = _table_columns(conn, legacy_table)
    if "source_id" not in legacy_columns:
        return

    rows = conn.execute(f"SELECT * FROM {legacy_table}").fetchall()
    for row in rows:
        source = row["source_id"]
        text = row["text"] or ""
        if not source or not text:
            continue
        raw_payload_hash = _hash_text(
            row["raw_json"]
            or json.dumps(
                {
                    "title": row["title"],
                    "text": text,
                    "url": row["url"],
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        external_or_hash = row["external_id"] or row["id"] or raw_payload_hash
        conn.execute(
            """
            INSERT OR IGNORE INTO news (
                news_id, source, published_at_msk, received_at_msk,
                title, text, url, raw_payload_hash, tickers_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                f"{source}:{external_or_hash}",
                source,
                _format_datetime(_parse_datetime(row["published_at"])),
                _format_datetime(_parse_datetime(row["saved_at"] or row["fetched_at"])),
                row["title"],
                text,
                row["url"],
                raw_payload_hash,
                "[]",
            ),
        )


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return {row["name"] for row in rows}


def _is_minimal_news_schema(columns: set[str]) -> bool:
    return {
        "news_id",
        "source",
        "published_at_msk",
        "received_at_msk",
        "title",
        "text",
        "url",
        "raw_payload_hash",
    }.issubset(columns)


def replace_news_tickers(
    conn: sqlite3.Connection,
    news_id: str,
    ticker_matches: Sequence[TickerMatchLike],
) -> None:
    tickers = [match.ticker for match in ticker_matches]
    now_msk = _format_datetime(datetime.now(UTC))
    conn.execute(
        """
        UPDATE news
        SET tickers_json = ?
        WHERE news_id = ?
        """,
        (json.dumps(tickers, ensure_ascii=False), news_id),
    )
    conn.execute("DELETE FROM news_tickers WHERE news_id = ?", (news_id,))
    conn.executemany(
        """
        INSERT INTO news_tickers (
            news_id,
            ticker,
            matched_by,
            matched_terms_json,
            created_at_msk
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        [
            (
                news_id,
                match.ticker,
                match.matched_by,
                json.dumps(list(match.matched_terms), ensure_ascii=False),
                now_msk,
            )
            for match in ticker_matches
        ],
    )


def _news_id(item: NewsItem, raw_payload_hash: str) -> str:
    external_or_hash = item.external_id or raw_payload_hash
    return f"{item.source_id}:{external_or_hash}"


def _raw_payload_hash(item: NewsItem) -> str:
    payload = item.raw if item.raw is not None else {
        "title": item.title,
        "text": item.text,
        "url": item.url,
    }
    return _hash_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str))


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _format_datetime(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=MOSCOW_TZ)
    return value.astimezone(MOSCOW_TZ).strftime(MSK_FORMAT)


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        try:
            parsed = datetime.strptime(value, MSK_FORMAT)
        except ValueError:
            return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=MOSCOW_TZ).astimezone(UTC)
    return parsed.astimezone(UTC)


def _normalize_datetime_storage(conn: sqlite3.Connection) -> None:
    for table, columns in (
        ("sources", ("last_seen_published_at", "last_polled_at")),
        ("news", ("published_at_msk", "received_at_msk")),
    ):
        selected_columns = ", ".join(columns)
        rows = conn.execute(f"SELECT rowid, {selected_columns} FROM {table}").fetchall()
        for row in rows:
            updates: dict[str, str] = {}
            for column in columns:
                parsed = _parse_datetime(row[column])
                formatted = _format_datetime(parsed)
                if formatted is not None and formatted != row[column]:
                    updates[column] = formatted
            if not updates:
                continue
            assignments = ", ".join(f"{column} = ?" for column in updates)
            conn.execute(
                f"UPDATE {table} SET {assignments} WHERE rowid = ?",
                (*updates.values(), row["rowid"]),
            )
