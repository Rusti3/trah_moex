from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
import sys
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

ARENA_DIR = Path(__file__).resolve().parents[1]
if str(ARENA_DIR) not in sys.path:
    sys.path.insert(0, str(ARENA_DIR))

from news_ingestion.config import SourceRegistry  # type: ignore  # noqa: E402
from news_ingestion.pipeline import IngestionPipeline, SourceRunStats  # type: ignore  # noqa: E402
from news_ingestion.scheduler import create_scheduler  # type: ignore  # noqa: E402
from news_ingestion.settings import Settings  # type: ignore  # noqa: E402
from news_ingestion.storage import connect, count_news, initialize_database  # type: ignore  # noqa: E402
from news_ingestion.tickers import TickerRegistry, tag_existing_news  # type: ignore  # noqa: E402

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None


def _format_ts(value: datetime | str) -> str:
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    return str(value)


def ensure_live_news_schema(database_path: str | Path) -> None:
    initialize_database(database_path)
    with connect(database_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS news_llm_tags (
                news_id TEXT NOT NULL,
                ticker TEXT NOT NULL,
                relation_type TEXT NOT NULL,
                confidence TEXT NOT NULL,
                relation_strength REAL NOT NULL,
                reason TEXT,
                tags_json TEXT NOT NULL DEFAULT '[]',
                created_at_msk TEXT NOT NULL,
                PRIMARY KEY(news_id, ticker, relation_type),
                FOREIGN KEY(news_id) REFERENCES news(news_id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS ix_news_llm_tags_ticker
                ON news_llm_tags(ticker);
            CREATE INDEX IF NOT EXISTS ix_news_received_at
                ON news(received_at_msk DESC);
            """
        )


class NewsBuffer:
    def __init__(self, database_path: str | Path):
        self.database_path = Path(database_path)
        ensure_live_news_schema(self.database_path)

    def get_context(self, as_of: datetime | str, tickers: list[str] | tuple[str, ...]) -> dict[str, Any]:
        as_of_s = _format_ts(as_of)
        day_start = as_of_s[:10] + " 00:00:00"
        per_ticker = {ticker: [] for ticker in tickers}
        marketwide = []
        with connect(self.database_path) as conn:
            rows = conn.execute(
                """
                SELECT
                    n.news_id, n.source, n.published_at_msk, n.received_at_msk,
                    n.title, n.text, n.url,
                    COALESCE(t.ticker, '') AS ticker,
                    COALESCE(t.relation_type, '') AS relation_type,
                    COALESCE(t.confidence, '') AS llm_confidence,
                    COALESCE(t.relation_strength, 0.0) AS relation_strength,
                    COALESCE(t.reason, '') AS llm_reason,
                    COALESCE(t.tags_json, '[]') AS tags_json,
                    sci.story_id,
                    COALESCE(sc.source_count, 1) AS source_count
                FROM news n
                LEFT JOIN news_llm_tags t ON t.news_id = n.news_id
                LEFT JOIN story_cluster_items sci ON sci.news_id = n.news_id
                LEFT JOIN story_clusters sc ON sc.story_id = sci.story_id
                WHERE n.received_at_msk >= ?
                  AND n.received_at_msk <= ?
                ORDER BY n.received_at_msk ASC, n.news_id ASC
                """,
                (day_start, as_of_s),
            ).fetchall()

        seen_story = set()
        for row in rows:
            story_key = row["story_id"] or row["news_id"]
            tag_ticker = str(row["ticker"] or "")
            relation_type = str(row["relation_type"] or "direct")
            dedupe_key = (story_key, tag_ticker, relation_type)
            if dedupe_key in seen_story:
                continue
            seen_story.add(dedupe_key)
            item = {
                "news_id": row["news_id"],
                "published_at": row["published_at_msk"],
                "received_at": row["received_at_msk"],
                "relation_type": relation_type,
                "source": row["source"],
                "source_count": int(row["source_count"] or 1),
                "title": row["title"] or "",
                "text": row["text"] or "",
                "url": row["url"] or "",
                "reason": row["llm_reason"] or "",
                "relation_strength": float(row["relation_strength"] or 0.0),
                "tags": _json_list(row["tags_json"]),
            }
            if tag_ticker in per_ticker and relation_type in {"direct", "cluster"}:
                if relation_type == "cluster":
                    item["relation_strength"] = min(float(item["relation_strength"] or 0.5), 0.6)
                per_ticker[tag_ticker].append(item)
            elif relation_type == "marketwide" or tag_ticker in {"", "MARKET", "MOEX_MARKET"}:
                marketwide.append(item)

        return {
            "rebalance_timestamp": as_of_s,
            "date": as_of_s[:10],
            "per_ticker_news": per_ticker,
            "marketwide_news": marketwide,
        }


def _json_list(value: str) -> list[Any]:
    try:
        parsed = json.loads(value or "[]")
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []


class NewsIngestionLogAggregator:
    """Rate-limited structured diagnostics for the 30s news parser loop."""

    def __init__(
        self,
        *,
        database_path: str | Path,
        logger: Any | None,
        interval_seconds: int = 1800,
    ):
        self.database_path = Path(database_path)
        self.logger = logger
        self.interval_seconds = max(int(interval_seconds), 1)
        self._lock = threading.Lock()
        self._last_emit = time.monotonic()
        self._window_started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._runs = 0
        self._totals = defaultdict(int)
        self._sources: dict[str, dict[str, Any]] = {}
        self._errors: list[dict[str, Any]] = []

    def observe(self, stats: SourceRunStats) -> None:
        should_emit = False
        with self._lock:
            self._runs += 1
            source = self._sources.setdefault(
                stats.source_id,
                {
                    "runs": 0,
                    "fetched": 0,
                    "selected": 0,
                    "saved": 0,
                    "duplicates": 0,
                    "skipped": 0,
                    "errors": 0,
                    "last_duration_seconds": None,
                    "last_finished_at": None,
                },
            )
            source["runs"] += 1
            for key in ("fetched", "selected", "saved", "duplicates", "skipped"):
                value = int(getattr(stats, key, 0) or 0)
                source[key] += value
                self._totals[key] += value
            source["errors"] += len(stats.errors or [])
            source["last_duration_seconds"] = stats.duration_seconds
            source["last_finished_at"] = _format_utc(stats.finished_at)
            self._totals["errors"] += len(stats.errors or [])
            for error in stats.errors or []:
                if len(self._errors) < 20:
                    self._errors.append({"source_id": stats.source_id, "error": str(error)[:500]})
            should_emit = (time.monotonic() - self._last_emit) >= self.interval_seconds
        if should_emit:
            self.emit()

    def emit(self, *, force: bool = False, event: str = "news_ingestion_summary") -> None:
        if self.logger is None:
            return
        with self._lock:
            if not force and (time.monotonic() - self._last_emit) < self.interval_seconds:
                return
            payload = {
                "window_started_at_msk": self._window_started_at,
                "window_seconds": round(time.monotonic() - self._last_emit, 3),
                "runs": self._runs,
                "totals": dict(self._totals),
                "sources": self._sources,
                "errors_sample": self._errors,
                **_news_db_summary(self.database_path),
            }
            errors = list(self._errors)
            self._last_emit = time.monotonic()
            self._window_started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self._runs = 0
            self._totals = defaultdict(int)
            self._sources = {}
            self._errors = []
        self.logger.write(event, payload, stream="news_ingestion")
        if errors:
            self.logger.error("news_ingestion_errors_summary", {**payload, "errors_sample": errors})

    def log_scheduler_error(self, payload: dict[str, Any]) -> None:
        if self.logger is not None:
            self.logger.error("news_scheduler_error", {**payload, **_news_db_summary(self.database_path)})


def _format_utc(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.strftime("%Y-%m-%d %H:%M:%S")


def _news_db_summary(database_path: str | Path) -> dict[str, Any]:
    summary = {
        "news_db_count": 0,
        "news_llm_tags_count": 0,
        "latest_received_at_msk": None,
        "latest_published_at_msk": None,
    }
    try:
        summary["news_db_count"] = count_news(database_path)
        with connect(database_path) as conn:
            latest = conn.execute(
                """
                SELECT received_at_msk, published_at_msk
                FROM news
                ORDER BY received_at_msk DESC
                LIMIT 1
                """
            ).fetchone()
            if latest is not None:
                summary["latest_received_at_msk"] = latest["received_at_msk"]
                summary["latest_published_at_msk"] = latest["published_at_msk"]
            tags = conn.execute("SELECT COUNT(*) AS c FROM news_llm_tags").fetchone()
            summary["news_llm_tags_count"] = int(tags["c"] if tags else 0)
    except Exception as exc:
        summary["news_db_error"] = str(exc)[:500]
    return summary


class LLMNewsTagger:
    """Tag freshly ingested news into direct/cluster/marketwide buckets.

    The source ingestion layer already writes append-only news rows, regex
    ticker matches, and story clusters. This layer adds the production LLM
    relation table used by NewsBuffer, while falling back to regex/source tags
    if Polza is unavailable.
    """

    def __init__(
        self,
        database_path: str | Path,
        *,
        base_url: str = "https://polza.ai/api/v1",
        model: str = "deepseek/deepseek-v4-pro",
        api_key_env: str = "POLZA_AI_API_KEY",
        timeout: float = 60.0,
        retries: int = 2,
    ):
        self.database_path = Path(database_path)
        self.model = model
        self.timeout = timeout
        self.retries = retries
        api_key = os.environ.get(api_key_env, "").strip()
        self.client = OpenAI(base_url=base_url, api_key=api_key) if api_key and OpenAI is not None else None
        ensure_live_news_schema(self.database_path)

    def tag_new_news(self, as_of: datetime | str, tickers: tuple[str, ...], *, limit: int = 30) -> dict[str, int | str]:
        if self.client is None:
            return {"mode": "regex_fallback", "inserted": seed_regex_tags_as_llm_tags(self.database_path)}
        rows = self._load_untagged(as_of, limit=limit)
        if not rows:
            return {"mode": "llm", "inserted": 0}
        payload = self._prompt_payload(rows, tickers)
        last_error = ""
        for attempt in range(self.retries + 1):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    temperature=0,
                    timeout=self.timeout,
                    messages=[
                        {"role": "system", "content": "You tag Russian market news. Return strict JSON only."},
                        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                    ],
                )
                raw = completion.choices[0].message.content or ""
                parsed = _extract_json(raw)
                inserted = self._insert_tags(parsed, rows, tickers)
                if inserted == 0:
                    inserted = seed_regex_tags_as_llm_tags(self.database_path)
                    return {"mode": "llm_empty_regex_fallback", "inserted": inserted}
                return {"mode": "llm", "inserted": inserted}
            except Exception as exc:
                last_error = str(exc)
                time.sleep(0.5 * (attempt + 1))
        inserted = seed_regex_tags_as_llm_tags(self.database_path)
        return {"mode": "regex_fallback_after_error", "inserted": inserted, "error": last_error[:500]}

    def _load_untagged(self, as_of: datetime | str, *, limit: int) -> list[dict[str, Any]]:
        as_of_s = _format_ts(as_of)
        day_start = as_of_s[:10] + " 00:00:00"
        with connect(self.database_path) as conn:
            rows = conn.execute(
                """
                SELECT
                    n.news_id, n.source, n.published_at_msk, n.received_at_msk,
                    n.title, n.text, n.url,
                    COALESCE(sci.story_id, n.news_id) AS story_key,
                    COALESCE(sc.source_count, 1) AS source_count,
                    COALESCE((
                        SELECT json_group_array(nt.ticker)
                        FROM news_tickers nt
                        WHERE nt.news_id = n.news_id
                    ), '[]') AS regex_tickers
                FROM news n
                LEFT JOIN story_cluster_items sci ON sci.news_id = n.news_id
                LEFT JOIN story_clusters sc ON sc.story_id = sci.story_id
                WHERE n.received_at_msk >= ?
                  AND n.received_at_msk <= ?
                  AND NOT EXISTS (
                    SELECT 1
                    FROM news_llm_tags lt
                    WHERE lt.news_id = n.news_id
                  )
                ORDER BY n.received_at_msk ASC, n.news_id ASC
                LIMIT ?
                """,
                (day_start, as_of_s, limit * 3),
            ).fetchall()
        out = []
        seen_story = set()
        for row in rows:
            story_key = row["story_key"] or row["news_id"]
            if story_key in seen_story:
                continue
            seen_story.add(story_key)
            out.append(
                {
                    "news_id": row["news_id"],
                    "source": row["source"],
                    "published_at": row["published_at_msk"],
                    "received_at": row["received_at_msk"],
                    "source_count": int(row["source_count"] or 1),
                    "title": row["title"] or "",
                    "text": row["text"] or "",
                    "url": row["url"] or "",
                    "regex_tickers": _json_list(row["regex_tickers"]),
                }
            )
            if len(out) >= limit:
                break
        return out

    def _prompt_payload(self, rows: list[dict[str, Any]], tickers: tuple[str, ...]) -> dict[str, Any]:
        return {
            "task": (
                "For each news item, add zero or more tags. Use ticker tags only from tickers. "
                "relation_type=direct for company-specific news, cluster for sector/supply-chain/peer impact, "
                "marketwide for broad MOEX/ruble/oil/rate/sanctions/geopolitical market impact. "
                "Use ticker='MARKET' for marketwide. Return strict JSON."
            ),
            "schema": {
                "items": [
                    {
                        "news_id": "same id",
                        "tags": [
                            {
                                "ticker": "SBER or MARKET",
                                "relation_type": "direct|cluster|marketwide",
                                "confidence": "low|medium|high",
                                "relation_strength": "0..1",
                                "reason": "short",
                                "tags": ["oil", "ruble"],
                            }
                        ],
                    }
                ]
            },
            "tickers": list(tickers),
            "news": [
                {
                    "news_id": row["news_id"],
                    "source": row["source"],
                    "published_at": row["published_at"],
                    "received_at": row["received_at"],
                    "source_count": row["source_count"],
                    "title": str(row["title"])[:300],
                    "text": str(row["text"])[:900],
                    "regex_tickers": row["regex_tickers"],
                }
                for row in rows
            ],
        }

    def _insert_tags(self, parsed: dict[str, Any], rows: list[dict[str, Any]], tickers: tuple[str, ...]) -> int:
        by_id = {row["news_id"]: row for row in rows}
        allowed = set(tickers) | {"MARKET", "MOEX_MARKET"}
        items = parsed.get("items", []) if isinstance(parsed, dict) else []
        if not isinstance(items, list):
            return 0
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        records = []
        for item in items:
            if not isinstance(item, dict):
                continue
            news_id = str(item.get("news_id", ""))
            if news_id not in by_id:
                continue
            tags = item.get("tags", [])
            if not isinstance(tags, list):
                continue
            for tag in tags:
                if not isinstance(tag, dict):
                    continue
                ticker = str(tag.get("ticker", "")).upper()
                relation_type = str(tag.get("relation_type", "")).lower()
                if relation_type not in {"direct", "cluster", "marketwide"}:
                    continue
                if relation_type == "marketwide":
                    ticker = "MARKET"
                if ticker not in allowed:
                    continue
                confidence = str(tag.get("confidence", "medium")).lower()
                if confidence not in {"low", "medium", "high"}:
                    confidence = "medium"
                strength = _clip_float(tag.get("relation_strength", 0.6 if relation_type == "cluster" else 0.8), 0.0, 1.0)
                records.append(
                    (
                        news_id,
                        ticker,
                        relation_type,
                        confidence,
                        strength,
                        str(tag.get("reason", ""))[:500],
                        json.dumps(tag.get("tags", []), ensure_ascii=False),
                        now,
                    )
                )
        if not records:
            return 0
        with connect(self.database_path) as conn:
            conn.executemany(
                """
                INSERT OR IGNORE INTO news_llm_tags(
                    news_id, ticker, relation_type, confidence, relation_strength,
                    reason, tags_json, created_at_msk
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                records,
            )
        return len(records)


def _extract_json(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end >= start:
        text = text[start : end + 1]
    return json.loads(text)


def _clip_float(value: Any, lo: float, hi: float) -> float:
    try:
        out = float(value)
    except Exception:
        return lo
    return max(lo, min(hi, out))


@dataclass
class NewsIngestionService:
    database_path: Path
    sources_config_path: Path
    tickers_config_path: Path
    bootstrap_enabled: bool = True
    logger: Any | None = None
    news_log_interval_seconds: int = 1800
    _stop_event: threading.Event = field(default_factory=threading.Event, init=False, repr=False)
    _thread: threading.Thread | None = field(default=None, init=False, repr=False)
    _scheduler: Any | None = field(default=None, init=False, repr=False)
    _log_aggregator: NewsIngestionLogAggregator | None = field(default=None, init=False, repr=False)

    def settings(self) -> Settings:
        return Settings(
            database_path=self.database_path,
            sources_config_path=self.sources_config_path,
            tickers_config_path=self.tickers_config_path,
            bootstrap_enabled=self.bootstrap_enabled,
        )

    def initialize(self) -> None:
        settings = self.settings()
        registry = SourceRegistry.load(settings.sources_config_path)
        ticker_registry = TickerRegistry.load(settings.tickers_config_path)
        initialize_database(settings.database_path)
        tag_existing_news(settings.database_path, ticker_registry, registry.sources)
        ensure_live_news_schema(settings.database_path)

    def start_background(self) -> threading.Thread:
        self._stop_event.clear()
        thread = threading.Thread(target=self._run_forever, name="news-ingestion", daemon=True)
        thread.start()
        self._thread = thread
        return thread

    def stop_background(self, timeout: float = 5.0) -> None:
        self._stop_event.set()
        if self._scheduler is not None:
            try:
                self._scheduler.shutdown(wait=False)
            except Exception:
                pass
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout)

    def _run_forever(self) -> None:
        asyncio.run(self._run_async())

    async def _run_async(self) -> None:
        logging.getLogger("apscheduler.executors.default").setLevel(logging.ERROR)
        logging.getLogger("apscheduler.scheduler").setLevel(logging.ERROR)
        settings = self.settings()
        registry = SourceRegistry.load(settings.sources_config_path)
        pipeline = IngestionPipeline(registry, settings)
        pipeline.initialize()
        self._log_aggregator = NewsIngestionLogAggregator(
            database_path=settings.database_path,
            logger=self.logger,
            interval_seconds=self.news_log_interval_seconds,
        )
        if self.logger is not None:
            self.logger.write(
                "news_ingestion_started",
                {
                    "database_path": str(settings.database_path),
                    "sources_config_path": str(settings.sources_config_path),
                    "tickers_config_path": str(settings.tickers_config_path),
                    "source_count": len(pipeline.enabled_sources()),
                    "log_interval_seconds": self.news_log_interval_seconds,
                    **_news_db_summary(settings.database_path),
                },
                stream="news_ingestion",
            )
        if settings.bootstrap_enabled:
            bootstrap_stats = await pipeline.bootstrap()
            for stats in bootstrap_stats:
                self._log_aggregator.observe(stats)
            self._log_aggregator.emit(force=True, event="news_ingestion_bootstrap_summary")
        scheduler = create_scheduler(
            pipeline,
            result_callback=self._log_aggregator.observe,
            error_callback=self._log_aggregator.log_scheduler_error,
        )
        self._scheduler = scheduler
        scheduler.start()
        try:
            while not self._stop_event.is_set():
                await asyncio.sleep(1)
        finally:
            try:
                scheduler.shutdown(wait=False)
            except Exception:
                pass
            self._scheduler = None


def seed_regex_tags_as_llm_tags(database_path: str | Path) -> int:
    """Convert regex/source ticker matches into direct tags for live scoring fallback."""
    ensure_live_news_schema(database_path)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with connect(database_path) as conn:
        rows = conn.execute(
            """
            SELECT nt.news_id, nt.ticker, nt.matched_by, nt.matched_terms_json
            FROM news_tickers nt
            LEFT JOIN news_llm_tags lt
              ON lt.news_id = nt.news_id AND lt.ticker = nt.ticker AND lt.relation_type = 'direct'
            WHERE lt.news_id IS NULL
            """
        ).fetchall()
        conn.executemany(
            """
            INSERT OR IGNORE INTO news_llm_tags(
                news_id, ticker, relation_type, confidence, relation_strength,
                reason, tags_json, created_at_msk
            )
            VALUES (?, ?, 'direct', 'medium', 0.65, ?, ?, ?)
            """,
            [
                (
                    row["news_id"],
                    row["ticker"],
                    f"regex/source match: {row['matched_by']}",
                    row["matched_terms_json"] or "[]",
                    now,
                )
                for row in rows
            ],
        )
    return len(rows)
