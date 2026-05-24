from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping


class StateStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    def initialize(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS bot_state (
                    key TEXT PRIMARY KEY,
                    value_json TEXT NOT NULL,
                    updated_at_msk TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS decisions (
                    decision_id TEXT PRIMARY KEY,
                    as_of_msk TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at_msk TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS orders (
                    idempotency_key TEXT PRIMARY KEY,
                    decision_id TEXT NOT NULL,
                    as_of_msk TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    response_json TEXT,
                    error TEXT,
                    created_at_msk TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS selector_returns (
                    timestamp_msk TEXT PRIMARY KEY,
                    selector_family_first REAL NOT NULL,
                    selector_news_aware REAL NOT NULL,
                    selector_marketwide_news REAL NOT NULL,
                    returns_json TEXT NOT NULL DEFAULT '{}',
                    created_at_msk TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS market_features (
                    timestamp_msk TEXT PRIMARY KEY,
                    features_json TEXT NOT NULL,
                    created_at_msk TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS paper_positions (
                    selector TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    weight REAL NOT NULL,
                    entry_price REAL NOT NULL,
                    updated_at_msk TEXT NOT NULL,
                    PRIMARY KEY(selector, ticker)
                );
                """
            )
            columns = {row["name"] for row in conn.execute("PRAGMA table_info(selector_returns)").fetchall()}
            if "returns_json" not in columns:
                conn.execute("ALTER TABLE selector_returns ADD COLUMN returns_json TEXT NOT NULL DEFAULT '{}'")

    @staticmethod
    def _now() -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def set_json(self, key: str, value: Mapping[str, Any]) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO bot_state(key, value_json, updated_at_msk)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value_json=excluded.value_json,
                    updated_at_msk=excluded.updated_at_msk
                """,
                (key, json.dumps(value, ensure_ascii=False, default=str), self._now()),
            )

    def get_json(self, key: str, default: dict | None = None) -> dict:
        with self.connect() as conn:
            row = conn.execute("SELECT value_json FROM bot_state WHERE key = ?", (key,)).fetchone()
        if row is None:
            return default or {}
        return json.loads(row["value_json"])

    def insert_decision(self, decision_id: str, as_of: str, payload: Mapping[str, Any]) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO decisions(decision_id, as_of_msk, payload_json, created_at_msk)
                VALUES (?, ?, ?, ?)
                """,
                (decision_id, as_of, json.dumps(payload, ensure_ascii=False, default=str), self._now()),
            )

    def insert_order_attempt(
        self,
        *,
        idempotency_key: str,
        decision_id: str,
        as_of: str,
        ticker: str,
        direction: str,
        quantity: int,
        status: str,
        request: Mapping[str, Any],
        response: Mapping[str, Any] | None = None,
        error: str | None = None,
    ) -> bool:
        with self.connect() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO orders(
                        idempotency_key, decision_id, as_of_msk, ticker, direction, quantity,
                        status, request_json, response_json, error, created_at_msk
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        idempotency_key,
                        decision_id,
                        as_of,
                        ticker,
                        direction,
                        int(quantity),
                        status,
                        json.dumps(request, ensure_ascii=False, default=str),
                        json.dumps(response, ensure_ascii=False, default=str) if response is not None else None,
                        error,
                        self._now(),
                    ),
                )
                return True
            except sqlite3.IntegrityError:
                return False

    def count_today_orders(self, date_prefix: str) -> int:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS c FROM orders WHERE as_of_msk LIKE ? AND status IN ('submitted','dry_run')",
                (f"{date_prefix}%",),
            ).fetchone()
        return int(row["c"])

    def append_selector_return(self, timestamp: str, returns: Mapping[str, float]) -> None:
        clean_returns = {str(k): float(v or 0.0) for k, v in returns.items()}
        with self.connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO selector_returns(
                    timestamp_msk, selector_family_first, selector_news_aware,
                    selector_marketwide_news, returns_json, created_at_msk
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    timestamp,
                    clean_returns.get("selector_family_first", 0.0),
                    clean_returns.get("selector_news_aware", 0.0),
                    clean_returns.get("selector_marketwide_news", 0.0),
                    json.dumps(clean_returns, ensure_ascii=False, default=str),
                    self._now(),
                ),
            )

    def count_selector_returns(self) -> int:
        with self.connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS c FROM selector_returns").fetchone()
        return int(row["c"] if row else 0)

    def count_lightgbm_training_rows(self, required_selectors: tuple[str, ...] | None = None) -> int:
        if required_selectors:
            rows = self.load_lightgbm_training_rows(limit=100000)
            required = set(required_selectors)
            return sum(1 for row in rows if required.issubset(set((row.get("returns") or {}).keys())))
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) AS c
                FROM market_features mf
                JOIN selector_returns sr
                  ON sr.timestamp_msk = mf.timestamp_msk
                """
            ).fetchone()
        return int(row["c"] if row else 0)

    def save_market_features(self, timestamp: str, features: Mapping[str, float]) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO market_features(timestamp_msk, features_json, created_at_msk)
                VALUES (?, ?, ?)
                ON CONFLICT(timestamp_msk) DO UPDATE SET
                    features_json=excluded.features_json,
                    created_at_msk=excluded.created_at_msk
                """,
                (timestamp, json.dumps(features, ensure_ascii=False, default=str), self._now()),
            )

    def load_lightgbm_training_rows(self, limit: int = 512) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    mf.timestamp_msk,
                    mf.features_json,
                    sr.returns_json,
                    sr.selector_family_first,
                    sr.selector_news_aware,
                    sr.selector_marketwide_news
                FROM market_features mf
                JOIN selector_returns sr
                  ON sr.timestamp_msk = mf.timestamp_msk
                ORDER BY mf.timestamp_msk DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        out = []
        for row in reversed(rows):
            try:
                features = json.loads(row["features_json"])
            except Exception:
                features = {}
            returns = _selector_returns_from_row(row)
            out.append(
                {
                    "timestamp": row["timestamp_msk"],
                    "features": features,
                    "returns": returns,
                }
            )
        return out

    def load_selector_history(self, limit: int = 512) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT timestamp_msk, returns_json, selector_family_first, selector_news_aware, selector_marketwide_news
                FROM selector_returns
                ORDER BY timestamp_msk DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        out = []
        for row in reversed(rows):
            returns = _selector_returns_from_row(row)
            out.append({"timestamp": row["timestamp_msk"], **returns, "returns": returns})
        return out

    def save_paper_positions(self, selector: str, weights: Mapping[str, float], prices: Mapping[str, float], as_of: str) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM paper_positions WHERE selector = ?", (selector,))
            conn.executemany(
                """
                INSERT INTO paper_positions(selector, ticker, weight, entry_price, updated_at_msk)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (selector, ticker, float(weight), float(prices.get(ticker, 0.0)), as_of)
                    for ticker, weight in weights.items()
                    if abs(float(weight)) > 1e-12 and float(prices.get(ticker, 0.0)) > 0
                ],
            )

    def load_paper_positions(self) -> dict[str, dict[str, dict[str, float]]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT selector, ticker, weight, entry_price FROM paper_positions"
            ).fetchall()
        out: dict[str, dict[str, dict[str, float]]] = {}
        for row in rows:
            out.setdefault(row["selector"], {})[row["ticker"]] = {
                "weight": float(row["weight"]),
                "entry_price": float(row["entry_price"]),
            }
        return out


def _selector_returns_from_row(row: sqlite3.Row) -> dict[str, float]:
    returns: dict[str, float] = {}
    try:
        parsed = json.loads(row["returns_json"] or "{}")
        if isinstance(parsed, dict):
            returns.update({str(k): float(v or 0.0) for k, v in parsed.items()})
    except Exception:
        returns = {}
    for key in ("selector_family_first", "selector_news_aware", "selector_marketwide_news"):
        if key not in returns:
            try:
                returns[key] = float(row[key] or 0.0)
            except Exception:
                returns[key] = 0.0
    return returns
