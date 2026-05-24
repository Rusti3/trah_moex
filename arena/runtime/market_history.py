from __future__ import annotations

import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd

KLINE_COLS = ["open", "high", "low", "close", "volume", "amount"]


class MarketHistoryCache:
    """Persistent OHLCV cache used by live Kronos and startup bootstrap."""

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
                CREATE TABLE IF NOT EXISTS ohlcv (
                    ticker TEXT NOT NULL,
                    interval_minutes INTEGER NOT NULL,
                    timestamp_msk TEXT NOT NULL,
                    end_msk TEXT NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL,
                    amount REAL NOT NULL,
                    updated_at_msk TEXT NOT NULL,
                    PRIMARY KEY(ticker, interval_minutes, timestamp_msk)
                );
                CREATE INDEX IF NOT EXISTS ix_ohlcv_lookup
                    ON ohlcv(ticker, interval_minutes, timestamp_msk DESC);
                """
            )

    def append_candles(self, ticker: str, interval_minutes: int, candles: pd.DataFrame) -> int:
        if candles.empty:
            return 0
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        df = candles.copy()
        if "timestamps" not in df.columns:
            raise ValueError("candles must include timestamps")
        if "end" not in df.columns:
            df["end"] = pd.to_datetime(df["timestamps"]) + pd.Timedelta(minutes=interval_minutes) - pd.Timedelta(seconds=1)
        df["timestamps"] = pd.to_datetime(df["timestamps"])
        df["end"] = pd.to_datetime(df["end"])
        rows = []
        for row in df.to_dict("records"):
            try:
                rows.append(
                    (
                        ticker,
                        int(interval_minutes),
                        _format_ts(row["timestamps"]),
                        _format_ts(row["end"]),
                        float(row.get("open", 0.0)),
                        float(row.get("high", 0.0)),
                        float(row.get("low", 0.0)),
                        float(row.get("close", 0.0)),
                        float(row.get("volume", 0.0) or 0.0),
                        float(row.get("amount", 0.0) or 0.0),
                        now,
                    )
                )
            except Exception:
                continue
        if not rows:
            return 0
        with self.connect() as conn:
            before = conn.total_changes
            conn.executemany(
                """
                INSERT OR REPLACE INTO ohlcv(
                    ticker, interval_minutes, timestamp_msk, end_msk,
                    open, high, low, close, volume, amount, updated_at_msk
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            return conn.total_changes - before

    def load_candles(
        self,
        ticker: str,
        *,
        interval_minutes: int,
        before: datetime | str | None = None,
        limit: int = 512,
    ) -> pd.DataFrame:
        params: list[object] = [ticker, int(interval_minutes)]
        where = "ticker = ? AND interval_minutes = ?"
        if before is not None:
            where += " AND timestamp_msk < ?"
            params.append(_format_ts(before))
        params.append(int(limit))
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT timestamp_msk, end_msk, open, high, low, close, volume, amount
                FROM ohlcv
                WHERE {where}
                ORDER BY timestamp_msk DESC
                LIMIT ?
                """,
                params,
            ).fetchall()
        if not rows:
            return pd.DataFrame()
        records = [dict(row) for row in reversed(rows)]
        df = pd.DataFrame(records).rename(columns={"timestamp_msk": "timestamps", "end_msk": "end"})
        df["timestamps"] = pd.to_datetime(df["timestamps"])
        df["end"] = pd.to_datetime(df["end"])
        for col in KLINE_COLS:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df[["timestamps", "end", *KLINE_COLS]].dropna(subset=["timestamps", "open", "high", "low", "close"])

    def latest_timestamp(self, ticker: str, *, interval_minutes: int) -> str | None:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT timestamp_msk
                FROM ohlcv
                WHERE ticker = ? AND interval_minutes = ?
                ORDER BY timestamp_msk DESC
                LIMIT 1
                """,
                (ticker, int(interval_minutes)),
            ).fetchone()
        return str(row["timestamp_msk"]) if row else None

    def ensure_history(
        self,
        ticker: str,
        *,
        as_of: datetime,
        days: int,
        interval_minutes: int,
        source_interval: int | None = None,
        drop_incomplete_last_candle: bool = True,
    ) -> int:
        from run_moex_hourly_rolling_backtest import fetch_moex_candles

        from_date = (as_of - timedelta(days=max(days, 1))).date().isoformat()
        till_date = as_of.date().isoformat()
        source = int(source_interval or interval_minutes)
        candles = fetch_moex_candles(
            ticker,
            from_date=from_date,
            till_date=till_date,
            interval=int(interval_minutes),
            source_interval=source,
            drop_incomplete_last_candle=drop_incomplete_last_candle,
        )
        if not candles.empty:
            candles = candles[pd.to_datetime(candles["timestamps"]) < pd.Timestamp(as_of)]
        return self.append_candles(ticker, interval_minutes, candles)

    def refresh(
        self,
        tickers: Iterable[str],
        *,
        as_of: datetime,
        days: int,
        intervals: tuple[int, ...] = (60,),
        time_budget_seconds: float | None = None,
    ) -> dict[str, int]:
        out: dict[str, int] = {}
        started = time.monotonic()
        for interval in intervals:
            source_interval = 10 if interval == 30 else interval
            for ticker in tickers:
                if time_budget_seconds is not None and time.monotonic() - started > time_budget_seconds:
                    return out
                key = f"{ticker}:{interval}"
                try:
                    out[key] = self.ensure_history(
                        ticker,
                        as_of=as_of,
                        days=days,
                        interval_minutes=interval,
                        source_interval=source_interval,
                    )
                except Exception:
                    out[key] = 0
        return out

    def timestamp_count(self, *, interval_minutes: int, before: datetime | str | None = None) -> int:
        params: list[object] = [int(interval_minutes)]
        where = "interval_minutes = ?"
        if before is not None:
            where += " AND timestamp_msk < ?"
            params.append(_format_ts(before))
        with self.connect() as conn:
            row = conn.execute(
                f"SELECT COUNT(DISTINCT timestamp_msk) AS c FROM ohlcv WHERE {where}",
                params,
            ).fetchone()
        return int(row["c"] if row else 0)


def _format_ts(value: datetime | str | pd.Timestamp) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d %H:%M:%S")
