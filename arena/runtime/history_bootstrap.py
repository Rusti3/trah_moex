from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping

import pandas as pd

from .feature_builder import build_live_features
from .market_history import MarketHistoryCache
from .news_service import NewsBuffer
from .portfolio import build_target_weights
from .schemas import BASE_SELECTORS, TOP20_TICKERS
from .state_store import StateStore


@dataclass(frozen=True)
class HistoryBootstrapResult:
    mode: str
    requested_intervals: int
    existing_rows_before: int
    rows_after: int
    inserted_rows: int
    market_rows_added: int
    elapsed_ms: int
    error: str = ""

    def to_log(self) -> dict[str, Any]:
        return {
            "history_bootstrap_mode": self.mode,
            "history_bootstrap_requested_intervals": self.requested_intervals,
            "history_bootstrap_existing_rows_before": self.existing_rows_before,
            "history_bootstrap_rows": self.rows_after,
            "history_bootstrap_inserted_rows": self.inserted_rows,
            "history_bootstrap_market_rows_added": self.market_rows_added,
            "history_bootstrap_elapsed_ms": self.elapsed_ms,
            "history_bootstrap_error": self.error,
        }


class HistoryBootstrapService:
    """Builds initial selector training rows from already-known market history."""

    def __init__(
        self,
        *,
        state: StateStore,
        market_history: MarketHistoryCache,
        news_buffer: NewsBuffer,
        tickers: tuple[str, ...] = TOP20_TICKERS,
        base_selectors: tuple[str, ...] = BASE_SELECTORS,
        base_selector_params: Mapping[str, Mapping[str, Any]] | None = None,
    ):
        self.state = state
        self.market_history = market_history
        self.news_buffer = news_buffer
        self.tickers = tickers
        self.base_selectors = tuple(base_selectors)
        self.base_selector_params = {k: dict(v) for k, v in (base_selector_params or {}).items()}
        self._thread: threading.Thread | None = None

    def bootstrap_initial(
        self,
        *,
        as_of: datetime,
        initial_intervals: int = 48,
        time_budget_seconds: float = 180.0,
        refresh_market_history: bool = True,
    ) -> HistoryBootstrapResult:
        existing = self.state.count_lightgbm_training_rows(required_selectors=self.base_selectors)
        if existing >= initial_intervals:
            return HistoryBootstrapResult(
                mode="already_ready",
                requested_intervals=initial_intervals,
                existing_rows_before=existing,
                rows_after=existing,
                inserted_rows=0,
                market_rows_added=0,
                elapsed_ms=0,
            )
        return self._bootstrap(
            as_of=as_of,
            requested_intervals=initial_intervals,
            existing_rows_before=existing,
            time_budget_seconds=time_budget_seconds,
            refresh_market_history=refresh_market_history,
        )

    def start_background(
        self,
        *,
        as_of: datetime,
        target_intervals: int = 512,
        time_budget_seconds: float = 900.0,
    ) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self._background_run,
            kwargs={"as_of": as_of, "target_intervals": target_intervals, "time_budget_seconds": time_budget_seconds},
            name="history-bootstrap",
            daemon=True,
        )
        self._thread.start()

    def _background_run(self, *, as_of: datetime, target_intervals: int, time_budget_seconds: float) -> None:
        try:
            existing = self.state.count_lightgbm_training_rows(required_selectors=self.base_selectors)
            if existing < target_intervals:
                self._bootstrap(
                    as_of=as_of,
                    requested_intervals=target_intervals,
                    existing_rows_before=existing,
                    time_budget_seconds=time_budget_seconds,
                    refresh_market_history=True,
                )
        except Exception:
            pass

    def _bootstrap(
        self,
        *,
        as_of: datetime,
        requested_intervals: int,
        existing_rows_before: int,
        time_budget_seconds: float,
        refresh_market_history: bool,
    ) -> HistoryBootstrapResult:
        started = time.monotonic()
        market_rows_added = 0
        try:
            if refresh_market_history:
                days = max(7, int(requested_intervals / 18) + 4)
                market_rows_added = self._refresh_market_history(
                    as_of=as_of,
                    days=days,
                    intervals=(30,),
                    started=started,
                    time_budget_seconds=time_budget_seconds,
                )
                if time.monotonic() - started > time_budget_seconds:
                    rows_after = self.state.count_lightgbm_training_rows(required_selectors=self.base_selectors)
                    return HistoryBootstrapResult(
                        mode="market_refresh_timeout",
                        requested_intervals=requested_intervals,
                        existing_rows_before=existing_rows_before,
                        rows_after=rows_after,
                        inserted_rows=max(rows_after - existing_rows_before, 0),
                        market_rows_added=market_rows_added,
                        elapsed_ms=_elapsed_ms(started),
                    )
            inserted = self._fill_selector_rows(as_of=as_of, requested_intervals=requested_intervals)
            rows_after = self.state.count_lightgbm_training_rows(required_selectors=self.base_selectors)
            return HistoryBootstrapResult(
                mode="market_proxy_bootstrap",
                requested_intervals=requested_intervals,
                existing_rows_before=existing_rows_before,
                rows_after=rows_after,
                inserted_rows=inserted,
                market_rows_added=market_rows_added,
                elapsed_ms=_elapsed_ms(started),
            )
        except Exception as exc:
            rows_after = self.state.count_lightgbm_training_rows(required_selectors=self.base_selectors)
            return HistoryBootstrapResult(
                mode="fallback_live_only",
                requested_intervals=requested_intervals,
                existing_rows_before=existing_rows_before,
                rows_after=rows_after,
                inserted_rows=max(rows_after - existing_rows_before, 0),
                market_rows_added=market_rows_added,
                elapsed_ms=_elapsed_ms(started),
                error=str(exc)[:500],
            )

    def _refresh_market_history(
        self,
        *,
        as_of: datetime,
        days: int,
        intervals: tuple[int, ...],
        started: float,
        time_budget_seconds: float,
    ) -> int:
        added = 0
        for interval in intervals:
            source_interval = 10 if interval == 30 else interval
            for ticker in self.tickers:
                if time.monotonic() - started > time_budget_seconds:
                    return added
                try:
                    added += int(
                        self.market_history.ensure_history(
                            ticker,
                            as_of=as_of,
                            days=days,
                            interval_minutes=interval,
                            source_interval=source_interval,
                            drop_incomplete_last_candle=True,
                        )
                    )
                except Exception:
                    continue
        return added

    def _fill_selector_rows(self, *, as_of: datetime, requested_intervals: int) -> int:
        frames = {
            ticker: self.market_history.load_candles(ticker, interval_minutes=30, before=as_of, limit=requested_intervals + 12)
            for ticker in self.tickers
        }
        timestamps = sorted(
            {
                pd.Timestamp(ts).to_pydatetime().replace(tzinfo=None)
                for df in frames.values()
                if not df.empty
                for ts in df["timestamps"].tolist()
                if pd.Timestamp(ts).to_pydatetime().replace(tzinfo=None) < as_of
            }
        )
        if len(timestamps) < 3:
            return 0
        selected = timestamps[-(requested_intervals + 1) :]
        prices_by_ts: dict[datetime, dict[str, float]] = {ts: {} for ts in selected}
        for ticker, df in frames.items():
            if df.empty:
                continue
            for row in df.to_dict("records"):
                ts = pd.Timestamp(row["timestamps"]).to_pydatetime().replace(tzinfo=None)
                if ts in prices_by_ts:
                    close = _safe_float(row.get("close"))
                    if close > 0:
                        prices_by_ts[ts][ticker] = close
        inserted = 0
        previous_prices: dict[str, float] = {}
        latest_base_decisions: dict[str, dict[str, Any]] = {}
        latest_prices: dict[str, float] = {}
        latest_as_of = ""
        for idx, ts in enumerate(selected[:-1]):
            next_ts = selected[idx + 1]
            prices = prices_by_ts.get(ts, {})
            next_prices = prices_by_ts.get(next_ts, {})
            if len(prices) < 3 or len(next_prices) < 3:
                previous_prices = prices or previous_prices
                continue
            kronos_scores = _percentile_scores(
                {
                    ticker: (price / previous_prices[ticker] - 1.0)
                    for ticker, price in prices.items()
                    if previous_prices.get(ticker, 0.0) > 0
                },
                self.tickers,
            )
            llm_raw = {
                ticker: {
                    "bullish_score": 0.5,
                    "confidence": 0.0,
                    "relation_strength": 0.0,
                    "direct_effect": "none",
                    "marketwide_effect": "none",
                    "already_priced_risk": 0.5,
                    "reason": "history bootstrap neutral",
                }
                for ticker in self.tickers
            }
            cost_depth = {
                ticker: {
                    "tradable": ticker in prices,
                    "last_price": prices.get(ticker, 0.0),
                    "estimated_cost_pct": 0.0,
                    "bbo_spread_pct": 0.0,
                    "source": "history_bootstrap",
                }
                for ticker in self.tickers
            }
            news_context = self.news_buffer.get_context(ts, self.tickers)
            features = build_live_features(
                as_of=ts,
                kronos_scores=kronos_scores,
                llm_raw=llm_raw,
                cost_depth=cost_depth,
                news_context=news_context,
                tickers=self.tickers,
            )
            base_decisions = self._build_base_decisions(ts, kronos_scores, cost_depth, latest_base_decisions)
            returns = {}
            for selector_name, decision in base_decisions.items():
                total = 0.0
                for ticker, weight in (decision.get("target_weights") or {}).items():
                    if prices.get(ticker, 0.0) > 0 and next_prices.get(ticker, 0.0) > 0:
                        total += float(weight) * (next_prices[ticker] / prices[ticker] - 1.0)
                returns[selector_name] = total
            ts_s = ts.strftime("%Y-%m-%d %H:%M:%S")
            self.state.save_market_features(ts_s, features)
            self.state.append_selector_return(ts_s, returns)
            inserted += 1
            latest_base_decisions = base_decisions
            latest_prices = prices
            latest_as_of = ts_s
            previous_prices = prices
        if latest_base_decisions and latest_prices and latest_as_of:
            for selector_name, decision in latest_base_decisions.items():
                self.state.save_paper_positions(selector_name, decision.get("target_weights", {}), latest_prices, latest_as_of)
            self.state.set_json("paper_last_as_of", {"as_of": latest_as_of})
        return inserted

    def _build_base_decisions(
        self,
        as_of: datetime,
        kronos_scores: Mapping[str, float],
        cost_depth: Mapping[str, Mapping[str, Any]],
        previous: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        neutral_llm = {ticker: 0.5 for ticker in self.tickers}
        previous = previous or {}
        for selector_name in self.base_selectors:
            params = dict(self.base_selector_params.get(selector_name, {}))
            if not _selector_due(as_of, int(float(params.get("rebalance_minutes", 30)))) and selector_name in previous:
                out[selector_name] = {**params, "target_weights": dict(previous[selector_name].get("target_weights") or {})}
                continue
            positions = build_target_weights(
                kronos_scores,
                neutral_llm,
                cost_depth,
                kronos_weight=float(params.get("kronos_weight", 1.0)),
                llm_weight=float(params.get("llm_weight", 1.0)),
                threshold=float(params.get("threshold", 0.65)),
                rank_power=float(params.get("rank_power", 2.0)),
                max_gross=float(params.get("max_gross", 1.0)),
                allow_short=bool(params.get("allow_short", True)),
                source=selector_name,
            )
            out[selector_name] = {**params, "target_weights": {p.ticker: p.weight for p in positions}}
        return out


def _percentile_scores(values: Mapping[str, float], tickers: tuple[str, ...]) -> dict[str, float]:
    clean = {ticker: float(value) for ticker, value in values.items() if value == value}
    if not clean:
        return {ticker: 0.5 for ticker in tickers}
    sorted_items = sorted(clean.items(), key=lambda kv: kv[1])
    n = len(sorted_items)
    out = {ticker: (idx + 1) / n for idx, (ticker, _) in enumerate(sorted_items)}
    for ticker in tickers:
        out.setdefault(ticker, 0.5)
    return out


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return 0.0
    return out if out == out else 0.0


def _elapsed_ms(started: float) -> int:
    return int(round((time.monotonic() - started) * 1000))


def _selector_due(as_of: datetime, rebalance_minutes: int) -> bool:
    if rebalance_minutes <= 30:
        return True
    anchor_minutes = 12 * 60
    current_minutes = as_of.hour * 60 + as_of.minute
    if current_minutes < anchor_minutes:
        return False
    return (current_minutes - anchor_minutes) % rebalance_minutes == 0
