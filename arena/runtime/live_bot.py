from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from .arena_go_client import ArenaGoClient
from .decision import make_decision
from .feature_builder import build_live_features
from .history_bootstrap import HistoryBootstrapService
from .jsonl_logger import JsonlLogger
from .kronos_provider import KronosTop20Provider
from .lightgbm_selector import LiveLightGBMSelector
from .llm_scorer import LLMNewsScorer
from .market_history import MarketHistoryCache
from .market_data_provider import MoexRealtimeProvider
from .market_hours import is_market_open, next_decision_time, now_msk, sleep_seconds_until
from .news_service import LLMNewsTagger, NewsBuffer, NewsIngestionService
from .order_manager import OrderManager
from .portfolio import build_target_weights
from .schemas import BASE_SELECTORS
from .selector import RollingRankWeightedSelector
from .settings import RuntimeSettings, load_settings, load_yaml
from .state_store import StateStore


DEFAULT_BASE_SELECTOR_PARAMS = {
    "selector_family_first": {"kronos_weight": 1.0, "llm_weight": 1.0, "threshold": 0.65, "rank_power": 2.0},
    "selector_news_aware": {"kronos_weight": 1.0, "llm_weight": 2.0, "threshold": 0.65, "rank_power": 2.0},
    "selector_marketwide_news": {"kronos_weight": 0.5, "llm_weight": 1.0, "threshold": 0.65, "rank_power": 2.0},
}


@dataclass
class PrecomputeSnapshot:
    as_of: str
    started_at: float
    positions: dict[str, int] | None = None
    bot_snapshot: Any | None = None
    news_context: dict[str, Any] | None = None
    news_context_key: str = ""
    llm_raw: dict[str, dict[str, Any]] | None = None
    cost_depth: dict[str, dict[str, Any]] | None = None
    market_history_rows_added: int = 0
    errors: dict[str, str] | None = None


def load_env_file(path: str | Path = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


class ArenaLiveBot:
    def __init__(self, settings: RuntimeSettings):
        self.settings = settings
        self.settings.data_dir.mkdir(parents=True, exist_ok=True)
        self.settings.logs_dir.mkdir(parents=True, exist_ok=True)
        self.logger = JsonlLogger(settings.logs_dir)
        self.state = StateStore(settings.state_db_path)
        self.market_history = MarketHistoryCache(settings.market_history_db_path)
        self.client = ArenaGoClient(
            settings.arena_api_key,
            base_url=settings.arena_base_url,
        )
        self.market = MoexRealtimeProvider(token=settings.moex_algo_token)
        self.kronos = KronosTop20Provider(
            weights_dir=settings.weights_dir,
            weights_mode=settings.weights_mode,
            device=settings.device,
            market_history=self.market_history,
        )
        self.llm = LLMNewsScorer(
            cache_path=settings.llm_cache_path,
            base_url=settings.polza_base_url,
            model=settings.polza_model,
        )
        self.news_ingestion = NewsIngestionService(
            database_path=settings.news_db_path,
            sources_config_path=settings.news_sources_config_path,
            tickers_config_path=settings.news_tickers_config_path,
        )
        self.news_tagger = LLMNewsTagger(
            settings.news_db_path,
            base_url=settings.polza_base_url,
            model=settings.polza_model,
        )
        self.news_buffer = NewsBuffer(settings.news_db_path)
        self.order_manager = OrderManager(
            client=self.client,
            state=self.state,
            bot_name=settings.bot_name,
            lot_sizes=settings.lot_sizes,
            live_orders=settings.live_orders,
            max_daily_trades=settings.max_daily_trades,
            min_order_value_rub=settings.min_order_value_rub,
        )
        cfg = load_yaml(settings.production_config_path)
        strategy = cfg.get("strategy", {})
        self.selector = RollingRankWeightedSelector(
            lookback=int(strategy.get("lookback_intervals", 24)),
            rank_power=float(strategy.get("rank_power", 2.0)),
        )
        live_selector = cfg.get("live_selector", {})
        self.live_selector_mode = str(live_selector.get("mode", "lightgbm_rank_weighted"))
        self.lightgbm_selector = LiveLightGBMSelector(
            min_train_intervals=int(live_selector.get("min_train_intervals", 48)),
            train_lookback_intervals=int(live_selector.get("train_lookback_intervals", 512)),
            rank_power=float(strategy.get("rank_power", 2.0)),
            n_estimators=int(live_selector.get("lightgbm_estimators", 60)),
        )
        self.base_selector_params = {
            **DEFAULT_BASE_SELECTOR_PARAMS,
            **(cfg.get("base_selector_params") or {}),
        }
        self.history_bootstrap = HistoryBootstrapService(
            state=self.state,
            market_history=self.market_history,
            news_buffer=self.news_buffer,
            tickers=settings.tickers,
            base_selector_params=self.base_selector_params,
        )

    def initialize(self) -> None:
        self.logger.write("startup", {"live_orders": self.settings.live_orders, "bot": self.settings.bot_name})
        self.news_ingestion.initialize()
        self.news_ingestion.start_background()
        bootstrap_result = self.history_bootstrap.bootstrap_initial(
            as_of=now_msk(),
            initial_intervals=self.settings.history_bootstrap_initial_intervals,
            time_budget_seconds=self.settings.history_bootstrap_time_budget_seconds,
        )
        self.logger.write("history_bootstrap_ready", bootstrap_result.to_log())
        self.history_bootstrap.start_background(
            as_of=now_msk(),
            target_intervals=self.settings.history_bootstrap_background_intervals,
        )
        try:
            print("[arena-live] kronos warmup start", flush=True)
            self.logger.write("kronos_warm_start", {"weights_dir": str(self.settings.weights_dir), "device": self.settings.device})
            self.kronos.warm()
            print("[arena-live] kronos warmup ok", flush=True)
            self.logger.write("kronos_warm_ok", {})
        except Exception as exc:
            print(f"[arena-live] kronos warmup error: {exc}", flush=True)
            self.logger.write("kronos_warm_error", {"error": str(exc)})
        positions = self.order_manager.reconcile_positions()
        self.logger.write("positions_reconciled", {"positions_count": len(positions), "positions": positions})

    def shutdown(self) -> None:
        self.news_ingestion.stop_background()

    async def run_forever(self) -> None:
        self.initialize()
        while True:
            current = now_msk()
            target = next_decision_time(
                current,
                interval_minutes=self.settings.decision_interval_minutes,
                decision_delay_seconds=self.settings.candle_close_wait_seconds,
            )
            precompute_at = target - timedelta(seconds=self.settings.precompute_seconds)
            sleep_seconds = sleep_seconds_until(precompute_at, current)
            self.logger.write(
                "loop_wait",
                {
                    "now": current.strftime("%Y-%m-%d %H:%M:%S"),
                    "precompute_at": precompute_at.strftime("%Y-%m-%d %H:%M:%S"),
                    "next_decision": target.strftime("%Y-%m-%d %H:%M:%S"),
                    "sleep_seconds": round(sleep_seconds, 3),
                },
            )
            await asyncio.sleep(sleep_seconds)
            precompute = None
            try:
                precompute = await self.precompute_once(target)
            except Exception as exc:
                self.logger.write("precompute_error", {"as_of": target.strftime("%Y-%m-%d %H:%M:%S"), "error": repr(exc)})
            await asyncio.sleep(sleep_seconds_until(target))
            try:
                self.logger.write("run_once_begin", {"as_of": target.strftime("%Y-%m-%d %H:%M:%S")})
                await self.run_once(target, precompute=precompute)
            except Exception as exc:
                self.logger.write("run_once_error", {"error": repr(exc)})
                await asyncio.sleep(5)

    async def precompute_once(self, as_of: datetime) -> PrecomputeSnapshot:
        as_of_s = as_of.strftime("%Y-%m-%d %H:%M:%S")
        started = time.monotonic()
        self.logger.write("precompute_start", {"as_of": as_of_s})
        snapshot = PrecomputeSnapshot(as_of=as_of_s, started_at=started, errors={})
        positions_task = asyncio.create_task(asyncio.to_thread(self.order_manager.reconcile_positions))
        bot_task = asyncio.create_task(asyncio.to_thread(self.order_manager.bot_snapshot))
        history_task = asyncio.create_task(
            asyncio.to_thread(
                self.kronos.prefetch_history,
                as_of,
                self.settings.tickers,
                time_budget_seconds=max(1.0, min(60.0, float(self.settings.precompute_seconds))),
            )
        )
        cost_task = asyncio.create_task(self.market.current_cost_depth(as_of, self.settings.tickers))
        try:
            tag_result = await asyncio.to_thread(self.news_tagger.tag_new_news, as_of, self.settings.tickers)
            snapshot.news_context = self.news_buffer.get_context(as_of, self.settings.tickers)
            snapshot.news_context_key = _context_key(snapshot.news_context)
            self.logger.write("precompute_news_ready", {"as_of": as_of_s, "news_tagging": tag_result})
            try:
                snapshot.llm_raw = await asyncio.wait_for(
                    self.llm.score_context(snapshot.news_context, self.settings.tickers),
                    timeout=self.settings.max_llm_wait_seconds,
                )
            except Exception as exc:
                snapshot.errors["llm"] = str(exc)
        except Exception as exc:
            snapshot.errors["news"] = str(exc)
        for name, task in {
            "positions": positions_task,
            "bot": bot_task,
            "history": history_task,
            "cost_depth": cost_task,
        }.items():
            try:
                result = await asyncio.wait_for(task, timeout=max(1, min(30, self.settings.precompute_seconds)))
                if name == "positions":
                    snapshot.positions = result
                elif name == "bot":
                    snapshot.bot_snapshot = result
                elif name == "history":
                    snapshot.market_history_rows_added = sum(int(v) for v in (result or {}).values())
                elif name == "cost_depth":
                    snapshot.cost_depth = result
            except Exception as exc:
                snapshot.errors[name] = str(exc)
        self.logger.write(
            "precompute_ready",
            {
                "as_of": as_of_s,
                "elapsed_ms": _elapsed_ms(started),
                "positions_count": len(snapshot.positions or {}),
                "has_bot_snapshot": snapshot.bot_snapshot is not None,
                "has_llm": snapshot.llm_raw is not None,
                "has_cost_depth": snapshot.cost_depth is not None,
                "market_history_rows_added": snapshot.market_history_rows_added,
                "errors": snapshot.errors or {},
            },
        )
        return snapshot

    async def run_once(self, as_of: datetime | None = None, *, precompute: PrecomputeSnapshot | None = None) -> dict[str, Any]:
        as_of = as_of or now_msk()
        as_of_s = as_of.strftime("%Y-%m-%d %H:%M:%S")
        run_started = time.monotonic()
        self.logger.write("run_once_start", {"as_of": as_of_s, "live_orders": self.settings.live_orders})
        if not is_market_open(as_of):
            self.logger.write("market_closed_skip", {"as_of": as_of_s})
            return {"status": "market_closed"}

        candle_started = time.monotonic()
        try:
            candle_rows = await asyncio.wait_for(
                asyncio.to_thread(
                    self.market_history.refresh,
                    self.settings.tickers,
                    as_of=as_of,
                    days=2,
                    intervals=(30, 60),
                    time_budget_seconds=min(20.0, max(1.0, float(self.settings.execution_deadline_seconds) / 3.0)),
                ),
                timeout=min(20, max(1, self.settings.execution_deadline_seconds // 3)),
            )
            candle_rows_added = sum(int(v) for v in candle_rows.values())
            self.logger.write("candle_append_ready", {"as_of": as_of_s, "rows_added": candle_rows_added, "elapsed_ms": _elapsed_ms(candle_started)})
        except Exception as exc:
            self.logger.write("candle_append_error", {"as_of": as_of_s, "error": str(exc), "elapsed_ms": _elapsed_ms(candle_started)})

        tag_result = await asyncio.to_thread(self.news_tagger.tag_new_news, as_of, self.settings.tickers)
        news_context = self.news_buffer.get_context(as_of, self.settings.tickers)
        news_rows = sum(len(v) for v in news_context.get("per_ticker_news", {}).values()) + len(news_context.get("marketwide_news", []))
        self.logger.write(
            "news_context_ready",
            {
                "as_of": as_of_s,
                "news_rows": news_rows,
                "per_ticker_news_rows": sum(len(v) for v in news_context.get("per_ticker_news", {}).values()),
                "marketwide_news_rows": len(news_context.get("marketwide_news", [])),
                "news_tagging": tag_result,
            },
        )

        kronos_task = asyncio.create_task(self.kronos.forecast_bullish_scores(as_of, self.settings.tickers))
        if precompute is not None and precompute.news_context_key == _context_key(news_context) and precompute.llm_raw is not None:
            llm_task = _completed_task(precompute.llm_raw)
            llm_source = "precompute_cache"
        else:
            llm_task = asyncio.create_task(self.llm.score_context(news_context, self.settings.tickers))
            llm_source = "final_call"
        moex_task = asyncio.create_task(self.market.current_cost_depth(as_of, self.settings.tickers))
        self.logger.write(
            "parallel_tasks_started",
            {
                "as_of": as_of_s,
                "max_kronos_wait_seconds": self.settings.max_kronos_wait_seconds,
                "max_llm_wait_seconds": self.settings.max_llm_wait_seconds,
                "max_moex_wait_seconds": self.settings.max_moex_wait_seconds,
                "execution_deadline_seconds": self.settings.execution_deadline_seconds,
                "llm_source": llm_source,
            },
        )

        kronos_stale = False
        try:
            kronos_scores = await _wait_component(
                kronos_task,
                name="kronos",
                run_started=run_started,
                deadline_seconds=self.settings.execution_deadline_seconds,
                component_timeout=self.settings.max_kronos_wait_seconds,
            )
        except Exception as exc:
            kronos_task.cancel()
            kronos_scores = self.state.get_json("last_good_kronos_scores", {})
            kronos_stale = True
            if not kronos_scores:
                kronos_scores = {ticker: 0.5 for ticker in self.settings.tickers}
            self.logger.write("kronos_fallback", {"as_of": as_of_s, "error": str(exc)})
        else:
            self.state.set_json("last_good_kronos_scores", dict(kronos_scores))
            self.logger.write("kronos_scores_ready", {**_score_summary(as_of_s, kronos_scores), "kronos_ready_ms": _elapsed_ms(run_started)})

        try:
            llm_raw = await _wait_component(
                llm_task,
                name="llm",
                run_started=run_started,
                deadline_seconds=self.settings.execution_deadline_seconds,
                component_timeout=self.settings.max_llm_wait_seconds,
            )
        except Exception as exc:
            llm_task.cancel()
            llm_raw = {
                ticker: {"bullish_score": 0.5, "confidence": 0.0, "reason": f"llm timeout: {exc}"}
                for ticker in self.settings.tickers
            }
            self.logger.write("llm_fallback", {"as_of": as_of_s, "error": str(exc)})
        llm_scores = {ticker: float(row.get("bullish_score", 0.5)) for ticker, row in llm_raw.items()}
        self.logger.write("llm_scores_ready", {**_score_summary(as_of_s, llm_scores), "llm_ready_ms": _elapsed_ms(run_started), "llm_source": llm_source})

        try:
            cost_depth = await _wait_component(
                moex_task,
                name="moex",
                run_started=run_started,
                deadline_seconds=self.settings.execution_deadline_seconds,
                component_timeout=self.settings.max_moex_wait_seconds,
            )
        except Exception as exc:
            moex_task.cancel()
            cost_depth = (
                (precompute.cost_depth if precompute is not None else None)
                or self.market.last_good_cost_depth
                or {ticker: {"tradable": True, "last_price": 0.0} for ticker in self.settings.tickers}
            )
            self.logger.write("moex_fallback", {"as_of": as_of_s, "error": str(exc)})
        else:
            self.logger.write(
                "moex_cost_depth_ready",
                {
                    "as_of": as_of_s,
                    "rows": len(cost_depth),
                    "tradable": sum(1 for row in cost_depth.values() if row.get("tradable", True)),
                    "priced": sum(1 for row in cost_depth.values() if float(row.get("last_price", 0.0) or 0.0) > 0),
                    "moex_ready_ms": _elapsed_ms(run_started),
                },
            )

        prices = {ticker: float(row.get("last_price", 0.0) or 0.0) for ticker, row in cost_depth.items()}
        self._update_paper_selector_returns(as_of_s, prices)
        market_features = build_live_features(
            as_of=as_of,
            kronos_scores=kronos_scores,
            llm_raw=llm_raw,
            cost_depth=cost_depth,
            news_context=news_context,
            tickers=self.settings.tickers,
        )
        self.state.save_market_features(as_of_s, market_features)
        self.logger.write(
            "market_features_saved",
            {
                "as_of": as_of_s,
                "feature_count": len(market_features),
                "total_news_count": market_features.get("total_news_count"),
                "kronos_spread": market_features.get("kronos_spread"),
                "llm_spread": market_features.get("llm_spread"),
            },
        )

        base_decisions = self._build_base_decisions(kronos_scores, llm_scores, cost_depth)
        self.logger.write("base_decisions_ready", {"as_of": as_of_s, "summary": _base_decision_summary(base_decisions)})
        lightgbm_result = None
        selector_weights_override = None
        if self.live_selector_mode == "lightgbm_rank_weighted":
            training_rows = self.state.load_lightgbm_training_rows(limit=self.lightgbm_selector.train_lookback_intervals)
            self.logger.write("lightgbm_training_rows_loaded", {"as_of": as_of_s, "rows": len(training_rows)})
            lightgbm_result = self.lightgbm_selector.predict_weights(
                current_features=market_features,
                training_rows=training_rows,
            )
            if lightgbm_result is not None and lightgbm_result.selector_weights:
                selector_weights_override = lightgbm_result.selector_weights
            self.logger.write(
                "selector_model_ready",
                {
                    "as_of": as_of_s,
                    "mode": lightgbm_result.mode if lightgbm_result is not None else "rolling_fallback",
                    "trained_rows": lightgbm_result.trained_rows if lightgbm_result is not None else 0,
                    "weights": lightgbm_result.selector_weights if lightgbm_result is not None else {},
                    "reason": lightgbm_result.reason if lightgbm_result is not None else "not_enough_history_or_disabled",
                },
            )
        history = {
            "selector_returns": self.state.load_selector_history(limit=512),
            "base_selector_decisions": base_decisions,
        }
        decision = make_decision(
            as_of=as_of_s,
            kronos_scores=kronos_scores,
            llm_scores=llm_scores,
            cost_depth=cost_depth,
            history=history,
            selector=self.selector,
            selector_weights_override=selector_weights_override,
            max_gross=self.settings.max_gross_exposure,
        )
        decision_id = _decision_id(as_of_s, decision.selector_weights)
        positions = precompute.positions if precompute is not None and precompute.positions is not None else self.order_manager.reconcile_positions()
        bot_snapshot = precompute.bot_snapshot if precompute is not None and precompute.bot_snapshot is not None else self.order_manager.bot_snapshot()
        cash_balance = bot_snapshot.cash_balance
        equity = self.order_manager.estimate_equity(positions, prices, cash_balance=cash_balance)
        planned = self.order_manager.plan_orders(
            decision,
            positions=positions,
            prices=prices,
            equity=equity,
            cash_balance=cash_balance,
        )
        cash_after_planned_buys = cash_balance - sum(o.order_value for o in planned if o.direction == "B")
        self.logger.write(
            "orders_planned",
            {
                "as_of": as_of_s,
                "equity": equity,
                "cash_balance": cash_balance,
                "portfolio_value_for_targets": equity,
                "cash_after_planned_buys": max(cash_after_planned_buys, 0.0),
                "positions_count": len(positions),
                "planned_count": len(planned),
                "live_orders": self.settings.live_orders,
                "orders": [
                    {
                        "ticker": o.ticker,
                        "direction": o.direction,
                        "quantity_lots": o.quantity,
                        "requested_lots": o.requested_quantity,
                        "capped_lots": o.capped_quantity,
                        "lot_size": o.lot_size,
                        "current_lots": o.current_position,
                        "target_lots": o.target_position,
                        "requested_target_lots": o.requested_target_position,
                        "price": o.price,
                        "order_value": o.order_value,
                        "cash_before_order": o.cash_before_order,
                        "cash_after_order": o.cash_after_order,
                        "cap_reason": o.cap_reason,
                    }
                    for o in planned
                ],
            },
        )
        self.logger.write("orders_submit_start", {"as_of": as_of_s, "orders": len(planned), "elapsed_ms": _elapsed_ms(run_started)})
        order_results = self.order_manager.execute_orders(decision_id, as_of_s, planned)
        order_log_payload = {
            "as_of": as_of_s,
            "results_count": len(order_results),
            "statuses": _status_counts(order_results),
            "elapsed_ms": _elapsed_ms(run_started),
        }
        self.logger.write(
            "orders_submit_done",
            order_log_payload,
        )
        self.logger.write("orders_executed", order_log_payload)

        self._save_paper_positions(base_decisions, prices, as_of_s)
        payload = {
            "decision_id": decision_id,
            "as_of": as_of_s,
            "kronos_stale": kronos_stale,
            "selector_weights": dict(decision.selector_weights),
            "selector_model": {
                "mode": lightgbm_result.mode if lightgbm_result is not None else "rolling_fallback",
                "trained_rows": lightgbm_result.trained_rows if lightgbm_result is not None else 0,
                "scores": lightgbm_result.selector_scores if lightgbm_result is not None else {},
                "reason": lightgbm_result.reason if lightgbm_result is not None else "",
            },
            "cash_balance": cash_balance,
            "portfolio_value_for_targets": equity,
            "cash_after_planned_buys": max(cash_after_planned_buys, 0.0),
            "target_positions": decision.to_order_targets(),
            "orders": order_results,
            "news_rows": news_rows,
            "news_tagging": tag_result,
            "latency_ms": _elapsed_ms(run_started),
        }
        self.state.insert_decision(decision_id, as_of_s, payload)
        self.logger.write("decision", payload)
        return payload

    def _build_base_decisions(self, kronos_scores, llm_scores, cost_depth) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for selector_name in BASE_SELECTORS:
            params = dict(self.base_selector_params.get(selector_name, DEFAULT_BASE_SELECTOR_PARAMS[selector_name]))
            positions = build_target_weights(
                kronos_scores,
                llm_scores,
                cost_depth,
                kronos_weight=float(params.get("kronos_weight", 1.0)),
                llm_weight=float(params.get("llm_weight", 1.0)),
                threshold=float(params.get("threshold", 0.65)),
                rank_power=float(params.get("rank_power", 2.0)),
                max_gross=float(params.get("max_gross", 1.0)),
                allow_short=bool(params.get("allow_short", True)),
                source=selector_name,
            )
            out[selector_name] = {
                **params,
                "target_weights": {p.ticker: p.weight for p in positions},
            }
        return out

    def _update_paper_selector_returns(self, as_of_s: str, prices: dict[str, float]) -> None:
        last = self.state.get_json("paper_last_as_of", {})
        last_ts = last.get("as_of")
        if not last_ts:
            return
        positions = self.state.load_paper_positions()
        returns = {}
        for selector_name in BASE_SELECTORS:
            total = 0.0
            for ticker, row in positions.get(selector_name, {}).items():
                entry = float(row.get("entry_price", 0.0) or 0.0)
                current = float(prices.get(ticker, 0.0) or 0.0)
                weight = float(row.get("weight", 0.0) or 0.0)
                if entry > 0 and current > 0:
                    total += weight * (current / entry - 1.0)
            returns[selector_name] = total
        self.state.append_selector_return(last_ts, returns)

    def _save_paper_positions(self, base_decisions: dict[str, dict[str, Any]], prices: dict[str, float], as_of_s: str) -> None:
        for selector_name, decision in base_decisions.items():
            self.state.save_paper_positions(selector_name, decision.get("target_weights", {}), prices, as_of_s)
        self.state.set_json("paper_last_as_of", {"as_of": as_of_s})


def _decision_id(as_of: str, selector_weights: dict[str, float]) -> str:
    raw = json.dumps({"as_of": as_of, "selector_weights": selector_weights}, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _score_summary(as_of: str, scores: dict[str, float]) -> dict[str, Any]:
    values = [float(v) for v in scores.values()]
    if not values:
        return {"as_of": as_of, "count": 0}
    top = sorted(scores.items(), key=lambda kv: float(kv[1]), reverse=True)[:3]
    bottom = sorted(scores.items(), key=lambda kv: float(kv[1]))[:3]
    return {
        "as_of": as_of,
        "count": len(scores),
        "min": min(values),
        "max": max(values),
        "top3": [{"ticker": ticker, "score": float(score)} for ticker, score in top],
        "bottom3": [{"ticker": ticker, "score": float(score)} for ticker, score in bottom],
    }


def _base_decision_summary(base_decisions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for selector, row in base_decisions.items():
        weights = {str(k): float(v) for k, v in (row.get("target_weights") or {}).items()}
        out[selector] = {
            "positions": len(weights),
            "long": sum(1 for value in weights.values() if value > 0),
            "short": sum(1 for value in weights.values() if value < 0),
            "gross": sum(abs(value) for value in weights.values()),
        }
    return out


def _status_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status", "unknown"))
        out[status] = out.get(status, 0) + 1
    return out


def _completed_task(value: Any) -> asyncio.Task:
    async def _done():
        return value

    return asyncio.create_task(_done())


async def _wait_component(
    task: asyncio.Task,
    *,
    name: str,
    run_started: float,
    deadline_seconds: int,
    component_timeout: int,
) -> Any:
    remaining = max(float(deadline_seconds) - (time.monotonic() - run_started), 0.0)
    timeout = min(float(component_timeout), remaining)
    if timeout <= 0:
        raise TimeoutError(f"{name} deadline exceeded")
    return await asyncio.wait_for(task, timeout=timeout)


def _context_key(context: dict[str, Any] | None) -> str:
    raw = json.dumps(context or {}, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _elapsed_ms(started: float) -> int:
    return int(round((time.monotonic() - started) * 1000))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ArenaGo live trading bot for rolling_rank_weighted_w24_p2.")
    p.add_argument("--config", default=os.environ.get("ARENA_CONFIG", "arena/config/production.yaml"))
    p.add_argument("--once", action="store_true", help="Run one decision cycle and exit.")
    p.add_argument("--as-of", default="", help="Optional MSK timestamp for --once, YYYY-mm-dd HH:MM:SS.")
    return p.parse_args()


def main() -> None:
    load_env_file()
    args = parse_args()
    settings = load_settings(args.config)
    bot = ArenaLiveBot(settings)
    if args.once:
        bot.initialize()
        try:
            as_of = datetime.fromisoformat(args.as_of) if args.as_of else now_msk()
            precompute = asyncio.run(bot.precompute_once(as_of))
            print(json.dumps(asyncio.run(bot.run_once(as_of, precompute=precompute)), ensure_ascii=False, indent=2, default=str))
        finally:
            bot.shutdown()
        return
    asyncio.run(bot.run_forever())


if __name__ == "__main__":
    main()
