from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from .arena_go_client import ArenaGoClient
from .decision import make_decision
from .jsonl_logger import JsonlLogger
from .kronos_provider import KronosTop20Provider
from .llm_scorer import LLMNewsScorer
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
        self.client = ArenaGoClient(
            settings.arena_api_key,
            base_url=settings.arena_base_url,
        )
        self.market = MoexRealtimeProvider(token=settings.moex_algo_token)
        self.kronos = KronosTop20Provider(
            weights_dir=settings.weights_dir,
            weights_mode=settings.weights_mode,
            device=settings.device,
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
        self.base_selector_params = {
            **DEFAULT_BASE_SELECTOR_PARAMS,
            **(cfg.get("base_selector_params") or {}),
        }

    def initialize(self) -> None:
        self.logger.write("startup", {"live_orders": self.settings.live_orders, "bot": self.settings.bot_name})
        self.news_ingestion.initialize()
        self.news_ingestion.start_background()
        try:
            self.kronos.warm()
            self.logger.write("kronos_warm_ok", {})
        except Exception as exc:
            self.logger.write("kronos_warm_error", {"error": str(exc)})
        self.order_manager.reconcile_positions()

    async def run_forever(self) -> None:
        self.initialize()
        while True:
            target = next_decision_time(
                now_msk(),
                interval_minutes=self.settings.decision_interval_minutes,
                decision_delay_seconds=self.settings.decision_delay_seconds,
            )
            await asyncio.sleep(sleep_seconds_until(target))
            try:
                await self.run_once(target)
            except Exception as exc:
                self.logger.write("run_once_error", {"error": repr(exc)})
                await asyncio.sleep(5)

    async def run_once(self, as_of: datetime | None = None) -> dict[str, Any]:
        as_of = as_of or now_msk()
        as_of_s = as_of.strftime("%Y-%m-%d %H:%M:%S")
        if not is_market_open(as_of):
            self.logger.write("market_closed_skip", {"as_of": as_of_s})
            return {"status": "market_closed"}

        tag_result = await asyncio.to_thread(self.news_tagger.tag_new_news, as_of, self.settings.tickers)
        news_context = self.news_buffer.get_context(as_of, self.settings.tickers)

        kronos_task = asyncio.create_task(self.kronos.forecast_bullish_scores(as_of, self.settings.tickers))
        llm_task = asyncio.create_task(self.llm.score_context(news_context, self.settings.tickers))
        moex_task = asyncio.create_task(self.market.current_cost_depth(as_of, self.settings.tickers))

        kronos_stale = False
        try:
            kronos_scores = await asyncio.wait_for(kronos_task, timeout=self.settings.max_kronos_wait_seconds)
        except Exception as exc:
            kronos_task.cancel()
            kronos_scores = self.state.get_json("last_good_kronos_scores", {})
            kronos_stale = True
            if not kronos_scores:
                kronos_scores = {ticker: 0.5 for ticker in self.settings.tickers}
            self.logger.write("kronos_fallback", {"as_of": as_of_s, "error": str(exc)})
        else:
            self.state.set_json("last_good_kronos_scores", dict(kronos_scores))

        try:
            llm_raw = await asyncio.wait_for(llm_task, timeout=self.settings.max_llm_wait_seconds)
        except Exception as exc:
            llm_task.cancel()
            llm_raw = {
                ticker: {"bullish_score": 0.5, "confidence": 0.0, "reason": f"llm timeout: {exc}"}
                for ticker in self.settings.tickers
            }
            self.logger.write("llm_fallback", {"as_of": as_of_s, "error": str(exc)})
        llm_scores = {ticker: float(row.get("bullish_score", 0.5)) for ticker, row in llm_raw.items()}

        try:
            cost_depth = await asyncio.wait_for(moex_task, timeout=self.settings.max_moex_wait_seconds)
        except Exception as exc:
            moex_task.cancel()
            cost_depth = self.market.last_good_cost_depth or {ticker: {"tradable": True, "last_price": 0.0} for ticker in self.settings.tickers}
            self.logger.write("moex_fallback", {"as_of": as_of_s, "error": str(exc)})

        prices = {ticker: float(row.get("last_price", 0.0) or 0.0) for ticker, row in cost_depth.items()}
        self._update_paper_selector_returns(as_of_s, prices)

        base_decisions = self._build_base_decisions(kronos_scores, llm_scores, cost_depth)
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
            max_gross=self.settings.max_gross_exposure,
        )
        decision_id = _decision_id(as_of_s, decision.selector_weights)
        positions = self.order_manager.reconcile_positions()
        equity = self.order_manager.estimate_equity(positions, prices)
        planned = self.order_manager.plan_orders(decision, positions=positions, prices=prices, equity=equity)
        order_results = self.order_manager.execute_orders(decision_id, as_of_s, planned)

        self._save_paper_positions(base_decisions, prices, as_of_s)
        payload = {
            "decision_id": decision_id,
            "as_of": as_of_s,
            "kronos_stale": kronos_stale,
            "selector_weights": dict(decision.selector_weights),
            "target_positions": decision.to_order_targets(),
            "orders": order_results,
            "news_rows": sum(len(v) for v in news_context.get("per_ticker_news", {}).values()) + len(news_context.get("marketwide_news", [])),
            "news_tagging": tag_result,
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
        as_of = datetime.fromisoformat(args.as_of) if args.as_of else now_msk()
        print(json.dumps(asyncio.run(bot.run_once(as_of)), ensure_ascii=False, indent=2, default=str))
        return
    asyncio.run(bot.run_forever())


if __name__ == "__main__":
    main()
