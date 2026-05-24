from __future__ import annotations

from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd

from arena.runtime.feature_builder import build_live_features
from arena.runtime.history_bootstrap import HistoryBootstrapService
from arena.runtime.jsonl_logger import JsonlLogger, redact
from arena.runtime.lightgbm_selector import LiveLightGBMSelector, rank_weights_from_scores
from arena.runtime import RollingRankWeightedSelector, build_target_weights, make_decision
from arena.runtime.llm_scorer import LLMNewsScorer, _hash_payload
from arena.runtime.market_history import MarketHistoryCache
from arena.runtime.news_service import NewsBuffer, NewsIngestionLogAggregator, ensure_live_news_schema
from arena.runtime.order_manager import OrderManager
from arena.runtime.schemas import DecisionResult, TargetPosition
from arena.runtime.settings import load_settings
from arena.runtime.state_store import StateStore
from news_ingestion.pipeline import SourceRunStats


class RuntimeTests(unittest.TestCase):
    def test_rank_power_two_weights(self):
        history = []
        for i in range(24):
            history.append(
                {
                    "timestamp": f"2026-05-01 12:{i:02d}:00",
                    "selector_family_first": 0.001,
                    "selector_news_aware": 0.003,
                    "selector_marketwide_news": -0.001,
                }
            )
        weights = RollingRankWeightedSelector(lookback=24, rank_power=2).weights(history, as_of="2026-05-01 13:00:00")
        self.assertAlmostEqual(sum(weights.values()), 1.0)
        self.assertGreater(weights["selector_news_aware"], weights["selector_family_first"])
        self.assertGreater(weights["selector_family_first"], weights["selector_marketwide_news"])
        self.assertAlmostEqual(weights["selector_news_aware"], 0.7346938775, places=6)

    def test_short_history_uses_fallback_order(self):
        history = [
            {
                "timestamp": "2026-05-01 12:00:00",
                "selector_family_first": -1.0,
                "selector_news_aware": 10.0,
                "selector_marketwide_news": 5.0,
            }
        ]
        order = RollingRankWeightedSelector(lookback=24, rank_power=2).rank_order(history, as_of="2026-05-01 12:30:00")
        self.assertEqual(order, ["selector_family_first", "selector_news_aware", "selector_marketwide_news"])

    def test_portfolio_has_no_long_short_overlap(self):
        positions = build_target_weights(
            kronos_scores={"SBER": 0.9, "LKOH": 0.1, "GAZP": 0.5},
            llm_scores={"SBER": 0.8, "LKOH": 0.2, "GAZP": 0.5},
            cost_depth={"SBER": {"tradable": True}, "LKOH": {"tradable": True}, "GAZP": {"tradable": True}},
            kronos_weight=1,
            llm_weight=1,
            threshold=0.65,
            rank_power=2,
        )
        by_ticker = {p.ticker: p.side for p in positions}
        self.assertEqual(by_ticker["SBER"], "long")
        self.assertEqual(by_ticker["LKOH"], "short")
        self.assertLessEqual(sum(abs(p.weight) for p in positions), 1.0)

    def test_decision_filters_future_history(self):
        history = {
            "selector_returns": [
                {
                    "timestamp": "2026-05-01 11:30:00",
                    "selector_family_first": 0.01,
                    "selector_news_aware": 0.0,
                    "selector_marketwide_news": 0.0,
                }
                for _ in range(24)
            ]
            + [
                {
                    "timestamp": "2026-05-01 12:30:00",
                    "selector_family_first": -1.0,
                    "selector_news_aware": 100.0,
                    "selector_marketwide_news": 0.0,
                }
            ],
            "base_selector_decisions": {
                "selector_family_first": {"target_weights": {"SBER": 1.0}, "kronos_weight": 1, "llm_weight": 1, "threshold": 0.6, "rank_power": 1},
                "selector_news_aware": {"target_weights": {"LKOH": 1.0}, "kronos_weight": 1, "llm_weight": 1, "threshold": 0.6, "rank_power": 1},
                "selector_marketwide_news": {"target_weights": {"GAZP": -1.0}, "kronos_weight": 1, "llm_weight": 1, "threshold": 0.6, "rank_power": 1},
            },
        }
        result = make_decision(
            as_of="2026-05-01 12:00:00",
            kronos_scores={},
            llm_scores={},
            cost_depth={},
            history=history,
        )
        self.assertGreater(result.selector_weights["selector_family_first"], result.selector_weights["selector_news_aware"])
        self.assertTrue(any(p.ticker == "SBER" for p in result.target_positions))

    def test_decision_accepts_lightgbm_selector_weight_override(self):
        history = {
            "selector_returns": [],
            "base_selector_decisions": {
                "selector_family_first": {"target_weights": {"SBER": 1.0}, "kronos_weight": 1, "llm_weight": 1, "threshold": 0.6, "rank_power": 1},
                "selector_news_aware": {"target_weights": {"LKOH": 1.0}, "kronos_weight": 1, "llm_weight": 1, "threshold": 0.6, "rank_power": 1},
                "selector_marketwide_news": {"target_weights": {"GAZP": -1.0}, "kronos_weight": 1, "llm_weight": 1, "threshold": 0.6, "rank_power": 1},
            },
        }
        weights = {"selector_family_first": 0.0, "selector_news_aware": 1.0, "selector_marketwide_news": 0.0}
        result = make_decision(
            as_of="2026-05-01 12:00:00",
            kronos_scores={},
            llm_scores={},
            cost_depth={},
            history=history,
            selector_weights_override=weights,
        )
        self.assertEqual(result.selector_weights, weights)
        self.assertEqual(result.target_positions[0].ticker, "LKOH")

    def test_lightgbm_rank_weights_from_scores(self):
        weights = rank_weights_from_scores(
            {
                "selector_family_first": 0.1,
                "selector_news_aware": 0.3,
                "selector_marketwide_news": -0.1,
            },
            rank_power=2,
        )
        self.assertGreater(weights["selector_news_aware"], weights["selector_family_first"])
        self.assertAlmostEqual(sum(weights.values()), 1.0)

    def test_lightgbm_rank_weights_support_custom_selector_list(self):
        selectors = ("a", "b", "c", "d")
        weights = rank_weights_from_scores({"a": 0.1, "b": 0.4, "c": -0.1, "d": 0.2}, base_selectors=selectors)
        self.assertAlmostEqual(sum(weights.values()), 1.0)
        self.assertGreater(weights["b"], weights["d"])
        selector = LiveLightGBMSelector(min_train_intervals=2, base_selectors=selectors)
        self.assertEqual(selector.base_selectors, selectors)

    def test_live_feature_builder_uses_known_current_inputs(self):
        features = build_live_features(
            as_of=datetime(2026, 5, 21, 12, 1, 30),
            kronos_scores={"SBER": 0.9, "LKOH": 0.1},
            llm_raw={"SBER": {"bullish_score": 0.8, "confidence": 0.7}},
            cost_depth={"SBER": {"estimated_cost_pct": 0.001, "bbo_spread_pct": 0.002, "tradable": True}},
            news_context={"per_ticker_news": {"SBER": [{"title": "x"}]}, "marketwide_news": [{"title": "m"}]},
            tickers=("SBER", "LKOH"),
        )
        self.assertEqual(features["hour"], 12.0)
        self.assertEqual(features["ticker_news_total"], 1.0)
        self.assertEqual(features["marketwide_news_count"], 1.0)
        self.assertGreater(features["kronos_spread"], 0.0)

    def test_llm_prompt_schema_is_per_ticker(self):
        with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            scorer = LLMNewsScorer(cache_path=f"{tmp}/llm_cache.jsonl", api_key_env="NO_SUCH_ENV")
            payload = scorer._prompt_payload(
                {
                    "rebalance_timestamp": "2026-05-24 12:31:30",
                    "date": "2026-05-24",
                    "per_ticker_news": {"SBER": [{"title": "news"}], "LKOH": []},
                    "marketwide_news": [],
                },
                ("SBER", "LKOH"),
            )
        self.assertIn("SBER", payload["schema"])
        self.assertIn("LKOH", payload["schema"])

    def test_settings_default_polza_model_is_pro(self):
        with patch.dict("os.environ", {}, clear=True):
            settings = load_settings("arena/config/production.yaml")
        self.assertEqual(settings.polza_model, "deepseek/deepseek-v4-pro")

    def test_llm_cache_key_depends_on_model(self):
        payload = {"tickers": ["SBER"], "per_ticker_news": {"SBER": []}}
        self.assertNotEqual(
            _hash_payload("deepseek/deepseek-v4-flash", payload),
            _hash_payload("deepseek/deepseek-v4-pro", payload),
        )

    def test_order_manager_sends_lots_not_shares_for_lot_tickers(self):
        manager = OrderManager(
            client=None,
            state=None,
            bot_name="test",
            lot_sizes={"ALRS": 10, "SBER": 1},
            live_orders=False,
        )
        decision = DecisionResult(
            as_of="2026-05-24 12:31:30",
            selector_weights={},
            target_positions=(
                TargetPosition("ALRS", "long", 0.50, 0.50),
                TargetPosition("SBER", "long", 0.10, 0.10),
            ),
        )
        planned = {
            order.ticker: order
            for order in manager.plan_orders(
                decision,
                positions={},
                prices={"ALRS": 25.0, "SBER": 300.0},
                equity=100000.0,
                cash_balance=100000.0,
            )
        }
        self.assertEqual(planned["ALRS"].quantity, 200)  # 2,000 shares / lot size 10
        self.assertEqual(planned["ALRS"].target_position, 200)
        self.assertEqual(planned["ALRS"].lot_size, 10)
        self.assertEqual(planned["SBER"].quantity, 33)

    def test_order_manager_reads_cash_balance_from_bots(self):
        class Client:
            def bots(self):
                class Response:
                    ok = True
                    payload = [{"name": "test", "cash_balance": 123456.78}]
                return Response()

        manager = OrderManager(
            client=Client(),
            state=None,
            bot_name="test",
            lot_sizes={"SBER": 1},
            live_orders=False,
        )
        self.assertEqual(manager.bot_snapshot().cash_balance, 123456.78)

    def test_order_manager_caps_buy_by_cash_balance(self):
        manager = OrderManager(
            client=None,
            state=None,
            bot_name="test",
            lot_sizes={"ALRS": 10},
            live_orders=False,
        )
        decision = DecisionResult(
            as_of="2026-05-24 12:31:30",
            selector_weights={},
            target_positions=(TargetPosition("ALRS", "long", 1.0, 1.0),),
        )
        planned = manager.plan_orders(
            decision,
            positions={},
            prices={"ALRS": 25.0},
            equity=1000000.0,
            cash_balance=500000.0,
        )
        self.assertEqual(len(planned), 1)
        self.assertEqual(planned[0].requested_quantity, 4000)
        self.assertEqual(planned[0].quantity, 2000)
        self.assertEqual(planned[0].capped_quantity, 2000)
        self.assertEqual(planned[0].cap_reason, "cash_cap")
        self.assertEqual(planned[0].cash_after_order, 0.0)

    def test_sells_do_not_fund_buy_budget_in_same_batch(self):
        manager = OrderManager(
            client=None,
            state=None,
            bot_name="test",
            lot_sizes={"ALRS": 10, "SBER": 1},
            live_orders=False,
        )
        decision = DecisionResult(
            as_of="2026-05-24 12:31:30",
            selector_weights={},
            target_positions=(TargetPosition("ALRS", "long", 1.0, 1.0),),
        )
        planned = {
            order.ticker: order
            for order in manager.plan_orders(
                decision,
                positions={"SBER": 10},
                prices={"ALRS": 10.0, "SBER": 100.0},
                equity=2000.0,
                cash_balance=1000.0,
            )
        }
        self.assertEqual(planned["SBER"].direction, "S")
        self.assertEqual(planned["SBER"].quantity, 10)
        self.assertEqual(planned["ALRS"].requested_quantity, 20)
        self.assertEqual(planned["ALRS"].quantity, 10)
        self.assertEqual(planned["ALRS"].cap_reason, "cash_cap")

    def test_order_manager_equity_uses_gross_lot_value(self):
        class Client:
            def bots(self):
                class Response:
                    ok = True
                    payload = [{"name": "test", "cash_balance": 50000.0}]
                return Response()

        manager = OrderManager(
            client=Client(),
            state=None,
            bot_name="test",
            lot_sizes={"ALRS": 10, "SBER": 1},
            live_orders=False,
        )
        equity = manager.estimate_equity({"ALRS": 100, "SBER": -10}, {"ALRS": 25.0, "SBER": 300.0})
        self.assertEqual(equity, 78000.0)

    def test_market_history_append_deduplicates_ticker_timestamp(self):
        with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            cache = MarketHistoryCache(f"{tmp}/market_history.sqlite3")
            candles = pd.DataFrame(
                [
                    {
                        "timestamps": "2026-05-21 12:00:00",
                        "end": "2026-05-21 12:29:59",
                        "open": 100,
                        "high": 101,
                        "low": 99,
                        "close": 100,
                        "volume": 10,
                        "amount": 1000,
                    }
                ]
            )
            cache.append_candles("SBER", 30, candles)
            cache.append_candles("SBER", 30, candles.assign(close=101))
            loaded = cache.load_candles("SBER", interval_minutes=30, before="2026-05-21 13:00:00", limit=10)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(float(loaded["close"].iloc[0]), 101.0)

    def test_history_bootstrap_fills_training_rows_from_past_only(self):
        with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            state = StateStore(f"{tmp}/state.sqlite3")
            news_db = f"{tmp}/news.sqlite3"
            ensure_live_news_schema(news_db)
            cache = MarketHistoryCache(f"{tmp}/market_history.sqlite3")
            rows = []
            for idx in range(6):
                ts = pd.Timestamp("2026-05-21 10:00:00") + pd.Timedelta(minutes=30 * idx)
                for ticker, offset in {"SBER": 1.0, "LKOH": 2.0, "GAZP": 3.0}.items():
                    base = 100 + idx + offset
                    rows.append(
                        {
                            "ticker": ticker,
                            "timestamps": ts,
                            "end": ts + pd.Timedelta(minutes=30) - pd.Timedelta(seconds=1),
                            "open": base,
                            "high": base + 1,
                            "low": base - 1,
                            "close": base,
                            "volume": 100,
                            "amount": 10000,
                        }
                    )
            df = pd.DataFrame(rows)
            for ticker in ("SBER", "LKOH", "GAZP"):
                cache.append_candles(ticker, 30, df[df["ticker"] == ticker])
            service = HistoryBootstrapService(
                state=state,
                market_history=cache,
                news_buffer=NewsBuffer(news_db),
                tickers=("SBER", "LKOH", "GAZP"),
                base_selector_params={
                    "selector_family_first": {"threshold": 0.55, "rank_power": 1},
                    "selector_news_aware": {"threshold": 0.55, "rank_power": 1},
                    "selector_marketwide_news": {"threshold": 0.55, "rank_power": 1},
                },
            )
            result = service.bootstrap_initial(
                as_of=datetime(2026, 5, 21, 13, 0, 0),
                initial_intervals=4,
                refresh_market_history=False,
            )
            rows = state.load_lightgbm_training_rows(limit=10)
        self.assertGreaterEqual(result.rows_after, 4)
        self.assertGreaterEqual(len(rows), 4)
        self.assertTrue(all(row["timestamp"] < "2026-05-21 13:00:00" for row in rows))

    def test_state_store_persists_dynamic_selector_returns(self):
        with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            state = StateStore(f"{tmp}/state.sqlite3")
            state.append_selector_return(
                "2026-05-21 12:00:00",
                {
                    "selector_family_first": 0.1,
                    "selector_nrh_00201": 0.2,
                    "selector_nrh_00891": -0.3,
                },
            )
            history = state.load_selector_history(limit=5)
            training = state.load_lightgbm_training_rows(limit=5)
            ready_rows = state.count_lightgbm_training_rows(required_selectors=("selector_nrh_00201", "selector_nrh_00891"))
        self.assertEqual(history[0]["selector_nrh_00201"], 0.2)
        self.assertEqual(history[0]["returns"]["selector_nrh_00891"], -0.3)
        self.assertEqual(training, [])
        self.assertEqual(ready_rows, 0)

    def test_production_config_has_thirty_base_selectors(self):
        settings = load_settings("arena/config/production.yaml")
        cfg = __import__("arena.runtime.settings", fromlist=["load_yaml"]).load_yaml(settings.production_config_path)
        self.assertEqual(len(cfg["strategy"]["base_selectors"]), 30)
        self.assertIn("selector_nrh_00201", cfg["base_selector_params"])

    def test_jsonl_logger_redacts_secrets(self):
        with TemporaryDirectory(ignore_cleanup_errors=True) as tmp, patch.dict("os.environ", {"ARENA_LOG_STDOUT": "false"}):
            logger = JsonlLogger(f"{tmp}/logs")
            logger.write(
                "secret_test",
                {
                    "api_key": "pza_testSecretToken",
                    "nested": {"authorization": "Bearer verysecretvalue"},
                    "message": "token pza_anotherSecret should be hidden",
                },
            )
            text = next((Path(tmp) / "logs").glob("arena_live_*.jsonl")).read_text(encoding="utf-8")
        self.assertNotIn("pza_testSecretToken", text)
        self.assertNotIn("verysecretvalue", text)
        self.assertNotIn("pza_anotherSecret", text)
        self.assertIn("[REDACTED]", text)
        self.assertEqual(redact({"token": "abc"})["token"], "[REDACTED]")

    def test_news_ingestion_log_aggregator_throttles_until_interval(self):
        with TemporaryDirectory(ignore_cleanup_errors=True) as tmp, patch.dict("os.environ", {"ARENA_LOG_STDOUT": "false"}):
            db = Path(tmp) / "news.sqlite3"
            ensure_live_news_schema(db)
            logger = JsonlLogger(Path(tmp) / "logs")
            aggregator = NewsIngestionLogAggregator(database_path=db, logger=logger, interval_seconds=1800)
            stats = SourceRunStats(source_id="src", fetched=2, selected=1, saved=1, duplicates=1)
            stats.finish()
            aggregator.observe(stats)
            self.assertFalse(list((Path(tmp) / "logs").glob("news_ingestion_*.jsonl")))
            aggregator._last_emit -= 1801
            stats2 = SourceRunStats(source_id="src", fetched=3, selected=2, saved=0, duplicates=2, errors=["temporary"])
            stats2.finish()
            aggregator.observe(stats2)
            news_logs = list((Path(tmp) / "logs").glob("news_ingestion_*.jsonl"))
            error_logs = list((Path(tmp) / "logs").glob("errors_*.jsonl"))
            payload = news_logs[0].read_text(encoding="utf-8")
        self.assertEqual(len(news_logs), 1)
        self.assertEqual(len(error_logs), 1)
        self.assertIn('"runs": 2', payload)
        self.assertIn('"news_db_count": 0', payload)


if __name__ == "__main__":
    unittest.main()
