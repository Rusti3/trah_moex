from __future__ import annotations

from datetime import datetime
import unittest

from arena.runtime.feature_builder import build_live_features
from arena.runtime.lightgbm_selector import rank_weights_from_scores
from arena.runtime import RollingRankWeightedSelector, build_target_weights, make_decision


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


if __name__ == "__main__":
    unittest.main()
