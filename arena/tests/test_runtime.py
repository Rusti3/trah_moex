from __future__ import annotations

import unittest

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


if __name__ == "__main__":
    unittest.main()
