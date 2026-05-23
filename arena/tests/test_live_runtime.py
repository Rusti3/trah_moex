from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from arena.runtime.arena_go_client import ArenaGoClient
from arena.runtime.news_service import LLMNewsTagger, NewsBuffer, ensure_live_news_schema
from arena.runtime.order_manager import OrderManager
from arena.runtime.schemas import DecisionResult, TargetPosition
from arena.runtime.state_store import StateStore


class FakeResponse:
    status_code = 200

    def __init__(self, payload):
        self._payload = payload
        self.text = str(payload)

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self):
        self.calls = []

    def request(self, method, url, headers=None, timeout=None, **kwargs):
        self.calls.append({"method": method, "url": url, "headers": headers, **kwargs})
        return FakeResponse({"success": True, "price": 100, "quantity": 10})


class FakeArenaClient:
    def positions(self, portfolio):
        return type("R", (), {"ok": True, "payload": []})()

    def bots(self):
        return type("R", (), {"ok": True, "payload": [{"name": "bot", "cash_balance": 100000}]})()

    def submit_order(self, **request):
        return type("R", (), {"ok": True, "payload": {"success": True}, "error": None})()


class LiveRuntimeTests(unittest.TestCase):
    def test_arena_go_submit_payload(self):
        session = FakeSession()
        client = ArenaGoClient("token", base_url="https://arenago.ru", session=session)
        response = client.submit_order(direction="B", secid="SBER", quantity=10, bot="bot")
        self.assertTrue(response.ok)
        self.assertEqual(session.calls[0]["json"], {"direction": "B", "secid": "SBER", "quantity": 10, "bot": "bot"})
        self.assertEqual(session.calls[0]["headers"]["Authorization"], "token")

    def test_order_manager_lot_rounding_and_idempotency(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            state = StateStore(Path(tmp) / "state.sqlite3")
            manager = OrderManager(
                client=FakeArenaClient(),
                state=state,
                bot_name="bot",
                lot_sizes={"GAZP": 10, "SBER": 1},
                live_orders=True,
            )
            decision = DecisionResult(
                as_of="2026-05-21 12:01:30",
                selector_weights={},
                target_positions=(
                    TargetPosition(ticker="GAZP", side="long", weight=0.123, score=0.8),
                    TargetPosition(ticker="SBER", side="short", weight=-0.051, score=0.7),
                ),
            )
            orders = manager.plan_orders(decision, positions={}, prices={"GAZP": 120, "SBER": 300}, equity=100000)
            by_ticker = {o.ticker: o for o in orders}
            self.assertEqual(by_ticker["GAZP"].quantity % 10, 0)
            self.assertEqual(by_ticker["GAZP"].direction, "B")
            self.assertEqual(by_ticker["SBER"].direction, "S")
            first = manager.execute_orders("d1", "2026-05-21 12:01:30", orders)
            second = manager.execute_orders("d1", "2026-05-21 12:01:30", orders)
            self.assertTrue(any(r["status"] == "submitted" for r in first))
            self.assertTrue(all(r["status"] == "duplicate_skipped" for r in second))

    def test_news_buffer_uses_received_at(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            db = Path(tmp) / "news.sqlite3"
            ensure_live_news_schema(db)
            import sqlite3

            with sqlite3.connect(db) as conn:
                conn.execute(
                    """
                    INSERT INTO news(news_id, source, published_at_msk, received_at_msk, title, text, url, raw_payload_hash, tickers_json)
                    VALUES ('n1', 'src', '2026-05-21 10:00:00', '2026-05-21 12:05:00', 'late received', 'text', '', 'h1', '[]')
                    """
                )
                conn.execute(
                    """
                    INSERT INTO news_llm_tags(news_id, ticker, relation_type, confidence, relation_strength, reason, tags_json, created_at_msk)
                    VALUES ('n1', 'SBER', 'direct', 'high', 1.0, 'reason', '[]', '2026-05-21 12:05:00')
                    """
                )
            ctx = NewsBuffer(db).get_context("2026-05-21 12:01:30", ("SBER",))
            self.assertEqual(ctx["per_ticker_news"]["SBER"], [])
            ctx2 = NewsBuffer(db).get_context("2026-05-21 12:06:00", ("SBER",))
            self.assertEqual(len(ctx2["per_ticker_news"]["SBER"]), 1)

    def test_news_buffer_splits_cluster_and_marketwide(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            db = Path(tmp) / "news.sqlite3"
            ensure_live_news_schema(db)
            import sqlite3

            with sqlite3.connect(db) as conn:
                conn.execute(
                    """
                    INSERT INTO news(news_id, source, published_at_msk, received_at_msk, title, text, url, raw_payload_hash, tickers_json)
                    VALUES ('n2', 'src', '2026-05-21 10:00:00', '2026-05-21 10:00:01', 'sector', 'text', '', 'h2', '[]')
                    """
                )
                conn.execute(
                    """
                    INSERT INTO news_llm_tags(news_id, ticker, relation_type, confidence, relation_strength, reason, tags_json, created_at_msk)
                    VALUES ('n2', 'LKOH', 'cluster', 'high', 0.9, 'sector read-through', '["oil"]', '2026-05-21 10:00:01')
                    """
                )
                conn.execute(
                    """
                    INSERT INTO news_llm_tags(news_id, ticker, relation_type, confidence, relation_strength, reason, tags_json, created_at_msk)
                    VALUES ('n2', 'MARKET', 'marketwide', 'high', 0.8, 'broad market', '["rate"]', '2026-05-21 10:00:01')
                    """
                )
            ctx = NewsBuffer(db).get_context("2026-05-21 12:01:30", ("LKOH",))
            self.assertEqual(len(ctx["per_ticker_news"]["LKOH"]), 1)
            self.assertLessEqual(ctx["per_ticker_news"]["LKOH"][0]["relation_strength"], 0.6)
            self.assertEqual(len(ctx["marketwide_news"]), 1)

    def test_llm_news_tagger_falls_back_to_regex_tags_without_key(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            db = Path(tmp) / "news.sqlite3"
            ensure_live_news_schema(db)
            import sqlite3

            with sqlite3.connect(db) as conn:
                conn.execute(
                    """
                    INSERT INTO news(news_id, source, published_at_msk, received_at_msk, title, text, url, raw_payload_hash, tickers_json)
                    VALUES ('n3', 'src', '2026-05-21 11:00:00', '2026-05-21 11:00:01', 'SBER news', 'text', '', 'h3', '["SBER"]')
                    """
                )
                conn.execute(
                    """
                    INSERT INTO news_tickers(news_id, ticker, matched_by, matched_terms_json, created_at_msk)
                    VALUES ('n3', 'SBER', 'regex', '["SBER"]', '2026-05-21 11:00:01')
                    """
                )
            result = LLMNewsTagger(db, api_key_env="MISSING_TEST_POLZA_KEY").tag_new_news(
                "2026-05-21 12:01:30",
                ("SBER",),
            )
            self.assertEqual(result["mode"], "regex_fallback")
            ctx = NewsBuffer(db).get_context("2026-05-21 12:01:30", ("SBER",))
            self.assertEqual(len(ctx["per_ticker_news"]["SBER"]), 1)


if __name__ == "__main__":
    unittest.main()
