from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping

from .arena_go_client import ArenaGoClient
from .schemas import DecisionResult
from .state_store import StateStore


@dataclass(frozen=True)
class PlannedOrder:
    ticker: str
    direction: str
    quantity: int
    current_position: int
    target_position: int
    price: float
    target_weight: float


class OrderManager:
    def __init__(
        self,
        *,
        client: ArenaGoClient,
        state: StateStore,
        bot_name: str,
        lot_sizes: Mapping[str, int],
        live_orders: bool = True,
        max_daily_trades: int = 950,
        min_order_value_rub: float = 100.0,
    ):
        self.client = client
        self.state = state
        self.bot_name = bot_name
        self.lot_sizes = {ticker: max(int(lot), 1) for ticker, lot in lot_sizes.items()}
        self.live_orders = live_orders
        self.max_daily_trades = max_daily_trades
        self.min_order_value_rub = min_order_value_rub

    def reconcile_positions(self) -> dict[str, int]:
        response = self.client.positions(self.bot_name)
        if not response.ok:
            return {}
        payload = response.payload
        rows = payload if isinstance(payload, list) else payload.get("positions", [])
        out: dict[str, int] = {}
        for row in rows or []:
            ticker = str(row.get("secid") or row.get("ticker") or "")
            if not ticker:
                continue
            try:
                out[ticker] = int(float(row.get("position", 0)))
            except Exception:
                out[ticker] = 0
        return out

    def estimate_equity(self, positions: Mapping[str, int], prices: Mapping[str, float]) -> float:
        cash = self._cash_balance()
        gross = sum(int(qty) * float(prices.get(ticker, 0.0) or 0.0) for ticker, qty in positions.items())
        return max(cash + gross, 0.0) if cash > 0 else max(gross, 100000.0)

    def plan_orders(
        self,
        decision: DecisionResult,
        *,
        positions: Mapping[str, int],
        prices: Mapping[str, float],
        equity: float,
    ) -> list[PlannedOrder]:
        target_by_ticker = {p.ticker: p.weight for p in decision.target_positions}
        tickers = sorted(set(positions) | set(target_by_ticker))
        planned = []
        for ticker in tickers:
            price = float(prices.get(ticker, 0.0) or 0.0)
            if price <= 0:
                continue
            lot = self.lot_sizes.get(ticker, 1)
            target_weight = float(target_by_ticker.get(ticker, 0.0))
            target_shares_raw = equity * target_weight / price
            target_shares = _round_to_lot(target_shares_raw, lot)
            current = int(positions.get(ticker, 0))
            delta = target_shares - current
            if delta == 0:
                continue
            quantity = abs(delta)
            if quantity * price < self.min_order_value_rub:
                continue
            planned.append(
                PlannedOrder(
                    ticker=ticker,
                    direction="B" if delta > 0 else "S",
                    quantity=quantity,
                    current_position=current,
                    target_position=target_shares,
                    price=price,
                    target_weight=target_weight,
                )
            )
        return planned

    def execute_orders(self, decision_id: str, as_of: str, orders: list[PlannedOrder]) -> list[dict[str, Any]]:
        results = []
        date_prefix = as_of[:10]
        already = self.state.count_today_orders(date_prefix)
        for order in orders:
            if already >= self.max_daily_trades:
                results.append({"ticker": order.ticker, "status": "blocked_daily_trade_limit"})
                continue
            request = {
                "direction": order.direction,
                "secid": order.ticker,
                "quantity": order.quantity,
                "bot": self.bot_name,
            }
            key = _idempotency_key(decision_id, request)
            status = "dry_run"
            response_payload = None
            error = None
            if self.live_orders:
                response = self.client.submit_order(**request)
                response_payload = response.payload
                status = "submitted" if response.ok else "error"
                error = response.error
            inserted = self.state.insert_order_attempt(
                idempotency_key=key,
                decision_id=decision_id,
                as_of=as_of,
                ticker=order.ticker,
                direction=order.direction,
                quantity=order.quantity,
                status=status,
                request=request,
                response=response_payload,
                error=error,
            )
            if inserted and status in {"submitted", "dry_run"}:
                already += 1
            results.append(
                {
                    "ticker": order.ticker,
                    "direction": order.direction,
                    "quantity": order.quantity,
                    "status": "duplicate_skipped" if not inserted else status,
                    "error": error,
                    "response": response_payload,
                }
            )
        return results

    def _cash_balance(self) -> float:
        response = self.client.bots()
        if not response.ok:
            return 0.0
        rows = response.payload if isinstance(response.payload, list) else response.payload.get("bots", [])
        for row in rows or []:
            if str(row.get("name")) == self.bot_name:
                try:
                    return float(row.get("cash_balance", 0.0))
                except Exception:
                    return 0.0
        return 0.0


def _round_to_lot(shares: float, lot: int) -> int:
    if not math.isfinite(shares) or shares == 0:
        return 0
    sign = 1 if shares > 0 else -1
    return sign * int(math.floor(abs(shares) / lot) * lot)


def _idempotency_key(decision_id: str, request: Mapping[str, Any]) -> str:
    raw = json.dumps({"decision_id": decision_id, **request}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
