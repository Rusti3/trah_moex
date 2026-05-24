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
    quantity: int  # ArenaGo quantity is lots.
    requested_quantity: int  # lots requested before cash capping.
    capped_quantity: int  # final lots after cash capping.
    current_position: int  # lots
    target_position: int  # lots
    requested_target_position: int  # lots before cash capping.
    price: float
    target_weight: float
    lot_size: int
    order_value: float
    cash_before_order: float | None = None
    cash_after_order: float | None = None
    cap_reason: str = ""
    liquidity_cap_reason: str = ""
    expected_cost_pct: float = 0.0
    bbo_spread_pct: float = 0.0
    liquidity_source: str = ""
    source_quality: str = ""
    liquidity_degraded: bool = False
    depth_shortfall: bool = False
    depth_unknown: bool = False


@dataclass(frozen=True)
class BotSnapshot:
    name: str
    cash_balance: float
    raw: Mapping[str, Any] | None = None


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

    def bot_snapshot(self) -> BotSnapshot:
        if self.client is None:
            return BotSnapshot(name=self.bot_name, cash_balance=0.0, raw=None)
        response = self.client.bots()
        if not response.ok:
            return BotSnapshot(name=self.bot_name, cash_balance=0.0, raw=None)
        rows = response.payload if isinstance(response.payload, list) else response.payload.get("bots", [])
        for row in rows or []:
            if str(row.get("name")) == self.bot_name:
                try:
                    cash = float(row.get("cash_balance", 0.0))
                except Exception:
                    cash = 0.0
                return BotSnapshot(name=self.bot_name, cash_balance=max(cash, 0.0), raw=row)
        return BotSnapshot(name=self.bot_name, cash_balance=0.0, raw=None)

    def cash_balance(self) -> float:
        return self.bot_snapshot().cash_balance

    def estimate_equity(
        self,
        positions: Mapping[str, int],
        prices: Mapping[str, float],
        *,
        cash_balance: float | None = None,
    ) -> float:
        cash = self.cash_balance() if cash_balance is None else max(float(cash_balance), 0.0)
        gross = sum(
            abs(int(lots)) * self.lot_sizes.get(ticker, 1) * float(prices.get(ticker, 0.0) or 0.0)
            for ticker, lots in positions.items()
        )
        return max(cash + gross, 0.0) if cash > 0 or gross > 0 else 100000.0

    def plan_orders(
        self,
        decision: DecisionResult,
        *,
        positions: Mapping[str, int],
        prices: Mapping[str, float],
        equity: float,
        cash_balance: float | None = None,
        cost_depth: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> list[PlannedOrder]:
        target_by_ticker = {p.ticker: p.weight for p in decision.target_positions}
        tickers = sorted(set(positions) | set(target_by_ticker))
        candidates = []
        for ticker in tickers:
            price = float(prices.get(ticker, 0.0) or 0.0)
            if price <= 0:
                continue
            lot = self.lot_sizes.get(ticker, 1)
            target_weight = float(target_by_ticker.get(ticker, 0.0))
            target_shares_raw = equity * target_weight / price
            target_lots = _shares_to_lots(target_shares_raw, lot)
            current = int(positions.get(ticker, 0))
            delta = target_lots - current
            if delta == 0:
                continue
            requested_quantity = abs(delta)
            requested_order_value = requested_quantity * lot * price
            if requested_order_value < self.min_order_value_rub:
                continue
            candidates.append(
                {
                    "ticker": ticker,
                    "direction": "B" if delta > 0 else "S",
                    "requested_quantity": requested_quantity,
                    "current_position": current,
                    "requested_target_position": target_lots,
                    "price": price,
                    "target_weight": target_weight,
                    "lot_size": lot,
                    "liquidity": (cost_depth or {}).get(ticker, {}),
                }
            )
        candidates.sort(key=lambda row: (0 if row["direction"] == "S" else 1, -abs(float(row["target_weight"])), str(row["ticker"])))
        buy_cash_remaining = self.cash_balance() if cash_balance is None else max(float(cash_balance), 0.0)
        planned: list[PlannedOrder] = []
        for row in candidates:
            requested_quantity = int(row["requested_quantity"])
            quantity = requested_quantity
            cap_reason = ""
            cash_before = buy_cash_remaining
            cash_after = buy_cash_remaining
            price = float(row["price"])
            lot = int(row["lot_size"])
            liquidity = row.get("liquidity") if isinstance(row.get("liquidity"), Mapping) else {}
            liquidity_quantity, liquidity_reason, liquidity_diag = _cap_by_liquidity(
                liquidity,
                direction=str(row["direction"]),
                requested_quantity=requested_quantity,
                current_position=int(row["current_position"]),
                requested_target_position=int(row["requested_target_position"]),
                lot_size=lot,
                price=price,
                equity=equity,
            )
            if liquidity_quantity < quantity:
                quantity = max(liquidity_quantity, 0)
                cap_reason = _append_reason(cap_reason, liquidity_reason)
            if row["direction"] == "B":
                lot_value = lot * price
                affordable_lots = int(math.floor(buy_cash_remaining / lot_value)) if lot_value > 0 else 0
                if affordable_lots < requested_quantity:
                    quantity = min(quantity, max(affordable_lots, 0))
                    cap_reason = _append_reason(cap_reason, "cash_cap")
                order_value = quantity * lot_value
                cash_after = max(buy_cash_remaining - order_value, 0.0)
                if quantity <= 0 or order_value < self.min_order_value_rub:
                    buy_cash_remaining = cash_after
                    continue
                buy_cash_remaining = cash_after
            else:
                order_value = quantity * lot * price
                if quantity <= 0 or order_value < self.min_order_value_rub:
                    continue
                cash_before = None
                cash_after = None
            current = int(row["current_position"])
            target_position = current + quantity if row["direction"] == "B" else current - quantity
            planned.append(
                PlannedOrder(
                    ticker=str(row["ticker"]),
                    direction=str(row["direction"]),
                    quantity=quantity,
                    requested_quantity=requested_quantity,
                    capped_quantity=quantity,
                    current_position=current,
                    target_position=target_position,
                    requested_target_position=int(row["requested_target_position"]),
                    price=price,
                    target_weight=float(row["target_weight"]),
                    lot_size=lot,
                    order_value=order_value,
                    cash_before_order=cash_before,
                    cash_after_order=cash_after,
                    cap_reason=cap_reason,
                    liquidity_cap_reason=liquidity_reason,
                    expected_cost_pct=float(liquidity_diag.get("expected_cost_pct", 0.0)),
                    bbo_spread_pct=float(liquidity_diag.get("bbo_spread_pct", 0.0)),
                    liquidity_source=str(liquidity_diag.get("source", "")),
                    source_quality=str(liquidity_diag.get("source_quality", "")),
                    liquidity_degraded=bool(liquidity_diag.get("liquidity_degraded", False)),
                    depth_shortfall=bool(liquidity_diag.get("depth_shortfall", False)),
                    depth_unknown=bool(liquidity_diag.get("depth_unknown", False)),
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
                    "requested_lots": order.requested_quantity,
                    "capped_lots": order.capped_quantity,
                    "current_lots": order.current_position,
                    "target_lots": order.target_position,
                    "requested_target_lots": order.requested_target_position,
                    "lot_size": order.lot_size,
                    "order_value": order.order_value,
                    "cash_before_order": order.cash_before_order,
                    "cash_after_order": order.cash_after_order,
                    "cap_reason": order.cap_reason,
                    "liquidity_cap_reason": order.liquidity_cap_reason,
                    "expected_cost_pct": order.expected_cost_pct,
                    "bbo_spread_pct": order.bbo_spread_pct,
                    "liquidity_source": order.liquidity_source,
                    "source_quality": order.source_quality,
                    "liquidity_degraded": order.liquidity_degraded,
                    "depth_shortfall": order.depth_shortfall,
                    "depth_unknown": order.depth_unknown,
                    "status": "duplicate_skipped" if not inserted else status,
                    "error": error,
                    "response": response_payload,
                }
            )
        return results

    def _cash_balance(self) -> float:
        return self.cash_balance()


def _shares_to_lots(shares: float, lot: int) -> int:
    if not math.isfinite(shares) or shares == 0:
        return 0
    sign = 1 if shares > 0 else -1
    return sign * int(math.floor(abs(shares) / lot))


def _cap_by_liquidity(
    liquidity: Mapping[str, Any],
    *,
    direction: str,
    requested_quantity: int,
    current_position: int,
    requested_target_position: int,
    lot_size: int,
    price: float,
    equity: float,
) -> tuple[int, str, dict[str, Any]]:
    side = "ask" if direction == "B" else "bid"
    levels = liquidity.get(f"{side}_levels") or []
    diag = {
        "source": liquidity.get("source", ""),
        "source_quality": liquidity.get("source_quality", ""),
        "liquidity_degraded": bool(liquidity.get("liquidity_degraded", False)),
        "depth_shortfall": bool(liquidity.get("depth_shortfall", False)),
        "depth_unknown": bool(liquidity.get("depth_unknown", False)),
        "bbo_spread_pct": _float(liquidity.get("bbo_spread_pct"), 0.0),
        "expected_cost_pct": _float(liquidity.get("estimated_cost_pct"), 0.0),
    }
    risk_increasing = _is_risk_increasing(current_position, requested_target_position)
    quantity = int(requested_quantity)
    reason = ""

    if levels:
        vwap_cost, max_lots, shortfall = _estimate_order_cost_from_levels(
            levels,
            side=side,
            requested_lots=requested_quantity,
            lot_size=lot_size,
            fallback_price=price,
            mid_price=_float(liquidity.get("mid_price"), price),
        )
        diag["expected_cost_pct"] = max(vwap_cost, diag["expected_cost_pct"])
        diag["depth_shortfall"] = shortfall
        if max_lots < requested_quantity:
            quantity = max(max_lots, 0)
            reason = _append_reason(reason, "depth_cap" if max_lots > 0 else "depth_shortfall")
    elif risk_increasing and bool(liquidity.get("liquidity_degraded") or liquidity.get("missing_bbo") or liquidity.get("unknown_cost")):
        max_value = max(float(equity), 0.0) * 0.20
        lot_value = max(price * lot_size, 0.0)
        max_lots = int(math.floor(max_value / lot_value)) if lot_value > 0 else 0
        if max_lots < requested_quantity:
            quantity = max(max_lots, 0)
            reason = _append_reason(reason, "degraded_liquidity_cap" if max_lots > 0 else "degraded_liquidity_skip")

    return quantity, reason, diag


def _estimate_order_cost_from_levels(
    levels: list[Mapping[str, Any]],
    *,
    side: str,
    requested_lots: int,
    lot_size: int,
    fallback_price: float,
    mid_price: float,
) -> tuple[float, int, bool]:
    requested_shares = max(int(requested_lots), 0) * max(int(lot_size), 1)
    if requested_shares <= 0:
        return 0.0, 0, False
    remaining = requested_shares
    notional = 0.0
    filled = 0.0
    available_shares = 0.0
    for level in levels:
        price = _float(level.get("price"), 0.0)
        quantity = _float(level.get("quantity"), 0.0)
        if price <= 0 or quantity <= 0:
            continue
        available_shares += quantity
        take = min(quantity, remaining)
        notional += take * price
        filled += take
        remaining -= take
        if remaining <= 0:
            break
    max_lots = int(math.floor(available_shares / max(int(lot_size), 1)))
    if filled <= 0:
        return 0.0, max_lots, True
    vwap = notional / filled
    reference = mid_price if mid_price > 0 else fallback_price
    if reference <= 0:
        return 0.0, max_lots, remaining > 0
    cost = (vwap - reference) / reference if side == "ask" else (reference - vwap) / reference
    return max(float(cost), 0.0), max_lots, remaining > 0


def _is_risk_increasing(current: int, target: int) -> bool:
    if target == current:
        return False
    if current == 0:
        return target != 0
    if (current > 0 and target > 0) or (current < 0 and target < 0):
        return abs(target) > abs(current)
    return target != 0


def _append_reason(existing: str, reason: str) -> str:
    if not reason:
        return existing
    if not existing:
        return reason
    if reason in existing.split("+"):
        return existing
    return f"{existing}+{reason}"


def _float(value: Any, default: float) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def _idempotency_key(decision_id: str, request: Mapping[str, Any]) -> str:
    raw = json.dumps({"decision_id": decision_id, **request}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
