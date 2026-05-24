from __future__ import annotations

import asyncio
import math
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import requests

from .schemas import TOP20_TICKERS

MOEX_ISS = "https://iss.moex.com/iss"
UNKNOWN_COST_FLOOR_PCT = 0.0015
BBO_COST_FLOOR_PCT = 0.0005
MAX_LEVELS = 10


class MoexRealtimeProvider:
    def __init__(self, *, token: str = "", timeout: float = 20.0):
        self.token = token
        self.timeout = timeout
        self.last_good_cost_depth: dict[str, dict[str, Any]] = {}

    async def current_cost_depth(self, as_of: datetime, tickers: tuple[str, ...] = TOP20_TICKERS) -> dict[str, dict[str, Any]]:
        try:
            data = await asyncio.to_thread(self._fetch_snapshot, tickers)
            if data:
                self.last_good_cost_depth = data
                return data
        except Exception:
            pass
        return self.last_good_cost_depth or {
            ticker: _missing_snapshot(ticker, source="fallback_missing")
            for ticker in tickers
        }

    def _fetch_snapshot(self, tickers: tuple[str, ...]) -> dict[str, dict[str, Any]]:
        out = self._fetch_with_moexalgo(tickers)
        if out:
            return out
        return self._fetch_with_iss(tickers)

    def _fetch_with_moexalgo(self, tickers: tuple[str, ...]) -> dict[str, dict[str, Any]]:
        try:
            from moexalgo import Ticker  # type: ignore
        except Exception:
            return {}
        out: dict[str, dict[str, Any]] = {}
        for ticker in tickers:
            try:
                instrument = Ticker(ticker)
                quote = _try_call(instrument, ["orderbook", "order_book", "quotes"])
                last_price = _extract_price(_try_call(instrument, ["marketdata", "trades", "candles"])) or _extract_price(quote)
                out[ticker] = _snapshot_from_quote(ticker, quote, last_price, source="moexalgo")
            except Exception:
                continue
        return out

    def _fetch_with_iss(self, tickers: tuple[str, ...]) -> dict[str, dict[str, Any]]:
        url = f"{MOEX_ISS}/engines/stock/markets/shares/boards/TQBR/securities.json"
        params = {
            "iss.meta": "off",
            "iss.only": "marketdata",
            "marketdata.columns": "SECID,LAST,BID,OFFER,OPEN,HIGH,LOW,LCURRENTPRICE",
        }
        response = requests.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()
        js = response.json()
        block = js.get("marketdata", {})
        cols = block.get("columns", [])
        rows = block.get("data", [])
        df = pd.DataFrame(rows, columns=cols)
        out = {}
        for row in df.to_dict("records"):
            ticker = str(row.get("SECID", ""))
            if ticker not in tickers:
                continue
            last = _first_float(row, ["LAST", "LCURRENTPRICE", "OPEN"])
            bid = _safe_float(row.get("BID"))
            ask = _safe_float(row.get("OFFER"))
            out[ticker] = _snapshot(
                ticker=ticker,
                last_price=last,
                bid=bid,
                ask=ask,
                bid_levels=[],
                ask_levels=[],
                source="iss_marketdata",
                source_quality="bbo_fallback" if bid > 0 and ask > 0 else "missing_bbo",
            )
        for ticker in tickers:
            out.setdefault(ticker, _missing_snapshot(ticker, source="missing"))
        return out

    async def latest_prices(self, tickers: tuple[str, ...] = TOP20_TICKERS) -> dict[str, float]:
        snapshot = await self.current_cost_depth(datetime.now(), tickers)
        return {ticker: float(row.get("last_price", 0.0) or 0.0) for ticker, row in snapshot.items()}


def _try_call(obj: Any, names: list[str]) -> Any:
    for name in names:
        if not hasattr(obj, name):
            continue
        attr = getattr(obj, name)
        try:
            return attr() if callable(attr) else attr
        except Exception:
            continue
    return None


def _extract_bid_ask(value: Any) -> tuple[float, float]:
    if value is None:
        return 0.0, 0.0
    if isinstance(value, pd.DataFrame) and not value.empty:
        row = value.iloc[0].to_dict()
    elif isinstance(value, dict):
        row = value
    else:
        return 0.0, 0.0
    return _first_float(row, ["bid", "BID", "best_bid"]), _first_float(row, ["ask", "OFFER", "best_ask"])


def _snapshot_from_quote(ticker: str, quote: Any, last_price: float, *, source: str) -> dict[str, Any]:
    bid_levels, ask_levels = _extract_orderbook_levels(quote)
    bid, ask = _extract_bid_ask(quote)
    if not bid and bid_levels:
        bid = bid_levels[0]["price"]
    if not ask and ask_levels:
        ask = ask_levels[0]["price"]
    quality = "depth" if bid_levels and ask_levels else ("bbo" if bid and ask else "missing_bbo")
    return _snapshot(
        ticker=ticker,
        last_price=last_price,
        bid=bid,
        ask=ask,
        bid_levels=bid_levels,
        ask_levels=ask_levels,
        source=source,
        source_quality=quality,
    )


def _snapshot(
    *,
    ticker: str,
    last_price: float,
    bid: float,
    ask: float,
    bid_levels: list[dict[str, float]],
    ask_levels: list[dict[str, float]],
    source: str,
    source_quality: str,
) -> dict[str, Any]:
    mid = ((bid + ask) / 2.0) if bid > 0 and ask > 0 and ask >= bid else float(last_price or 0.0)
    spread = (ask - bid) / mid if mid > 0 and bid > 0 and ask >= bid else 0.0
    missing_bbo = not (bid > 0 and ask > 0 and ask >= bid)
    unknown_cost = missing_bbo or source_quality in {"missing", "missing_bbo"}
    if unknown_cost:
        estimated_cost = UNKNOWN_COST_FLOOR_PCT
    else:
        estimated_cost = max(spread / 2.0, BBO_COST_FLOOR_PCT)
    bid_depth_value = _levels_value(bid_levels)
    ask_depth_value = _levels_value(ask_levels)
    depth_unknown = not (bid_levels and ask_levels)
    return {
        "ticker": ticker,
        "tradable": bool((last_price or mid) > 0),
        "last_price": float(last_price or mid or 0.0),
        "bid": float(bid or 0.0),
        "ask": float(ask or 0.0),
        "mid_price": float(mid or 0.0),
        "estimated_cost_pct": float(max(estimated_cost, 0.0)),
        "bbo_spread_pct": float(max(spread, 0.0)),
        "source": source,
        "source_quality": source_quality,
        "missing_bbo": bool(missing_bbo),
        "unknown_cost": bool(unknown_cost),
        "depth_unknown": bool(depth_unknown),
        "liquidity_degraded": bool(source_quality != "depth" or missing_bbo),
        "depth_shortfall": False,
        "bid_depth_value_rub": bid_depth_value,
        "ask_depth_value_rub": ask_depth_value,
        "bid_levels": bid_levels[:MAX_LEVELS],
        "ask_levels": ask_levels[:MAX_LEVELS],
    }


def _missing_snapshot(ticker: str, *, source: str) -> dict[str, Any]:
    return {
        "ticker": ticker,
        "tradable": False,
        "last_price": 0.0,
        "bid": 0.0,
        "ask": 0.0,
        "mid_price": 0.0,
        "estimated_cost_pct": UNKNOWN_COST_FLOOR_PCT,
        "bbo_spread_pct": 0.0,
        "source": source,
        "source_quality": "missing",
        "missing_bbo": True,
        "unknown_cost": True,
        "depth_unknown": True,
        "liquidity_degraded": True,
        "depth_shortfall": True,
        "bid_depth_value_rub": 0.0,
        "ask_depth_value_rub": 0.0,
        "bid_levels": [],
        "ask_levels": [],
    }


def _extract_orderbook_levels(value: Any) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    if value is None:
        return [], []
    if isinstance(value, dict):
        bid_levels = _levels_from_sequence(value.get("bids") or value.get("bid") or value.get("buy") or value.get("BUY"))
        ask_levels = _levels_from_sequence(value.get("asks") or value.get("ask") or value.get("sell") or value.get("SELL") or value.get("offer"))
        if bid_levels or ask_levels:
            return _sort_levels(bid_levels, reverse=True), _sort_levels(ask_levels, reverse=False)
        records = [value]
    elif isinstance(value, pd.DataFrame):
        records = value.to_dict("records")
    elif isinstance(value, list):
        records = value
    else:
        return [], []

    bids: list[dict[str, float]] = []
    asks: list[dict[str, float]] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        price = _first_float(record, ["PRICE", "price", "Price", "bid", "BID", "ask", "ASK", "OFFER", "offer"])
        quantity = _first_float(record, ["QUANTITY", "quantity", "QTY", "qty", "VOLUME", "volume", "SIZE", "size"])
        if price <= 0:
            continue
        side = _side_from_record(record)
        if side == "bid":
            bids.append({"price": price, "quantity": quantity})
        elif side == "ask":
            asks.append({"price": price, "quantity": quantity})
    return _sort_levels(bids, reverse=True), _sort_levels(asks, reverse=False)


def _levels_from_sequence(value: Any) -> list[dict[str, float]]:
    if value is None:
        return []
    if isinstance(value, pd.DataFrame):
        value = value.to_dict("records")
    if not isinstance(value, list):
        value = [value]
    out = []
    for item in value:
        if isinstance(item, dict):
            price = _first_float(item, ["PRICE", "price", "Price"])
            quantity = _first_float(item, ["QUANTITY", "quantity", "QTY", "qty", "VOLUME", "volume", "SIZE", "size"])
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            price = _safe_float(item[0])
            quantity = _safe_float(item[1])
        else:
            continue
        if price > 0:
            out.append({"price": price, "quantity": max(quantity, 0.0)})
    return out


def _side_from_record(record: dict[str, Any]) -> str:
    raw = " ".join(str(record.get(key, "")) for key in ("BUYSELL", "buy_sell", "side", "SIDE", "type", "TYPE", "operation"))
    raw = raw.lower()
    if any(token in raw for token in ("bid", "buy", "b", "покуп")):
        return "bid"
    if any(token in raw for token in ("ask", "offer", "sell", "s", "прод")):
        return "ask"
    if _safe_float(record.get("bid") or record.get("BID")) > 0:
        return "bid"
    if _safe_float(record.get("ask") or record.get("ASK") or record.get("OFFER") or record.get("offer")) > 0:
        return "ask"
    return ""


def _sort_levels(levels: list[dict[str, float]], *, reverse: bool) -> list[dict[str, float]]:
    return sorted(levels, key=lambda item: item["price"], reverse=reverse)[:MAX_LEVELS]


def _levels_value(levels: list[dict[str, float]]) -> float:
    return float(sum(max(row.get("price", 0.0), 0.0) * max(row.get("quantity", 0.0), 0.0) for row in levels))


def _extract_price(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, pd.DataFrame) and not value.empty:
        row = value.iloc[-1].to_dict()
    elif isinstance(value, dict):
        row = value
    else:
        return 0.0
    return _first_float(row, ["LAST", "last", "close", "LCURRENTPRICE", "price"])


def _first_float(row: dict[str, Any], names: list[str]) -> float:
    for name in names:
        value = _safe_float(row.get(name))
        if value > 0:
            return value
    return 0.0


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return 0.0
    return out if math.isfinite(out) else 0.0


def hourly_from_date(as_of: datetime, days: int = 20) -> tuple[str, str]:
    start = (as_of - timedelta(days=days)).date().isoformat()
    till = as_of.date().isoformat()
    return start, till
