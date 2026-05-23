from __future__ import annotations

import asyncio
import math
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import requests

from .schemas import TOP20_TICKERS

MOEX_ISS = "https://iss.moex.com/iss"


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
        return self.last_good_cost_depth or {ticker: {"tradable": True, "last_price": 0.0, "estimated_cost_pct": 0.0} for ticker in tickers}

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
                bid, ask = _extract_bid_ask(quote)
                spread = (ask - bid) / ((ask + bid) / 2.0) if bid and ask and ask >= bid else 0.0
                out[ticker] = {
                    "tradable": True,
                    "last_price": float(last_price or 0.0),
                    "estimated_cost_pct": float(max(spread / 2.0, 0.0)),
                    "bbo_spread_pct": float(max(spread, 0.0)),
                    "source": "moexalgo",
                }
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
            spread = (ask - bid) / ((ask + bid) / 2.0) if bid and ask and ask >= bid else 0.0
            out[ticker] = {
                "tradable": bool(last > 0),
                "last_price": float(last),
                "estimated_cost_pct": float(max(spread / 2.0, 0.0)),
                "bbo_spread_pct": float(max(spread, 0.0)),
                "source": "iss_marketdata",
            }
        for ticker in tickers:
            out.setdefault(ticker, {"tradable": True, "last_price": 0.0, "estimated_cost_pct": 0.0, "source": "missing"})
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
