from __future__ import annotations

from datetime import datetime
from statistics import mean, pstdev
from typing import Any, Mapping

from .schemas import TOP20_TICKERS


def build_live_features(
    *,
    as_of: datetime,
    kronos_scores: Mapping[str, float],
    llm_raw: Mapping[str, Mapping[str, Any]],
    cost_depth: Mapping[str, Mapping[str, Any]],
    news_context: Mapping[str, Any],
    tickers: tuple[str, ...] = TOP20_TICKERS,
) -> dict[str, float]:
    """Build live-safe interval features for selector inference.

    These features use only data known at the current rebalance timestamp:
    Kronos scores, LLM/news scores for known news, current cost/depth snapshot,
    and session calendar fields.
    """

    kronos = [_score(kronos_scores.get(ticker), 0.5) for ticker in tickers]
    llm = [_score((llm_raw.get(ticker) or {}).get("bullish_score"), 0.5) for ticker in tickers]
    confidence = [_score((llm_raw.get(ticker) or {}).get("confidence"), 0.0) for ticker in tickers]
    priced_risk = [_score((llm_raw.get(ticker) or {}).get("already_priced_risk"), 0.5) for ticker in tickers]
    costs = [_float((cost_depth.get(ticker) or {}).get("estimated_cost_pct"), 0.0) for ticker in tickers]
    spreads = [_float((cost_depth.get(ticker) or {}).get("bbo_spread_pct"), 0.0) for ticker in tickers]
    tradable = [1.0 if (cost_depth.get(ticker) or {}).get("tradable", True) else 0.0 for ticker in tickers]
    degraded = [1.0 if (cost_depth.get(ticker) or {}).get("liquidity_degraded", False) else 0.0 for ticker in tickers]
    depth_unknown = [1.0 if (cost_depth.get(ticker) or {}).get("depth_unknown", False) else 0.0 for ticker in tickers]
    missing_bbo = [1.0 if (cost_depth.get(ticker) or {}).get("missing_bbo", False) else 0.0 for ticker in tickers]
    per_ticker_news = news_context.get("per_ticker_news", {}) if isinstance(news_context, Mapping) else {}
    news_counts = [float(len(per_ticker_news.get(ticker, []) or [])) for ticker in tickers]
    marketwide_count = float(len(news_context.get("marketwide_news", []) or [])) if isinstance(news_context, Mapping) else 0.0
    disagreement = [abs(a - b) for a, b in zip(kronos, llm)]

    return {
        "hour": float(as_of.hour),
        "minute": float(as_of.minute),
        "day_of_week": float(as_of.weekday()),
        "is_weekend": 1.0 if as_of.weekday() >= 5 else 0.0,
        "kronos_mean": _mean(kronos),
        "kronos_std": _std(kronos),
        "kronos_spread": max(kronos) - min(kronos),
        "kronos_top": max(kronos),
        "kronos_bottom": min(kronos),
        "llm_mean": _mean(llm),
        "llm_std": _std(llm),
        "llm_spread": max(llm) - min(llm),
        "llm_top": max(llm),
        "llm_bottom": min(llm),
        "llm_confidence_mean": _mean(confidence),
        "llm_confidence_max": max(confidence),
        "llm_already_priced_risk_mean": _mean(priced_risk),
        "llm_already_priced_risk_max": max(priced_risk),
        "kronos_llm_disagreement_mean": _mean(disagreement),
        "kronos_llm_disagreement_max": max(disagreement),
        "estimated_cost_mean": _mean(costs),
        "estimated_cost_max": max(costs),
        "bbo_spread_mean": _mean(spreads),
        "bbo_spread_max": max(spreads),
        "tradable_count": sum(tradable),
        "liquidity_degraded_count": sum(degraded),
        "depth_unknown_count": sum(depth_unknown),
        "missing_bbo_count": sum(missing_bbo),
        "ticker_news_total": sum(news_counts),
        "ticker_news_max": max(news_counts) if news_counts else 0.0,
        "ticker_news_coverage": sum(1.0 for value in news_counts if value > 0.0),
        "marketwide_news_count": marketwide_count,
        "total_news_count": sum(news_counts) + marketwide_count,
    }


def _score(value: Any, default: float) -> float:
    out = _float(value, default)
    return max(0.0, min(1.0, out))


def _float(value: Any, default: float) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if out == out else default


def _mean(values: list[float]) -> float:
    return float(mean(values)) if values else 0.0


def _std(values: list[float]) -> float:
    return float(pstdev(values)) if len(values) > 1 else 0.0
