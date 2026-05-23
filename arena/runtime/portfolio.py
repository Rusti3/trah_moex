from __future__ import annotations

from typing import Mapping

from .schemas import TOP20_TICKERS, TargetPosition


def _score(value: object, default: float = 0.5) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    if out != out:
        return default
    return min(1.0, max(0.0, out))


def _tradable(cost_depth: Mapping[str, Mapping[str, object]] | None, ticker: str) -> bool:
    if not cost_depth:
        return True
    row = cost_depth.get(ticker, {})
    return bool(row.get("tradable", True))


def build_target_weights(
    kronos_scores: Mapping[str, float],
    llm_scores: Mapping[str, float] | None = None,
    cost_depth: Mapping[str, Mapping[str, object]] | None = None,
    *,
    kronos_weight: float,
    llm_weight: float,
    threshold: float,
    rank_power: float,
    max_gross: float = 1.0,
    allow_short: bool = True,
    tickers: tuple[str, ...] = TOP20_TICKERS,
    source: str = "runtime",
) -> tuple[TargetPosition, ...]:
    """Build long/short target weights from Kronos and LLM bullish scores."""

    if kronos_weight < 0 or llm_weight < 0:
        raise ValueError("kronos_weight and llm_weight must be non-negative")
    denom = kronos_weight + llm_weight
    if denom <= 0:
        raise ValueError("at least one of kronos_weight or llm_weight must be positive")

    llm_scores = llm_scores or {}
    candidates: list[TargetPosition] = []
    for ticker in tickers:
        if ticker not in kronos_scores or not _tradable(cost_depth, ticker):
            continue
        kronos = _score(kronos_scores.get(ticker))
        llm = _score(llm_scores.get(ticker), default=0.5)
        bull = (kronos_weight * kronos + llm_weight * llm) / denom
        bear = (kronos_weight * (1.0 - kronos) + llm_weight * (1.0 - llm)) / denom

        if bull >= threshold and (not allow_short or bull > bear):
            candidates.append(TargetPosition(ticker=ticker, side="long", weight=0.0, score=bull, source=source))
        elif allow_short and bear >= threshold and bear > bull:
            candidates.append(TargetPosition(ticker=ticker, side="short", weight=0.0, score=bear, source=source))

    if not candidates:
        return tuple()

    candidates.sort(key=lambda item: item.score, reverse=True)
    raw = [(rank + 1) ** (-rank_power) for rank in range(len(candidates))]
    if rank_power == 0:
        raw = [1.0 for _ in candidates]
    total = sum(raw)
    positions: list[TargetPosition] = []
    for candidate, raw_weight in zip(candidates, raw):
        gross_weight = max_gross * raw_weight / total
        signed_weight = gross_weight if candidate.side == "long" else -gross_weight
        positions.append(
            TargetPosition(
                ticker=candidate.ticker,
                side=candidate.side,
                weight=signed_weight,
                score=candidate.score,
                source=source,
            )
        )
    return tuple(positions)


def normalize_gross(weights: Mapping[str, float], max_gross: float = 1.0) -> dict[str, float]:
    gross = sum(abs(float(v)) for v in weights.values())
    if gross <= 0 or gross <= max_gross:
        return {k: float(v) for k, v in weights.items()}
    scale = max_gross / gross
    return {k: float(v) * scale for k, v in weights.items()}


def positions_from_weights(weights: Mapping[str, float], source: str = "combined") -> tuple[TargetPosition, ...]:
    out = []
    for ticker, weight in sorted(weights.items()):
        if abs(weight) < 1e-12:
            continue
        out.append(
            TargetPosition(
                ticker=ticker,
                side="long" if weight > 0 else "short",
                weight=float(weight),
                score=abs(float(weight)),
                source=source,
            )
        )
    return tuple(out)
