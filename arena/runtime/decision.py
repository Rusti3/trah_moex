from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping

from .portfolio import build_target_weights, normalize_gross, positions_from_weights
from .schemas import BASE_SELECTORS, BaseSelectorDecision, DecisionResult
from .selector import RollingRankWeightedSelector


def _history_rows(history: Mapping[str, Any] | list[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    if isinstance(history, Mapping):
        rows = history.get("selector_returns", [])
    else:
        rows = history
    return list(rows or [])


def _base_decisions(history: Mapping[str, Any] | list[Mapping[str, Any]]) -> dict[str, BaseSelectorDecision]:
    if not isinstance(history, Mapping):
        raise ValueError("history must contain base_selector_decisions for live target construction")
    raw = history.get("base_selector_decisions")
    if not isinstance(raw, Mapping):
        raise ValueError("history['base_selector_decisions'] is required")
    out: dict[str, BaseSelectorDecision] = {}
    for name, value in raw.items():
        if isinstance(value, BaseSelectorDecision):
            out[name] = value
            continue
        if not isinstance(value, Mapping):
            raise ValueError(f"base selector decision {name!r} must be a mapping")
        out[name] = BaseSelectorDecision(
            name=name,
            kronos_weight=float(value.get("kronos_weight", 1.0)),
            llm_weight=float(value.get("llm_weight", 1.0)),
            threshold=float(value.get("threshold", 0.7)),
            rank_power=float(value.get("rank_power", 2.0)),
            max_gross=float(value.get("max_gross", 1.0)),
            allow_short=bool(value.get("allow_short", True)),
            target_weights=value.get("target_weights"),
        )
    missing = [name for name in BASE_SELECTORS if name not in out]
    if missing:
        raise ValueError(f"missing base selector decisions: {missing}")
    return out


def make_decision(
    as_of: datetime | str,
    kronos_scores: Mapping[str, float],
    llm_scores: Mapping[str, float] | None,
    cost_depth: Mapping[str, Mapping[str, object]] | None,
    history: Mapping[str, Any] | list[Mapping[str, Any]],
    *,
    selector: RollingRankWeightedSelector | None = None,
    selector_weights_override: Mapping[str, float] | None = None,
    max_gross: float = 1.0,
) -> DecisionResult:
    """Make a production target portfolio decision.

    `history` must include:
    - `selector_returns`: rows strictly before `as_of` or rows with timestamps
      that can be filtered by the selector.
    - `base_selector_decisions`: params or precomputed target weights for the
      three base selectors.
    """

    selector = selector or RollingRankWeightedSelector()
    rows = _history_rows(history)
    base_decisions = _base_decisions(history)
    selector_weights = (
        {name: float(weight) for name, weight in selector_weights_override.items()}
        if selector_weights_override is not None
        else selector.weights(rows, as_of=as_of)
    )

    blended: dict[str, float] = {}
    source_parts: list[str] = []
    for selector_name, selector_weight in selector_weights.items():
        decision = base_decisions[selector_name]
        if decision.target_weights is not None:
            selector_targets = {k: float(v) for k, v in decision.target_weights.items()}
        else:
            positions = build_target_weights(
                kronos_scores,
                llm_scores,
                cost_depth,
                kronos_weight=decision.kronos_weight,
                llm_weight=decision.llm_weight,
                threshold=decision.threshold,
                rank_power=decision.rank_power,
                max_gross=decision.max_gross,
                allow_short=decision.allow_short,
                source=selector_name,
            )
            selector_targets = {p.ticker: p.weight for p in positions}
        source_parts.append(f"{selector_name}:{selector_weight:.6f}")
        for ticker, weight in selector_targets.items():
            blended[ticker] = blended.get(ticker, 0.0) + selector_weight * weight

    normalized = normalize_gross(blended, max_gross=max_gross)
    return DecisionResult(
        as_of=as_of,
        selector_weights=selector_weights,
        target_positions=positions_from_weights(normalized, source="rolling_rank_weighted_w24_p2"),
        metadata={
            "strategy": "rolling_rank_weighted_w24_p2",
            "lookback": selector.lookback,
            "rank_power": selector.rank_power,
            "selector_weight_debug": ";".join(source_parts),
        },
    )
