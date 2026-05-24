from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from .schemas import BASE_SELECTORS

try:
    import lightgbm as lgb
except Exception:  # pragma: no cover
    lgb = None


@dataclass(frozen=True)
class LightGBMSelectorResult:
    selector_scores: dict[str, float]
    selector_weights: dict[str, float]
    trained_rows: int
    mode: str
    reason: str = ""


class LiveLightGBMSelector:
    def __init__(
        self,
        *,
        min_train_intervals: int = 48,
        train_lookback_intervals: int = 512,
        rank_power: float = 2.0,
        n_estimators: int = 60,
        base_selectors: tuple[str, ...] = BASE_SELECTORS,
    ):
        self.min_train_intervals = min_train_intervals
        self.train_lookback_intervals = train_lookback_intervals
        self.rank_power = rank_power
        self.n_estimators = n_estimators
        self.base_selectors = tuple(base_selectors)

    def predict_weights(
        self,
        *,
        current_features: Mapping[str, float],
        training_rows: list[Mapping[str, Any]],
    ) -> LightGBMSelectorResult | None:
        if lgb is None:
            return None
        rows = list(training_rows)[-self.train_lookback_intervals :]
        if len(rows) < self.min_train_intervals:
            return None
        feature_names = sorted({key for row in rows for key in (row.get("features") or {}).keys()} | set(current_features.keys()))
        x_train: list[list[float]] = []
        y_train: list[float] = []
        for row in rows:
            features = row.get("features") or {}
            returns = row.get("returns") or {}
            for selector_index, selector in enumerate(self.base_selectors):
                x_train.append(_vector(features, feature_names, selector_index, len(self.base_selectors)))
                y_train.append(_float(returns.get(selector), 0.0))
        if len(set(round(v, 12) for v in y_train)) <= 1:
            return None
        try:
            model = lgb.LGBMRegressor(
                n_estimators=self.n_estimators,
                learning_rate=0.05,
                max_depth=3,
                min_child_samples=8,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
                n_jobs=1,
                verbosity=-1,
            )
            model.fit(x_train, y_train)
            scores = {
                selector: float(model.predict([_vector(current_features, feature_names, idx, len(self.base_selectors))])[0])
                for idx, selector in enumerate(self.base_selectors)
            }
        except Exception as exc:
            return LightGBMSelectorResult({}, {}, len(rows), mode="fallback", reason=str(exc)[:300])
        return LightGBMSelectorResult(
            selector_scores=scores,
            selector_weights=rank_weights_from_scores(scores, rank_power=self.rank_power, base_selectors=self.base_selectors),
            trained_rows=len(rows),
            mode="lightgbm",
        )


def rank_weights_from_scores(
    scores: Mapping[str, float],
    *,
    rank_power: float = 2.0,
    base_selectors: tuple[str, ...] = BASE_SELECTORS,
) -> dict[str, float]:
    order = sorted(base_selectors, key=lambda selector: float(scores.get(selector, 0.0)), reverse=True)
    raw = [(rank + 1) ** (-rank_power) for rank in range(len(order))]
    total = sum(raw)
    return {selector: value / total for selector, value in zip(order, raw)}


def _vector(features: Mapping[str, Any], feature_names: list[str], selector_index: int, selector_count: int) -> list[float]:
    selector_one_hot = [1.0 if selector_index == idx else 0.0 for idx in range(selector_count)]
    return [_float(features.get(name), 0.0) for name in feature_names] + [float(selector_index)] + selector_one_hot


def _float(value: Any, default: float) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default
