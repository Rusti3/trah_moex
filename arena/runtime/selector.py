from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Mapping, Sequence

import pandas as pd

from .schemas import BASE_SELECTORS


def _coerce_timestamp(value) -> pd.Timestamp | None:
    if value is None:
        return None
    ts = pd.Timestamp(value)
    return None if pd.isna(ts) else ts


def _row_timestamp(row: Mapping[str, object]) -> pd.Timestamp | None:
    return _coerce_timestamp(row.get("timestamp") or row.get("as_of"))


def _row_return(row: Mapping[str, object], selector: str) -> float:
    if selector in row:
        return float(row.get(selector) or 0.0)
    returns = row.get("returns")
    if isinstance(returns, Mapping):
        return float(returns.get(selector, 0.0) or 0.0)
    return 0.0


@dataclass(frozen=True)
class RollingRankWeightedSelector:
    """Live-safe combiner for base selector interval returns.

    It mirrors `rolling_rank_weighted_w24_p2`: only rows strictly before
    `as_of` are eligible, the three base selectors are ranked by their rolling
    return sum, and weights are proportional to `rank^-rank_power`.
    """

    lookback: int = 24
    rank_power: float = 2.0
    base_selectors: Sequence[str] = BASE_SELECTORS
    min_history: int | None = None

    def history_cutoff(self) -> int:
        return self.min_history if self.min_history is not None else max(6, self.lookback // 4)

    def rank_order(
        self,
        history: Iterable[Mapping[str, object]],
        as_of: datetime | str | pd.Timestamp | None = None,
    ) -> list[str]:
        rows = list(history)
        as_of_ts = _coerce_timestamp(as_of)
        if as_of_ts is not None:
            rows = [r for r in rows if (_row_timestamp(r) is None or _row_timestamp(r) < as_of_ts)]
        rows = rows[-self.lookback :]
        if len(rows) < self.history_cutoff():
            return list(self.base_selectors)
        scores = {
            selector: sum(_row_return(row, selector) for row in rows)
            for selector in self.base_selectors
        }
        return sorted(self.base_selectors, key=lambda name: scores[name], reverse=True)

    def weights(
        self,
        history: Iterable[Mapping[str, object]],
        as_of: datetime | str | pd.Timestamp | None = None,
    ) -> dict[str, float]:
        order = self.rank_order(history, as_of=as_of)
        raw = [(rank + 1) ** (-self.rank_power) for rank in range(len(order))]
        total = sum(raw)
        return {selector: value / total for selector, value in zip(order, raw)}

    def combine_returns(
        self,
        current_returns: Mapping[str, float],
        history: Iterable[Mapping[str, object]],
        as_of: datetime | str | pd.Timestamp | None = None,
    ) -> float:
        weights = self.weights(history, as_of=as_of)
        return sum(float(current_returns.get(selector, 0.0) or 0.0) * weight for selector, weight in weights.items())
