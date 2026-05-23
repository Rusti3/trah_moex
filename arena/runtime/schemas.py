from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping


TOP20_TICKERS: tuple[str, ...] = (
    "LKOH",
    "SBER",
    "ROSN",
    "GAZP",
    "VTBR",
    "YDEX",
    "PLZL",
    "T",
    "NVTK",
    "X5",
    "GMKN",
    "MGNT",
    "ALRS",
    "AFLT",
    "CHMF",
    "NLMK",
    "MOEX",
    "SNGSP",
    "MTSS",
    "PIKK",
)


BASE_SELECTORS: tuple[str, ...] = (
    "selector_family_first",
    "selector_news_aware",
    "selector_marketwide_news",
)


@dataclass(frozen=True)
class SelectorReturn:
    timestamp: datetime | str
    returns: Mapping[str, float]


@dataclass(frozen=True)
class BaseSelectorDecision:
    name: str
    kronos_weight: float
    llm_weight: float
    threshold: float
    rank_power: float
    max_gross: float = 1.0
    allow_short: bool = True
    target_weights: Mapping[str, float] | None = None


@dataclass(frozen=True)
class TargetPosition:
    ticker: str
    side: str
    weight: float
    score: float
    source: str = "combined"


@dataclass(frozen=True)
class DecisionResult:
    as_of: datetime | str
    selector_weights: Mapping[str, float]
    target_positions: tuple[TargetPosition, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_order_targets(self) -> list[dict[str, Any]]:
        return [
            {
                "ticker": p.ticker,
                "side": p.side,
                "target_weight": p.weight,
                "score": p.score,
                "source": p.source,
            }
            for p in self.target_positions
        ]
