from __future__ import annotations

from datetime import datetime
from typing import Mapping, Protocol, Sequence


class KronosForecastProvider(Protocol):
    async def forecast_bullish_scores(self, as_of: datetime, tickers: Sequence[str]) -> Mapping[str, float]:
        """Return ticker -> bullish score in [0, 1]."""


class LLMNewsScorer(Protocol):
    async def score_daily_news(self, as_of: datetime, tickers: Sequence[str]) -> Mapping[str, float]:
        """Return ticker -> bullish news-background score in [0, 1]."""


class MoexCostDepthProvider(Protocol):
    async def current_cost_depth(self, as_of: datetime, tickers: Sequence[str]) -> Mapping[str, Mapping[str, object]]:
        """Return ticker -> liquidity/cost metadata known at decision time."""
