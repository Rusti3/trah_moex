from __future__ import annotations

import asyncio
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .schemas import TOP20_TICKERS


class KronosTop20Provider:
    def __init__(
        self,
        *,
        weights_dir: str | Path,
        weights_mode: str = "auto",
        device: str = "cpu",
        model_key: str = "base",
        context_days: int = 15,
        sample_count: int = 10,
        temperature: float = 0.6,
        top_p: float = 0.90,
    ):
        self.weights_dir = Path(weights_dir)
        self.weights_mode = weights_mode
        self.device = device
        self.model_key = model_key
        self.context_days = context_days
        self.sample_count = sample_count
        self.temperature = temperature
        self.top_p = top_p
        self._predictor = None
        self.last_good_scores: dict[str, float] = {}

    async def forecast_bullish_scores(self, as_of: datetime, tickers: tuple[str, ...] = TOP20_TICKERS) -> dict[str, float]:
        try:
            scores = await asyncio.to_thread(self._forecast_sync, as_of, tickers)
            if scores:
                self.last_good_scores = scores
                return scores
        except Exception:
            pass
        if self.last_good_scores:
            return dict(self.last_good_scores)
        return {ticker: 0.5 for ticker in tickers}

    def warm(self) -> None:
        self._ensure_predictor()

    def _ensure_predictor(self):
        if self._predictor is not None:
            return self._predictor
        from run_moex_hourly_rolling_backtest import make_predictor_with_mode

        self._predictor = make_predictor_with_mode(
            model_key=self.model_key,
            max_context=512,
            weights_dir=self.weights_dir,
            device=self.device,
            weights_mode=self.weights_mode,
        )
        return self._predictor

    def _forecast_sync(self, as_of: datetime, tickers: tuple[str, ...]) -> dict[str, float]:
        predictor = self._ensure_predictor()
        from run_moex_hourly_rolling_backtest import KLINE_COLS, fetch_moex_candles

        from_date = (as_of - timedelta(days=self.context_days + 5)).date().isoformat()
        till_date = as_of.date().isoformat()
        pred_returns: dict[str, float] = {}
        for ticker in tickers:
            try:
                df = fetch_moex_candles(
                    ticker,
                    from_date=from_date,
                    till_date=till_date,
                    interval=60,
                    source_interval=60,
                    drop_incomplete_last_candle=True,
                )
                df = df[df["timestamps"] < pd.Timestamp(as_of).floor("h")].tail(512).copy()
                if len(df) < 30:
                    continue
                y_ts = pd.Series([pd.Timestamp(df["timestamps"].iloc[-1]) + pd.Timedelta(hours=1)])
                sample_preds = predictor.predict_samples(
                    df=df[KLINE_COLS],
                    x_timestamp=df["timestamps"],
                    y_timestamp=y_ts,
                    pred_len=1,
                    T=self.temperature,
                    top_p=self.top_p,
                    sample_count=self.sample_count,
                    verbose=False,
                )
                pred_close = float(sample_preds[:, 0, KLINE_COLS.index("close")].mean())
                last_close = float(df["close"].iloc[-1])
                pred_returns[ticker] = pred_close / last_close - 1.0 if last_close > 0 else 0.0
            except Exception:
                continue
        return _percentile_scores(pred_returns, tickers)


def _percentile_scores(values: dict[str, float], tickers: tuple[str, ...]) -> dict[str, float]:
    if not values:
        return {ticker: 0.5 for ticker in tickers}
    sorted_items = sorted(values.items(), key=lambda kv: kv[1])
    n = len(sorted_items)
    scores = {ticker: (idx + 1) / n for idx, (ticker, _) in enumerate(sorted_items)}
    for ticker in tickers:
        scores.setdefault(ticker, 0.5)
    return scores
