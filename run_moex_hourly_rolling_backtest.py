import argparse
import math
import os
import random
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import torch
except ImportError:
    torch = None

from model import Kronos, KronosPredictor, KronosTokenizer
from run_moex_baseline import (
    KLINE_COLS,
    MOEX_BASE,
    MODEL_CONFIGS,
    TOKENIZER_CONFIGS,
    default_weights_dir,
    iss_block_to_df,
    iss_get_json,
    make_predictor,
    parse_models,
    resolve_device,
)


HF_SOURCES = {
    "mini": {
        "model": "NeoQuasar/Kronos-mini",
        "tokenizer": "NeoQuasar/Kronos-Tokenizer-2k",
        "native_context": 2048,
    },
    "small": {
        "model": "NeoQuasar/Kronos-small",
        "tokenizer": "NeoQuasar/Kronos-Tokenizer-base",
        "native_context": 512,
    },
    "base": {
        "model": "NeoQuasar/Kronos-base",
        "tokenizer": "NeoQuasar/Kronos-Tokenizer-base",
        "native_context": 512,
    },
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def local_weights_available(weights_dir: Path, model_key: str) -> bool:
    model_cfg = MODEL_CONFIGS[model_key]
    tokenizer_cfg = TOKENIZER_CONFIGS[model_cfg["tokenizer_key"]]
    return (weights_dir / model_cfg["weights_file"]).exists() and (weights_dir / tokenizer_cfg["weights_file"]).exists()


def make_predictor_with_mode(
    model_key: str,
    max_context: int,
    weights_dir: Path,
    device: str,
    weights_mode: str,
):
    if weights_mode == "local":
        return make_predictor(model_key=model_key, lookback=max_context, weights_dir=weights_dir, device=device)

    if weights_mode == "auto" and local_weights_available(weights_dir, model_key):
        return make_predictor(model_key=model_key, lookback=max_context, weights_dir=weights_dir, device=device)

    if weights_mode not in {"hf", "auto"}:
        raise ValueError("--weights-mode must be one of: local, hf, auto")

    src = HF_SOURCES[model_key]
    tokenizer = KronosTokenizer.from_pretrained(src["tokenizer"])
    model = Kronos.from_pretrained(src["model"])
    tokenizer.eval()
    model.eval()
    return KronosPredictor(model, tokenizer, device=device, max_context=min(src["native_context"], max_context))


def get_issue_cap_candidates(limit: int) -> pd.DataFrame:
    url = f"{MOEX_BASE}/engines/stock/markets/shares/securities.json"
    params = {
        "iss.meta": "off",
        "iss.only": "securities,marketdata",
        "securities.columns": "SECID,SHORTNAME,SECNAME,ISIN,BOARDID,MARKETCODE,SECTYPE,LISTLEVEL",
        "marketdata.columns": "SECID,BOARDID,ISSUECAPITALIZATION,ISSUECAPITALIZATION_UPDATETIME,VALTODAY,NUMTRADES",
    }
    js = iss_get_json(url, params=params)
    securities = iss_block_to_df(js, "securities")
    marketdata = iss_block_to_df(js, "marketdata")

    required = {"SECID", "BOARDID"}
    if securities.empty or marketdata.empty or not required.issubset(securities.columns) or not required.issubset(marketdata.columns):
        raise RuntimeError("Could not read MOEX securities and marketdata blocks.")

    df = securities.merge(marketdata, on=["SECID", "BOARDID"], how="inner")
    df["ISSUECAPITALIZATION"] = pd.to_numeric(df["ISSUECAPITALIZATION"], errors="coerce")
    df = df.dropna(subset=["SECID", "ISSUECAPITALIZATION"])
    df = df[df["ISSUECAPITALIZATION"] > 0]

    if "SECTYPE" in df.columns:
        df = df[df["SECTYPE"].astype(str) == "1"]

    if df.empty:
        raise RuntimeError("MOEX returned no share rows with positive ISSUECAPITALIZATION.")

    return (
        df.sort_values("ISSUECAPITALIZATION", ascending=False)
        .drop_duplicates(subset=["SECID"], keep="first")
        .head(limit)
        .reset_index(drop=True)
    )


def is_incomplete_last_candle(row: pd.Series, interval: int) -> bool:
    begin_ts = pd.Timestamp(row["timestamps"])
    end_ts = pd.Timestamp(row["end"])

    if interval == 24:
        expected_end = begin_ts.normalize() + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        tolerance = pd.Timedelta(minutes=2)
    else:
        expected_end = begin_ts + pd.Timedelta(minutes=interval) - pd.Timedelta(seconds=1)
        tolerance = pd.Timedelta(minutes=1)

    return end_ts < expected_end - tolerance


def fetch_moex_candles(
    secid: str,
    from_date: str,
    till_date: str,
    interval: int,
    source_interval: int,
    drop_incomplete_last_candle: bool,
) -> pd.DataFrame:
    url = f"{MOEX_BASE}/engines/stock/markets/shares/securities/{secid}/candles.json"
    chunks = []
    start = 0

    while True:
        params = {
            "interval": source_interval,
            "from": from_date,
            "till": till_date,
            "start": start,
            "iss.meta": "off",
        }
        part = iss_block_to_df(iss_get_json(url, params=params), "candles")
        if part.empty:
            break
        chunks.append(part)
        start += len(part)

    if not chunks:
        return pd.DataFrame()

    df = pd.concat(chunks, ignore_index=True).rename(columns={"begin": "timestamps", "value": "amount"})
    needed = ["timestamps", "end", "open", "high", "low", "close", "volume", "amount"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"{secid}: missing MOEX candle columns {missing}; got {list(df.columns)}")

    df = df[needed].copy()
    df["timestamps"] = pd.to_datetime(df["timestamps"])
    df["end"] = pd.to_datetime(df["end"])
    for c in KLINE_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["timestamps", "end", "open", "high", "low", "close"])
    df["volume"] = df["volume"].fillna(0.0)
    df["amount"] = df["amount"].fillna(0.0)
    df = df.sort_values("timestamps").drop_duplicates(subset=["timestamps"]).reset_index(drop=True)

    if source_interval != interval:
        df = resample_candles(df, interval)

    if drop_incomplete_last_candle and not df.empty and is_incomplete_last_candle(df.iloc[-1], interval):
        print(f"INFO: {secid}: drop incomplete last candle begin={df['timestamps'].iloc[-1]} end={df['end'].iloc[-1]}")
        df = df.iloc[:-1].copy()

    return df


def resample_candles(df: pd.DataFrame, interval: int) -> pd.DataFrame:
    if interval <= 0:
        raise ValueError("interval must be positive.")

    freq = f"{interval}min"
    tmp = df.copy()
    tmp["bucket"] = tmp["timestamps"].dt.floor(freq)

    grouped = tmp.groupby("bucket", sort=True)
    out = grouped.agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        amount=("amount", "sum"),
        end=("end", "max"),
    ).reset_index()
    out = out.rename(columns={"bucket": "timestamps"})
    return out[["timestamps", "end", "open", "high", "low", "close", "volume", "amount"]].reset_index(drop=True)


def make_timedelta(minutes: int | None, days: int | None, name: str) -> pd.Timedelta:
    if minutes is not None:
        if minutes <= 0:
            raise ValueError(f"--{name}-minutes must be positive.")
        return pd.Timedelta(minutes=minutes)
    if days is None or days <= 0:
        raise ValueError(f"--{name}-days must be positive when --{name}-minutes is not set.")
    return pd.Timedelta(days=days)


def get_reference_target_timestamps(
    candidates: pd.DataFrame,
    from_date: str,
    till_date: str,
    interval: int,
    source_interval: int,
    backtest_delta: pd.Timedelta,
    context_delta: pd.Timedelta,
    min_context_bars: int,
    drop_incomplete_last_candle: bool,
) -> tuple[str, list[pd.Timestamp]]:
    min_rows = min_context_bars + 1
    for _, row in candidates.iterrows():
        ticker = str(row["SECID"])
        try:
            df = fetch_moex_candles(ticker, from_date, till_date, interval, source_interval, drop_incomplete_last_candle)
        except Exception as exc:
            print(f"WARN: {ticker}: failed as timestamp reference: {exc}")
            continue

        if len(df) < min_rows:
            continue

        last_ts = pd.Timestamp(df["timestamps"].iloc[-1])
        start_ts = last_ts - backtest_delta
        target_times = list(df.loc[df["timestamps"] > start_ts, "timestamps"])
        target_times = [
            ts
            for ts in target_times
            if len(df[(df["timestamps"] >= ts - context_delta) & (df["timestamps"] < ts)]) >= min_context_bars
        ]
        if target_times:
            return ticker, target_times

    raise RuntimeError("Could not find a reference ticker with enough intraday candles.")


def expected_target_bars(interval: int, backtest_delta: pd.Timedelta) -> int:
    bar_delta = pd.Timedelta(minutes=interval)
    return max(1, int(backtest_delta / bar_delta))


def latest_valid_target_timestamps(
    df: pd.DataFrame,
    target_bar_count: int,
    context_delta: pd.Timedelta,
    min_context_bars: int,
) -> list[pd.Timestamp]:
    if df.empty:
        return []

    valid = [
        ts
        for ts in df["timestamps"]
        if len(df[(df["timestamps"] >= ts - context_delta) & (df["timestamps"] < ts)]) >= min_context_bars
    ]
    return valid[-target_bar_count:]


def valid_target_timestamps_for_ticker(
    df: pd.DataFrame,
    reference_targets: list[pd.Timestamp],
    context_delta: pd.Timedelta,
    min_context_bars: int,
) -> list[pd.Timestamp]:
    timestamps = set(df["timestamps"])
    valid = []

    for target_ts in reference_targets:
        if target_ts not in timestamps:
            continue
        context = df[(df["timestamps"] >= target_ts - context_delta) & (df["timestamps"] < target_ts)]
        if len(context) >= min_context_bars:
            valid.append(target_ts)

    return valid


def select_universe_with_intraday_data(
    candidates: pd.DataFrame,
    top_n: int,
    from_date: str,
    till_date: str,
    interval: int,
    source_interval: int,
    reference_targets: list[pd.Timestamp] | None,
    target_mode: str,
    backtest_delta: pd.Timedelta,
    context_delta: pd.Timedelta,
    min_context_bars: int,
    min_target_coverage: float,
    drop_incomplete_last_candle: bool,
    out_dir: Path,
    save_candles: bool,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, list[pd.Timestamp]]]:
    selected_rows = []
    data: dict[str, pd.DataFrame] = {}
    ticker_targets: dict[str, list[pd.Timestamp]] = {}
    if target_mode == "common":
        if not reference_targets:
            raise ValueError("reference_targets must be provided when target_mode='common'.")
        target_bar_count = len(reference_targets)
    else:
        target_bar_count = expected_target_bars(interval, backtest_delta)
    min_target_count = max(1, math.ceil(target_bar_count * min_target_coverage))

    for _, row in candidates.iterrows():
        ticker = str(row["SECID"])
        try:
            df = fetch_moex_candles(ticker, from_date, till_date, interval, source_interval, drop_incomplete_last_candle)
        except Exception as exc:
            print(f"WARN: {ticker}: failed to fetch candles: {exc}")
            continue

        if df.empty:
            print(f"WARN: {ticker}: no candles, skip.")
            continue

        if target_mode == "common":
            valid_targets = valid_target_timestamps_for_ticker(df, reference_targets or [], context_delta, min_context_bars)
        else:
            valid_targets = latest_valid_target_timestamps(df, target_bar_count, context_delta, min_context_bars)

        if len(valid_targets) < min_target_count:
            print(f"WARN: {ticker}: only {len(valid_targets)}/{target_bar_count} target bars, skip.")
            continue

        enriched = row.to_dict()
        enriched["rows"] = len(df)
        enriched["first_candle"] = df["timestamps"].iloc[0]
        enriched["last_candle"] = df["timestamps"].iloc[-1]
        enriched["valid_target_bars"] = len(valid_targets)
        enriched["reference_target_bars"] = target_bar_count
        enriched["target_coverage"] = len(valid_targets) / target_bar_count
        selected_rows.append(enriched)
        data[ticker] = df
        ticker_targets[ticker] = valid_targets

        if save_candles:
            df.to_csv(out_dir / f"candles_{ticker}.csv", index=False)

        if len(selected_rows) >= top_n:
            break

    if len(selected_rows) < top_n:
        raise RuntimeError(
            f"Only selected {len(selected_rows)} valid tickers out of requested {top_n}. "
            "Increase --candidate-n or lower --min-target-coverage."
        )

    return pd.DataFrame(selected_rows), data, ticker_targets


def predict_one_bar(
    predictor,
    df: pd.DataFrame,
    target_ts: pd.Timestamp,
    context_delta: pd.Timedelta,
    sample_count: int,
    temperature: float,
    top_p: float,
    direction_confidence_threshold: float,
) -> dict:
    x = df[(df["timestamps"] >= target_ts - context_delta) & (df["timestamps"] < target_ts)].copy()
    y = df[df["timestamps"] == target_ts].copy()

    if x.empty or len(y) != 1:
        raise ValueError(f"Bad target/context for {target_ts}: context={len(x)}, target_rows={len(y)}")

    start_time = time.time()
    sample_preds = predictor.predict_samples(
        df=x[KLINE_COLS],
        x_timestamp=x["timestamps"],
        y_timestamp=y["timestamps"],
        pred_len=1,
        T=temperature,
        top_p=top_p,
        sample_count=sample_count,
        verbose=False,
    )
    pred_df = pd.DataFrame(
        sample_preds.mean(axis=0),
        columns=KLINE_COLS,
        index=y["timestamps"],
    )
    elapsed = time.time() - start_time

    last_close = float(x["close"].iloc[-1])
    actual_close = float(y["close"].iloc[0])
    pred_close = float(pred_df["close"].iloc[0])
    error = pred_close - actual_close
    abs_error = abs(error)
    pct_error = abs_error / actual_close * 100.0 if actual_close != 0 else np.nan
    actual_direction = float(np.sign(actual_close - last_close))
    pred_direction = float(np.sign(pred_close - last_close))
    sample_close_dirs = np.sign(sample_preds[:, 0, KLINE_COLS.index("close")] - last_close)
    unique_dirs, dir_counts = np.unique(sample_close_dirs, return_counts=True)
    best_idx = int(np.argmax(dir_counts))
    confident_direction = float(unique_dirs[best_idx])
    direction_confidence = float(dir_counts[best_idx] / sample_count)
    is_direction_confident = bool(direction_confidence >= direction_confidence_threshold)
    confident_direction_hit = bool(is_direction_confident and confident_direction == actual_direction)

    return {
        "target_timestamp": y["timestamps"].iloc[0],
        "context_from": x["timestamps"].iloc[0],
        "context_till": x["timestamps"].iloc[-1],
        "context_bars": len(x),
        "last_close": last_close,
        "actual_close": actual_close,
        "pred_close": pred_close,
        "pred_open": float(pred_df["open"].iloc[0]),
        "pred_high": float(pred_df["high"].iloc[0]),
        "pred_low": float(pred_df["low"].iloc[0]),
        "pred_volume": float(pred_df["volume"].iloc[0]),
        "pred_amount": float(pred_df["amount"].iloc[0]),
        "error_close": float(error),
        "abs_error_close": float(abs_error),
        "pct_error_close": float(pct_error),
        "actual_direction": actual_direction,
        "pred_direction": pred_direction,
        "direction_hit": bool(actual_direction == pred_direction),
        "direction_confidence": direction_confidence,
        "confident_direction": confident_direction,
        "is_direction_confident": is_direction_confident,
        "confident_direction_hit": confident_direction_hit,
        "seconds": float(elapsed),
    }


def run_model_backtest(
    model_name: str,
    predictor,
    data: dict[str, pd.DataFrame],
    ticker_targets: dict[str, list[pd.Timestamp]],
    context_delta: pd.Timedelta,
    sample_count: int,
    temperature: float,
    top_p: float,
    direction_confidence_threshold: float,
) -> list[dict]:
    rows = []
    for ticker, df in tqdm(data.items(), desc=f"Intraday {model_name}"):
        for step, target_ts in enumerate(ticker_targets[ticker], start=1):
            pred = predict_one_bar(
                predictor=predictor,
                df=df,
                target_ts=target_ts,
                context_delta=context_delta,
                sample_count=sample_count,
                temperature=temperature,
                top_p=top_p,
                direction_confidence_threshold=direction_confidence_threshold,
            )
            rows.append({"model": model_name, "ticker": ticker, "rolling_step": step, **pred})
    return rows


def summarize(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    def rmse(series: pd.Series) -> float:
        return float(np.sqrt(np.mean(np.square(series))))

    def add_confident_only(summary: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
        confident = predictions[predictions["is_direction_confident"]]
        metric_name = "direction_acc_confident_only"
        if confident.empty:
            summary[metric_name] = np.nan
            return summary

        confident_acc = (
            confident.groupby(group_cols, as_index=False)["confident_direction_hit"]
            .mean()
            .rename(columns={"confident_direction_hit": metric_name})
        )
        return summary.merge(confident_acc, on=group_cols, how="left")

    common_aggs = dict(
        mae_close=("abs_error_close", "mean"),
        rmse_close=("error_close", rmse),
        mape_close_pct=("pct_error_close", "mean"),
        direction_acc=("direction_hit", "mean"),
        direction_confidence=("direction_confidence", "mean"),
        direction_confidence_coverage=("is_direction_confident", "mean"),
        direction_acc_confident_strict=("confident_direction_hit", "mean"),
        avg_seconds=("seconds", "mean"),
        avg_context_bars=("context_bars", "mean"),
        rows=("ticker", "size"),
    )

    by_model = (
        predictions.groupby("model", as_index=False)
        .agg(**common_aggs)
        .sort_values("mae_close")
        .reset_index(drop=True)
    )
    by_model = add_confident_only(by_model, ["model"])
    by_ticker = (
        predictions.groupby(["model", "ticker"], as_index=False)
        .agg(**common_aggs)
        .sort_values(["model", "mae_close"])
        .reset_index(drop=True)
    )
    by_ticker = add_confident_only(by_ticker, ["model", "ticker"])
    by_timestamp = (
        predictions.groupby(["model", "target_timestamp"], as_index=False)
        .agg(**common_aggs)
        .sort_values(["model", "target_timestamp"])
        .reset_index(drop=True)
    )
    by_timestamp = add_confident_only(by_timestamp, ["model", "target_timestamp"])
    return by_model, by_ticker, by_timestamp


def target_timestamps_frame(ticker_targets: dict[str, list[pd.Timestamp]], target_mode: str) -> pd.DataFrame:
    rows = []
    for ticker, targets in ticker_targets.items():
        for step, target_ts in enumerate(targets, start=1):
            rows.append(
                {
                    "ticker": ticker,
                    "rolling_step": step,
                    "target_timestamp": target_ts,
                    "target_mode": target_mode,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Intraday Kronos rolling backtest on MOEX candles.")
    parser.add_argument("--from-date", default=str(date.today() - timedelta(days=7)))
    parser.add_argument("--till-date", default=str(date.today()))
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--candidate-n", type=int, default=300)
    parser.add_argument("--interval", type=int, default=1)
    parser.add_argument("--source-interval", type=int, default=None)
    parser.add_argument("--context-days", type=int, default=None)
    parser.add_argument("--context-minutes", type=int, default=None)
    parser.add_argument("--backtest-days", type=int, default=None)
    parser.add_argument("--backtest-minutes", type=int, default=None)
    parser.add_argument("--min-context-bars", type=int, default=30)
    parser.add_argument("--min-target-coverage", type=float, default=0.8)
    parser.add_argument("--target-mode", choices=["per-ticker", "common"], default="per-ticker")
    parser.add_argument("--max-context", type=int, default=512)
    parser.add_argument("--models", default="mini,small,base", help="Comma-separated subset: mini,small,base")
    parser.add_argument("--sample-count", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--direction-confidence-threshold", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs_moex_kronos_1m_30m_10m"))
    parser.add_argument("--weights-dir", type=Path, default=default_weights_dir())
    parser.add_argument("--weights-mode", choices=["local", "hf", "auto"], default="local")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda:0, mps, ...")
    parser.add_argument("--drop-incomplete-last-candle", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-candles", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if args.context_minutes is None and args.context_days is None:
        args.context_minutes = 30
    if args.backtest_minutes is None and args.backtest_days is None:
        args.backtest_minutes = 10

    if args.top_n <= 0 or args.candidate_n <= 0:
        raise ValueError("--top-n and --candidate-n must be positive.")
    if args.candidate_n < args.top_n:
        raise ValueError("--candidate-n must be at least --top-n.")
    if args.interval <= 0:
        raise ValueError("--interval must be positive.")
    source_interval = args.source_interval if args.source_interval is not None else args.interval
    if source_interval <= 0:
        raise ValueError("--source-interval must be positive.")
    if args.min_context_bars <= 0:
        raise ValueError("--min-context-bars must be positive.")
    if not 0 < args.min_target_coverage <= 1:
        raise ValueError("--min-target-coverage must be in (0, 1].")
    if args.sample_count <= 0:
        raise ValueError("--sample-count must be positive.")
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive.")
    if not 0 < args.top_p <= 1:
        raise ValueError("--top-p must be in (0, 1].")
    if not 0 < args.direction_confidence_threshold <= 1:
        raise ValueError("--direction-confidence-threshold must be in (0, 1].")

    set_seed(args.seed)
    models = parse_models(args.models)
    device = resolve_device(args.device)
    weights_dir = args.weights_dir.resolve()
    out_dir = args.out_dir.resolve()
    os.makedirs(out_dir, exist_ok=True)
    context_delta = make_timedelta(args.context_minutes, args.context_days, "context")
    backtest_delta = make_timedelta(args.backtest_minutes, args.backtest_days, "backtest")

    print(f"Weights dir: {weights_dir}")
    print(f"Weights mode: {args.weights_mode}")
    print(f"Models: {models}")
    print(f"Device: {device}")
    print(f"Output dir: {out_dir}")
    print(
        f"Intraday setup: interval={args.interval}, source_interval={source_interval}, "
        f"context={context_delta}, backtest={backtest_delta}, sample_count={args.sample_count}, "
        f"T={args.temperature}, top_p={args.top_p}, target_mode={args.target_mode}, "
        f"direction_confidence_threshold={args.direction_confidence_threshold}"
    )

    candidates = get_issue_cap_candidates(args.candidate_n)
    candidates.to_csv(out_dir / "candidate_universe.csv", index=False)

    reference_targets = None
    if args.target_mode == "common":
        reference_ticker, reference_targets = get_reference_target_timestamps(
            candidates=candidates,
            from_date=args.from_date,
            till_date=args.till_date,
            interval=args.interval,
            source_interval=source_interval,
            backtest_delta=backtest_delta,
            context_delta=context_delta,
            min_context_bars=args.min_context_bars,
            drop_incomplete_last_candle=args.drop_incomplete_last_candle,
        )
        print(f"Reference target bars: {len(reference_targets)} (reference={reference_ticker})")
    else:
        expected_bars = expected_target_bars(args.interval, backtest_delta)
        print(f"Using each ticker's own latest target bars; expected bars per ticker: {expected_bars}")

    selected, data, ticker_targets = select_universe_with_intraday_data(
        candidates=candidates,
        top_n=args.top_n,
        from_date=args.from_date,
        till_date=args.till_date,
        interval=args.interval,
        source_interval=source_interval,
        reference_targets=reference_targets,
        target_mode=args.target_mode,
        backtest_delta=backtest_delta,
        context_delta=context_delta,
        min_context_bars=args.min_context_bars,
        min_target_coverage=args.min_target_coverage,
        drop_incomplete_last_candle=args.drop_incomplete_last_candle,
        out_dir=out_dir,
        save_candles=args.save_candles,
    )
    selected_path = out_dir / "selected_universe.csv"
    target_path = out_dir / "target_timestamps.csv"
    selected.to_csv(selected_path, index=False)
    target_timestamps_frame(ticker_targets, args.target_mode).to_csv(target_path, index=False)
    print(f"Selected {len(selected)} tickers.")

    predictions_path = out_dir / "intraday_predictions.csv"
    partial_predictions_path = out_dir / "intraday_predictions_partial.csv"
    by_model_path = out_dir / "intraday_summary_by_model.csv"
    by_ticker_path = out_dir / "intraday_summary_by_ticker.csv"
    by_timestamp_path = out_dir / "intraday_summary_by_timestamp.csv"
    model_errors_path = out_dir / "model_errors.csv"

    all_rows = []
    model_errors = []
    for model_name in models:
        print(f"\nLoading model: {model_name}")
        predictor = None
        try:
            predictor = make_predictor_with_mode(
                model_key=model_name,
                max_context=args.max_context,
                weights_dir=weights_dir,
                device=device,
                weights_mode=args.weights_mode,
            )
            all_rows.extend(
                run_model_backtest(
                    model_name=model_name,
                    predictor=predictor,
                    data=data,
                    ticker_targets=ticker_targets,
                    context_delta=context_delta,
                    sample_count=args.sample_count,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    direction_confidence_threshold=args.direction_confidence_threshold,
                )
            )
            pd.DataFrame(all_rows).to_csv(partial_predictions_path, index=False)
        except Exception as exc:
            model_errors.append({"model": model_name, "error": repr(exc)})
            print(f"WARN: {model_name}: failed: {exc}")
        finally:
            if predictor is not None:
                del predictor
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()

    predictions = pd.DataFrame(all_rows)
    if predictions.empty:
        raise RuntimeError("No intraday predictions were produced.")

    by_model, by_ticker, by_timestamp = summarize(predictions)

    predictions.to_csv(predictions_path, index=False)
    by_model.to_csv(by_model_path, index=False)
    by_ticker.to_csv(by_ticker_path, index=False)
    by_timestamp.to_csv(by_timestamp_path, index=False)
    if model_errors:
        pd.DataFrame(model_errors).to_csv(model_errors_path, index=False)

    if not by_model["direction_acc"].between(0, 1).all():
        raise RuntimeError("direction_acc is outside [0, 1].")
    if not by_model["mae_close"].is_monotonic_increasing:
        raise RuntimeError("intraday_summary_by_model.csv is not sorted by mae_close.")

    print("\n=== SUMMARY BY MODEL ===")
    print(by_model)
    print("\nSaved:")
    print(selected_path)
    print(target_path)
    print(predictions_path)
    print(by_model_path)
    print(by_ticker_path)
    print(by_timestamp_path)
    if model_errors:
        print(model_errors_path)


if __name__ == "__main__":
    main()
