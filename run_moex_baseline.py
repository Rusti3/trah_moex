import argparse
import os
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from pandas.tseries.offsets import BDay
from safetensors.torch import load_file
from tqdm import tqdm

try:
    import torch
except ImportError:
    torch = None

from model import Kronos, KronosTokenizer, KronosPredictor


MOEX_BASE = "https://iss.moex.com/iss"

MODEL_CONFIGS = {
    "mini": {
        "class": Kronos,
        "weights_file": "Kronos-mini.safetensors",
        "tokenizer_key": "2k",
        "native_context": 2048,
        "config": {
            "attn_dropout_p": 0.0,
            "d_model": 256,
            "ff_dim": 512,
            "ffn_dropout_p": 0.2,
            "learn_te": True,
            "n_heads": 4,
            "n_layers": 4,
            "resid_dropout_p": 0.2,
            "s1_bits": 10,
            "s2_bits": 10,
            "token_dropout_p": 0.0,
        },
    },
    "small": {
        "class": Kronos,
        "weights_file": "Kronos-small.safetensors",
        "tokenizer_key": "base",
        "native_context": 512,
        "config": {
            "attn_dropout_p": 0.1,
            "d_model": 512,
            "ff_dim": 1024,
            "ffn_dropout_p": 0.25,
            "learn_te": True,
            "n_heads": 8,
            "n_layers": 8,
            "resid_dropout_p": 0.25,
            "s1_bits": 10,
            "s2_bits": 10,
            "token_dropout_p": 0.1,
        },
    },
    "base": {
        "class": Kronos,
        "weights_file": "Kronos-base.safetensors",
        "tokenizer_key": "base",
        "native_context": 512,
        "config": {
            "attn_dropout_p": 0.0,
            "d_model": 832,
            "ff_dim": 2048,
            "ffn_dropout_p": 0.2,
            "learn_te": True,
            "n_heads": 16,
            "n_layers": 12,
            "resid_dropout_p": 0.2,
            "s1_bits": 10,
            "s2_bits": 10,
            "token_dropout_p": 0.0,
        },
    },
}

TOKENIZER_CONFIGS = {
    "2k": {
        "class": KronosTokenizer,
        "weights_file": "Kronos-Tokenizer-2k.safetensors",
        "config": {
            "attn_dropout_p": 0.0,
            "beta": 0.05,
            "d_in": 6,
            "d_model": 256,
            "ff_dim": 512,
            "ffn_dropout_p": 0.0,
            "gamma": 1.1,
            "gamma0": 1.0,
            "group_size": 5,
            "n_dec_layers": 4,
            "n_enc_layers": 4,
            "n_heads": 4,
            "resid_dropout_p": 0.0,
            "s1_bits": 10,
            "s2_bits": 10,
            "zeta": 0.05,
        },
    },
    "base": {
        "class": KronosTokenizer,
        "weights_file": "Kronos-Tokenizer-base.safetensors",
        "config": {
            "attn_dropout_p": 0.0,
            "beta": 0.05,
            "d_in": 6,
            "d_model": 256,
            "ff_dim": 512,
            "ffn_dropout_p": 0.0,
            "gamma": 1.1,
            "gamma0": 1.0,
            "group_size": 4,
            "n_dec_layers": 4,
            "n_enc_layers": 4,
            "n_heads": 4,
            "resid_dropout_p": 0.0,
            "s1_bits": 10,
            "s2_bits": 10,
            "zeta": 0.05,
        },
    },
}

FALLBACK_TICKERS = ["SBER", "GAZP", "LKOH", "ROSN", "NVTK", "GMKN", "TATN", "PLZL", "YDEX", "MOEX"]
KLINE_COLS = ["open", "high", "low", "close", "volume", "amount"]


def default_weights_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def iss_get_json(url: str, params: dict | None = None, timeout: int = 30) -> dict:
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()


def iss_block_to_df(js: dict, block_name: str) -> pd.DataFrame:
    block = js.get(block_name)
    if not block or "columns" not in block or "data" not in block:
        return pd.DataFrame()
    return pd.DataFrame(block["data"], columns=block["columns"])


def get_top_by_issue_capitalization(limit: int = 10) -> tuple[list[str], pd.DataFrame]:
    """
    Use the shares market board snapshot because the statistics capitalization endpoint
    returns aggregate market capitalization, not per-security SECID rows.
    """
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

    required_sec_cols = {"SECID", "BOARDID"}
    required_md_cols = {"SECID", "BOARDID", "ISSUECAPITALIZATION"}
    if not required_sec_cols.issubset(securities.columns) or not required_md_cols.issubset(marketdata.columns):
        print("WARN: Не получилось получить ISSUECAPITALIZATION из MOEX ISS, использую fallback-тикеры.")
        fallback_df = pd.DataFrame({"SECID": FALLBACK_TICKERS[:limit], "source": "fallback"})
        return FALLBACK_TICKERS[:limit], fallback_df

    df = securities.merge(marketdata, on=["SECID", "BOARDID"], how="inner")
    df["ISSUECAPITALIZATION"] = pd.to_numeric(df["ISSUECAPITALIZATION"], errors="coerce")
    df = df.dropna(subset=["SECID", "ISSUECAPITALIZATION"])
    df = df[df["ISSUECAPITALIZATION"] > 0]

    if "SECTYPE" in df.columns:
        df = df[df["SECTYPE"].astype(str) == "1"]

    if df.empty:
        print("WARN: MOEX ISS вернул пустой список капитализации, использую fallback-тикеры.")
        fallback_df = pd.DataFrame({"SECID": FALLBACK_TICKERS[:limit], "source": "fallback"})
        return FALLBACK_TICKERS[:limit], fallback_df

    top = (
        df.sort_values("ISSUECAPITALIZATION", ascending=False)
        .drop_duplicates(subset=["SECID"], keep="first")
        .head(limit)
        .reset_index(drop=True)
    )
    return top["SECID"].astype(str).tolist(), top


def is_incomplete_daily_candle(row: pd.Series) -> bool:
    if "timestamps" not in row or "end" not in row:
        return False

    begin_ts = pd.Timestamp(row["timestamps"])
    end_ts = pd.Timestamp(row["end"])
    today = pd.Timestamp(date.today())

    if begin_ts.normalize() < today:
        return False

    return end_ts.time() < pd.Timestamp("23:59:00").time()


def fetch_moex_daily_candles(
    secid: str,
    from_date: str,
    till_date: str,
    drop_incomplete_last_candle: bool = True,
) -> pd.DataFrame:
    url = f"{MOEX_BASE}/engines/stock/markets/shares/securities/{secid}/candles.json"

    chunks = []
    start = 0

    while True:
        params = {
            "interval": 24,
            "from": from_date,
            "till": till_date,
            "start": start,
            "iss.meta": "off",
        }
        js = iss_get_json(url, params=params)
        part = iss_block_to_df(js, "candles")

        if part.empty:
            break

        chunks.append(part)
        start += len(part)

    if not chunks:
        return pd.DataFrame()

    df = pd.concat(chunks, ignore_index=True)
    df = df.rename(
        columns={
            "begin": "timestamps",
            "value": "amount",
        }
    )

    need = ["timestamps", "end", "open", "high", "low", "close", "volume", "amount"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"{secid}: в свечах MOEX нет колонок: {missing}. Есть: {list(df.columns)}")

    df = df[need].copy()
    df["timestamps"] = pd.to_datetime(df["timestamps"])
    df["end"] = pd.to_datetime(df["end"])

    for c in KLINE_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["timestamps", "end", "open", "high", "low", "close"])
    df["volume"] = df["volume"].fillna(0.0)
    df["amount"] = df["amount"].fillna(0.0)
    df = df.sort_values("timestamps").drop_duplicates(subset=["timestamps"]).reset_index(drop=True)

    if drop_incomplete_last_candle and not df.empty and is_incomplete_daily_candle(df.iloc[-1]):
        last_begin = df["timestamps"].iloc[-1]
        last_end = df["end"].iloc[-1]
        print(f"INFO: {secid}: drop incomplete last candle begin={last_begin} end={last_end}")
        df = df.iloc[:-1].copy()

    return df


def resolve_device(requested_device: str) -> str:
    if torch is None:
        raise RuntimeError("PyTorch не установлен.")

    if requested_device != "auto":
        return requested_device

    if torch.cuda.is_available():
        return "cuda:0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_local_module(module_class, config: dict, weights_path: Path):
    if not weights_path.exists():
        raise FileNotFoundError(f"Не найден файл весов: {weights_path}")

    module = module_class(**config)
    state_dict = load_file(str(weights_path), device="cpu")
    missing, unexpected = module.load_state_dict(state_dict, strict=False)

    if missing or unexpected:
        raise RuntimeError(
            f"Не удалось строго загрузить {weights_path.name}: "
            f"missing={missing[:10]}, unexpected={unexpected[:10]}"
        )

    return module


def make_predictor(model_key: str, lookback: int, weights_dir: Path, device: str):
    cfg = MODEL_CONFIGS[model_key]
    tok_cfg = TOKENIZER_CONFIGS[cfg["tokenizer_key"]]

    tokenizer = load_local_module(
        tok_cfg["class"],
        tok_cfg["config"],
        weights_dir / tok_cfg["weights_file"],
    )
    model = load_local_module(
        cfg["class"],
        cfg["config"],
        weights_dir / cfg["weights_file"],
    )

    if hasattr(model, "eval"):
        model.eval()
    if hasattr(tokenizer, "eval"):
        tokenizer.eval()

    max_context = min(cfg["native_context"], lookback)
    predictor = KronosPredictor(model, tokenizer, device=device, max_context=max_context)
    return predictor


def predict_one_window(
    predictor,
    df: pd.DataFrame,
    end_idx: int,
    lookback: int,
    pred_len: int,
    sample_count: int,
):
    x = df.iloc[end_idx - lookback:end_idx].copy()
    y = df.iloc[end_idx:end_idx + pred_len].copy()

    pred_df = predictor.predict(
        df=x[KLINE_COLS],
        x_timestamp=x["timestamps"],
        y_timestamp=y["timestamps"],
        pred_len=len(y),
        T=1.0,
        top_p=0.9,
        sample_count=sample_count,
        verbose=False,
    )

    return x, y, pred_df


def calc_metrics(x: pd.DataFrame, y: pd.DataFrame, pred_df: pd.DataFrame) -> dict:
    actual = y["close"].to_numpy(dtype=float)
    pred = pred_df["close"].to_numpy(dtype=float)

    n = min(len(actual), len(pred))
    actual = actual[:n]
    pred = pred[:n]

    last_close = float(x["close"].iloc[-1])

    err = pred - actual
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    nonzero = actual != 0
    mape = float(np.mean(np.abs(err[nonzero] / actual[nonzero])) * 100.0) if np.any(nonzero) else np.nan

    actual_dir = np.sign(actual - last_close)
    pred_dir = np.sign(pred - last_close)
    direction_acc = float(np.mean(actual_dir == pred_dir))

    return {
        "mae_close": mae,
        "rmse_close": rmse,
        "mape_close_pct": mape,
        "direction_acc": direction_acc,
    }


def evaluate_model_on_ticker(
    model_name: str,
    predictor,
    ticker: str,
    df: pd.DataFrame,
    lookback: int,
    pred_len: int,
    eval_windows: int,
    sample_count: int,
) -> list[dict]:
    rows = []
    possible_ends = list(range(lookback, len(df) - pred_len + 1, pred_len))
    possible_ends = possible_ends[-eval_windows:]

    for end_idx in possible_ends:
        start_time = time.time()
        x, y, pred_df = predict_one_window(
            predictor=predictor,
            df=df,
            end_idx=end_idx,
            lookback=lookback,
            pred_len=pred_len,
            sample_count=sample_count,
        )
        elapsed = time.time() - start_time

        m = calc_metrics(x, y, pred_df)
        rows.append(
            {
                "model": model_name,
                "ticker": ticker,
                "train_until": x["timestamps"].iloc[-1],
                "test_from": y["timestamps"].iloc[0],
                "test_till": y["timestamps"].iloc[-1],
                "pred_len": len(y),
                "seconds": elapsed,
                **m,
            }
        )

    return rows


def forecast_latest(
    model_name: str,
    predictor,
    ticker: str,
    df: pd.DataFrame,
    lookback: int,
    pred_len: int,
    sample_count: int,
) -> pd.DataFrame:
    x = df.iloc[-lookback:].copy()
    last_ts = pd.Timestamp(x["timestamps"].iloc[-1]).normalize()
    future_ts = pd.bdate_range(last_ts + BDay(1), periods=pred_len)

    pred_df = predictor.predict(
        df=x[KLINE_COLS],
        x_timestamp=x["timestamps"],
        y_timestamp=pd.Series(future_ts),
        pred_len=pred_len,
        T=1.0,
        top_p=0.9,
        sample_count=sample_count,
        verbose=False,
    ).copy()

    pred_df.insert(0, "date", future_ts)
    pred_df.insert(0, "ticker", ticker)
    pred_df.insert(0, "model", model_name)
    pred_df.insert(3, "last_known_close", float(x["close"].iloc[-1]))

    return pred_df


def parse_models(models_arg: str) -> list[str]:
    requested = [m.strip() for m in models_arg.split(",") if m.strip()]
    unknown = [m for m in requested if m not in MODEL_CONFIGS]
    if unknown:
        raise ValueError(f"Неизвестные модели: {unknown}. Доступны: {list(MODEL_CONFIGS)}")
    if not requested:
        raise ValueError("Список моделей пуст.")
    return requested


def main():
    parser = argparse.ArgumentParser(description="Zero-shot Kronos baseline on MOEX daily candles.")
    parser.add_argument("--from-date", default="2020-01-01")
    parser.add_argument("--till-date", default=str(date.today()))
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--lookback", type=int, default=512)
    parser.add_argument("--pred-len", type=int, default=10)
    parser.add_argument("--eval-windows", type=int, default=3)
    parser.add_argument("--sample-count", type=int, default=1)
    parser.add_argument("--out-dir", default="outputs_moex_kronos")
    parser.add_argument("--weights-dir", type=Path, default=default_weights_dir())
    parser.add_argument("--models", default="mini,small,base", help="Comma-separated subset: mini,small,base")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda:0, mps, ...")
    parser.add_argument(
        "--drop-incomplete-last-candle",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop today's incomplete MOEX daily candle before evaluation and latest forecast.",
    )
    args = parser.parse_args()

    if args.lookback <= 0 or args.pred_len <= 0 or args.eval_windows <= 0:
        raise ValueError("--lookback, --pred-len and --eval-windows must be positive.")

    weights_dir = args.weights_dir.resolve()
    models = parse_models(args.models)
    device = resolve_device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Weights dir: {weights_dir}")
    print(f"Models: {models}")
    print(f"Device: {device}")
    print(f"Drop incomplete last candle: {args.drop_incomplete_last_candle}")

    print("Getting top tickers by MOEX issue capitalization...")
    tickers, universe_df = get_top_by_issue_capitalization(limit=args.top_n)
    universe_path = os.path.join(args.out_dir, "selected_universe.csv")
    universe_df.to_csv(universe_path, index=False)
    print("Tickers:", tickers)

    print("Downloading MOEX candles...")
    data = {}
    for ticker in tqdm(tickers):
        try:
            df = fetch_moex_daily_candles(
                ticker,
                args.from_date,
                args.till_date,
                drop_incomplete_last_candle=args.drop_incomplete_last_candle,
            )
            if len(df) < args.lookback + args.pred_len + 5:
                print(f"WARN: {ticker}: мало данных: {len(df)} rows, skip.")
                continue
            data[ticker] = df
            df.to_csv(os.path.join(args.out_dir, f"candles_{ticker}.csv"), index=False)
        except Exception as e:
            print(f"WARN: {ticker}: failed to fetch candles: {e}")

    if not data:
        raise RuntimeError("Нет данных ни по одному тикеру.")

    all_metric_rows = []
    all_latest_preds = []

    for model_name in models:
        print(f"\nLoading model: {model_name}")
        predictor = make_predictor(model_name, lookback=args.lookback, weights_dir=weights_dir, device=device)

        for ticker, df in tqdm(data.items(), desc=f"Evaluating {model_name}"):
            try:
                rows = evaluate_model_on_ticker(
                    model_name=model_name,
                    predictor=predictor,
                    ticker=ticker,
                    df=df,
                    lookback=args.lookback,
                    pred_len=args.pred_len,
                    eval_windows=args.eval_windows,
                    sample_count=args.sample_count,
                )
                all_metric_rows.extend(rows)

                latest = forecast_latest(
                    model_name=model_name,
                    predictor=predictor,
                    ticker=ticker,
                    df=df,
                    lookback=args.lookback,
                    pred_len=args.pred_len,
                    sample_count=args.sample_count,
                )
                all_latest_preds.append(latest)
            except Exception as e:
                print(f"WARN: {model_name}/{ticker}: failed: {e}")

        del predictor
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not all_metric_rows:
        raise RuntimeError("Не удалось посчитать метрики ни для одного окна.")
    if not all_latest_preds:
        raise RuntimeError("Не удалось сформировать свежие прогнозы.")

    metrics = pd.DataFrame(all_metric_rows)
    metrics_path = os.path.join(args.out_dir, "metrics_by_window.csv")
    metrics.to_csv(metrics_path, index=False)

    summary = (
        metrics.groupby("model", as_index=False)
        .agg(
            mae_close=("mae_close", "mean"),
            rmse_close=("rmse_close", "mean"),
            mape_close_pct=("mape_close_pct", "mean"),
            direction_acc=("direction_acc", "mean"),
            avg_seconds=("seconds", "mean"),
        )
        .sort_values("mae_close")
    )
    summary_path = os.path.join(args.out_dir, "metrics_summary.csv")
    summary.to_csv(summary_path, index=False)

    latest_preds = pd.concat(all_latest_preds, ignore_index=True)
    preds_path = os.path.join(args.out_dir, "latest_predictions.csv")
    latest_preds.to_csv(preds_path, index=False)

    print("\n=== SUMMARY ===")
    print(summary)

    print("\nSaved:")
    print(universe_path)
    print(metrics_path)
    print(summary_path)
    print(preds_path)


if __name__ == "__main__":
    main()
