from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .schemas import TOP20_TICKERS


def _bool_env(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class RuntimeSettings:
    data_dir: Path
    bot_name: str
    arena_base_url: str
    arena_api_key: str
    live_orders: bool
    tickers: tuple[str, ...]
    lot_sizes: dict[str, int]
    decision_interval_minutes: int
    decision_delay_seconds: int
    precompute_seconds: int
    candle_close_wait_seconds: int
    execution_deadline_seconds: int
    history_bootstrap_initial_intervals: int
    history_bootstrap_background_intervals: int
    history_bootstrap_time_budget_seconds: int
    max_kronos_wait_seconds: int
    max_llm_wait_seconds: int
    max_moex_wait_seconds: int
    max_gross_exposure: float
    max_daily_trades: int
    min_order_value_rub: float
    state_db_path: Path
    market_history_db_path: Path
    news_db_path: Path
    llm_cache_path: Path
    logs_dir: Path
    news_log_interval_seconds: int
    health_log_interval_seconds: int
    production_config_path: Path
    news_sources_config_path: Path
    news_tickers_config_path: Path
    polza_base_url: str
    polza_model: str
    moex_algo_token: str
    weights_dir: Path
    weights_mode: str
    device: str


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_settings(config_path: str | Path | None = None) -> RuntimeSettings:
    config = Path(config_path or os.environ.get("ARENA_CONFIG", "arena/config/production.yaml"))
    cfg = load_yaml(config)
    data_dir = Path(os.environ.get("DATA_DIR", "/data"))
    tickers = tuple(cfg.get("universe", {}).get("tickers") or TOP20_TICKERS)
    lot_sizes = {ticker: 1 for ticker in tickers}
    lot_sizes.update({str(k): int(v) for k, v in (cfg.get("lot_sizes") or {}).items()})
    lot_sizes.update(
        {
            "GAZP": 10,
            "GMKN": 10,
            "ALRS": 10,
            "AFLT": 10,
            "NLMK": 10,
            "MOEX": 10,
            "SNGSP": 10,
            "MTSS": 10,
        }
    )

    rebalance = cfg.get("rebalance", {})
    env = cfg.get("environment", {})
    return RuntimeSettings(
        data_dir=data_dir,
        bot_name=os.environ.get("ARENA_BOT_NAME", "ArenaKronosTop20"),
        arena_base_url=os.environ.get("ARENA_BASE_URL", "https://arenago.ru"),
        arena_api_key=os.environ.get("SANDBOX_API_KEY", ""),
        live_orders=_bool_env("ARENA_LIVE_ORDERS", True),
        tickers=tickers,
        lot_sizes=lot_sizes,
        decision_interval_minutes=int(os.environ.get("ARENA_DECISION_INTERVAL_MINUTES", rebalance.get("decision_interval_minutes", 30))),
        decision_delay_seconds=int(os.environ.get("ARENA_DECISION_DELAY_SECONDS", rebalance.get("decision_delay_seconds", 90))),
        precompute_seconds=int(os.environ.get("ARENA_PRECOMPUTE_SECONDS", rebalance.get("precompute_seconds", 120))),
        candle_close_wait_seconds=int(os.environ.get("ARENA_CANDLE_CLOSE_WAIT_SECONDS", rebalance.get("candle_close_wait_seconds", rebalance.get("decision_delay_seconds", 90)))),
        execution_deadline_seconds=int(os.environ.get("ARENA_EXECUTION_DEADLINE_SECONDS", rebalance.get("execution_deadline_seconds", 90))),
        history_bootstrap_initial_intervals=int(os.environ.get("ARENA_HISTORY_BOOTSTRAP_INITIAL_INTERVALS", cfg.get("history_bootstrap", {}).get("initial_intervals", 48))),
        history_bootstrap_background_intervals=int(os.environ.get("ARENA_HISTORY_BOOTSTRAP_BACKGROUND_INTERVALS", cfg.get("history_bootstrap", {}).get("background_intervals", 512))),
        history_bootstrap_time_budget_seconds=int(os.environ.get("ARENA_HISTORY_BOOTSTRAP_TIME_BUDGET_SECONDS", cfg.get("history_bootstrap", {}).get("time_budget_seconds", 180))),
        max_kronos_wait_seconds=int(os.environ.get("ARENA_MAX_KRONOS_WAIT_SECONDS", 300)),
        max_llm_wait_seconds=int(os.environ.get("ARENA_MAX_LLM_WAIT_SECONDS", 60)),
        max_moex_wait_seconds=int(os.environ.get("ARENA_MAX_MOEX_WAIT_SECONDS", 30)),
        max_gross_exposure=float(os.environ.get("ARENA_MAX_GROSS_EXPOSURE", cfg.get("risk", {}).get("max_gross_exposure", 1.0))),
        max_daily_trades=int(os.environ.get("ARENA_MAX_DAILY_TRADES", 950)),
        min_order_value_rub=float(os.environ.get("ARENA_MIN_ORDER_VALUE_RUB", 100.0)),
        state_db_path=Path(os.environ.get("ARENA_STATE_DB", str(data_dir / "arena_state.sqlite3"))),
        market_history_db_path=Path(os.environ.get("ARENA_MARKET_HISTORY_DB", str(data_dir / "market_history.sqlite3"))),
        news_db_path=Path(os.environ.get("DATABASE_PATH", str(data_dir / "news.sqlite3"))),
        llm_cache_path=Path(os.environ.get("ARENA_LLM_CACHE", str(data_dir / "llm_cache.jsonl"))),
        logs_dir=Path(os.environ.get("ARENA_LOGS_DIR", str(data_dir / "logs"))),
        news_log_interval_seconds=int(os.environ.get("ARENA_NEWS_LOG_INTERVAL_SECONDS", cfg.get("logging", {}).get("news_log_interval_seconds", 1800))),
        health_log_interval_seconds=int(os.environ.get("ARENA_HEALTH_LOG_INTERVAL_SECONDS", cfg.get("logging", {}).get("health_log_interval_seconds", 300))),
        production_config_path=config,
        news_sources_config_path=Path(os.environ.get("SOURCES_CONFIG_PATH", "arena/config/news/sources.yaml")),
        news_tickers_config_path=Path(os.environ.get("TICKERS_CONFIG_PATH", "arena/config/news/tickers.yaml")),
        polza_base_url=os.environ.get("POLZA_BASE_URL", "https://polza.ai/api/v1"),
        polza_model=os.environ.get("POLZA_MODEL", "deepseek/deepseek-v4-pro"),
        moex_algo_token=os.environ.get(env.get("moex_algo_token_env", "MOEX_ALGO_TOKEN"), ""),
        weights_dir=Path(os.environ.get("KRONOS_WEIGHTS_DIR", "/data/model_cache")),
        weights_mode=os.environ.get("KRONOS_WEIGHTS_MODE", "auto"),
        device=os.environ.get("KRONOS_DEVICE", "cpu"),
    )
