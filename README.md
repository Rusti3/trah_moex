## Arena Kronos Live Trading Runtime

Production container for autonomous ArenaGo trading on the fixed MOEX top-20
universe.

The live strategy:

```text
candle closed
  -> async Kronos top20 forecast
  -> async news ingestion + Polza LLM news scoring
  -> async MOEX cost/depth snapshot
  -> family_first / news_aware / marketwide base selector portfolios
  -> live LightGBM scores the three selector families when enough history exists
  -> fallback rolling_rank_weighted_w24_p2 while history is short
  -> lot-rounded ArenaGo market orders
```

## Runtime Scope

This repository intentionally contains production logic only:

- `arena/runtime` - live bot, selector inference, order manager, providers.
- `arena/news_ingestion` - append-only news ingestion, ticker tagging, dedupe.
- `arena/config` - production and news-source configuration.
- `model`, `run_moex_baseline.py`, `run_moex_hourly_rolling_backtest.py` -
  minimal Kronos dependencies used by the live provider.
- `Dockerfile` - single-container deployment entrypoint.

Research runners, notebooks, historical outputs, minute candles, LLM caches,
and ALGOPACK caches are not included.

## Persistent Data

All mutable runtime data is stored under `/data`:

- `/data/news.sqlite3`
- `/data/arena_state.sqlite3`
- `/data/llm_cache.jsonl`
- `/data/logs/*.jsonl`
- `/data/model_cache`
- `/data/cache`

The bot is designed to reconcile positions and continue after container
restart.

## Environment

The deployment reads credentials from `.env` / environment variables:

- `SANDBOX_API_KEY` - ArenaGo API token.
- `POLZA_AI_API_KEY` - Polza AI token.
- `MOEX_ALGO_TOKEN` - MOEX ALGOPACK token.

Runtime defaults are in `arena/config/production.yaml`.

## Run

```powershell
docker build -t arena-kronos-live .
docker run --env-file .env -v arena-data:/data arena-kronos-live
```

The container command is:

```text
python -m arena.runtime.live_bot
```

Smoke check:

```powershell
python -m unittest discover -s arena\tests
python -m arena.runtime.live_bot --help
```
