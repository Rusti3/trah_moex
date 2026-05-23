# Arena Production Package

`arena/` is the portable production package for the current preferred top-20
MOEX strategy: **`rolling_rank_weighted_w24_p2`**.

The package is split into two layers:

- `runtime/` - clean production API for live decisions.
- `research_snapshot/` - copied research runners used to reproduce the current
  results.

Heavy outputs, minute candles, ALGOPACK cache, LLM cache, `.env`, API keys, and
token files are intentionally not copied.

## Current Reference

Preferred live-safe mode:

```text
rolling_rank_weighted_w24_p2
lookback = 24 intervals
rank_power = 2.0
base selectors = family_first, news_aware, marketwide_news
```

Reference performance:

| Window | Return |
|---|---:|
| Apr 1-14 | +13.36% |
| Apr 14-30 | +10.46% |
| May 1-21 | +20.42% |
| Compounded | +50.79% |

Latency stress on Apr 1-7 + May 1-7:

| Delay | Compounded return | Delta |
|---:|---:|---:|
| 0m | +10.27% | 0.00 pp |
| 1m | +10.04% | -0.23 pp |
| 2m | +9.31% | -0.96 pp |

## Live Flow

```text
candle closed
  -> async Kronos top20 forecast
  -> async LLM daily/news-background scoring
  -> async MOEX cost/depth snapshot
  -> base selector decisions
  -> RollingRankWeightedSelector(lookback=24, rank_power=2)
  -> target portfolio weights / orders
```

The runtime does not call external services directly.  Production should inject
providers implementing:

- `KronosForecastProvider`
- `LLMNewsScorer`
- `MoexCostDepthProvider`

## Minimal Runtime Use

```python
from arena.runtime import RollingRankWeightedSelector, make_decision

selector = RollingRankWeightedSelector(lookback=24, rank_power=2.0)

result = make_decision(
    as_of="2026-05-21 12:00:00",
    kronos_scores={"SBER": 0.82, "LKOH": 0.31},
    llm_scores={"SBER": 0.70, "LKOH": 0.45},
    cost_depth={"SBER": {"tradable": True}, "LKOH": {"tradable": True}},
    history={
        "selector_returns": [
            {
                "timestamp": "2026-05-21 11:30:00",
                "selector_family_first": 0.001,
                "selector_news_aware": 0.0004,
                "selector_marketwide_news": -0.0001,
            }
        ],
        "base_selector_decisions": {
            "selector_family_first": {"kronos_weight": 1, "llm_weight": 1, "threshold": 0.65, "rank_power": 2},
            "selector_news_aware": {"kronos_weight": 1, "llm_weight": 2, "threshold": 0.65, "rank_power": 2},
            "selector_marketwide_news": {"kronos_weight": 0.5, "llm_weight": 1, "threshold": 0.65, "rank_power": 2},
        },
    },
)

orders = result.to_order_targets()
```

## Reproduction

Use `offline/` and `research_snapshot/` to reproduce research tables:

```powershell
python arena/offline/reproduce_combiner.py `
  --out-dir outputs_moex_kronos_top20_selector_v2_combined_family_news
```

For the exact historical research commands, use the copied scripts in
`research_snapshot/` and the external output/cache folders listed in
`config/production.yaml`.

## Required Environment

No secrets are stored in this folder.

- `POLZA_AI_API_KEY` for Polza LLM calls.
- `MOEX_ALGO_TOKEN` for ALGOPACK downloads/repricing.

If a provider cannot score news, pass neutral LLM scores (`0.5`) and keep a
separate `news_data_available` flag in upstream features.

## Live Container

Root `Dockerfile` starts the live bot:

```powershell
docker build -t arena-kronos .
docker run --rm -v arena_data:/data arena-kronos
```

Entrypoint:

```text
python -m arena.runtime.live_bot
```

Live defaults:

- ArenaGo orders are enabled by default: `ARENA_LIVE_ORDERS=true`.
- State is persisted in `/data/arena_state.sqlite3`.
- News are persisted in `/data/news.sqlite3`.
- LLM cache is persisted in `/data/llm_cache.jsonl`.
- Logs are appended to `/data/logs/*.jsonl`.

The live loop waits for the decision timestamp, runs Kronos, LLM news scoring
and MOEX cost/depth concurrently, waits up to 5 minutes for Kronos, and then
falls back to the last good Kronos scores if needed.

Lot handling:

```text
GAZP, GMKN, ALRS, AFLT, NLMK, MOEX, SNGSP, MTSS = multiples of 10
all other top-20 tickers = multiples of 1
```

News ingestion is vendored from `D:\news_fuck` into `arena/news_ingestion`.
`NewsBuffer` uses `received_at_msk <= as_of`, not `published_at_msk`, so the
strategy only sees news that the system could actually know at decision time.
