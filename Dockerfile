FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app:/app/arena \
    DATA_DIR=/data \
    ARENA_CONFIG=arena/config/production.yaml \
    DATABASE_PATH=/data/news.sqlite3 \
    ARENA_STATE_DB=/data/arena_state.sqlite3 \
    ARENA_MARKET_HISTORY_DB=/data/market_history.sqlite3 \
    ARENA_LLM_CACHE=/data/llm_cache.jsonl \
    ARENA_LOGS_DIR=/data/logs \
    XDG_CACHE_HOME=/data/cache \
    HF_HOME=/data/model_cache/huggingface \
    HUGGINGFACE_HUB_CACHE=/data/model_cache/huggingface/hub \
    TRANSFORMERS_CACHE=/data/model_cache/huggingface/transformers \
    TORCH_HOME=/data/model_cache/torch \
    ARENA_RUNTIME_PYTHON_PACKAGES=/data/python_packages \
    ARENA_TORCH_SPEC=torch==2.5.1+cpu \
    ARENA_TORCH_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu \
    KRONOS_WEIGHTS_DIR=/data/model_cache \
    SOURCES_CONFIG_PATH=arena/config/news/sources.yaml \
    TICKERS_CONFIG_PATH=arena/config/news/tickers.yaml \
    ARENA_LIVE_ORDERS=true

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY arena/requirements-runtime.txt /app/arena/requirements-runtime.txt
COPY arena/requirements-live.txt /app/arena/requirements-live.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --prefer-binary -r /app/arena/requirements-runtime.txt \
    && pip install --no-cache-dir --prefer-binary -r /app/arena/requirements-live.txt

COPY . /app

RUN mkdir -p /data/logs /data/cache /data/model_cache/huggingface /data/model_cache/torch /data/python_packages

CMD ["python", "-m", "arena.runtime.bootstrap_live_bot"]
