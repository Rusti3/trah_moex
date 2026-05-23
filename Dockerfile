FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app:/app/arena \
    DATA_DIR=/data \
    ARENA_CONFIG=arena/config/production.yaml \
    DATABASE_PATH=/data/news.sqlite3 \
    ARENA_STATE_DB=/data/arena_state.sqlite3 \
    ARENA_LLM_CACHE=/data/llm_cache.jsonl \
    ARENA_LOGS_DIR=/data/logs \
    XDG_CACHE_HOME=/data/cache \
    HF_HOME=/data/model_cache/huggingface \
    HUGGINGFACE_HUB_CACHE=/data/model_cache/huggingface/hub \
    TRANSFORMERS_CACHE=/data/model_cache/huggingface/transformers \
    TORCH_HOME=/data/model_cache/torch \
    KRONOS_WEIGHTS_DIR=/data/model_cache \
    SOURCES_CONFIG_PATH=arena/config/news/sources.yaml \
    TICKERS_CONFIG_PATH=arena/config/news/tickers.yaml \
    ARENA_LIVE_ORDERS=true

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        libxml2 \
        libxslt1.1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
COPY arena/requirements-live.txt /app/arena/requirements-live.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt \
    && pip install --no-cache-dir -r /app/arena/requirements-live.txt

COPY . /app

RUN mkdir -p /data/logs /data/cache /data/model_cache/huggingface /data/model_cache/torch

CMD ["python", "-m", "arena.runtime.live_bot"]
