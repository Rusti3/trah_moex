from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None

from .schemas import TOP20_TICKERS


def _hash_payload(model: str, payload: dict[str, Any]) -> str:
    raw = json.dumps({"model": model, "payload": payload}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _extract_json(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end >= start:
        text = text[start : end + 1]
    return json.loads(text)


class LLMNewsScorer:
    def __init__(
        self,
        *,
        cache_path: str | Path,
        base_url: str = "https://polza.ai/api/v1",
        model: str = "deepseek/deepseek-v4-pro",
        api_key_env: str = "POLZA_AI_API_KEY",
        timeout: float = 60.0,
        retries: int = 2,
    ):
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.base_url = base_url
        self.model = model
        self.api_key = os.environ.get(api_key_env, "").strip()
        self.timeout = timeout
        self.retries = retries
        self.client = OpenAI(base_url=base_url, api_key=self.api_key) if self.api_key and OpenAI is not None else None
        self.cache = self._load_cache()

    def _load_cache(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        if not self.cache_path.exists():
            return out
        with self.cache_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    item = json.loads(line)
                except Exception:
                    continue
                key = item.get("cache_key")
                if key:
                    out[str(key)] = item
        return out

    def _append_cache(self, item: dict[str, Any]) -> None:
        with self.cache_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")
        self.cache[str(item["cache_key"])] = item

    async def score_context(self, context: dict[str, Any], tickers: tuple[str, ...] = TOP20_TICKERS) -> dict[str, dict[str, Any]]:
        total_news = sum(len(v) for v in context.get("per_ticker_news", {}).values()) + len(context.get("marketwide_news", []))
        if total_news <= 0:
            return {
                ticker: {"bullish_score": 0.5, "confidence": 0.0, "relation_strength": 0.0, "reason": "no known same-day news"}
                for ticker in tickers
            }
        payload = self._prompt_payload(context, tickers)
        cache_key = _hash_payload(self.model, payload)
        if cache_key in self.cache:
            parsed = self.cache[cache_key].get("parsed", {})
            return self._normalize(parsed, tickers)
        if self.client is None:
            return {
                ticker: {"bullish_score": 0.5, "confidence": 0.0, "relation_strength": 0.0, "reason": "POLZA_AI_API_KEY missing"}
                for ticker in tickers
            }
        last_error = ""
        for attempt in range(self.retries + 1):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    temperature=0,
                    timeout=self.timeout,
                    messages=[
                        {"role": "system", "content": "You score Russian stock news. Return strict JSON only."},
                        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                    ],
                )
                raw = completion.choices[0].message.content or ""
                parsed = _extract_json(raw)
                normalized = self._normalize(parsed, tickers)
                self._append_cache({"cache_key": cache_key, "payload": payload, "raw": raw, "parsed": normalized})
                return normalized
            except Exception as exc:
                last_error = str(exc)
                time.sleep(0.5 * (attempt + 1))
        return {
            ticker: {"bullish_score": 0.5, "confidence": 0.0, "relation_strength": 0.0, "reason": f"llm fallback: {last_error}"}
            for ticker in tickers
        }

    def _prompt_payload(self, context: dict[str, Any], tickers: tuple[str, ...]) -> dict[str, Any]:
        return {
            "task": "For each ticker, score expected intraday bullish impact from known news only. 0=strong bearish, 0.5=neutral, 1=strong bullish.",
            "schema": {
                ticker: {"bullish_score": "0..1", "confidence": "0..1", "relation_strength": "0..1", "reason": "short"}
                for ticker in tickers
            },
            "tickers": list(tickers),
            "rebalance_timestamp": context.get("rebalance_timestamp"),
            "date": context.get("date"),
            "per_ticker_news": _trim_news(context.get("per_ticker_news", {})),
            "marketwide_news": _trim_news_list(context.get("marketwide_news", [])),
        }

    def _normalize(self, parsed: dict[str, Any], tickers: tuple[str, ...]) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for ticker in tickers:
            item = parsed.get(ticker, {}) if isinstance(parsed, dict) else {}
            if not isinstance(item, dict):
                item = {}
            out[ticker] = {
                "bullish_score": _clip(item.get("bullish_score", 0.5), 0.5),
                "confidence": _clip(item.get("confidence", 0.0), 0.0),
                "relation_strength": _clip(item.get("relation_strength", 0.0), 0.0),
                "reason": str(item.get("reason", ""))[:500],
            }
        return out


def _clip(value: Any, default: float) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    if out != out:
        return default
    return max(0.0, min(1.0, out))


def _trim_news(per_ticker: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    return {ticker: _trim_news_list(items[-8:]) for ticker, items in per_ticker.items()}


def _trim_news_list(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for item in items[-20:]:
        out.append(
            {
                "published_at": item.get("published_at"),
                "received_at": item.get("received_at"),
                "relation_type": item.get("relation_type"),
                "source": item.get("source"),
                "source_count": item.get("source_count", 1),
                "title": str(item.get("title", ""))[:260],
                "reason": str(item.get("reason", ""))[:260],
                "text": str(item.get("text", ""))[:500],
                "tags": item.get("tags", []),
            }
        )
    return out
