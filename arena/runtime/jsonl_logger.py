from __future__ import annotations

import json
import os
import re
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

MSK = timezone(timedelta(hours=3))

SECRET_KEY_RE = re.compile(r"(api[_-]?key|token|authorization|secret|password|bearer)", re.IGNORECASE)
SECRET_VALUE_RE = re.compile(
    r"(pza_[A-Za-z0-9_\-]{8,}|sk-or-[A-Za-z0-9_\-]{8,}|Bearer\s+[A-Za-z0-9._\-]+|"
    r"eyJ[A-Za-z0-9_\-]{20,}\.[A-Za-z0-9_\-]{20,}\.[A-Za-z0-9_\-]{20,})"
)
LONG_SECRET_RE = re.compile(r"^[A-Za-z0-9_\-\.]{96,}$")


def redact(value: Any, *, key: str = "") -> Any:
    if SECRET_KEY_RE.search(key):
        return "[REDACTED]"
    if isinstance(value, dict):
        return {str(k): redact(v, key=str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact(item) for item in value]
    if isinstance(value, str):
        if SECRET_VALUE_RE.search(value) or LONG_SECRET_RE.match(value):
            return "[REDACTED]"
        return SECRET_VALUE_RE.sub("[REDACTED]", value)
    return value


class JsonlLogger:
    def __init__(self, logs_dir: str | Path):
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.stdout = os.environ.get("ARENA_LOG_STDOUT", "true").strip().lower() in {"1", "true", "yes", "y", "on"}
        self.log_level = os.environ.get("ARENA_LOG_LEVEL", "info").strip().lower()
        self._lock = threading.Lock()

    def write(self, event: str, payload: dict[str, Any] | None = None, *, stream: str = "arena_live") -> None:
        if self.log_level in {"error", "errors"} and stream not in {"errors"}:
            return
        now = datetime.now(MSK).replace(tzinfo=None)
        row = {
            "ts_msk": now.strftime("%Y-%m-%d %H:%M:%S"),
            "event": event,
            **redact(payload or {}),
        }
        safe_stream = re.sub(r"[^A-Za-z0-9_\-]+", "_", stream or "arena_live").strip("_") or "arena_live"
        path = self.logs_dir / f"{safe_stream}_{now:%Y%m%d}.jsonl"
        line = json.dumps(row, ensure_ascii=False, default=str)
        with self._lock:
            with path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
            if self.stdout:
                print(line, flush=True)

    def error(self, event: str, payload: dict[str, Any] | None = None) -> None:
        self.write(event, payload, stream="errors")

    def health(self, event: str, payload: dict[str, Any] | None = None) -> None:
        self.write(event, payload, stream="arena_health")
