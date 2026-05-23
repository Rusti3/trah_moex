from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

MSK = timezone(timedelta(hours=3))


class JsonlLogger:
    def __init__(self, logs_dir: str | Path):
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, event: str, payload: dict[str, Any] | None = None) -> None:
        now = datetime.now(MSK).replace(tzinfo=None)
        row = {
            "ts_msk": now.strftime("%Y-%m-%d %H:%M:%S"),
            "event": event,
            **(payload or {}),
        }
        path = self.logs_dir / f"arena_live_{now:%Y%m%d}.jsonl"
        with self._lock:
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
