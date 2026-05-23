from __future__ import annotations

from datetime import datetime, time, timedelta, timezone

MSK = timezone(timedelta(hours=3))


def now_msk() -> datetime:
    return datetime.now(MSK).replace(tzinfo=None)


def is_market_open(value: datetime | None = None) -> bool:
    dt = value or now_msk()
    weekday = dt.weekday()
    current = dt.time()
    if weekday < 5:
        return time(7, 0) <= current <= time(23, 50)
    return time(10, 0) <= current <= time(19, 0)


def next_decision_time(
    value: datetime | None = None,
    *,
    interval_minutes: int = 30,
    decision_delay_seconds: int = 90,
) -> datetime:
    dt = value or now_msk()
    base = dt.replace(second=0, microsecond=0)
    minute_bucket = (base.minute // interval_minutes) * interval_minutes
    bucket_start = base.replace(minute=minute_bucket)
    candidate = bucket_start + timedelta(seconds=decision_delay_seconds)
    if candidate <= dt:
        candidate = bucket_start + timedelta(minutes=interval_minutes, seconds=decision_delay_seconds)
    while not is_market_open(candidate):
        candidate += timedelta(minutes=interval_minutes)
    return candidate


def sleep_seconds_until(target: datetime, now: datetime | None = None) -> float:
    current = now or now_msk()
    return max(0.0, (target - current).total_seconds())
