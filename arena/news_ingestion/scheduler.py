import logging
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from apscheduler.events import EVENT_JOB_ERROR
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from news_ingestion.pipeline import IngestionPipeline, SourceRunStats


async def _run_source_logged(
    pipeline: IngestionPipeline,
    source_id: str,
    result_callback: Callable[[SourceRunStats], None] | None = None,
) -> SourceRunStats:
    stats = await pipeline.run_source(source_id)
    if result_callback is not None:
        result_callback(stats)
    return stats


def create_scheduler(
    pipeline: IngestionPipeline,
    *,
    result_callback: Callable[[SourceRunStats], None] | None = None,
    error_callback: Callable[[dict[str, Any]], None] | None = None,
) -> AsyncIOScheduler:
    logging.getLogger("apscheduler.executors.default").setLevel(logging.ERROR)
    logging.getLogger("apscheduler.scheduler").setLevel(logging.ERROR)
    scheduler = AsyncIOScheduler(timezone=UTC)

    if error_callback is not None:
        def _on_error(event: Any) -> None:
            error_callback(
                {
                    "job_id": getattr(event, "job_id", ""),
                    "exception": repr(getattr(event, "exception", "")),
                    "traceback": str(getattr(event, "traceback", ""))[:2000],
                }
            )

        scheduler.add_listener(_on_error, EVENT_JOB_ERROR)

    for source in pipeline.enabled_sources():
        scheduler.add_job(
            _run_source_logged,
            "interval",
            seconds=source.interval_seconds,
            args=[pipeline, source.id, result_callback],
            id=f"ingest:{source.id}",
            replace_existing=True,
            max_instances=1,
            coalesce=True,
            misfire_grace_time=max(5, int(source.interval_seconds)),
            next_run_time=datetime.now(UTC),
        )
    return scheduler
