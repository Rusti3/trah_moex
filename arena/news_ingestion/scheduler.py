from datetime import UTC, datetime

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from news_ingestion.pipeline import IngestionPipeline


def create_scheduler(pipeline: IngestionPipeline) -> AsyncIOScheduler:
    scheduler = AsyncIOScheduler(timezone=UTC)
    for source in pipeline.enabled_sources():
        scheduler.add_job(
            pipeline.run_source,
            "interval",
            seconds=source.interval_seconds,
            args=[source.id],
            id=f"ingest:{source.id}",
            replace_existing=True,
            max_instances=1,
            coalesce=True,
            next_run_time=datetime.now(UTC),
        )
    return scheduler
