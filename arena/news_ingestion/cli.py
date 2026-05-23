from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Iterable

from news_ingestion.config import SourceRegistry
from news_ingestion.dedup import (
    StoryDedupStats,
    rebuild_story_clusters,
    story_cluster_report,
    story_dedup_stats,
)
from news_ingestion.pipeline import IngestionPipeline, SourceRunStats
from news_ingestion.scheduler import create_scheduler
from news_ingestion.settings import Settings, get_settings
from news_ingestion.storage import count_news, initialize_database, sync_sources
from news_ingestion.tickers import (
    TickerRegistry,
    TickerStats,
    tag_existing_news,
    ticker_stats,
)


def main() -> None:
    _configure_output_encoding()

    parser = argparse.ArgumentParser(prog="news-pars")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("init-db")
    subparsers.add_parser("list-sources")
    subparsers.add_parser("bootstrap")
    subparsers.add_parser("dedup")
    subparsers.add_parser("tag-tickers")
    subparsers.add_parser("ticker-stats")

    dedup_stats_parser = subparsers.add_parser("dedup-stats")
    dedup_stats_parser.add_argument("--limit", type=int, default=20)

    ingest_parser = subparsers.add_parser("ingest")
    ingest_parser.add_argument("source_id", nargs="?")

    subparsers.add_parser("run")

    args = parser.parse_args()
    settings = get_settings()

    if args.command == "dedup":
        stats = rebuild_story_clusters(settings.database_path)
        _print_dedup_stats(stats)
        return

    if args.command == "dedup-stats":
        stats = story_dedup_stats(settings.database_path)
        _print_dedup_stats(stats)
        _print_story_clusters(settings.database_path, args.limit)
        return

    if args.command == "ticker-stats":
        _print_ticker_stats(ticker_stats(settings.database_path))
        return

    registry = SourceRegistry.load(settings.sources_config_path)
    ticker_registry = TickerRegistry.load(settings.tickers_config_path)

    if args.command == "tag-tickers":
        _print_ticker_stats(
            tag_existing_news(settings.database_path, ticker_registry, registry.sources)
        )
        return

    if args.command == "init-db":
        initialize_database(settings.database_path)
        synced = sync_sources(settings.database_path, registry.sources)
        print(f"initialized {settings.database_path} ({synced} sources synced)")
        return

    if args.command == "list-sources":
        _print_sources(registry)
        return

    pipeline = IngestionPipeline(registry, settings)

    if args.command == "ingest":
        results = asyncio.run(_run_ingest(pipeline, args.source_id))
        _print_stats(results)
        print(f"total rows: {count_news(settings.database_path)}")
        return

    if args.command == "bootstrap":
        results = asyncio.run(pipeline.bootstrap())
        _print_stats(results)
        print(f"total rows: {count_news(settings.database_path)}")
        return

    if args.command == "run":
        asyncio.run(_run_scheduler(pipeline, settings))
        return


async def _run_ingest(
    pipeline: IngestionPipeline,
    source_id: str | None,
) -> list[SourceRunStats]:
    if source_id:
        return [await pipeline.run_source(source_id)]
    return await pipeline.run()


async def _run_scheduler(pipeline: IngestionPipeline, settings: Settings) -> None:
    pipeline.initialize()
    if settings.bootstrap_enabled:
        print(
            f"bootstrap started: last {settings.bootstrap_lookback_days} day(s), "
            f"fallback={settings.bootstrap_fallback_items}"
        )
        _print_stats(await pipeline.bootstrap())
        print(f"bootstrap finished: total rows={count_news(settings.database_path)}")
    scheduler = create_scheduler(pipeline)
    scheduler.start()
    print(
        f"scheduler started: {len(pipeline.enabled_sources())} sources, "
        f"database={settings.database_path}"
    )
    try:
        while True:
            await asyncio.sleep(3600)
    except (KeyboardInterrupt, asyncio.CancelledError):
        scheduler.shutdown(wait=False)


def _print_sources(registry: SourceRegistry) -> None:
    for source in registry.sources:
        marker = "enabled" if source.enabled else "disabled"
        print(f"{source.id}\t{source.method}\t{marker}\t{source.interval_seconds}s\t{source.name}")


def _print_stats(results: Iterable[SourceRunStats]) -> None:
    for result in results:
        print(
            f"{result.source_id} [{result.mode}]: fetched={result.fetched} "
            f"selected={result.selected} saved={result.saved} "
            f"duplicates={result.duplicates} skipped={result.skipped} "
            f"errors={len(result.errors)} duration={result.duration_seconds}s"
        )
        for error in result.errors:
            print(f"  error: {error}")


def _print_dedup_stats(stats: StoryDedupStats) -> None:
    print(
        "dedup: "
        f"total_news={stats.total_news} "
        f"story_clusters={stats.story_clusters} "
        f"clustered_items={stats.clustered_items} "
        f"deduplicated_items={stats.deduplicated_items} "
        f"unique_stories={stats.unique_stories}"
    )


def _print_story_clusters(database_path, limit: int) -> None:
    for index, cluster in enumerate(story_cluster_report(database_path, limit=limit), start=1):
        print(
            f"#{index} {cluster.story_id}: "
            f"items={cluster.item_count} sources={cluster.source_count} "
            f"first={cluster.first_published_at_msk} last={cluster.last_published_at_msk}"
        )
        print(f"  canonical: {cluster.canonical_title}")
        for item in cluster.items:
            print(
                f"  - {item.event_at_msk} | {item.source} | "
                f"score={item.score:g} | {item.title}"
            )


def _print_ticker_stats(stats: TickerStats) -> None:
    print(
        "tickers: "
        f"total_news={stats.total_news} "
        f"tagged_news={stats.tagged_news} "
        f"untagged_news={stats.untagged_news} "
        f"multi_ticker_news={stats.multi_ticker_news}"
    )
    for ticker, count in stats.ticker_counts.items():
        print(f"  {ticker}: {count}")


def _configure_output_encoding() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


if __name__ == "__main__":
    main()
