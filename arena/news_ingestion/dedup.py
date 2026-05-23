from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from news_ingestion.storage import MOSCOW_TZ, MSK_FORMAT, connect, initialize_database

WINDOW_SECONDS = 60 * 60
MATCH_THRESHOLD = 70

STOPWORDS = {
    "что",
    "как",
    "это",
    "для",
    "или",
    "при",
    "над",
    "под",
    "без",
    "его",
    "она",
    "они",
    "оно",
    "уже",
    "еще",
    "ещё",
    "после",
    "из-за",
    "из",
    "за",
    "со",
    "во",
    "на",
    "по",
    "об",
    "от",
    "до",
    "а",
    "и",
    "в",
    "с",
    "к",
    "о",
    "у",
    "не",
    "но",
    "же",
    "ли",
    "бы",
    "мы",
    "вы",
    "их",
    "ее",
    "её",
    "он",
    "все",
    "всё",
    "новости",
    "новость",
    "сообщил",
    "сообщила",
    "сообщили",
    "заявил",
    "заявила",
    "заявили",
    "рассказал",
    "рассказала",
    "стало",
    "известно",
    "может",
    "могут",
    "будет",
    "будут",
    "года",
    "году",
    "лет",
    "руб",
    "рублей",
    "млн",
    "млрд",
    "тыс",
    "рф",
    "россии",
    "россия",
    "российский",
    "российская",
    "российские",
    "совет",
    "директоров",
    "рекомендовал",
    "рекомендовала",
    "рекомендовали",
    "пресс",
    "релиз",
    "релизы",
    "финансовые",
    "финансовый",
    "финансовая",
    "операционные",
    "операционный",
    "операционная",
    "результаты",
    "результат",
    "группа",
    "группы",
    "квартал",
    "квартала",
}

STEM_ENDINGS = (
    "иями",
    "ями",
    "ами",
    "ого",
    "ему",
    "ими",
    "ыми",
    "ее",
    "ие",
    "ые",
    "ое",
    "ая",
    "яя",
    "ий",
    "ый",
    "ой",
    "ую",
    "юю",
    "ых",
    "их",
    "ам",
    "ям",
    "ом",
    "ем",
    "ах",
    "ях",
    "ов",
    "ев",
    "ей",
    "ии",
    "ия",
    "ья",
    "иям",
    "ием",
    "иях",
    "ию",
    "ью",
    "а",
    "я",
    "ы",
    "и",
    "е",
    "у",
    "ю",
    "о",
)


@dataclass(frozen=True)
class StoryDedupStats:
    total_news: int
    story_clusters: int
    clustered_items: int
    deduplicated_items: int
    unique_stories: int


@dataclass(frozen=True)
class StoryClusterItemReport:
    news_id: str
    source: str
    title: str | None
    event_at_msk: str | None
    score: float
    matched_news_id: str | None
    url: str | None


@dataclass(frozen=True)
class StoryClusterReport:
    story_id: str
    canonical_title: str | None
    first_published_at_msk: str | None
    last_published_at_msk: str | None
    item_count: int
    source_count: int
    sources: list[str]
    items: list[StoryClusterItemReport]


@dataclass(frozen=True)
class _NewsRecord:
    news_id: str
    source: str
    title: str | None
    url: str | None
    raw_payload_hash: str | None
    published_at_msk: str | None
    received_at_msk: str | None
    event_time: datetime | None
    normalized_title: str
    tokens: frozenset[str]
    numbers: frozenset[str]


@dataclass(frozen=True)
class _ClusterItem:
    record: _NewsRecord
    score: float
    matched_news_id: str | None


@dataclass
class _WorkingCluster:
    items: list[_ClusterItem]

    @property
    def sources(self) -> set[str]:
        return {item.record.source for item in self.items}

    @property
    def first_time(self) -> datetime:
        return min(
            (item.record.event_time for item in self.items if item.record.event_time),
            default=datetime.max,
        )


def rebuild_story_clusters(database_path: str | Path) -> StoryDedupStats:
    initialize_database(database_path)
    records = _load_news_records(database_path)
    clusters = _build_story_clusters(records)
    _replace_story_clusters(database_path, clusters)
    return story_dedup_stats(database_path)


def story_dedup_stats(database_path: str | Path) -> StoryDedupStats:
    initialize_database(database_path)
    with connect(database_path) as conn:
        total_news = int(conn.execute("SELECT COUNT(*) FROM news").fetchone()[0])
        row = conn.execute(
            """
            SELECT
                COUNT(*) AS story_clusters,
                COALESCE(SUM(item_count), 0) AS clustered_items,
                COALESCE(SUM(item_count - 1), 0) AS deduplicated_items
            FROM story_clusters
            """
        ).fetchone()

    deduplicated_items = int(row["deduplicated_items"])
    return StoryDedupStats(
        total_news=total_news,
        story_clusters=int(row["story_clusters"]),
        clustered_items=int(row["clustered_items"]),
        deduplicated_items=deduplicated_items,
        unique_stories=total_news - deduplicated_items,
    )


def story_cluster_report(
    database_path: str | Path,
    *,
    limit: int = 20,
) -> list[StoryClusterReport]:
    initialize_database(database_path)
    with connect(database_path) as conn:
        cluster_rows = conn.execute(
            """
            SELECT *
            FROM story_clusters
            ORDER BY item_count DESC, first_published_at_msk ASC, story_id ASC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        reports = []
        for cluster in cluster_rows:
            item_rows = conn.execute(
                """
                SELECT
                    sci.news_id,
                    sci.source,
                    sci.score,
                    sci.matched_news_id,
                    n.title,
                    COALESCE(n.published_at_msk, n.received_at_msk) AS event_at_msk,
                    n.url
                FROM story_cluster_items AS sci
                JOIN news AS n ON n.news_id = sci.news_id
                WHERE sci.story_id = ?
                ORDER BY event_at_msk ASC, sci.source ASC, sci.news_id ASC
                """,
                (cluster["story_id"],),
            ).fetchall()
            reports.append(
                StoryClusterReport(
                    story_id=cluster["story_id"],
                    canonical_title=cluster["canonical_title"],
                    first_published_at_msk=cluster["first_published_at_msk"],
                    last_published_at_msk=cluster["last_published_at_msk"],
                    item_count=int(cluster["item_count"]),
                    source_count=int(cluster["source_count"]),
                    sources=json.loads(cluster["sources_json"]),
                    items=[
                        StoryClusterItemReport(
                            news_id=row["news_id"],
                            source=row["source"],
                            title=row["title"],
                            event_at_msk=row["event_at_msk"],
                            score=float(row["score"]),
                            matched_news_id=row["matched_news_id"],
                            url=row["url"],
                        )
                        for row in item_rows
                    ],
                )
            )
    return reports


def _load_news_records(database_path: str | Path) -> list[_NewsRecord]:
    with connect(database_path) as conn:
        rows = conn.execute(
            """
            SELECT
                news_id,
                source,
                title,
                url,
                raw_payload_hash,
                published_at_msk,
                received_at_msk
            FROM news
            ORDER BY COALESCE(published_at_msk, received_at_msk) ASC, news_id ASC
            """
        ).fetchall()

    records = []
    for row in rows:
        title = row["title"] or ""
        tokens = frozenset(_tokenize(title))
        records.append(
            _NewsRecord(
                news_id=row["news_id"],
                source=row["source"],
                title=row["title"],
                url=row["url"],
                raw_payload_hash=row["raw_payload_hash"],
                published_at_msk=row["published_at_msk"],
                received_at_msk=row["received_at_msk"],
                event_time=_parse_msk(row["published_at_msk"])
                or _parse_msk(row["received_at_msk"]),
                normalized_title=_normalize_title(title),
                tokens=tokens,
                numbers=frozenset(re.findall(r"\d+(?:[,.]\d+)?", title)),
            )
        )
    return records


def _build_story_clusters(records: list[_NewsRecord]) -> list[_WorkingCluster]:
    rarity = _token_rarity(records)
    processed: list[_NewsRecord] = []
    assigned: dict[str, _WorkingCluster] = {}
    clusters: list[_WorkingCluster] = []

    for record in records:
        best_score = 0.0
        best_match: _NewsRecord | None = None
        best_cluster: _WorkingCluster | None = None

        for candidate in processed:
            if not _within_window(record, candidate):
                continue
            score = _match_score(record, candidate, rarity)
            if score < MATCH_THRESHOLD:
                continue
            candidate_cluster = assigned.get(candidate.news_id)
            if candidate_cluster is not None and record.source in candidate_cluster.sources:
                continue
            if _is_better_match(
                score,
                candidate,
                candidate_cluster,
                best_score,
                best_match,
                best_cluster,
            ):
                best_score = score
                best_match = candidate
                best_cluster = candidate_cluster

        if best_match is not None:
            if best_cluster is None:
                best_cluster = _WorkingCluster(
                    items=[_ClusterItem(best_match, score=100.0, matched_news_id=None)]
                )
                clusters.append(best_cluster)
                assigned[best_match.news_id] = best_cluster

            best_cluster.items.append(
                _ClusterItem(record, score=best_score, matched_news_id=best_match.news_id)
            )
            assigned[record.news_id] = best_cluster

        processed.append(record)

    return clusters


def _replace_story_clusters(
    database_path: str | Path,
    clusters: list[_WorkingCluster],
) -> None:
    now_msk = _format_msk(datetime.now(MOSCOW_TZ))
    with connect(database_path) as conn:
        existing_created_at = {
            row["story_id"]: row["created_at_msk"]
            for row in conn.execute("SELECT story_id, created_at_msk FROM story_clusters")
        }
        conn.execute("DELETE FROM story_cluster_items")
        conn.execute("DELETE FROM story_clusters")

        for cluster in clusters:
            ordered = sorted(cluster.items, key=lambda item: _record_sort_key(item.record))
            records = [item.record for item in ordered]
            story_id = _story_id(records[0])
            sources = sorted({record.source for record in records})
            event_times = [record.event_time for record in records if record.event_time]
            first_event = min(event_times) if event_times else None
            last_event = max(event_times) if event_times else None

            conn.execute(
                """
                INSERT INTO story_clusters (
                    story_id,
                    canonical_title,
                    first_published_at_msk,
                    last_published_at_msk,
                    item_count,
                    source_count,
                    sources_json,
                    created_at_msk,
                    updated_at_msk
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    story_id,
                    _canonical_title(records),
                    _format_msk(first_event),
                    _format_msk(last_event),
                    len(records),
                    len(sources),
                    json.dumps(sources, ensure_ascii=False),
                    existing_created_at.get(story_id, now_msk),
                    now_msk,
                ),
            )

            for item in ordered:
                conn.execute(
                    """
                    INSERT INTO story_cluster_items (
                        story_id,
                        news_id,
                        source,
                        score,
                        matched_news_id,
                        created_at_msk
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        story_id,
                        item.record.news_id,
                        item.record.source,
                        round(item.score, 3),
                        item.matched_news_id,
                        now_msk,
                    ),
                )


def _match_score(
    first: _NewsRecord,
    second: _NewsRecord,
    rarity: dict[str, int],
) -> float:
    if first.source == second.source:
        return 0.0
    if first.url and first.url == second.url:
        return 100.0
    if first.raw_payload_hash and first.raw_payload_hash == second.raw_payload_hash:
        return 100.0
    if not first.tokens or not second.tokens:
        return 0.0
    if not _has_enough_signal(first) or not _has_enough_signal(second):
        return 0.0

    common = first.tokens & second.tokens
    union = first.tokens | second.tokens
    jaccard = len(common) / len(union)
    overlap = len(common) / min(len(first.tokens), len(second.tokens))
    rare_common = {token for token in common if rarity[token] <= 8}

    score = 0.0
    if first.normalized_title and first.normalized_title == second.normalized_title:
        score += 100.0
    if len(common) >= 3:
        score += 25.0
    if len(rare_common) >= 2:
        score += 25.0
    if jaccard >= 0.45:
        score += 30.0
    elif jaccard >= 0.32:
        score += 18.0
    if overlap >= 0.65:
        score += 20.0
    elif overlap >= 0.5:
        score += 10.0
    if first.numbers and second.numbers and first.numbers & second.numbers:
        score += 10.0
    if first.numbers and second.numbers and not first.numbers & second.numbers and len(common) < 4:
        score -= 20.0
    return score


def _is_better_match(
    score: float,
    candidate: _NewsRecord,
    candidate_cluster: _WorkingCluster | None,
    best_score: float,
    best_match: _NewsRecord | None,
    best_cluster: _WorkingCluster | None,
) -> bool:
    if score > best_score:
        return True
    if score < best_score or best_match is None:
        return False
    return _match_sort_time(candidate, candidate_cluster) < _match_sort_time(
        best_match,
        best_cluster,
    )


def _match_sort_time(
    candidate: _NewsRecord,
    candidate_cluster: _WorkingCluster | None,
) -> datetime:
    if candidate_cluster is not None:
        return candidate_cluster.first_time
    return candidate.event_time or datetime.max


def _within_window(first: _NewsRecord, second: _NewsRecord) -> bool:
    if first.event_time is None or second.event_time is None:
        return False
    return abs((first.event_time - second.event_time).total_seconds()) <= WINDOW_SECONDS


def _token_rarity(records: list[_NewsRecord]) -> dict[str, int]:
    rarity: dict[str, int] = defaultdict(int)
    for record in records:
        for token in record.tokens:
            rarity[token] += 1
    return rarity


def _has_enough_signal(record: _NewsRecord) -> bool:
    word_tokens = {token for token in record.tokens if not token.isdigit()}
    return len(word_tokens) >= 3 or (len(word_tokens) >= 2 and bool(record.numbers))


def _tokenize(title: str) -> list[str]:
    tokens = []
    for token in re.findall(r"[a-zа-я0-9]+", title.lower().replace("ё", "е")):
        if len(token) < 3 or token in STOPWORDS:
            continue
        tokens.append(_stem(token))
    return tokens


def _stem(token: str) -> str:
    if len(token) <= 5:
        return token
    for ending in STEM_ENDINGS:
        if token.endswith(ending) and len(token) - len(ending) >= 4:
            return token[: -len(ending)]
    return token


def _normalize_title(title: str) -> str:
    return " ".join(_tokenize(title))


def _canonical_title(records: list[_NewsRecord]) -> str | None:
    for record in records:
        if record.title:
            return record.title
    return None


def _story_id(record: _NewsRecord) -> str:
    digest = hashlib.sha1(record.news_id.encode("utf-8")).hexdigest()[:16]
    return f"story:{digest}"


def _record_sort_key(record: _NewsRecord) -> tuple[datetime, str]:
    return (record.event_time or datetime.max, record.news_id)


def _parse_msk(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.strptime(value, MSK_FORMAT)
    except ValueError:
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            return parsed
        return parsed.astimezone(MOSCOW_TZ).replace(tzinfo=None)


def _format_msk(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is not None:
        value = value.astimezone(MOSCOW_TZ).replace(tzinfo=None)
    return value.strftime(MSK_FORMAT)
