from collections.abc import AsyncIterator
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

import feedparser
import httpx
import trafilatura
from bs4 import BeautifulSoup

from news_ingestion.adapters.base import NewsSourceAdapter
from news_ingestion.cleaning import clean_text
from news_ingestion.schemas import NewsItem

FULL_TEXT_MIN_LENGTH = 900
DEFAULT_RSS_MAX_ITEMS = 2


class RSSAdapter(NewsSourceAdapter):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._known_external_ids: set[str] = set()
        self._last_seen_external_id: str | None = None
        self._last_seen_published_at: datetime | None = None

    def set_known_external_ids(self, values: set[str]) -> None:
        self._known_external_ids = values

    def set_watermark(
        self,
        last_seen_external_id: str | None,
        last_seen_published_at: datetime | None,
    ) -> None:
        self._last_seen_external_id = last_seen_external_id
        self._last_seen_published_at = last_seen_published_at

    async def fetch(self) -> list[NewsItem]:
        return [item async for item in self.iter_items()]

    async def iter_items(self) -> AsyncIterator[NewsItem]:
        async with httpx.AsyncClient(
            timeout=self.settings.request_timeout_seconds, follow_redirects=True
        ) as client:
            response = await client.get(
                str(self.config.url),
                headers={"User-Agent": "Mozilla/5.0 news-monitor/0.1"},
            )
            response.raise_for_status()
            items = self.parse_feed(response.content)
            for item in items:
                await self._enrich_item_with_full_text(client, item)
                yield item

    def parse_feed(self, content: bytes | str) -> list[NewsItem]:
        feed = feedparser.parse(content)
        fetched_at = datetime.now(UTC)
        items: list[NewsItem] = []

        for entry in feed.entries[: self._max_items()]:
            title = clean_text(entry.get("title"))
            summary = clean_text(entry.get("summary") or entry.get("description"))
            content_text = _entry_content_text(entry)
            body = content_text or summary
            text = clean_text("\n\n".join(part for part in [title, body] if part))
            if not text:
                continue

            item = NewsItem(
                source_id=self.config.id,
                source_type=self.config.type,
                external_id=str(
                    entry.get("id") or entry.get("guid") or entry.get("link") or ""
                ),
                published_at=_parse_entry_datetime(entry),
                fetched_at=fetched_at,
                title=title or None,
                summary=summary or None,
                text=text,
                url=entry.get("link"),
                confidence=self.config.trust_score,
                raw=_entry_to_dict(entry),
            )
            if self._should_skip_incremental(item):
                continue
            items.append(item)
        return items

    def _max_items(self) -> int:
        if self.config.parser and self.config.parser.max_items:
            return self.config.parser.max_items
        return DEFAULT_RSS_MAX_ITEMS

    def _should_skip_incremental(self, item: NewsItem) -> bool:
        if item.external_id and item.external_id in self._known_external_ids:
            return True
        if (
            item.external_id
            and self._last_seen_external_id
            and item.external_id == self._last_seen_external_id
        ):
            return True
        return bool(
            item.published_at is not None
            and self._last_seen_published_at is not None
            and item.published_at < self._last_seen_published_at
        )

    async def _enrich_item_with_full_text(
        self,
        client: httpx.AsyncClient,
        item: NewsItem,
    ) -> None:
        if not item.url or not _needs_full_text(item.text):
            return
        try:
            response = await client.get(
                item.url,
                headers={"User-Agent": "Mozilla/5.0 news-monitor/0.1"},
            )
            response.raise_for_status()
        except httpx.HTTPError:
            return

        extracted = clean_text(
            trafilatura.extract(
                response.text,
                include_comments=False,
                include_tables=False,
                favor_precision=True,
            )
        )
        if not extracted:
            soup = BeautifulSoup(response.text, "html.parser")
            article = soup.find("article")
            extracted = clean_text(article.get_text("\n", strip=True) if article else None)
        if len(extracted) > len(item.text):
            item.text = clean_text("\n\n".join(part for part in [item.title, extracted] if part))
            if item.raw is not None:
                item.raw["full_text_extracted"] = True
                item.raw["full_text_length"] = len(extracted)


def _parse_entry_datetime(entry: Any) -> datetime | None:
    published = entry.get("published") or entry.get("updated")
    if published:
        try:
            parsed = parsedate_to_datetime(published)
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=UTC)
            return parsed
        except (TypeError, ValueError):
            return None

    parsed_struct = entry.get("published_parsed") or entry.get("updated_parsed")
    if parsed_struct:
        return datetime(*parsed_struct[:6], tzinfo=UTC)
    return None


def _entry_to_dict(entry: Any) -> dict[str, Any]:
    return {key: entry.get(key) for key in entry}


def _entry_content_text(entry: Any) -> str:
    content = entry.get("content")
    if not content:
        return ""
    parts: list[str] = []
    for item in content:
        value = item.get("value") if hasattr(item, "get") else None
        if value:
            parts.append(_html_to_text(value))
    return clean_text("\n\n".join(parts))


def _html_to_text(value: str) -> str:
    soup = BeautifulSoup(value, "html.parser")
    return clean_text(soup.get_text("\n", strip=True))


def _needs_full_text(text: str) -> bool:
    stripped = text.strip()
    return len(stripped) < FULL_TEXT_MIN_LENGTH or stripped.endswith(("...", "…"))
