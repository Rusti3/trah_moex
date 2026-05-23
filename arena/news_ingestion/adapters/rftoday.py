from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from urllib.parse import urldefrag, urljoin
from zoneinfo import ZoneInfo

import httpx
import trafilatura
from bs4 import BeautifulSoup
from dateutil.parser import parse as parse_datetime

from news_ingestion.adapters.base import NewsSourceAdapter
from news_ingestion.cleaning import clean_text
from news_ingestion.schemas import NewsItem

DEFAULT_MAX_ITEMS_PER_SECTION = 2
MOSCOW_TZ = ZoneInfo("Europe/Moscow")


@dataclass(frozen=True)
class RFTodaySection:
    id: str
    name: str
    url: str


@dataclass(frozen=True)
class RFTodayListEntry:
    section: RFTodaySection
    archive_url: str
    title: str
    summary: str
    source_name: str | None
    time_text: str | None


RFTODAY_SECTIONS: tuple[RFTodaySection, ...] = (
    RFTodaySection("country", "Страна", "https://www.rftoday.ru/"),
    RFTodaySection("oil", "Нефть", "https://oil.rftoday.ru/"),
    RFTodaySection("gas", "Газ", "https://gas.rftoday.ru/"),
    RFTodaySection("metal", "Металл", "https://metal.rftoday.ru/"),
    RFTodaySection("agro", "Агро", "https://agro.rftoday.ru/"),
    RFTodaySection("finance", "Финансы", "https://finance.rftoday.ru/"),
    RFTodaySection("hitech", "Hi-Tech", "https://hitech.rftoday.ru/"),
)


class RFTodayAdapter(NewsSourceAdapter):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._known_external_ids: set[str] = set()

    def set_known_external_ids(self, external_ids: set[str]) -> None:
        self._known_external_ids = external_ids

    async def fetch(self) -> list[NewsItem]:
        return [item async for item in self.iter_items()]

    async def iter_items(self) -> AsyncIterator[NewsItem]:
        seen: set[str] = set()
        section_errors: list[str] = []
        loaded_sections = 0

        async with httpx.AsyncClient(
            timeout=self.settings.request_timeout_seconds,
            follow_redirects=True,
        ) as client:
            for section in self._section_urls():
                try:
                    html, _final_url = await self._fetch_html(client, section.url)
                except httpx.HTTPError as exc:
                    section_errors.append(f"{section.id}: {exc}")
                    continue

                loaded_sections += 1
                for entry in self._parse_list_page(section, html):
                    if entry.archive_url in seen or entry.archive_url in self._known_external_ids:
                        continue
                    seen.add(entry.archive_url)
                    yield await self._build_item(client, entry)

        if loaded_sections == 0 and section_errors:
            raise RuntimeError("; ".join(section_errors))

    def _section_urls(self) -> tuple[RFTodaySection, ...]:
        return RFTODAY_SECTIONS

    def _parse_list_page(
        self,
        section: RFTodaySection,
        html: str,
    ) -> list[RFTodayListEntry]:
        soup = BeautifulSoup(html, "html.parser")
        entries: list[RFTodayListEntry] = []

        for element in soup.select("div.item"):
            link = element.select_one("a.source[href]")
            title_element = element.select_one("span.title")
            if link is None or title_element is None:
                continue

            href = link.get("href")
            title = clean_text(title_element.get_text(" ", strip=True))
            if not href or not title:
                continue

            summary_element = element.select_one("p")
            time_element = element.select_one("b")
            archive_url = _normalize_url(urljoin(section.url, href))

            entries.append(
                RFTodayListEntry(
                    section=section,
                    archive_url=archive_url,
                    title=title,
                    summary=clean_text(
                        summary_element.get_text(" ", strip=True)
                        if summary_element is not None
                        else None
                    ),
                    source_name=clean_text(link.get("title") or link.get_text(" ", strip=True))
                    or None,
                    time_text=clean_text(
                        time_element.get_text(" ", strip=True)
                        if time_element is not None
                        else None
                    )
                    or None,
                )
            )
            if len(entries) >= self._max_items_per_section():
                break

        return entries

    async def _build_item(
        self,
        client: httpx.AsyncClient,
        entry: RFTodayListEntry,
    ) -> NewsItem:
        final_url = entry.archive_url
        article_text = ""
        article_published_at: datetime | None = None
        detail_error: str | None = None

        try:
            html, final_url = await self._fetch_html(client, entry.archive_url)
            article_text = _extract_full_text(html)
            article_published_at = _extract_article_datetime(html)
        except httpx.HTTPError as exc:
            detail_error = str(exc)

        body = article_text or entry.summary
        text = clean_text("\n\n".join(part for part in [entry.title, body] if part))

        raw = {
            "archive_url": entry.archive_url,
            "final_url": final_url,
            "section": entry.section.id,
            "section_name": entry.section.name,
            "source_name": entry.source_name,
        }
        if detail_error:
            raw["detail_error"] = detail_error

        list_published_at = _parse_list_time(entry.time_text)
        if (
            article_published_at is not None
            and list_published_at is not None
            and _looks_like_date_only(article_published_at)
        ):
            published_at = list_published_at
        else:
            published_at = article_published_at or list_published_at

        return NewsItem(
            source_id=self.config.id,
            source_type=self.config.type,
            external_id=entry.archive_url,
            url=final_url,
            published_at=published_at,
            fetched_at=datetime.now(UTC),
            title=entry.title,
            summary=entry.summary or None,
            text=text,
            confidence=self.config.trust_score,
            raw=raw,
        )

    async def _fetch_html(
        self,
        client: httpx.AsyncClient,
        url: str,
    ) -> tuple[str, str]:
        response = await client.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 news-monitor/0.1"},
        )
        response.raise_for_status()
        return response.text, _normalize_url(str(response.url))

    def _max_items_per_section(self) -> int:
        if self.config.parser and self.config.parser.max_items:
            return self.config.parser.max_items
        return DEFAULT_MAX_ITEMS_PER_SECTION


def _extract_full_text(html: str) -> str:
    extracted = clean_text(
        trafilatura.extract(
            html,
            include_comments=False,
            include_tables=False,
            favor_precision=True,
        )
    )
    if extracted:
        return extracted

    soup = BeautifulSoup(html, "html.parser")
    article = (
        soup.find("article")
        or soup.select_one("[itemprop='articleBody']")
        or soup.select_one("main")
    )
    return clean_text(article.get_text("\n", strip=True) if article else None)


def _extract_article_datetime(html: str) -> datetime | None:
    soup = BeautifulSoup(html, "html.parser")
    for selector in (
        'meta[property="article:published_time"]',
        'meta[name="pubdate"]',
        'meta[itemprop="datePublished"]',
        "time[datetime]",
    ):
        element = soup.select_one(selector)
        if element is None:
            continue
        value = element.get("content") or element.get("datetime")
        parsed = _parse_datetime_value(value)
        if parsed is not None:
            return parsed

    metadata = trafilatura.extract_metadata(html)
    if metadata and metadata.date:
        parsed = _parse_datetime_value(metadata.date)
        if parsed is not None:
            return parsed
    return None


def _parse_datetime_value(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = parse_datetime(value)
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _parse_list_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed_time = datetime.strptime(value, "%H:%M").time()
    except ValueError:
        return None

    now = datetime.now(MOSCOW_TZ)
    candidate = datetime.combine(now.date(), parsed_time, tzinfo=MOSCOW_TZ)
    if candidate > now + timedelta(minutes=10):
        candidate -= timedelta(days=1)
    return candidate.astimezone(UTC)


def _looks_like_date_only(value: datetime) -> bool:
    return (
        value.hour == 0
        and value.minute == 0
        and value.second == 0
        and value.microsecond == 0
    )


def _normalize_url(url: str) -> str:
    clean, _fragment = urldefrag(url)
    return clean.rstrip("/")
