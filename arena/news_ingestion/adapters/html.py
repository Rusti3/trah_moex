from collections.abc import AsyncIterator
from datetime import UTC, datetime
from urllib.parse import urldefrag, urljoin, urlparse

import httpx
import trafilatura
from bs4 import BeautifulSoup
from dateutil.parser import parse as parse_datetime

from news_ingestion.adapters.base import NewsSourceAdapter
from news_ingestion.cleaning import clean_text
from news_ingestion.filters import is_news_url
from news_ingestion.schemas import NewsItem


class HTMLNewsAdapter(NewsSourceAdapter):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._known_external_ids: set[str] = set()

    def set_known_external_ids(self, values: set[str]) -> None:
        self._known_external_ids = values

    async def fetch(self) -> list[NewsItem]:
        return [item async for item in self.iter_items()]

    async def iter_items(self) -> AsyncIterator[NewsItem]:
        links = await self.discover_article_links()
        for link in links:
            try:
                article = await self.extract_article(link)
            except httpx.HTTPError:
                continue
            if article is not None:
                yield article

    async def discover_article_links(self) -> list[str]:
        html = await self._fetch_html(str(self.config.url))
        soup = BeautifulSoup(html, "html.parser")
        parser = self.config.parser
        selector = parser.list_item_selector if parser else None
        anchors = soup.select(selector) if selector else soup.select("a[href]")
        max_items = parser.max_items if parser else 2
        base_host = urlparse(str(self.config.url)).netloc
        source_url = _normalize_url(str(self.config.url))

        candidates: list[str] = []
        seen: set[str] = set()
        for anchor in anchors:
            href = anchor.get("href")
            if not href:
                continue
            absolute = urljoin(str(self.config.url), href)
            absolute = _normalize_url(absolute)
            parsed = urlparse(absolute)
            if parsed.netloc != base_host:
                continue
            if (
                absolute == source_url
                or absolute in seen
                or not self._is_allowed_link(absolute)
            ):
                continue
            candidates.append(absolute)
            seen.add(absolute)
            if len(candidates) >= max_items:
                break
        return [link for link in candidates if link not in self._known_external_ids]

    async def extract_article(self, url: str) -> NewsItem | None:
        html = await self._fetch_html(url)
        soup = BeautifulSoup(html, "html.parser")
        parser = self.config.parser

        text = clean_text(
            trafilatura.extract(
                html,
                include_comments=False,
                include_tables=False,
                favor_precision=True,
            )
        )
        if not text and parser and parser.article_selector:
            article_element = soup.select_one(parser.article_selector)
            text = clean_text(
                article_element.get_text("\n", strip=True) if article_element else None
            )

        title = self._extract_title(soup)
        if not text and title:
            text = title
        if not text:
            return None

        return NewsItem(
            source_id=self.config.id,
            source_type=self.config.type,
            external_id=url,
            published_at=self._extract_date(soup),
            fetched_at=datetime.now(UTC),
            title=title,
            text=text,
            url=url,
            confidence=self.config.trust_score,
            raw={"url": url},
        )

    async def _fetch_html(self, url: str) -> str:
        async with httpx.AsyncClient(
            timeout=self.settings.request_timeout_seconds, follow_redirects=True
        ) as client:
            response = await client.get(
                url,
                headers={"User-Agent": "Mozilla/5.0 news-monitor/0.1"},
            )
            response.raise_for_status()
            return response.text

    def _is_allowed_link(self, url: str) -> bool:
        parser = self.config.parser
        lowered = url.lower()
        if parser and parser.link_block_patterns and any(
            pattern.lower() in lowered for pattern in parser.link_block_patterns
        ):
            return False
        if parser and parser.link_allow_patterns:
            return any(pattern.lower() in lowered for pattern in parser.link_allow_patterns)
        return is_news_url(url)

    def _extract_title(self, soup: BeautifulSoup) -> str | None:
        selector = self.config.parser.title_selector if self.config.parser else None
        element = soup.select_one(selector) if selector else None
        if element is None:
            element = soup.find("h1") or soup.find("title")
        return clean_text(element.get_text(" ", strip=True) if element else None) or None

    def _extract_date(self, soup: BeautifulSoup) -> datetime | None:
        selector = self.config.parser.date_selector if self.config.parser else None
        element = soup.select_one(selector) if selector else soup.find("time")
        if element is None:
            return None
        raw_value = element.get("datetime") or element.get_text(" ", strip=True)
        if not raw_value:
            return None
        try:
            parsed = parse_datetime(raw_value)
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
        except (TypeError, ValueError):
            return None


def _normalize_url(url: str) -> str:
    clean, _fragment = urldefrag(url)
    return clean
