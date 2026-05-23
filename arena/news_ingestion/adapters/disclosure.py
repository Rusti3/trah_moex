from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime
from urllib.parse import parse_qs, urlencode, urljoin, urlparse, urlunparse
from zoneinfo import ZoneInfo

import httpx
from bs4 import BeautifulSoup
from dateutil.parser import parse as parse_datetime

from news_ingestion.adapters.base import NewsSourceAdapter
from news_ingestion.cleaning import clean_text
from news_ingestion.schemas import NewsItem

MOSCOW_TZ = ZoneInfo("Europe/Moscow")
DEFAULT_MAX_ITEMS = 2


class DisclosureMessagesAdapter(NewsSourceAdapter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._known_external_ids: set[str] = set()

    def set_known_external_ids(self, values: set[str]) -> None:
        self._known_external_ids = values

    async def fetch(self) -> list[NewsItem]:
        return [item async for item in self.iter_items()]

    async def iter_items(self) -> AsyncIterator[NewsItem]:
        day = _current_day()
        max_items = self.config.parser.max_items if self.config.parser else DEFAULT_MAX_ITEMS
        max_items = max_items or DEFAULT_MAX_ITEMS
        base_url = _messages_url(str(self.config.url), day)
        yielded = 0

        async with httpx.AsyncClient(
            timeout=self.settings.request_timeout_seconds,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 news-monitor/0.1"},
        ) as client:
            page = 1
            while yielded < max_items:
                page_url = _page_url(base_url, day, page)
                html = await self._fetch_html(client, page_url)
                page_items = _parse_message_list(
                    html,
                    source_id=self.config.id,
                    source_type=self.config.type,
                    trust_score=self.config.trust_score,
                    page_url=page_url,
                )
                if not page_items:
                    break
                all_page_items_known = all(
                    item.external_id in self._known_external_ids for item in page_items
                )

                for item in page_items:
                    if yielded >= max_items:
                        break
                    if item.external_id not in self._known_external_ids:
                        item = await self._enrich_item_detail(client, item)
                    yield item
                    yielded += 1

                if all_page_items_known or yielded >= max_items or not _has_next_page(html, page):
                    break
                page += 1

    async def _fetch_html(self, client: httpx.AsyncClient, url: str) -> str:
        response = await client.get(url)
        response.raise_for_status()
        if "servicepipe.ru" in response.text:
            raise RuntimeError(
                "e-disclosure.ru returned anti-bot challenge; use e-disclosure.azipi.ru mirror"
            )
        return response.text

    async def _enrich_item_detail(self, client: httpx.AsyncClient, item: NewsItem) -> NewsItem:
        if not item.url:
            return item
        try:
            html = await self._fetch_html(client, item.url)
        except httpx.HTTPError as exc:
            item.raw = {**(item.raw or {}), "detail_error": str(exc)}
            return item
        detail = _parse_message_detail(html)
        if detail.title:
            item.title = detail.title
        if detail.text and len(detail.text) > len(item.text):
            item.text = detail.text
            item.raw = {
                **(item.raw or {}),
                "detail_extracted": True,
                "detail_text_length": len(detail.text),
            }
        return item


@dataclass(frozen=True)
class MessageDetail:
    title: str | None
    text: str


def _current_day() -> str:
    return datetime.now(MOSCOW_TZ).strftime("%d.%m.%Y")


def _messages_url(config_url: str, day: str) -> str:
    parsed = urlparse(config_url)
    host = parsed.netloc.lower()
    if "azipi.ru" not in host:
        return f"https://e-disclosure.azipi.ru/messages/list/day-{day}/"
    return urljoin(config_url, f"/messages/list/day-{day}/")


def _page_url(base_url: str, day: str, page: int) -> str:
    if page <= 1:
        return base_url
    parsed = urlparse(base_url)
    query = parse_qs(parsed.query)
    query["MESSAGE_BY_DAY"] = [day]
    query["PAGEN_2"] = [str(page)]
    return urlunparse(parsed._replace(query=urlencode(query, doseq=True)))


def _parse_message_list(
    html: str,
    *,
    source_id: str,
    source_type: str,
    trust_score: float,
    page_url: str,
) -> list[NewsItem]:
    soup = BeautifulSoup(html, "html.parser")
    fetched_at = datetime.now(UTC)
    result: list[NewsItem] = []
    for element in soup.select(".messages-subjects .item"):
        link = element.select_one(".link a[href]") or element.select_one("a[href]")
        if link is None:
            continue
        href = link.get("href")
        if not href:
            continue
        title = clean_text(link.get_text(" ", strip=True) or link.get("alt"))
        text = clean_text(element.get_text(" ", strip=True))
        if not text:
            continue
        published_at = _parse_published_at(element)
        url = urljoin(page_url, href)
        result.append(
            NewsItem(
                source_id=source_id,
                source_type=source_type,
                external_id=url,
                url=url,
                published_at=published_at,
                fetched_at=fetched_at,
                title=title or None,
                text=text,
                confidence=trust_score,
                raw={"page_url": page_url},
            )
        )
    return result


def _parse_message_detail(html: str) -> MessageDetail:
    soup = BeautifulSoup(html, "html.parser")
    title_element = soup.find("h1")
    title = clean_text(title_element.get_text(" ", strip=True) if title_element else None) or None
    container = soup.select_one(".main-col") or soup.find("main") or soup.body
    text = clean_text(container.get_text("\n", strip=True) if container else None)
    return MessageDetail(title=title, text=text)


def _parse_published_at(element) -> datetime | None:
    date_element = element.select_one(".date")
    raw_value = clean_text(date_element.get_text(" ", strip=True) if date_element else None)
    if not raw_value:
        return None
    try:
        parsed = parse_datetime(raw_value, dayfirst=True)
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=MOSCOW_TZ)
    return parsed.astimezone(UTC)


def _has_next_page(html: str, current_page: int) -> bool:
    soup = BeautifulSoup(html, "html.parser")
    next_page = str(current_page + 1)
    for link in soup.select("a[href]"):
        href = link.get("href") or ""
        if parse_qs(urlparse(href).query).get("PAGEN_2") == [next_page]:
            return True
    return False
