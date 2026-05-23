from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from datetime import datetime

from news_ingestion.schemas import NewsItem, SourceConfig
from news_ingestion.settings import Settings


class NewsSourceAdapter(ABC):
    def __init__(self, config: SourceConfig, settings: Settings):
        self.config = config
        self.settings = settings

    def set_known_external_ids(self, values: set[str]) -> None:
        """Provide ids already accepted/rejected so adapters can avoid detail fetches."""
        return None

    def set_watermark(
        self,
        last_seen_external_id: str | None,
        last_seen_published_at: datetime | None,
    ) -> None:
        """Provide source watermark for incremental fetches."""
        return None

    @abstractmethod
    async def fetch(self) -> list[NewsItem]:
        """Fetch normalized news items from a source."""

    async def iter_items(self) -> AsyncIterator[NewsItem]:
        """Stream news items from a source as soon as they are available."""
        for item in await self.fetch():
            yield item
