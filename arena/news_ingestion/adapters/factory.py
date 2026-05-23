from news_ingestion.adapters.base import NewsSourceAdapter
from news_ingestion.adapters.disclosure import DisclosureMessagesAdapter
from news_ingestion.adapters.html import HTMLNewsAdapter
from news_ingestion.adapters.rftoday import RFTodayAdapter
from news_ingestion.adapters.rss import RSSAdapter
from news_ingestion.schemas import SourceConfig
from news_ingestion.settings import Settings


def build_adapter(config: SourceConfig, settings: Settings) -> NewsSourceAdapter:
    if config.method == "rss":
        return RSSAdapter(config, settings)
    if config.method == "html":
        return HTMLNewsAdapter(config, settings)
    if config.method == "azipi_disclosure":
        return DisclosureMessagesAdapter(config, settings)
    if config.method == "rftoday":
        return RFTodayAdapter(config, settings)
    raise ValueError(f"unsupported source method: {config.method}")
