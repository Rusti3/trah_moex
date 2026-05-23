from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator, model_validator

SourceType = Literal[
    "official_company",
    "fast_agency",
    "media_analysis",
    "disclosure",
]
SourceMethod = Literal["rss", "html", "azipi_disclosure", "rftoday"]


class ParserConfig(BaseModel):
    list_item_selector: str | None = None
    title_selector: str | None = None
    date_selector: str | None = None
    article_selector: str | None = None
    link_allow_patterns: list[str] = Field(default_factory=list)
    link_block_patterns: list[str] = Field(default_factory=list)
    max_items: int = 2


class SourceConfig(BaseModel):
    id: str
    name: str
    type: SourceType
    method: SourceMethod
    tickers: list[str] = Field(default_factory=lambda: ["ALL"])
    url: HttpUrl | None = None
    interval_seconds: int
    trust_score: float = Field(ge=0.0, le=1.0)
    enabled: bool = True
    parser: ParserConfig | None = None
    last_seen_external_id: str | None = None
    last_seen_published_at: datetime | None = None
    last_polled_at: datetime | None = None

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        if not value.replace("_", "").replace("-", "").isalnum():
            raise ValueError("source id must contain only letters, numbers, '_' or '-'")
        return value

    @model_validator(mode="after")
    def validate_access_shape(self) -> "SourceConfig":
        if self.method in {"rss", "html", "azipi_disclosure", "rftoday"} and self.url is None:
            raise ValueError(f"{self.method} source requires url")
        return self


class NewsItem(BaseModel):
    source_id: str
    source_type: SourceType
    external_id: str | None = None
    url: str | None = None

    published_at: datetime | None = None
    fetched_at: datetime
    discovered_at: datetime | None = None
    saved_at: datetime | None = None

    title: str | None = None
    text: str
    summary: str | None = None

    language: str = "ru"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    raw: dict[str, Any] | None = None

    model_config = ConfigDict(from_attributes=True)
