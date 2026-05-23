from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from news_ingestion.schemas import SourceConfig


class SourceRegistry(BaseModel):
    sources: list[SourceConfig] = Field(default_factory=list)

    @classmethod
    def load(cls, path: str | Path) -> "SourceRegistry":
        config_path = Path(path)
        with config_path.open("r", encoding="utf-8") as file:
            payload = yaml.safe_load(file) or {}
        return cls.model_validate(payload)

    def enabled_sources(self) -> list[SourceConfig]:
        return [source for source in self.sources if source.enabled]

    def get(self, source_id: str) -> SourceConfig:
        for source in self.sources:
            if source.id == source_id:
                return source
        raise KeyError(f"unknown source_id: {source_id}")
