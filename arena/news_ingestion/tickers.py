from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from news_ingestion.schemas import NewsItem, SourceConfig
from news_ingestion.storage import connect, initialize_database, replace_news_tickers

WORD_CHAR_CLASS = "0-9A-Za-zА-Яа-яЁё"


@dataclass(frozen=True)
class TickerMatch:
    ticker: str
    matched_by: str
    matched_terms: tuple[str, ...]


@dataclass(frozen=True)
class TickerStats:
    total_news: int
    tagged_news: int
    untagged_news: int
    multi_ticker_news: int
    ticker_counts: dict[str, int]


@dataclass(frozen=True)
class _CompiledTerm:
    term: str
    pattern: re.Pattern[str]


class TickerRegistry:
    def __init__(self, terms_by_ticker: dict[str, list[str]]):
        self.terms_by_ticker = {
            ticker: list(dict.fromkeys(terms))
            for ticker, terms in sorted(terms_by_ticker.items())
        }
        self._compiled = {
            ticker: [_compile_term(term) for term in terms]
            for ticker, terms in self.terms_by_ticker.items()
        }

    @classmethod
    def load(cls, path: str | Path) -> TickerRegistry:
        config_path = Path(path)
        with config_path.open("r", encoding="utf-8") as file:
            payload = yaml.safe_load(file) or {}
        raw_tickers = payload.get("tickers") or {}
        return cls(
            {
                str(ticker): [str(term) for term in terms]
                for ticker, terms in raw_tickers.items()
            }
        )

    def tag_item(self, item: NewsItem, source: SourceConfig) -> list[TickerMatch]:
        return self.tag_text(
            title=item.title,
            text=item.text,
            fallback_tickers=_source_fallback_tickers(source),
            fallback_source_id=source.id,
        )

    def tag_text(
        self,
        *,
        title: str | None,
        text: str | None,
        fallback_tickers: list[str] | None = None,
        fallback_source_id: str | None = None,
    ) -> list[TickerMatch]:
        haystack = f"{title or ''}\n{text or ''}"
        matches: dict[str, dict[str, set[str]]] = {}

        for ticker, terms in self._compiled.items():
            matched_terms = {
                term.term
                for term in terms
                if term.pattern.search(haystack)
            }
            if matched_terms:
                matches.setdefault(ticker, {"matched_by": set(), "matched_terms": set()})
                matches[ticker]["matched_by"].add("regex")
                matches[ticker]["matched_terms"].update(matched_terms)

        for ticker in fallback_tickers or []:
            if ticker not in self.terms_by_ticker:
                continue
            matches.setdefault(ticker, {"matched_by": set(), "matched_terms": set()})
            matches[ticker]["matched_by"].add("source")
            if fallback_source_id:
                matches[ticker]["matched_terms"].add(f"source:{fallback_source_id}")

        return [
            TickerMatch(
                ticker=ticker,
                matched_by=",".join(sorted(payload["matched_by"])),
                matched_terms=tuple(sorted(payload["matched_terms"])),
            )
            for ticker, payload in sorted(matches.items())
        ]


def tag_existing_news(
    database_path: str | Path,
    registry: TickerRegistry,
    sources: list[SourceConfig],
) -> TickerStats:
    initialize_database(database_path)
    sources_by_id = {source.id: source for source in sources}
    fallback_by_source = {
        source.id: _source_fallback_tickers(source)
        for source in sources
    }

    with connect(database_path) as conn:
        rows = conn.execute(
            """
            SELECT news_id, source, title, text
            FROM news
            ORDER BY received_at_msk ASC, news_id ASC
            """
        ).fetchall()
        for row in rows:
            fallback_tickers = fallback_by_source.get(row["source"], [])
            matches = registry.tag_text(
                title=row["title"],
                text=row["text"],
                fallback_tickers=fallback_tickers,
                fallback_source_id=row["source"] if row["source"] in sources_by_id else None,
            )
            replace_news_tickers(conn, row["news_id"], matches)

    return ticker_stats(database_path)


def ticker_stats(database_path: str | Path) -> TickerStats:
    initialize_database(database_path)
    with connect(database_path) as conn:
        total_news = int(conn.execute("SELECT COUNT(*) FROM news").fetchone()[0])
        tagged_news = int(
            conn.execute(
                """
                SELECT COUNT(*)
                FROM news
                WHERE tickers_json != '[]'
                """
            ).fetchone()[0]
        )
        multi_ticker_news = int(
            conn.execute(
                """
                SELECT COUNT(*)
                FROM (
                    SELECT news_id
                    FROM news_tickers
                    GROUP BY news_id
                    HAVING COUNT(*) > 1
                )
                """
            ).fetchone()[0]
        )
        rows = conn.execute(
            """
            SELECT ticker, COUNT(*) AS count
            FROM news_tickers
            GROUP BY ticker
            ORDER BY ticker ASC
            """
        ).fetchall()

    return TickerStats(
        total_news=total_news,
        tagged_news=tagged_news,
        untagged_news=total_news - tagged_news,
        multi_ticker_news=multi_ticker_news,
        ticker_counts={row["ticker"]: int(row["count"]) for row in rows},
    )


def _source_fallback_tickers(source: SourceConfig) -> list[str]:
    return sorted({ticker for ticker in source.tickers if ticker != "ALL"})


def _compile_term(term: str) -> _CompiledTerm:
    pattern = _term_pattern(term)
    return _CompiledTerm(
        term=term,
        pattern=re.compile(
            rf"(?<![{WORD_CHAR_CLASS}]){pattern}(?![{WORD_CHAR_CLASS}])",
            re.IGNORECASE,
        ),
    )


def _term_pattern(term: str) -> str:
    parts = []
    for char in term:
        if char.isspace():
            parts.append(r"\s+")
        elif char in {"е", "Е", "ё", "Ё"}:
            parts.append("[её]")
        else:
            parts.append(re.escape(char))
    return "".join(parts)
