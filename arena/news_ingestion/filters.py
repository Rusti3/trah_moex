ALLOW_PATTERNS = [
    "/news",
    "/press",
    "/press-release",
    "/press-releases",
    "/press-center",
    "/media",
    "/media-center",
    "/news-and-media",
    "/investors/news",
    "/investors/disclosure",
    "/ir/news",
    "/ir-releases",
    "/disclosure",
    "/releases",
    "/corporate_releases",
    "/investments/news",
    "/about/news-and-reports/news",
]

BLOCK_PATTERNS = [
    "/quote",
    "/quotes",
    "/stocks",
    "/chart",
    "/trading",
    "/marketdata",
    "/securities",
    "/candles",
    "/orderbook",
]

BAD_TEXT_PHRASES = [
    "открыть график",
    "стакан заявок",
    "изменение за день",
]


def is_news_url(url: str) -> bool:
    lowered = url.lower()
    if any(pattern in lowered for pattern in BLOCK_PATTERNS):
        return False
    return any(pattern in lowered for pattern in ALLOW_PATTERNS)


def is_valid_news_text(text: str | None, min_length: int = 300) -> bool:
    if not text:
        return False

    clean = text.strip()
    if len(clean) < min_length:
        return False

    lowered = clean.lower()
    return not any(phrase in lowered for phrase in BAD_TEXT_PHRASES)
