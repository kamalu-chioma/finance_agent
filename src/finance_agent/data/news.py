"""Yahoo Finance's XHR news API (yfinance's `.news`) has been returning 404s from Yahoo's side
regardless of auth/crumb handling, so this uses Yahoo's classic, still-functioning RSS feed
instead — same free data, no key, no bot-detection dance required."""

import logging

import requests
from defusedxml import ElementTree
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

_RSS_URL = "https://feeds.finance.yahoo.com/rss/2.0/headline"


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8), reraise=True)
def _fetch_rss(ticker: str) -> bytes:
    resp = requests.get(
        _RSS_URL,
        params={"s": ticker, "region": "US", "lang": "en-US"},
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.content


def get_recent_news(ticker: str, limit: int = 10) -> list[dict]:
    """Fetch recent news headlines for a ticker via Yahoo Finance's RSS feed. Never raises; returns [] on failure."""
    try:
        raw = _fetch_rss(ticker)
        root = ElementTree.fromstring(raw)
    except Exception:
        logger.exception("Failed to fetch news for %s", ticker)
        return []

    items = []
    for item in root.findall(".//item")[:limit]:
        title = item.findtext("title")
        if not title:
            continue
        items.append(
            {
                "title": title,
                "publisher": _source_from_link(item.findtext("link") or ""),
                "link": item.findtext("link") or "",
                "published": item.findtext("pubDate") or "",
                "summary": item.findtext("description") or "",
            }
        )
    return items


def _source_from_link(link: str) -> str:
    try:
        return link.split("//", 1)[1].split("/", 1)[0]
    except IndexError:
        return "unknown"
