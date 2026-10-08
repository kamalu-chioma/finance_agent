"""Free SEC EDGAR access: ticker -> CIK lookup, recent filings, and filing text extraction.

All requests must set a descriptive User-Agent per SEC's fair-access policy
(https://www.sec.gov/os/accessing-edgar-data).
"""

import logging
import re

import requests
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential

from finance_agent.config import settings

logger = logging.getLogger(__name__)

_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
_ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_nodash}/{doc}"

_ticker_to_cik_cache: dict[str, str] | None = None


def _headers() -> dict:
    return {"User-Agent": settings.sec_user_agent}


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8), reraise=True)
def _load_ticker_to_cik() -> dict[str, str]:
    global _ticker_to_cik_cache
    if _ticker_to_cik_cache is not None:
        return _ticker_to_cik_cache
    resp = requests.get(_TICKERS_URL, headers=_headers(), timeout=15)
    resp.raise_for_status()
    data = resp.json()
    rows = data.values() if isinstance(data, dict) else data
    _ticker_to_cik_cache = {row["ticker"].upper(): str(row["cik_str"]).zfill(10) for row in rows}
    return _ticker_to_cik_cache


def get_cik(ticker: str) -> str | None:
    try:
        mapping = _load_ticker_to_cik()
    except Exception:
        logger.exception("Failed to load SEC ticker->CIK mapping")
        return None
    return mapping.get(ticker.upper())


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8), reraise=True)
def get_recent_filings(ticker: str, forms: tuple[str, ...] = ("10-K", "10-Q", "8-K"), limit: int = 5) -> list[dict]:
    """Return recent filing metadata for a ticker. Returns [] if ticker/CIK unresolvable."""
    cik = get_cik(ticker)
    if not cik:
        return []
    resp = requests.get(_SUBMISSIONS_URL.format(cik=cik), headers=_headers(), timeout=15)
    resp.raise_for_status()
    data = resp.json()
    recent = data.get("filings", {}).get("recent", {})
    out = []
    for form, acc, date, doc in zip(
        recent.get("form", []),
        recent.get("accessionNumber", []),
        recent.get("filingDate", []),
        recent.get("primaryDocument", []),
    ):
        if form in forms:
            out.append({"form": form, "accessionNumber": acc, "filingDate": date, "primaryDocument": doc, "cik": cik})
        if len(out) >= limit:
            break
    return out


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8), reraise=True)
def get_filing_text(cik: str, accession_number: str, primary_document: str, max_chars: int = 20000) -> str:
    """Download a filing document and return cleaned plain text, truncated to max_chars."""
    cik_int = str(int(cik))
    accession_nodash = accession_number.replace("-", "")
    url = _ARCHIVE_URL.format(cik_int=cik_int, accession_nodash=accession_nodash, doc=primary_document)
    resp = requests.get(url, headers=_headers(), timeout=20)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.content, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator=" ")
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars]
