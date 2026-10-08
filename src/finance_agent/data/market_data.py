import logging

import pandas as pd
import yfinance as yf
from curl_cffi import requests as cffi_requests
from tenacity import retry, stop_after_attempt, wait_exponential

from finance_agent.exceptions import TickerNotFoundError

logger = logging.getLogger(__name__)


def _yf_session() -> cffi_requests.Session:
    """Yahoo Finance blocks plain `requests`/urllib3 TLS fingerprints with 429s; impersonating
    a real browser's TLS handshake via curl_cffi routes around that bot check."""
    return cffi_requests.Session(impersonate="chrome")


def _ticker(symbol: str) -> yf.Ticker:
    return yf.Ticker(symbol, session=_yf_session())


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8), reraise=True)
def get_price_history(ticker: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    """Fetch OHLCV history for a ticker. Raises TickerNotFoundError if empty."""
    df = _ticker(ticker).history(period=period, interval=interval)
    if df.empty:
        raise TickerNotFoundError(f"No price data for ticker '{ticker}'")
    df = df.rename(columns=str.lower)
    return df


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8), reraise=True)
def get_quote_info(ticker: str) -> dict:
    """Fetch basic company/quote info. Returns {} on failure rather than raising."""
    try:
        info = _ticker(ticker).get_info()
    except Exception:
        logger.exception("Failed to fetch quote info for %s", ticker)
        return {}
    keys = [
        "shortName",
        "sector",
        "industry",
        "currency",
        "marketCap",
        "trailingPE",
        "forwardPE",
        "fiftyTwoWeekHigh",
        "fiftyTwoWeekLow",
        "currentPrice",
    ]
    return {k: info.get(k) for k in keys if k in info}
