"""Deterministic technical indicator math. No LLM calls here on purpose: numbers must be
reproducible and auditable. The LLM layer (agents/graph.py) only interprets these outputs."""

import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator
from ta.volatility import BollingerBands

from finance_agent.analysis.schemas import TechnicalSignal


def compute_indicators(ticker: str, price_df: pd.DataFrame) -> TechnicalSignal:
    """Compute SMA/RSI/MACD/Bollinger from an OHLCV dataframe (lowercase columns)."""
    close = price_df["close"]

    sma_20 = SMAIndicator(close, window=20).sma_indicator()
    sma_50 = SMAIndicator(close, window=50).sma_indicator() if len(close) >= 50 else pd.Series(dtype=float)
    rsi = RSIIndicator(close, window=14).rsi()
    macd_ind = MACD(close)
    macd_line = macd_ind.macd()
    macd_signal_line = macd_ind.macd_signal()
    bb = BollingerBands(close, window=20, window_dev=2)
    bb_high = bb.bollinger_hband()
    bb_low = bb.bollinger_lband()

    last_close = float(close.iloc[-1])
    last_sma20 = _last_or_none(sma_20)
    last_sma50 = _last_or_none(sma_50)
    last_rsi = _last_or_none(rsi)
    last_macd = _last_or_none(macd_line)
    last_macd_signal = _last_or_none(macd_signal_line)

    bb_pct = None
    if not bb_high.empty and not bb_low.empty:
        hi, lo = bb_high.iloc[-1], bb_low.iloc[-1]
        if hi != lo:
            bb_pct = float((last_close - lo) / (hi - lo))

    trend_bias = _rule_based_bias(last_close, last_sma20, last_sma50, last_rsi, last_macd, last_macd_signal)
    volatility = _rule_based_volatility(bb_pct, last_rsi)

    return TechnicalSignal(
        ticker=ticker,
        close=last_close,
        sma_20=last_sma20,
        sma_50=last_sma50,
        rsi_14=last_rsi,
        macd=last_macd,
        macd_signal=last_macd_signal,
        bollinger_pct=bb_pct,
        trend_bias=trend_bias,
        volatility=volatility,
    )


def _last_or_none(series: pd.Series) -> float | None:
    if series is None or series.empty:
        return None
    value = series.iloc[-1]
    return None if pd.isna(value) else float(value)


def _rule_based_bias(
    close: float,
    sma20: float | None,
    sma50: float | None,
    rsi: float | None,
    macd: float | None,
    macd_signal: float | None,
) -> str:
    score = 0
    if sma20 is not None and close > sma20:
        score += 1
    elif sma20 is not None:
        score -= 1
    if sma50 is not None:
        score += 1 if close > sma50 else -1
    if rsi is not None:
        if rsi > 60:
            score += 1
        elif rsi < 40:
            score -= 1
    if macd is not None and macd_signal is not None:
        score += 1 if macd > macd_signal else -1

    if score >= 2:
        return "bullish"
    if score <= -2:
        return "bearish"
    return "neutral"


def _rule_based_volatility(bollinger_pct: float | None, rsi: float | None) -> str:
    if bollinger_pct is None:
        return "medium"
    extreme = bollinger_pct <= 0.05 or bollinger_pct >= 0.95
    if extreme and rsi is not None and (rsi >= 70 or rsi <= 30):
        return "high"
    if extreme:
        return "medium"
    return "low"
