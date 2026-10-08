import numpy as np
import pandas as pd
import pytest

from finance_agent.analysis.technical import compute_indicators


def _price_df(closes: list[float]) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=len(closes), freq="D")
    closes = np.array(closes, dtype=float)
    return pd.DataFrame(
        {
            "open": closes,
            "high": closes * 1.01,
            "low": closes * 0.99,
            "close": closes,
            "volume": np.full(len(closes), 1_000_000),
        },
        index=idx,
    )


def test_uptrend_is_bullish():
    closes = list(np.linspace(100, 160, 60))
    signal = compute_indicators("TEST", _price_df(closes))

    assert signal.trend_bias == "bullish"
    assert signal.rsi_14 is not None and signal.rsi_14 > 50
    assert signal.close == pytest.approx(closes[-1])


def test_downtrend_is_bearish():
    closes = list(np.linspace(160, 100, 60))
    signal = compute_indicators("TEST", _price_df(closes))

    assert signal.trend_bias == "bearish"
    assert signal.rsi_14 is not None and signal.rsi_14 < 50


def test_flat_series_is_neutral_and_handles_short_history():
    closes = [100.0] * 30  # shorter than the 50-period SMA window
    signal = compute_indicators("TEST", _price_df(closes))

    assert signal.sma_50 is None
    assert signal.trend_bias == "neutral"
