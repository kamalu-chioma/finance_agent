import pandas as pd
import pytest

from finance_agent.data import market_data
from finance_agent.exceptions import TickerNotFoundError


def test_get_price_history_raises_on_empty(mocker):
    fake_ticker = mocker.Mock()
    fake_ticker.history.return_value = pd.DataFrame()
    mocker.patch("finance_agent.data.market_data.yf.Ticker", return_value=fake_ticker)

    with pytest.raises(TickerNotFoundError):
        market_data.get_price_history("BADTICKER")


def test_get_price_history_lowercases_columns(mocker):
    fake_ticker = mocker.Mock()
    fake_ticker.history.return_value = pd.DataFrame({"Close": [1.0, 2.0], "Open": [1.0, 2.0]})
    mocker.patch("finance_agent.data.market_data.yf.Ticker", return_value=fake_ticker)

    df = market_data.get_price_history("AAPL")
    assert "close" in df.columns
    assert "open" in df.columns


def test_get_quote_info_returns_empty_dict_on_failure(mocker):
    mocker.patch("finance_agent.data.market_data.yf.Ticker", side_effect=RuntimeError("boom"))

    assert market_data.get_quote_info("AAPL") == {}
