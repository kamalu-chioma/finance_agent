class FinanceAgentError(Exception):
    """Base error for the finance agent package."""


class TickerNotFoundError(FinanceAgentError):
    """Raised when a ticker has no price data."""


class DataSourceError(FinanceAgentError):
    """Raised when an external data source (SEC, Yahoo) fails after retries."""
