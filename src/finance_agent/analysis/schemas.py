from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field

Bias = Literal["bullish", "bearish", "neutral"]
Volatility = Literal["low", "medium", "high"]
SentimentLabel = Literal["positive", "neutral", "negative"]
Recommendation = Literal["BUY", "HOLD", "SELL"]


class TechnicalNarrative(BaseModel):
    """What the LLM is asked to produce — the numeric indicators are computed deterministically
    in Python and handed to the model as context, so the model only interprets, never invents numbers."""

    trend_bias: Bias
    volatility: Volatility
    notes: str = Field(description="2-4 sentence plain-English read of the indicators")


class TechnicalSignal(BaseModel):
    ticker: str
    close: float
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    rsi_14: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    bollinger_pct: Optional[float] = Field(default=None, description="0=lower band, 1=upper band")
    trend_bias: Bias = "neutral"
    volatility: Volatility = "medium"
    notes: str = ""


class SentimentResult(BaseModel):
    ticker: str
    sentiment_score: float = Field(ge=-1.0, le=1.0)
    sentiment_label: SentimentLabel
    key_drivers: list[str] = Field(default_factory=list)
    risk_flags: list[str] = Field(default_factory=list)
    sources_used: int = 0


class FinalReportDraft(BaseModel):
    """What the LLM is asked to produce. `ticker` and `generated_at` are filled in by code
    afterwards — leaving them in the schema lets the model hallucinate a wrong ticker/date."""

    recommendation: Recommendation
    confidence: float = Field(ge=0.0, le=1.0)
    price_target: Optional[float] = None
    rationale: str
    key_risks: list[str] = Field(default_factory=list)
    technical_summary: str
    sentiment_summary: str


class FinalReport(FinalReportDraft):
    ticker: str
    generated_at: datetime = Field(default_factory=datetime.utcnow)
