import operator
from typing import Annotated, Any, Optional, TypedDict

from finance_agent.analysis.schemas import FinalReport, SentimentResult, TechnicalSignal


class AgentState(TypedDict):
    ticker: str
    price_df: Any  # pandas.DataFrame, kept out of the type system to avoid a hard pandas import here
    quote_info: dict
    news_items: list[dict]
    filing_meta: list[dict]
    kb_chunks: int
    technical: Optional[TechnicalSignal]
    sentiment: Optional[SentimentResult]
    final_report: Optional[FinalReport]
    log: Annotated[list[str], operator.add]
    errors: Annotated[list[str], operator.add]
