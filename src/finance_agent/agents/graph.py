import logging
from functools import lru_cache

from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from finance_agent.agents import prompts
from finance_agent.agents.state import AgentState
from finance_agent.analysis.schemas import (
    FinalReport,
    FinalReportDraft,
    SentimentResult,
    TechnicalNarrative,
    TechnicalSignal,
)
from finance_agent.analysis.technical import compute_indicators
from finance_agent.config import settings
from finance_agent.data import filings, market_data, news
from finance_agent.rag.ingest import build_knowledge_base
from finance_agent.rag.retriever import retrieve_context

logger = logging.getLogger(__name__)

MAX_FILINGS_FOR_RAG = 3


def get_llm(temperature: float = 0.0) -> ChatOpenAI:
    return ChatOpenAI(model=settings.openai_model, temperature=temperature, api_key=settings.openai_api_key)


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def fetch_data(state: AgentState) -> dict:
    ticker = state["ticker"]
    errors: list[str] = []
    log = [f"Fetching price history for {ticker}..."]

    price_df = market_data.get_price_history(ticker)  # raises on truly invalid ticker

    try:
        quote_info = market_data.get_quote_info(ticker)
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("quote_info failed")
        quote_info, errors = {}, errors + [f"quote_info: {exc}"]

    try:
        news_items = news.get_recent_news(ticker)
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("news fetch failed")
        news_items, errors = [], errors + [f"news: {exc}"]

    filing_meta: list[dict] = []
    filing_docs: list[dict] = []
    try:
        filing_meta = filings.get_recent_filings(ticker)
        for meta in filing_meta[:MAX_FILINGS_FOR_RAG]:
            try:
                text = filings.get_filing_text(meta["cik"], meta["accessionNumber"], meta["primaryDocument"])
                filing_docs.append({**meta, "text": text})
            except Exception as exc:
                errors.append(f"filing_text({meta.get('form')}): {exc}")
    except Exception as exc:
        logger.exception("filings fetch failed")
        errors.append(f"filings: {exc}")

    log.append(f"News items: {len(news_items)}. Filings pulled for RAG: {len(filing_docs)}.")

    try:
        kb_chunks = build_knowledge_base(ticker, news_items, filing_docs)
        log.append(f"Knowledge base built: {kb_chunks} chunks.")
    except Exception as exc:
        logger.exception("knowledge base build failed")
        kb_chunks = 0
        errors.append(f"rag_ingest: {exc}")

    return {
        "price_df": price_df,
        "quote_info": quote_info,
        "news_items": news_items,
        "filing_meta": filing_meta,
        "kb_chunks": kb_chunks,
        "log": log,
        "errors": errors,
    }


def technical_agent(state: AgentState) -> dict:
    ticker = state["ticker"]
    signal: TechnicalSignal = compute_indicators(ticker, state["price_df"])

    try:
        llm = get_llm().with_structured_output(TechnicalNarrative)
        narrative: TechnicalNarrative = llm.invoke(
            [
                ("system", prompts.TECHNICAL_SYSTEM_PROMPT),
                (
                    "user",
                    prompts.TECHNICAL_USER_TEMPLATE.format(
                        ticker=ticker,
                        close=signal.close,
                        sma_20=signal.sma_20,
                        sma_50=signal.sma_50,
                        rsi_14=signal.rsi_14,
                        macd=signal.macd,
                        macd_signal=signal.macd_signal,
                        bollinger_pct=signal.bollinger_pct,
                        trend_bias=signal.trend_bias,
                        volatility=signal.volatility,
                    ),
                ),
            ]
        )
        signal.trend_bias = narrative.trend_bias
        signal.volatility = narrative.volatility
        signal.notes = narrative.notes
        log = [f"Technical agent: {signal.trend_bias}/{signal.volatility}."]
        errors: list[str] = []
    except Exception as exc:
        logger.exception("technical LLM narrative failed, keeping rule-based bias")
        signal.notes = "LLM narrative unavailable; showing rule-based read only."
        log = ["Technical agent: LLM narrative failed, used rule-based fallback."]
        errors = [f"technical_agent: {exc}"]

    return {"technical": signal, "log": log, "errors": errors}


def sentiment_agent(state: AgentState) -> dict:
    ticker = state["ticker"]
    docs = retrieve_context(ticker, query=f"{ticker} outlook, risks, and recent developments", k=6)

    if not docs:
        result = SentimentResult(
            ticker=ticker,
            sentiment_score=0.0,
            sentiment_label="neutral",
            key_drivers=[],
            risk_flags=["No news or filing context available for this run."],
            sources_used=0,
        )
        return {"sentiment": result, "log": ["Sentiment agent: no RAG context, returned neutral."], "errors": []}

    context = "\n\n".join(f"[{d.metadata.get('type')}] {d.page_content[:600]}" for d in docs)

    try:
        llm = get_llm().with_structured_output(SentimentResult)
        result: SentimentResult = llm.invoke(
            [
                ("system", prompts.SENTIMENT_SYSTEM_PROMPT),
                (
                    "user",
                    prompts.SENTIMENT_USER_TEMPLATE.format(ticker=ticker, n_sources=len(docs), context=context),
                ),
            ]
        )
        result.ticker = ticker
        result.sources_used = len(docs)
        log = [f"Sentiment agent: {result.sentiment_label} ({result.sentiment_score:+.2f}) from {len(docs)} sources."]
        errors: list[str] = []
    except Exception as exc:
        logger.exception("sentiment LLM call failed")
        result = SentimentResult(
            ticker=ticker,
            sentiment_score=0.0,
            sentiment_label="neutral",
            risk_flags=["Sentiment LLM call failed; treat as unknown, not neutral-confirmed."],
            sources_used=len(docs),
        )
        log = ["Sentiment agent: LLM call failed, returned degraded neutral result."]
        errors = [f"sentiment_agent: {exc}"]

    return {"sentiment": result, "log": log, "errors": errors}


def synthesis_agent(state: AgentState) -> dict:
    ticker = state["ticker"]
    technical = state["technical"]
    sentiment = state["sentiment"]

    try:
        llm = get_llm().with_structured_output(FinalReportDraft)
        draft: FinalReportDraft = llm.invoke(
            [
                ("system", prompts.SYNTHESIS_SYSTEM_PROMPT),
                (
                    "user",
                    prompts.SYNTHESIS_USER_TEMPLATE.format(
                        ticker=ticker,
                        close=technical.close,
                        trend_bias=technical.trend_bias,
                        volatility=technical.volatility,
                        technical_notes=technical.notes,
                        sentiment_score=sentiment.sentiment_score,
                        sentiment_label=sentiment.sentiment_label,
                        key_drivers=sentiment.key_drivers,
                        risk_flags=sentiment.risk_flags,
                    ),
                ),
            ]
        )
        report = FinalReport(ticker=ticker, **draft.model_dump())
        log = [f"Synthesis agent: {report.recommendation} (confidence {report.confidence:.0%})."]
        errors: list[str] = []
    except Exception as exc:
        logger.exception("synthesis LLM call failed")
        report = FinalReport(
            ticker=ticker,
            recommendation="HOLD",
            confidence=0.1,
            rationale="Synthesis LLM call failed; defaulting to HOLD with low confidence.",
            key_risks=["Synthesis step failed — treat this report as incomplete."],
            technical_summary=technical.notes or technical.trend_bias,
            sentiment_summary=sentiment.sentiment_label,
        )
        log = ["Synthesis agent: LLM call failed, returned degraded HOLD."]
        errors = [f"synthesis_agent: {exc}"]

    return {"final_report": report, "log": log, "errors": errors}


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("fetch_data", fetch_data)
    graph.add_node("technical_agent", technical_agent)
    graph.add_node("sentiment_agent", sentiment_agent)
    graph.add_node("synthesis_agent", synthesis_agent)

    graph.set_entry_point("fetch_data")
    graph.add_edge("fetch_data", "technical_agent")
    graph.add_edge("fetch_data", "sentiment_agent")
    graph.add_edge("technical_agent", "synthesis_agent")
    graph.add_edge("sentiment_agent", "synthesis_agent")
    graph.add_edge("synthesis_agent", END)

    return graph.compile()


def initial_state(ticker: str) -> AgentState:
    return AgentState(
        ticker=ticker.upper(),
        price_df=None,
        quote_info={},
        news_items=[],
        filing_meta=[],
        kb_chunks=0,
        technical=None,
        sentiment=None,
        final_report=None,
        log=[],
        errors=[],
    )


def run_analysis(ticker: str) -> AgentState:
    """Run the full pipeline synchronously and return the final state."""
    graph = build_graph()
    return graph.invoke(initial_state(ticker))
