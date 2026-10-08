# Agentic Financial Analyst

A small multi-agent research pipeline that pulls live stock data, grounds itself in recent news
and SEC filings via RAG, runs deterministic technical analysis, and has a team of LLM agents
(technical analyst, sentiment/fundamentals analyst, synthesizing portfolio manager) collaborate
over **LangGraph** to produce a structured BUY/HOLD/SELL report.

## Architecture

```
                ┌──────────────┐
                │  fetch_data   │  yfinance price/news, SEC EDGAR filings,
                │ (no LLM call) │  builds a per-ticker Chroma knowledge base
                └───────┬──────┘
                        │
            ┌───────────┴───────────┐
            ▼                       ▼
  ┌──────────────────┐   ┌───────────────────┐
  │ technical_agent    │   │ sentiment_agent    │
  │ deterministic math │   │ RAG retrieval over │
  │ + LLM interprets    │   │ news/filings + LLM │
  └─────────┬──────────┘   └──────────┬─────────┘
            └────────────┬────────────┘
                          ▼
                ┌──────────────────┐
                │ synthesis_agent    │  weighs both views into one
                │ (LLM, structured)  │  FinalReport (recommendation,
                └──────────────────┘  confidence, price target, risks)
```

Each LLM node returns a validated **Pydantic** model (`with_structured_output`), not free text —
downstream code never parses prose. Every external call (Yahoo Finance, SEC EDGAR, OpenAI) is
wrapped with retries and a degraded fallback, and the pipeline accumulates non-fatal errors into
the final report instead of crashing on a flaky API.

Numbers you can audit (SMA/RSI/MACD/Bollinger) are computed in plain pandas — the LLM is only ever
asked to *interpret* those numbers, never to recompute them, which keeps the technical view
reproducible.

## Stack

- **LangGraph** — multi-agent orchestration with a parallel fan-out/fan-in (`fetch_data` →
  `{technical_agent, sentiment_agent}` → `synthesis_agent`)
- **LangChain + OpenAI** — structured-output LLM calls, `OpenAIEmbeddings`
- **Chroma** — local vector store for the RAG layer (no external infra)
- **yfinance** — price history, quote info, news headlines (free)
- **SEC EDGAR** (`data.sec.gov`, `sec.gov/Archives`) — 10-K/10-Q/8-K filings (free, no key)
- **pandas + ta** — technical indicators
- **Streamlit** — web UI; **Rich** — CLI output

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
cp .env.example .env            # then fill in OPENAI_API_KEY
```

## Usage

CLI:

```bash
python -m finance_agent AAPL
```

Writes a markdown report to `output/AAPL_report.md` and prints a live log of each agent's step.

Web UI:

```bash
streamlit run app/streamlit_app.py
```

## Tests

```bash
pytest
ruff check src app tests
```

Technical-indicator tests run on synthetic price series (no network). Market-data tests mock
`yfinance` so CI doesn't depend on live APIs.

## Design notes / trade-offs

- **Free data sources only** — no NewsAPI/Alpha Vantage key required. News comes from Yahoo
  Finance headlines; filings come straight from SEC EDGAR's public JSON + document archive.
- **RAG scope** — only the latest few 10-K/10-Q/8-K documents are pulled and chunked per ticker,
  truncated to a bounded character count, to keep embedding cost and latency predictable.
- **Deterministic math / LLM-for-judgment split** — technical indicators are never left to the
  model; sentiment scoring is left to the model because it's inherently a judgment call, but it's
  instructed to ground itself only in retrieved context and say so when context is thin.
- **Known limitation** — this is a research/demo tool, not investment advice, and the free news
  feed is less comprehensive than a paid API.
