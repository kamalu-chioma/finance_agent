import os
import sys

import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# On Streamlit Community Cloud, secrets are configured via the dashboard's secrets.toml editor
# (st.secrets), not a .env file. Mirror them into os.environ before importing finance_agent.config
# so the same pydantic Settings class works unchanged in both local dev and on Cloud.
try:
    for _key, _value in st.secrets.items():
        os.environ.setdefault(_key, str(_value))
except Exception:
    pass  # no secrets.toml present (e.g. local dev using .env) — fine

from finance_agent.agents.graph import build_graph, initial_state  # noqa: E402
from finance_agent.config import settings  # noqa: E402
from finance_agent.exceptions import TickerNotFoundError  # noqa: E402
from finance_agent.report import render_markdown  # noqa: E402

st.set_page_config(page_title="Agentic Financial Analyst", page_icon="📈", layout="wide")
st.title("📈 Agentic Financial Analyst")
st.caption("LangGraph multi-agent team: technical analysis + RAG-grounded sentiment over SEC filings & news.")

if not settings.openai_api_key:
    st.error("OPENAI_API_KEY is not set. Copy .env.example to .env and fill it in, then restart.")
    st.stop()

ticker = st.text_input("Ticker", value="AAPL").strip().upper()
run = st.button("Run analysis", type="primary")

if run and ticker:
    graph = build_graph()
    state = initial_state(ticker)
    log_box = st.status("Running agent pipeline...", expanded=True)

    try:
        for update in graph.stream(state):
            for node, delta in update.items():
                for line in delta.get("log", []):
                    log_box.write(f"**{node}** — {line}")
                state.update(delta)
        log_box.update(label="Pipeline complete", state="complete")
    except TickerNotFoundError as exc:
        log_box.update(label="Failed", state="error")
        st.error(str(exc))
        st.stop()

    report = state["final_report"]
    technical = state["technical"]
    sentiment = state["sentiment"]

    col1, col2, col3 = st.columns(3)
    col1.metric("Recommendation", report.recommendation)
    col2.metric("Confidence", f"{report.confidence:.0%}")
    col3.metric("Price target", report.price_target if report.price_target is not None else "—")

    st.subheader("Rationale")
    st.write(report.rationale)

    tech_col, sent_col = st.columns(2)
    with tech_col:
        st.subheader("Technical signal")
        st.write(f"Bias: **{technical.trend_bias}** · Volatility: **{technical.volatility}**")
        st.json(
            {
                "close": technical.close,
                "sma_20": technical.sma_20,
                "sma_50": technical.sma_50,
                "rsi_14": technical.rsi_14,
                "macd": technical.macd,
                "macd_signal": technical.macd_signal,
                "bollinger_pct": technical.bollinger_pct,
            }
        )
        st.caption(technical.notes)

    with sent_col:
        st.subheader("Sentiment signal")
        st.write(f"Score: **{sentiment.sentiment_score:+.2f}** ({sentiment.sentiment_label}), {sentiment.sources_used} sources")
        st.write("**Key drivers:**")
        for d in sentiment.key_drivers or ["none identified"]:
            st.write(f"- {d}")
        st.write("**Risk flags:**")
        for f in sentiment.risk_flags or ["none identified"]:
            st.write(f"- {f}")

    if state.get("errors"):
        with st.expander("Pipeline warnings"):
            for e in state["errors"]:
                st.write(f"- {e}")

    md = render_markdown(state)
    st.download_button("Download markdown report", md, file_name=f"{ticker}_report.md")
