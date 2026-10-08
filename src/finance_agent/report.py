import os

from finance_agent.agents.state import AgentState


def render_markdown(state: AgentState) -> str:
    t = state["technical"]
    s = state["sentiment"]
    r = state["final_report"]
    quote = state.get("quote_info", {})

    lines = [
        f"# {r.ticker} — Analyst Report",
        "",
        f"_Generated {r.generated_at.isoformat()}Z_",
        "",
        f"## Recommendation: {r.recommendation} (confidence {r.confidence:.0%})",
        "",
        r.rationale,
        "",
    ]
    if r.price_target is not None:
        lines.append(f"**Price target:** {r.price_target}")
        lines.append("")

    if quote:
        lines += ["## Snapshot", ""]
        for k, v in quote.items():
            lines.append(f"- **{k}:** {v}")
        lines.append("")

    lines += [
        "## Technical View",
        f"- Trend bias: **{t.trend_bias}** | Volatility: **{t.volatility}**",
        f"- Close: {t.close} | SMA20: {t.sma_20} | SMA50: {t.sma_50} | RSI14: {t.rsi_14}",
        f"- MACD: {t.macd} (signal {t.macd_signal}) | Bollinger %B: {t.bollinger_pct}",
        "",
        t.notes,
        "",
        "## Sentiment View",
        f"- Score: **{s.sentiment_score:+.2f}** ({s.sentiment_label}) from {s.sources_used} sources",
        "",
        "**Key drivers:**",
    ]
    lines += [f"- {d}" for d in s.key_drivers] or ["- none identified"]
    lines += ["", "**Risk flags:**"]
    risk_flags = list(dict.fromkeys(r.key_risks + s.risk_flags))
    lines += [f"- {flag}" for flag in risk_flags] or ["- none identified"]

    if state.get("errors"):
        lines += ["", "## Pipeline Warnings", ""]
        lines += [f"- {e}" for e in state["errors"]]

    return "\n".join(lines)


def save_report(ticker: str, content: str, output_dir: str = "output") -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{ticker.upper()}_report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path
