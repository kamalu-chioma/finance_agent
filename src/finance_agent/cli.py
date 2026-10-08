import argparse
import logging
import sys

from rich.console import Console
from rich.table import Table

from finance_agent.agents.graph import build_graph, initial_state
from finance_agent.config import settings
from finance_agent.exceptions import TickerNotFoundError
from finance_agent.report import render_markdown, save_report

console = Console()


def main() -> int:
    parser = argparse.ArgumentParser(description="Agentic financial analyst")
    parser.add_argument("ticker", help="Stock ticker, e.g. AAPL")
    parser.add_argument("--output", default="output", help="Directory to write the markdown report to")
    parser.add_argument("--verbose", action="store_true", help="Show debug logging")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING)

    if not settings.openai_api_key:
        console.print("[red]OPENAI_API_KEY is not set. Copy .env.example to .env and fill it in.[/red]")
        return 1

    graph = build_graph()
    state = initial_state(args.ticker)

    console.print(f"[bold]Running multi-agent analysis for {args.ticker.upper()}[/bold]")
    try:
        for update in graph.stream(state):
            for node, delta in update.items():
                for line in delta.get("log", []):
                    console.print(f"  [cyan]{node}[/cyan]: {line}")
                state.update(delta)
    except TickerNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        return 1

    final_state = state
    report = final_state["final_report"]

    table = Table(title=f"{report.ticker} — {report.recommendation} ({report.confidence:.0%} confidence)")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Technical bias", final_state["technical"].trend_bias)
    table.add_row("Sentiment", f"{final_state['sentiment'].sentiment_label} ({final_state['sentiment'].sentiment_score:+.2f})")
    table.add_row("Price target", str(report.price_target))
    console.print(table)
    console.print(report.rationale)

    content = render_markdown(final_state)
    path = save_report(args.ticker, content, args.output)
    console.print(f"[green]Report saved to {path}[/green]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
