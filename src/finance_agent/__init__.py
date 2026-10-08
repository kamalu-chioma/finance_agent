"""Agentic financial analyst: RAG + technical analysis + multi-agent LangGraph pipeline."""

import os

# Must run before chromadb is imported anywhere in the package (its telemetry call is broken
# in some versions — a capture() signature mismatch — which otherwise logs noisy errors).
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

__version__ = "0.1.0"
