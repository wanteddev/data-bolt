"""LangGraph CLI entrypoints for local dev server."""

from __future__ import annotations

from data_bolt.tasks.bigquery_agent import graph as legacy_graph_module
from data_bolt.tasks.bigquery_agent import loop_runtime

# Default graph for local dev mirrors production default runtime.
loop_graph = loop_runtime._build_graph().compile()

# Legacy graph is kept for rollback and side-by-side inspection.
legacy_graph = legacy_graph_module._build_graph().compile()
