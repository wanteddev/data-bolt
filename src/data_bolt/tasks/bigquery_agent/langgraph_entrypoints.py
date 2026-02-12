"""LangGraph CLI entrypoints for local dev server."""

from __future__ import annotations

from data_bolt.tasks.bigquery_agent import loop_runtime

# Project's only runtime graph.
loop_graph = loop_runtime._build_graph().compile()
