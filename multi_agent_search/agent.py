"""LangGraph Studio entrypoint for Deep Research mode.

This file makes the deep research agent visible in LangGraph Studio.
Run: `langgraph dev` in the project root to launch the studio.
"""

from src.agents.deep_graph import graph

__all__ = ["graph"]
