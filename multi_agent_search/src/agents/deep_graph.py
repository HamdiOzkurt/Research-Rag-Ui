"""Legacy entry point for Deep Research.
Redirects to new modular structure in `deep_research/`.
"""
# Use the FIXED deep researcher architecture
from src.agents.deep_research.deep_researcher import graph

def setup_langsmith(project: str = None):
    """Compatibility shim."""
    import os
    if project:
        os.environ["LANGCHAIN_PROJECT"] = project

# Re-export key components
__all__ = ["graph", "setup_langsmith"]
