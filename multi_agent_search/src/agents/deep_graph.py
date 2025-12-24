"""Legacy entry point for Deep Research.
Redirects to new modular structure in `deep_research/`.
"""
# Temporarily use simple test graph for debugging
from .deep_research.simple_test_graph import graph

def setup_langsmith(project: str = None):
    """Compatibility shim."""
    pass

# Re-export key components
__all__ = ["graph", "setup_langsmith"]
