from .deep_researcher import graph
from .configuration import Configuration
import os

def setup_langsmith(project: str = None):
    """Configure LangSmith tracing."""
    # Ensure environment variables are set if they exist in settings
    # This is a compatibility shim. 
    # In a real setup, we might set os.environ["LANGCHAIN_PROJECT"] = project
    if project:
        os.environ["LANGCHAIN_PROJECT"] = project

__all__ = ["graph", "Configuration", "setup_langsmith"]
