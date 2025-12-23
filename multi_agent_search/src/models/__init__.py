"""Models package for type-safe RAG system"""

# ✅ RAG Models (Pydantic AI)
from .rag_models import (
    ChunkMetadata,
    ChunkContent,
    ChunkDecision,
    ChunkSummary,
    ChunkTitle,
    SearchParams,
    RetrievedChunk,
    RetrievalResult,
    VisionAnalysis,
    RAGError,
)

# ✅ BACKWARD COMPATIBILITY: Re-export from old models.py location
# These are imported by simple_agent.py and other modules
import sys
import os

# Import from the OLD models.py file (now at parent level)
# We need to import it directly since it's not a package
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_old_models_path = os.path.join(_parent_dir, "models.py")

if os.path.exists(_old_models_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("_old_models", _old_models_path)
    _old_models = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_old_models)
    
    # Re-export the functions
    get_llm_model = _old_models.get_llm_model
    get_llm_model_with_retry = _old_models.get_llm_model_with_retry
    sanitize_tool_schema = _old_models.sanitize_tool_schema
    rotate_key_on_error = _old_models.rotate_key_on_error
    
    # Re-export data models
    ResearchTask = _old_models.ResearchTask
    Source = _old_models.Source
    ResearchResult = _old_models.ResearchResult
    GROQ_MODEL_ALIASES = _old_models.GROQ_MODEL_ALIASES

__all__ = [
    # Pydantic AI models
    "ChunkMetadata",
    "ChunkContent",
    "ChunkDecision",
    "ChunkSummary",
    "ChunkTitle",
    "SearchParams",
    "RetrievedChunk",
    "RetrievalResult",
    "VisionAnalysis",
    "RAGError",
    # Legacy exports (backward compatibility)
    "get_llm_model",
    "get_llm_model_with_retry",
    "sanitize_tool_schema",
    "rotate_key_on_error",
    "ResearchTask",
    "Source",
    "ResearchResult",
    "GROQ_MODEL_ALIASES",
]

