import os
import logging
from langchain.chat_models import init_chat_model
from ...config import settings

logger = logging.getLogger(__name__)

def setup_langsmith(project: str = "ai-research-deep") -> bool:
    """Configure LangSmith tracing env vars consistently."""
    tracing_enabled = os.getenv("LANGSMITH_TRACING", "").strip().lower() in {"1", "true", "yes", "on"}
    if not tracing_enabled:
        return False
    if not settings.langsmith_api_key:
        return False

    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = project
    os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
    os.environ.setdefault("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
    return True

def get_ollama_model():
    """Force Ollama usage as requested."""
    # Ollama host configuration
    if settings.ollama_base_url:
        os.environ.setdefault("OLLAMA_HOST", settings.ollama_base_url)
    
    # Fallback/Default model name for Ollama
    # Trying generic 'llama3.1' or checking env
    model_name = os.getenv("OLLAMA_MODEL", "llama3.1") 
    
    logger.info(f"[DEEP] Using Ollama model: {model_name}")
    
    return init_chat_model(
        model_name,
        model_provider="ollama",
        temperature=0.0  # Low temp for analysis
    )
