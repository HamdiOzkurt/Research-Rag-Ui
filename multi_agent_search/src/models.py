"""
Models ve Helper Functions
Multi API Key Rotation desteği
"""
from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import logging
import os

logger = logging.getLogger(__name__)


# =============================================================================
# GROQ MODEL ALIASES (deprecations)
# =============================================================================

# Groq sometimes decommissions older model IDs. Map known deprecated IDs to
# recommended replacements to keep demos running.
GROQ_MODEL_ALIASES = {
    # Per Groq deprecations: llama-3.1-70b-versatile -> llama-3.3-70b-versatile
    "llama-3.1-70b-versatile": "llama-3.3-70b-versatile",
    "llama-3.1-70b-specdec": "llama-3.3-70b-specdec",
}


# =============================================================================
# DATA MODELS
# =============================================================================

class ResearchTask(BaseModel):
    """Araştırma görevi veri modeli"""
    query: str
    priority: int = 1
    depth: int = 3  # 1-5 arası derinlik
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Source(BaseModel):
    """Bulunan kaynak veri modeli"""
    url: str
    title: str
    content: str
    source_type: str  # web, reddit, github, etc.
    reliability_score: float = 0.0
    publish_date: Optional[str] = None
    key_points: List[str] = Field(default_factory=list)


class ResearchResult(BaseModel):
    """Nihai araştırma sonucu"""
    task: ResearchTask
    report: str
    sources: List[Source]
    metadata: Dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# HELPER FUNCTIONS - Multi API Key Support
# =============================================================================

def get_llm_model(retry_on_failure: bool = True, max_retries: int = 3):
    """
    LLM modelini döndürür - Multi API Key Rotation desteği
    
    Args:
        retry_on_failure: Hata durumunda başka key dene
        max_retries: Maksimum deneme sayısı
    """
    from src.config import settings
    from langchain.chat_models import init_chat_model
    
    model_string = settings.get_available_model()
    provider, model_name = settings.get_model_provider(model_string)
    
    if provider == "google_genai":
        api_key = settings.google_api_key
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
            key_count = len(settings.google_api_keys)
            key_index = settings._current_key_index + 1
            logger.info(f"[MODEL] Model: {model_name} (Key {key_index}/{key_count})")
        else:
            raise ValueError("Google API key not available")

    elif provider == "groq":
        api_key = getattr(settings, "groq_api_key", None)
        if api_key:
            os.environ["GROQ_API_KEY"] = api_key
            resolved_model_name = GROQ_MODEL_ALIASES.get(model_name, model_name)
            if resolved_model_name != model_name:
                logger.warning(
                    f"[MODEL] Groq model '{model_name}' deprecated; using '{resolved_model_name}'"
                )
            logger.info(f"[MODEL] Model: {resolved_model_name} (Groq)")
            try:
                from langchain_groq import ChatGroq
                return ChatGroq(model=resolved_model_name, api_key=api_key, temperature=0.3)
            except Exception as e:
                logger.warning(f"[WARN] ChatGroq init failed, falling back to init_chat_model: {e}")
        else:
            raise ValueError("Groq API key not available")
    
    elif provider == "ollama":
        os.environ["OLLAMA_HOST"] = settings.ollama_base_url
        logger.info(f"[MODEL] Model: {model_name} (Ollama Local)")
        return init_chat_model(model_string, temperature=0.3, model_kwargs={"num_gpu": -1})
    
    return init_chat_model(model_string, temperature=0.3)


def get_llm_model_with_retry(max_retries: int = 3):
    """
    LLM modelini retry logic ile döndürür
    429 hatası alınırsa sonraki key'e geçer
    """
    from src.config import settings
    
    for attempt in range(max_retries):
        try:
            model = get_llm_model()
            return model
        except Exception as e:
            error_msg = str(e)
            
            # 429 veya quota hatası
            if "429" in error_msg or "quota" in error_msg.lower() or "rate" in error_msg.lower():
                logger.warning(f"[WARN] Rate limit hit, rotating key (attempt {attempt + 1}/{max_retries})")
                settings.rotate_api_key(mark_failed=True)
            else:
                raise e
    
    raise ValueError("Tüm API key'ler denendi, hepsi başarısız!")


def sanitize_tool_schema(tool):
    """
    Tool schema'sını temizler ve bazı özel tool'lar için (örn. Firecrawl)
    argüman uyumsuzluklarını giderir.
    """
    # Ortak şema temizliği (Gemini vb. için)
    if hasattr(tool, "args_schema") and tool.args_schema:
        schema = tool.args_schema
        if hasattr(schema, "schema"):
            schema_dict = schema.schema()
            # Bazı modellerin sevmediği alanları kaldır
            if "$schema" in schema_dict:
                del schema_dict["$schema"]
            if "additionalProperties" in schema_dict:
                del schema_dict["additionalProperties"]

    # Firecrawl özel fix: sources.0 expected object, received string
    # DeepAgents / LLM bazen `sources: ["google"]` gibi string list veriyor.
    # Firecrawl MCP ise `[{ "source": "google" }]` bekliyor.
    try:
        if getattr(tool, "name", "") == "firecrawl_search":
            # Her iki invoke metodunu da patch'le
            for method_name in ["ainvoke", "invoke", "_run", "run"]:
                original_method = getattr(tool, method_name, None)
                
                if original_method is not None and callable(original_method):
                    import asyncio
                    
                    def _fix_sources(input_data):
                        """Sources parametresini düzelt"""
                        if isinstance(input_data, dict) and "sources" in input_data:
                            sources_val = input_data["sources"]
                            # Eğer ["google", "news"] gibi string list ise dönüştür
                            if isinstance(sources_val, list) and sources_val:
                                if isinstance(sources_val[0], str):
                                    input_data["sources"] = [
                                        {"source": s} for s in sources_val
                                    ]
                        return input_data
                    
                    if method_name.startswith("a"):  # async method
                        async def patched_async(
                            input_data, *args, _orig=original_method, **kwargs
                        ):
                            input_data = _fix_sources(input_data)
                            return await _orig(input_data, *args, **kwargs)
                        setattr(tool, method_name, patched_async)
                    else:  # sync method
                        def patched_sync(
                            input_data, *args, _orig=original_method, **kwargs
                        ):
                            input_data = _fix_sources(input_data)
                            return _orig(input_data, *args, **kwargs)
                        setattr(tool, method_name, patched_sync)
    except Exception as e:
        # Bu yardımcı fonksiyon, hata yüzünden tüm agent'ı patlatmamalı.
        import logging
        logging.getLogger(__name__).warning(f"[WARN] Tool patch failed for {getattr(tool, 'name', 'unknown')}: {e}")
        pass

    return tool


def rotate_key_on_error():
    """429 hatası sonrası key'i rotate et"""
    from src.config import settings
    settings.rotate_api_key(mark_failed=True)
