from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

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
# HELPER FUNCTIONS
# =============================================================================

def get_llm_model():
    """LLM modelini döndürür - Rate limit için optimize edilmiş"""
    import os
    from src.config import settings
    from langchain.chat_models import init_chat_model
    
    model_string = settings.get_available_model()
    provider, model_name = settings.get_model_provider(model_string)
    
    print(f"🤖 Model: {provider}:{model_name}")
    
    if provider == "google_genai" and settings.google_api_key:
        os.environ["GOOGLE_API_KEY"] = settings.google_api_key
    elif provider == "ollama":
        os.environ["OLLAMA_HOST"] = settings.ollama_base_url
    
    return init_chat_model(model_string, temperature=0.3)


def sanitize_tool_schema(tool):
    """Tool schema'sını Gemini uyumlu hale getirir"""
    if hasattr(tool, 'args_schema') and tool.args_schema:
        schema = tool.args_schema
        if hasattr(schema, 'schema'):
            schema_dict = schema.schema()
            # Gemini desteklemeyen alanları kaldır
            if '$schema' in schema_dict:
                del schema_dict['$schema']
            if 'additionalProperties' in schema_dict:
                del schema_dict['additionalProperties']
    return tool
