"""LangGraph-based Deep Research agent (ReAct pattern).

This is intended to match LangChain/LangGraph's proven agent pattern:
request -> model <-> tools -> result.

Uses DeepAgents library for planning and file system tools.
Exported symbol: `graph` (used by LangGraph Studio).
"""

from __future__ import annotations

import os
import json
import logging
from typing import Any, Dict, List, Optional

import httpx
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from deepagents import create_deep_agent

from ..config import settings

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


async def _firecrawl_search(query: str, limit: int = 5) -> Dict[str, Any]:
    api_key = settings.firecrawl_api_key
    if not api_key:
        return {"provider": "firecrawl", "text": "Firecrawl API key bulunamadı", "sources": []}

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            "https://api.firecrawl.dev/v1/search",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "query": query,
                "limit": int(limit),
                "scrapeOptions": {"formats": ["markdown"], "onlyMainContent": True},
            },
        )

    if response.status_code != 200:
        return {
            "provider": "firecrawl",
            "text": f"Firecrawl search failed: HTTP {response.status_code}",
            "sources": [],
        }

    data = response.json() or {}
    results = data.get("data", []) or []
    sources = []
    chunks = []
    for r in results:
        url = r.get("url") or r.get("link")
        title = r.get("title")
        if url:
            sources.append({"title": title, "url": url, "provider": "firecrawl"})
        snippet = r.get("markdown") or r.get("content") or r.get("description") or ""
        if snippet:
            chunks.append(f"- {title or url}: {snippet[:1200]}")

    text = "\n".join(chunks) if chunks else "Firecrawl: sonuç bulunamadı"
    return {"provider": "firecrawl", "text": text, "sources": sources}


async def _tavily_search(query: str, limit: int = 5) -> Dict[str, Any]:
    api_key = settings.tavily_api_key
    if not api_key:
        return {"provider": "tavily", "text": "Tavily API key bulunamadı", "sources": []}

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            "https://api.tavily.com/search",
            json={
                "api_key": api_key,
                "query": query,
                "max_results": int(limit),
                "include_raw_content": False,
            },
        )

    if response.status_code != 200:
        return {
            "provider": "tavily",
            "text": f"Tavily search failed: HTTP {response.status_code}",
            "sources": [],
        }

    data = response.json() or {}
    results = data.get("results", []) or []
    sources = []
    chunks = []
    for r in results:
        url = r.get("url")
        title = r.get("title")
        if url:
            sources.append({"title": title, "url": url, "provider": "tavily"})
        snippet = r.get("content") or ""
        if snippet:
            chunks.append(f"- {title or url}: {snippet[:1200]}")

    text = "\n".join(chunks) if chunks else "Tavily: sonuç bulunamadı"
    return {"provider": "tavily", "text": text, "sources": sources}


@tool
async def web_search(query: str, limit: int = 5, provider: str = "both") -> str:
    """Web search for deep research.

    Args:
        query: Search query
        limit: Max results per provider
        provider: one of: both | tavily | firecrawl

    Returns:
        JSON string with fields: { results: [..], sources: [..] }
    """
    provider = (provider or "both").lower().strip()

    sources: List[Dict[str, Any]] = []
    results: List[Dict[str, Any]] = []

    if provider in ("both", "firecrawl"):
        fc = await _firecrawl_search(query, limit=limit)
        results.append(fc)
        sources.extend(fc.get("sources") or [])

    if provider in ("both", "tavily"):
        tv = await _tavily_search(query, limit=limit)
        results.append(tv)
        sources.extend(tv.get("sources") or [])

    return json.dumps({"results": results, "sources": sources}, ensure_ascii=False)


def _get_deep_model():
    """Deep research kritik -> Groq tercih et"""
    if settings.groq_api_key:
        return init_chat_model(
            "llama-3.3-70b-versatile",
            model_provider="groq",
            temperature=0.25
        )
    
    # Fallback: available model
    model_string = settings.get_available_model()
    if settings.google_api_key:
        os.environ["GOOGLE_API_KEY"] = settings.google_api_key
    if settings.ollama_base_url:
        os.environ.setdefault("OLLAMA_HOST", settings.ollama_base_url)
    return init_chat_model(model_string, temperature=0.25)


DEEP_SYSTEM_PROMPT = """Sen Türkçe Deep Research Agent'ısın. Kullanıcıların sorularını derinlemesine araştırıp profesyonel raporlar yazıyorsun.

🎯 GÖREVLERİN:

1. **Planlama (write_todos kullan)**:
   - Karmaşık görevleri küçük adımlara böl
   - Her adımın durumunu takip et (pending → in_progress → completed)
   - Plan değiştikçe güncelle

2. **Araştırma (web_search kullan)**:
   - Güncel bilgi topla (web_search tool'unu kullan)
   - Çoklu kaynak tarama yap
   - Kaynakları mutlaka URL ile cite et

3. **Context Yönetimi (write_file, read_file kullan)**:
   - Uzun search sonuçlarını dosyaya kaydet (context overflow önle)
   - Gerektiğinde dosyadan geri oku ve analiz et
   - Kullanıcı dosya verirse: read_file ile oku, analiz et

4. **Subagent Delegation (task kullan)**:
   - Çok karmaşık alt görevleri subagent'a delege et
   - Context izolasyonu için kullan

5. **Final Report**:
   - Profesyonel Markdown formatında yaz
   - Kaynakları URL ile belirt
   - En az 1000 kelime, detaylı ve kapsamlı

🛠️ KULLANILABILIR TOOLS:
- `write_todos` - Task listesi oluştur/güncelle
- `read_file` - Dosya oku (kullanıcı dosyası veya workspace)
- `write_file` - Dosyaya kaydet
- `ls` - Dosyaları listele
- `edit_file` - Dosya düzenle
- `web_search` - Web araması (Firecrawl + Tavily)
- `task` - Subagent'a delege et

📂 DOSYA OKUMA:
Kullanıcı "bu dosyayı analiz et" derse:
1. read_file ile dosyayı oku
2. İçeriği analiz et
3. Bulguları rapor et

⚡ ÖNEMLI:
- Her zaman write_todos ile başla (planlama)
- Uzun tool output'larını write_file ile kaydet
- Context window'u temiz tut
- Markdown formatında profesyonel rapor yaz
- Kaynakları cite et
"""


# Exported graph for LangGraph Studio
setup_langsmith(project="ai-research-deep")
_model = _get_deep_model()

# Use DeepAgents create_deep_agent - includes built-in planning, file system, and subagents
# We only need to provide custom tools (web_search)
graph = create_deep_agent(
    model=_model,
    tools=[web_search],  # Custom tools - DeepAgents adds write_todos, read_file, write_file, ls, edit_file automatically
    system_prompt=DEEP_SYSTEM_PROMPT  # System prompt for agent (DeepAgents version compatibility)
)
