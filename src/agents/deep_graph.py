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
from langgraph.prebuilt import create_react_agent

from ..config import settings
# Use DeepAgents built-in tools (no need to reimplement!)
from deepagents.tools import write_todos, read_file, write_file, ls, edit_file  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)


def setup_langsmith(project: str = "ai-research-deep") -> bool:
    """Configure LangSmith tracing env vars consistently."""
    if not settings.langsmith_api_key:
        return False

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
    model_string = settings.get_available_model()
    if settings.google_api_key:
        os.environ["GOOGLE_API_KEY"] = settings.google_api_key
    if settings.ollama_base_url:
        os.environ.setdefault("OLLAMA_HOST", settings.ollama_base_url)
    return init_chat_model(model_string, temperature=0.25)


DEEP_SYSTEM_PROMPT = """Sen Türkçe Deep Research Agent'ısın.

🎯 GÖREV AKIŞI:
1. **write_todos**: Karmaşık görevleri adımlara ayır (plan yap)
2. **web_search**: Güncel bilgi topla, kaynakları kontrol et
3. **write_file**: Büyük search sonuçlarını dosyaya kaydet (context overflow'u önle)
4. **Analiz ve Sentez**: Dosyadan oku, cross-check yap
5. **Final Report**: Profesyonel Markdown rapor yaz

🛠️ TOOLS (DeepAgents):
- write_todos([{"title": "adım1", "state": "pending"}, ...]) - Plan oluştur ve yönet
- web_search(query, limit, provider) - Web araması
- write_file(file_path, content) - Context'i dosyaya kaydet
- read_file(file_path) - Dosyadan oku
- ls(directory) - Workspace dosyalarını listele
- edit_file(file_path, old_string, new_string) - Dosya düzenle

📋 ÖRNEK WORKFLOW:
1. write_todos([{"title": "Web'de araştır", "state": "in_progress"}, {"title": "Rapor yaz", "state": "pending"}])
2. web_search("Python FastAPI best practices", limit=5)
3. write_file("research.md", "<search results>")  # Context'i koru
4. write_todos([{"title": "Web'de araştır", "state": "completed"}, {"title": "Rapor yaz", "state": "in_progress"}])  # Update status
5. read_file("research.md")  # Analiz için geri oku
6. Profesyonel rapor yaz
7. write_todos([{"title": "Web'de araştır", "state": "completed"}, {"title": "Rapor yaz", "state": "completed"}])

⚡ KURALLAR:
- Karmaşık görevde MUTLAKA write_todos ile başla
- Uzun search sonuçlarını write_file ile kaydet
- Kaynakları URL ile cite et
- Markdown format kullan
- write_todos her çağrıldığında TÜM task'ları güncelle (append değil, replace)
"""


# Exported graph for LangGraph Studio
setup_langsmith(project="ai-research-deep")
_model = _get_deep_model()
# Use DeepAgents built-in tools
_deepagent_tools = [write_todos, read_file, write_file, ls, edit_file]
_tools = [web_search] + _deepagent_tools  # Web search + DeepAgents planning + file tools

# Create ReAct agent with system prompt via `prompt` parameter
graph = create_react_agent(_model, _tools, prompt=DEEP_SYSTEM_PROMPT)
