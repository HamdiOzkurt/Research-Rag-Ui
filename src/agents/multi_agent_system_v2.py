"""
Multi-Agent Research System - Simplified Version
Direct LLM calls without complex tool calling (Ollama compatible)
"""

import os
import asyncio
import httpx
import json
import uuid
from typing import Optional, AsyncGenerator, Any
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from src.config import settings
from src.models import get_llm_model, sanitize_tool_schema
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============ LANGSMITH ============
def setup_langsmith():
    """LangSmith tracing'i multi-agent için aktifleştir"""
    if settings.langsmith_api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = "ai-research-multi-agent-v2"
        os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
        os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
        logger.info("[LANGSMITH] Multi-Agent v2 aktif - https://smith.langchain.com/o/personal/projects/p/ai-research-multi-agent-v2")
        return True
    return False


# =============================================================================
# FIRECRAWL DIRECT API (MCP yerine doğrudan API çağrısı)
# =============================================================================

async def firecrawl_search(query: str, limit: int = 5) -> dict:
    """Firecrawl API ile doğrudan web araması.

    Returns a dict with:
      - provider: str
      - text: str (LLM-facing)
      - sources: list[{title,url}]
    """
    api_key = settings.firecrawl_api_key
    if not api_key:
        return {"provider": "firecrawl", "text": "Firecrawl API key bulunamadı", "sources": []}
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://api.firecrawl.dev/v1/search",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "query": query,
                    "limit": limit,
                    "scrapeOptions": {
                        "formats": ["markdown"],
                        "onlyMainContent": True
                    }
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                results = data.get("data", [])
                if not results:
                    return {"provider": "firecrawl", "text": f"'{query}' için sonuç bulunamadı", "sources": []}
                
                # Sonuçları formatla
                output = []
                sources = []
                for i, item in enumerate(results[:limit], 1):
                    title = item.get("title", "Başlık yok")
                    url = item.get("url", "")
                    content = item.get("markdown", item.get("description", ""))[:1500]
                    if url:
                        sources.append({"title": title, "url": url})
                    output.append(f"### {i}. {title}\nURL: {url}\n\n{content}\n")

                return {"provider": "firecrawl", "text": "\n---\n".join(output), "sources": sources}
            else:
                logger.warning(f"Firecrawl API hatası: {response.status_code}")
                return {"provider": "firecrawl", "text": f"Firecrawl API hatası: {response.status_code}", "sources": []}
    except Exception as e:
        logger.error(f"Firecrawl hatası: {e}")
        return {"provider": "firecrawl", "text": f"Arama hatası: {str(e)}", "sources": []}


async def tavily_search(query: str, limit: int = 5) -> dict:
    """Tavily API ile web araması (yedek).

    Returns a dict with:
      - provider: str
      - text: str (LLM-facing)
      - sources: list[{title,url}]
    """
    api_key = settings.tavily_api_key if hasattr(settings, 'tavily_api_key') else os.getenv("TAVILY_API_KEY")
    if not api_key:
        return {"provider": "tavily", "text": "", "sources": []}
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": api_key,
                    "query": query,
                    "max_results": limit,
                    "include_answer": True
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])
                answer = data.get("answer", "")
                
                output = []
                sources = []
                if answer:
                    output.append(f"**Özet:** {answer}\n")
                
                for i, item in enumerate(results[:limit], 1):
                    title = item.get("title", "")
                    url = item.get("url", "")
                    content = item.get("content", "")[:800]
                    if url:
                        sources.append({"title": title, "url": url})
                    output.append(f"### {i}. {title}\nURL: {url}\n\n{content}\n")

                return {"provider": "tavily", "text": "\n---\n".join(output), "sources": sources}
    except Exception as e:
        logger.warning(f"Tavily hatası: {e}")
    
    return {"provider": "tavily", "text": "", "sources": []}


# =============================================================================
# HYBRID LLM WRAPPER
# - Groq: Router / tool selection / final synthesis
# - Ollama: Small steps (research draft, code draft)
# =============================================================================

_groq_model = None
_local_model = None


def _get_groq_model():
    """Groq model lazy loading (router + final synthesis)."""
    global _groq_model
    if _groq_model is not None:
        return _groq_model

    provider, model_name = settings.get_model_provider(settings.default_model)
    if provider != "groq":
        # Misconfig fallback
        _groq_model = get_llm_model()
        return _groq_model

    api_key = getattr(settings, "groq_api_key", None)
    if not api_key:
        # Misconfig fallback
        _groq_model = get_llm_model()
        return _groq_model

    from langchain_groq import ChatGroq
    _groq_model = ChatGroq(model=model_name, api_key=api_key, temperature=0.2)
    return _groq_model


def _get_local_model():
    """Local Ollama model lazy loading (cheap small steps)."""
    global _local_model
    if _local_model is not None:
        return _local_model

    # Prefer SECONDARY_MODEL if it's ollama:...
    provider, model_name = settings.get_model_provider(settings.secondary_model)
    if provider != "ollama":
        model_name = os.getenv("LOCAL_MODEL", "llama3.1:8b")

    from langchain_ollama import ChatOllama
    _local_model = ChatOllama(
        model=model_name,
        base_url=settings.ollama_base_url,
        temperature=0.3,
    )
    return _local_model


def _is_retryable_llm_error(err: Exception) -> bool:
    msg = str(err).lower()
    return (
        "429" in msg
        or "rate" in msg
        or "quota" in msg
        or "resource_exhausted" in msg
        or "temporarily" in msg
        or "timeout" in msg
    )


async def call_llm(
    system_prompt: str,
    user_prompt: str,
    role: str = "agent",
    retries: int = 2
) -> str:
    """
    LLM call with hybrid model routing (Groq for critical, Ollama for heavy lifting).
    
    Args:
        system_prompt: System message
        user_prompt: User message
        role: 'synthesis' for final (Groq), 'agent' for heavy work (Ollama)
        retries: Retry count
    """
    # Hybrid routing: Groq for final synthesis, Ollama for token-heavy agent work
    if role == "synthesis":
        try:
            model = _get_groq_model()
            logger.info("[HYBRID] Using Groq for final synthesis")
        except Exception as e:
            logger.warning(f"[HYBRID] Groq unavailable, falling back to local: {e}")
            model = _get_local_model()
    else:
        model = _get_local_model()
        logger.info(f"[HYBRID] Using Ollama for {role} (token-heavy work)")
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    # Generate proper UUID for LangSmith tracing
    run_id = str(uuid.uuid4())
    config = {"run_id": run_id, "run_name": f"multi-agent-{role}"}
    
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = await model.ainvoke(messages, config=config)
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            last_err = e
            if attempt < retries and _is_retryable_llm_error(e):
                delay = 1.0 * (2 ** attempt)
                logger.warning(f"[WARN] LLM retryable error, retrying in {delay:.1f}s: {e}")
                await asyncio.sleep(delay)
                continue
            break

    logger.error(f"LLM hatası: {last_err}")
    return f"Hata: {str(last_err)}"


# =============================================================================
# AGENT PROMPTS (Simple, direct prompts for Ollama)
# =============================================================================

RESEARCHER_PROMPT = """Sen uzman bir web araştırmacısısın. 

Aşağıdaki web arama sonuçlarını kullanarak kullanıcının sorusuna kapsamlı bir araştırma özeti hazırla.

📋 Görevlerin:
1. Web sonuçlarını dikkatlice analiz et
2. En önemli bilgileri çıkar ve özetle
3. Farklı kaynakları karşılaştır
4. Güncel ve doğrulanmış bilgileri vurgula
5. Kaynak linklerini belirt

⚠️ Kurallar:
- EN AZ 500 kelime yaz
- Madde madde ve organize ol
- Kaynak linklerini mutlaka ekle
- Spesifik veriler, tarihler, istatistikler kullan
- "Araştırma sonucunda..." diye başlama, direkt bilgiyi ver"""

CODER_PROMPT = """Sen uzman bir yazılım geliştiricisisin.

Aşağıdaki araştırma sonuçlarını baz alarak konuyla ilgili pratik, çalışan kod örnekleri yaz.

📋 Görevlerin:
1. Temel kullanım örneği (yeni başlayanlar için)
2. Orta seviye örnek (gerçek dünya senaryosu)
3. İleri seviye örnek (best practices)
4. Her kod bloğunu açıklayıcı yorumlarla destekle

⚠️ Kurallar:
- Kod ÇALIŞMALI (syntax hatası olmasın)
- Her örneği kısa açıklamayla tanıt
- Import statement'larını dahil et
- Modern syntax kullan
- Minimum 3 farklı örnek ver"""

WRITER_PROMPT = """Sen uzman bir teknik yazar ve eğitmensin. Verilen araştırma ve kod örneklerinden yola çıkarak profesyonel, kapsamlı ve anlaşılır bir Türkçe makale oluşturacaksın.

Markdown formatında yaz. ChatGPT tarzında temiz, modern ve akıcı bir yapı kullan:

# [Konu Başlığı]

Konuya giriş paragrafı (2-3 cümle) - ne, neden önemli?

## Genel Bakış

Konunun temellerini açıkla. Okuyucunun neyi öğreneceğini net şekilde belirt.

## Ana Kavramlar

Her bir önemli kavramı ayrı alt başlık altında detaylı açıkla:

### [Kavram 1]
Açıklama ve detaylar...

### [Kavram 2]  
Açıklama ve detaylar...

## Karşılaştırma (eğer uygunsa)

Alternatifleri veya farklı yaklaşımları karşılaştır. Avantaj/dezavantajları dengeli şekilde sun.

## Pratik Kullanım

Gerçek dünya senaryolarında nasıl kullanılır? Somut örnekler ver.

## Kod Örnekleri

```kod-dili
// Açıklamalı, anlaşılır kod örnekleri
// Her örneği kısa açıklamayla sun
```

## En İyi Uygulamalar

- Liste formatında, pratik öneriler
- Her madde somut ve uygulanabilir olmalı
- Yaygın hatalardan kaçınma yolları

## Kaynaklar

Araştırmadan gelen güvenilir kaynakları listele.

---

**Kurallar:**
- Akıcı, doğal Türkçe kullan
- Emoji kullanma (token israfı)
- Gereksiz formatlamadan kaçın
- Her bölümü anlamlı içerikle doldur
- Minimum 800 kelime hedefle"""


ROUTER_PROMPT = """Sen bir supervisor/router'sın. Amaç: Groq'u minimum kullanıp işleri yerelde (Ollama) yaptırmak.

Sadece aşağıdaki JSON'u döndür (başka hiçbir şey yazma):

{
    "web_search": "none" | "tavily" | "both",
    "need_code": true | false,
    "need_long_report": true | false
}

Kurallar:
- Soru güncel bilgi/versiyon/istatistik içeriyorsa web_search='both'
- Basit tanım/özet sorularında web_search='tavily'
- Tamamen genel ve küçük bir işse web_search='none'
- Kod isteniyorsa need_code=true
"""


# =============================================================================
# MAIN PIPELINE (Simplified - No DeepAgents, Direct LLM calls)
# =============================================================================

async def run_multi_agent_research(
    query: str,
    verbose: bool = True,
    options: Optional[dict] = None,
) -> AsyncGenerator:
    """
    Simplified Multi-Agent Pipeline - Ollama Compatible
    
    1. Web Search (Firecrawl + Tavily)
    2. Researcher LLM (analyze search results)
    3. Coder LLM (generate code examples)
    4. Writer LLM (create final report)
    """
    setup_langsmith()
    
    logger.info(f"[PIPELINE] Başlatılıyor: {query[:50]}...")
    
    try:
        # 0. PLANNING
        yield {
            "status": "planning",
            "message": "Plan oluşturuluyor...",
            "agent": "supervisor"
        }

        # Router decision (Groq - synthesis role for critical decisions)
        router_raw = await call_llm(
            ROUTER_PROMPT,
            f"Kullanıcı sorusu: {query}\nJSON üret.",
            role="synthesis",
            retries=1,
        )

        route = {"web_search": "both", "need_code": True, "need_long_report": True}
        try:
            # some models wrap in ```json blocks
            cleaned = router_raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.strip("`")
                cleaned = cleaned.replace("json", "", 1).strip()
            route = {**route, **json.loads(cleaned)}
        except Exception:
            pass

        # Apply optional user overrides (shared state from UI)
        if isinstance(options, dict):
            if options.get("web_search") in ("none", "tavily", "both"):
                route["web_search"] = options["web_search"]
            if isinstance(options.get("need_code"), bool):
                route["need_code"] = bool(options["need_code"])
            if isinstance(options.get("need_long_report"), bool):
                route["need_long_report"] = bool(options["need_long_report"])

        web_search_mode = route.get("web_search", "both")
        if web_search_mode not in ("none", "tavily", "both"):
            web_search_mode = "both"
        need_code = bool(route.get("need_code", True))
        need_long_report = bool(route.get("need_long_report", True))
        
        # 1. WEB SEARCH (Direct API calls - costs credits; router can reduce)
        yield {
            "status": "searching",
            "message": (
                "Web araması atlandı (router kararı)" if web_search_mode == "none" else
                ("Web araması yapılıyor (Tavily)" if web_search_mode == "tavily" else "Web araması yapılıyor (Firecrawl + Tavily)")
            ),
            "agent": "search"
        }
        logger.info("[1/4] Web Search başlıyor...")

        search_results = ""
        sources: list[dict] = []
        if web_search_mode == "none":
            search_results = "Web araması router tarafından atlandı. Genel bilgiyle devam." 
        else:
            tasks = []
            # Tavily (cheaper) always when searching
            tasks.append(tavily_search(query, limit=3))
            # Firecrawl only on 'both'
            if web_search_mode == "both":
                tasks.append(firecrawl_search(query, limit=5))

            results = await asyncio.gather(*tasks, return_exceptions=True)

            tavily_pack = results[0] if results else {"provider": "tavily", "text": "", "sources": []}
            firecrawl_pack = results[1] if len(results) > 1 else {"provider": "firecrawl", "text": "", "sources": []}

            if isinstance(tavily_pack, dict) and tavily_pack.get("text"):
                search_results += f"## Tavily Sonuçları:\n{tavily_pack.get('text','')}\n\n"
                if isinstance(tavily_pack.get("sources"), list):
                    for s in tavily_pack["sources"]:
                        if isinstance(s, dict) and s.get("url"):
                            sources.append({"provider": "tavily", **s})

            if isinstance(firecrawl_pack, dict) and firecrawl_pack.get("text"):
                search_results += f"## Firecrawl Sonuçları:\n{firecrawl_pack.get('text','')}\n\n"
                if isinstance(firecrawl_pack.get("sources"), list):
                    for s in firecrawl_pack["sources"]:
                        if isinstance(s, dict) and s.get("url"):
                            sources.append({"provider": "firecrawl", **s})
        logger.info(f"[1/4] Web Search tamamlandı: {len(search_results)} karakter")

        # Emit structured sources for tool-card rendering in the UI
        if sources:
            yield {
                "status": "searching",
                "message": "Kaynaklar derlendi",
                "agent": "search",
                "meta": {"sources": sources[:20]},
            }
        
        # 2. RESEARCHER - Analyze search results
        yield {
            "status": "researching",
            "message": "Araştırma sonuçları analiz ediliyor",
            "agent": "researcher"
        }
        logger.info("[2/4] Researcher başlıyor... (LOCAL)")
        
        researcher_input = f"""Kullanıcı Sorusu: {query}

Web Arama Sonuçları:
{search_results[:8000]}

Yukarıdaki kaynaklara dayanarak kapsamlı bir araştırma özeti hazırla."""
        
        # Researcher uses Ollama (agent role for token-heavy work)
        research_result = await call_llm(
            RESEARCHER_PROMPT,
            researcher_input,
            role="agent",
            retries=0,
        )
        logger.info(f"[2/4] Researcher tamamlandı: {len(research_result)} karakter (Ollama - token-heavy)")
        
        # 3. CODER - Generate code examples
        yield {
            "status": "coding",
            "message": "Kod örnekleri hazırlanıyor",
            "agent": "coder"
        }
        logger.info("[3/4] Coder başlıyor... (LOCAL)")
        
        coder_input = f"""Konu: {query}

Araştırma Özeti:
{research_result[:4000]}

Bu konuyla ilgili pratik kod örnekleri yaz."""
        
        if need_code:
            # Coder uses Ollama (agent role for token-heavy work)
            code_result = await call_llm(
                CODER_PROMPT,
                coder_input,
                role="agent",
                retries=0,
            )
            logger.info(f"[3/4] Coder tamamlandı: {len(code_result)} karakter (Ollama - token-heavy)")
        else:
            code_result = "(Router kararıyla kod örnekleri atlandı.)"
            logger.info("[3/4] Coder atlandı (router)")
        
        # 4. WRITER - Create final report
        yield {
            "status": "writing",
            "message": "Final rapor yazılıyor",
            "agent": "writer"
        }
        logger.info("[4/4] Writer başlıyor... (GROQ)")
        
        writer_input = f"""Konu: {query}

## Araştırma Sonuçları:
{research_result[:5000]}

## Kod Örnekleri:
{code_result[:3000]}

## Web Kaynakları:
{search_results[:2000]}

Yukarıdaki tüm bilgileri kullanarak kapsamlı bir Türkçe eğitim makalesi yaz."""
        
        # Writer uses Groq (synthesis role for quality output)
        final_report = await call_llm(
            WRITER_PROMPT,
            writer_input,
            role="synthesis",
            retries=2,
        )
        logger.info(f"[4/4] Writer tamamlandı: {len(final_report)} karakter (Groq - final synthesis)")
        
        # 5. DONE
        logger.info("[OK] Multi-Agent pipeline tamamlandı!")
        yield {
            "status": "done",
            "message": "Tamamlandı",
            "content": final_report
        }
    
    except Exception as e:
        error_msg = f"Multi-Agent hatası: {str(e)}"
        logger.error(f"[ERROR] {error_msg}", exc_info=True)
        yield {
            "status": "error",
            "message": f"Hata: {error_msg}",
            "content": f"# {query}\n\nHata: {error_msg}"
        }


# =============================================================================
# BACKWARD COMPATIBILITY
# =============================================================================

async def run_multi_agent_research_old(query: str, verbose: bool = True) -> str:
    """Eski API - yeni versiyonu çağırır"""
    async for update in run_multi_agent_research(query, verbose):
        if update.get("status") == "done":
            return update.get("content", "")
    return ""

