"""
Multi-Agent Research System - LangChain Best Practices
Tool Calling Pattern - Supervisor koordine eder, agent'lar tool olarak çalışır
Kaynak: https://docs.langchain.com/oss/python/langchain/multi-agent#tool-calling
"""

import os
import asyncio
from typing import Optional
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from deepagents import create_deep_agent
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
        logger.info("[LANGSMITH] Multi-Agent v2 aktif - https://smith.langchain.com/o/personal/projects/p/ai-research-multi-agent-v2")
        return True
    return False


# =============================================================================
# SUB-AGENTS (Başlangıçta bir kere oluşturulur)
# =============================================================================

# Global değişkenler - agent'lar ilk çağrıda oluşturulacak
_researcher_agent = None
_coder_agent = None
_writer_agent = None
_mcp_tools = None


def _get_tool_calling_model():
    """Multi-Agent için tool calling optimize Ollama model
    
    qwen2.5:7b - Tool calling için en iyi ücretsiz Ollama modeli
    Alternatifler: mistral:7b, llama3.1:8b
    
    .env'de MULTI_AGENT_MODEL ile override edilebilir
    """
    from langchain_ollama import ChatOllama
    
    # Tool calling için optimize model
    tool_model = os.getenv("MULTI_AGENT_MODEL", "qwen2.5:7b")
    
    try:
        model = ChatOllama(
            model=tool_model,
            base_url=settings.ollama_base_url,
            temperature=0.7,
        )
        logger.info(f"[MODEL] Multi-Agent: {tool_model} (tool calling optimized)")
        return model
    except Exception as e:
        logger.warning(f"[WARN] {tool_model} init hatası: {e}, default'a fallback")
        return get_llm_model()


async def _init_agents():
    """Agent'ları ve MCP tool'larını başlat (lazy initialization)"""
    global _researcher_agent, _coder_agent, _writer_agent, _mcp_tools
    
    if _researcher_agent is not None:
        return  # Zaten başlatılmış
    
    logger.info("[INIT] Agent'lar başlatılıyor...")
    
    # MCP Tools (Firecrawl)
    try:
        mcp_servers = {
            "firecrawl": {
                "command": settings.firecrawl_mcp_command,
                "args": settings.firecrawl_mcp_args,
                "env": settings.get_firecrawl_env(),
                "transport": "stdio"
            }
        }
        mcp_client = MultiServerMCPClient(mcp_servers)
        _mcp_tools = await mcp_client.get_tools()
        for tool_obj in _mcp_tools:
            sanitize_tool_schema(tool_obj)
        logger.info(f"[OK] {len(_mcp_tools)} MCP tool yüklendi")
    except Exception as e:
        logger.warning(f"[WARN] MCP başlatılamadı: {e}")
        _mcp_tools = []
    
    # Model - Multi-Agent için tool calling optimize (qwen2.5 default)
    model = _get_tool_calling_model()
    
    # 1. RESEARCHER AGENT (DeepAgent with Planning)
    researcher_prompt = """Sen bir Web Araştırma Uzmanısın (DeepAgent).

Görevlerin:
- Kullanıcının sorusunu ayrıntılı anlamlandır.
- Gerekirse web'de araştırma yap (özellikle güncel, spesifik, istatistik veya kütüphane dokümantasyonu gereken konularda).
- En az 5–10 adet güvenilir kaynaktan fikir topla, karşılaştır ve sentez yap.

🛠️ Tool'ların:
- write_todos: Ayrıntılı araştırma planı yap (alt başlıklar, adımlar).
- firecrawl_search: Web araması (query, limit, lang, country, scrapeOptions).
- read_file/write_file: Araştırma notlarını kaydet ve gerektiğinde tekrar kullan.
- task: Çok büyük araştırmalarda alt araştırmalar için subagent spawn et.

📋 İş Akışı:
1. write_todos ile detaylı bir araştırma planı çıkar (alt başlıklar, yapılacaklar).
2. firecrawl_search ile web'de birden fazla arama yap, farklı açılardan veri topla.
   Örnek argüman: {"query": "...", "limit": 5, "lang": "en", "country": "us", "scrapeOptions": {"formats": ["markdown"], "onlyMainContent": true}}.
3. Önemli bulguları "research_notes.md" dosyasına kaydet (kaynak linkleri dahil).
4. Son olarak, kullanıcı sorusuna yönelik net, madde madde bir araştırma özeti hazırla.

⚡ Önemli:
- Yüzeysel 2–3 cümlelik cevap verme; kavramı, nerede kullanıldığını, iyi/kötü yanlarını açıkla.
- Eğitim amaçlı sorularda (ör: kütüphane nedir, neden kullanılır?) örnek senaryolar ve kısa kod parçaları önerebilirsin, ama asıl kod Coder agent'a bırakılacak."""
    
    # Firecrawl MCP tool'larını tekrar etkinleştir (yalnızca search)
    search_tools = [t for t in _mcp_tools if t.name == "firecrawl_search"]
    _researcher_agent = create_deep_agent(
        model=model,
        tools=search_tools,
        system_prompt=researcher_prompt,
    )
    
    # 2. CODER AGENT (DeepAgent with File System)
    coder_prompt = """Sen bir Kod Uzmanısın (DeepAgent).

Görevin:
- Researcher'ın notlarını ve kullanıcının sorusunu temel alarak, öğretici ve gerçekten çalışabilir örnek kodlar yazmak.
- Kodun yanına kısa açıklamalar eklemek (yorum satırı veya metin olarak) ama asıl açıklamayı Writer'a bırakmak.

🛠️ Tool'ların:
- write_todos: Kod yazma planı (örnek sayısı, adımlar, hangi konular gösterilecek).
- read_file: Araştırma notlarını oku ("research_notes.md").
- write_file: Kod'u "code_examples.py" dosyasına kaydet.
- edit_file: Kodu daha sonra geliştir veya düzenle.
- task: Çok kapsamlı örnekler için alt kod agent'ları oluştur.

📋 İş Akışı:
1. write_todos ile hangi örnekleri yazacağını planla (ör: temel kullanım, orta seviye kullanım, iyi pratikler).
2. read_file ile "research_notes.md" içeriğini incele.
3. Kullanıcının seviyesini başlangıç/orta seviye varsayarak okunabilir, açıklamalı örnekler yaz.
4. Örnekleri "code_examples.py" içine kaydet, özetini kullanıcıya döndür.

⚡ Tercihen Python kullan; kod gerçekten çalışabilir, minimum bağımlılık gerektirmeli ve hata içermemeli."""
    
    _coder_agent = create_deep_agent(
        model=model,
        tools=[],
        system_prompt=coder_prompt,
    )
    
    # 3. WRITER AGENT (DeepAgent with Context Management)
    writer_prompt = """Sen bir Teknik Yazarsın (DeepAgent).

Amaç:
- Researcher ve Coder'ın çıktılarından faydalanarak, kullanıcının seviyesine uygun (başlangıç/orta seviye) bir eğitim notu/mini makale yazmak.
- Cevapları Türkçe ve çok net yaz; kullanıcı kavramı ilk defa duyuyormuş gibi düşün.

🛠️ Tool'ların:
- write_todos: Yazı planı (bölümler, alt başlıklar).
- read_file: Araştırma ve kod dosyalarını oku.
- write_file: Final raporu "final_report.md" kaydet.
- ls: Dosyaları listele.
- task: Kompleks editöryal iş için subagent.

📋 İş Akışı:
1. write_todos ile makale yapısını planla (Giriş, Temel Kavramlar, Kullanım Alanları, Örnek, Sonuç vb.).
2. ls ile mevcut dosyaları kontrol et, ardından read_file ile "research_notes.md" ve "code_examples.py" dosyalarını oku.
3. Bu içerikleri birleştirerek, kullanıcı için anlaşılır ve akıcı bir anlatım oluştur.
4. Raporu "final_report.md" olarak kaydet ve özetini kullanıcıya Markdown formatında döndür.

📄 Önerilen Format (Markdown):
# [Konu Başlığı]

## Kısa Özet
2–4 cümlede temel fikri anlat.

## Temel Kavramlar
- Kavram 1: Açıklama
- Kavram 2: Açıklama

## Neden Önemli / Nerede Kullanılır?
- Gerçek dünyadan 2–3 senaryo örneği.

## Basit Kod Örneği
```python
[Kısa ve odaklı kod]
```

## İyi Pratikler / Dikkat Edilecek Noktalar
- Madde madde.

## İleri Okuma
- Kütüphane dokümantasyonu, resmi rehberler, kaliteli blog yazıları.

⚡ Profesyonel, detaylı, ama gereksiz akademik jargon kullanmadan, sade ve öğretici yaz."""
    
    _writer_agent = create_deep_agent(
        model=model,
        tools=[],
        system_prompt=writer_prompt,
    )
    
    logger.info("[OK] Tüm agent'lar hazır!")


# =============================================================================
# TOOL WRAPPERS (LangChain Best Practice)
# =============================================================================

@tool(
    "researcher",
    description="Web araştırması yapar. Firecrawl ile web'den bilgi toplar. Kullanım: Bilgi eksikse, genel sorularda."
)
async def researcher_tool(query: str) -> str:
    """Web araştırması tool'u - Supervisor tarafından çağrılır"""
    await _init_agents()
    
    logger.info(f"[RESEARCHER] Çalışıyor: {query[:50]}...")
    
    try:
        # Firecrawl schema uyumu için ipucu (docs v2'ye göre):
        # firecrawl_search argümanları: query, limit, lang, country, scrapeOptions.
        # sources vb. ekstra alanları KULLANMA.
        hint = (
            'Firecrawl (firecrawl_search) kullanacaksan, SADECE şu argümanları kullan:\n'
            '{"query": "%s", "limit": 5, "lang": "en", "country": "us", '
            '"scrapeOptions": {"formats": ["markdown"], "onlyMainContent": true}}\n'
            'sources, urls vb. ek alanlar EKLEME; schema hatasına sebep olur.'
        ) % query

        result = await _researcher_agent.ainvoke(
            {"messages": [{"role": "user", "content": f"{query}\n\n{hint}"}]},
            config={"recursion_limit": 20}
        )
        
        # Son mesajı al
        response = ""
        if "messages" in result:
            for msg in reversed(result["messages"]):
                if hasattr(msg, 'content') and isinstance(msg.content, str) and msg.content.strip():
                    response = msg.content.strip()
                    break
        
        if not response:
            return "Araştırma sonucu bulunamadı."
        
        logger.info(f"[OK] Researcher tamamlandı: {len(response)} karakter")
        return response
    
    except Exception as e:
        error_msg = f"Araştırma hatası: {str(e)}"
        logger.error(f"[ERROR] {error_msg}")
        return error_msg


@tool(
    "coder",
    description="Kod örnekleri oluşturur. Python, JavaScript gibi dillerle çalışan kod yazar. Kullanım: Kod istendiğinde."
)
async def coder_tool(task: str, research_context: str = "") -> str:
    """Kod üretme tool'u - Supervisor tarafından çağrılır"""
    await _init_agents()
    
    logger.info(f"[CODER] Çalışıyor: {task[:50]}...")
    
    prompt = task
    if research_context:
        prompt = f"Araştırma sonuçları:\n{research_context}\n\nGörev: {task}"
    
    try:
        result = await _coder_agent.ainvoke(
            {"messages": [{"role": "user", "content": prompt}]},
            config={"recursion_limit": 25}  # Kod yazma iteratif olabilir
        )
        
        response = ""
        if "messages" in result:
            for msg in reversed(result["messages"]):
                if hasattr(msg, 'content') and isinstance(msg.content, str) and msg.content.strip():
                    response = msg.content.strip()
                    break
        
        if not response:
            return "Kod oluşturulamadı."
        
        logger.info(f"[OK] Coder tamamlandı: {len(response)} karakter")
        return response
    
    except Exception as e:
        error_msg = f"Kod üretme hatası: {str(e)}"
        logger.error(f"[ERROR] {error_msg}")
        return error_msg


@tool(
    "writer",
    description="Final rapor yazar. Araştırma ve kod sonuçlarını birleştirip profesyonel Markdown rapor oluşturur."
)
async def writer_tool(research: str = "", code: str = "", query: str = "") -> str:
    """Rapor yazma tool'u - Supervisor tarafından çağrılır"""
    await _init_agents()
    
    logger.info(f"[WRITER] Rapor yazılıyor...")
    
    prompt = f"""Konu: {query}

Araştırma Sonuçları:
{research if research else "Yok"}

Kod Örnekleri:
{code if code else "Yok"}

Profesyonel Markdown rapor oluştur."""
    
    try:
        result = await _writer_agent.ainvoke(
            {"messages": [{"role": "user", "content": prompt}]},
            config={"recursion_limit": 15}  # Rapor yazma genelde hızlı
        )
        
        response = ""
        if "messages" in result:
            for msg in reversed(result["messages"]):
                if hasattr(msg, 'content') and isinstance(msg.content, str) and msg.content.strip():
                    response = msg.content.strip()
                    break
        
        if not response:
            return f"# {query}\n\nRapor oluşturulamadı."
        
        logger.info(f"[OK] Writer tamamlandı: {len(response)} karakter")
        return response
    
    except Exception as e:
        error_msg = f"Rapor yazma hatası: {str(e)}"
        logger.error(f"[ERROR] {error_msg}")
        return f"# {query}\n\n{error_msg}"


# =============================================================================
# SUPERVISOR AGENT (Tool Calling Pattern)
# =============================================================================

SUPERVISOR_PROMPT = """Sen bir Araştırma Yöneticisisin (DeepAgent Supervisor).

🛠️ Built-in Tool'lar (Otomatik):
- write_todos: Genel plan yap
- read_file/write_file/edit_file/ls: Dosya sistemi
- task: Subagent spawn et

🧑‍💼 Subagent Tool'lar:
- researcher: Web araştırması (DeepAgent + MCP/Firecrawl)
- coder: Kod örnekleri (DeepAgent)
- writer: Final rapor (DeepAgent)

🚨 ZORUNLU KURAL:
HER SORUDA MUTLAKA ŞUNU YAP:
1. researcher tool'unu çağır (web'den güncel bilgi topla)
2. coder tool'unu çağır (kod örnekleri oluştur)
3. writer tool'unu çağır (final rapor yaz)

❌ ASLA base knowledge'ını kullanma
❌ ASLA researcher'ı atlama
✅ HER ZAMAN 3 tool'u sırayla çağır

📋 İş Akışı:
1. write_todos: ["Web araştır", "Kod yaz", "Rapor hazırla"]
2. researcher(query) → MCP/Firecrawl ile web'den araştır
3. coder(task, research_context) → Araştırma sonuçlarını kullanarak kod yaz
4. writer(research, code, query) → Final raporu oluştur

💡 Önemli:
- Researcher MUTLAKA çağrılmalı (MCP tool'ları orada)
- Her agent kendi dosya sistemini kullanır
- Subagent'lar otomatik planning yapar"""


async def run_multi_agent_research(query: str, verbose: bool = True) -> str:
    """
    Sequential Multi-Agent Pipeline
    
    Ollama tool calling uyumsuzluğu nedeniyle sıralı çalıştırma:
    Researcher → Coder → Writer
    
    Bu yaklaşım:
    - Her agent garantili çağrılır
    - MCP tool'lar kesinlikle kullanılır
    - LangSmith'te tüm trace'ler görünür
    """
    # LangSmith'i bu mod için ayarla
    setup_langsmith()
    
    await _init_agents()
    
    logger.info(f"[PIPELINE] Başlatılıyor: {query[:50]}...")
    
    try:
        # 1. RESEARCHER - Web'den bilgi topla
        logger.info("[1/3] Researcher başlıyor...")
        research_result = await researcher_tool.ainvoke(query)
        logger.info(f"[1/3] Researcher tamamlandı: {len(research_result)} karakter")
        
        # 2. CODER - Kod örnekleri oluştur
        logger.info("[2/3] Coder başlıyor...")
        code_result = await coder_tool.ainvoke({
            "task": query,
            "research_context": research_result[:2000]  # Context overflow önle
        })
        logger.info(f"[2/3] Coder tamamlandı: {len(code_result)} karakter")
        
        # 3. WRITER - Final rapor oluştur
        logger.info("[3/3] Writer başlıyor...")
        final_report = await writer_tool.ainvoke({
            "research": research_result[:3000],
            "code": code_result[:2000],
            "query": query
        })
        logger.info(f"[3/3] Writer tamamlandı: {len(final_report)} karakter")
        
        logger.info("[OK] Multi-Agent pipeline tamamlandı!")
        return final_report
    
    except Exception as e:
        error_msg = f"Multi-Agent hatası: {str(e)}"
        logger.error(f"[ERROR] {error_msg}", exc_info=True)
        return f"# {query}\n\n❌ Hata: {error_msg}"


# =============================================================================
# BACKWARD COMPATIBILITY
# =============================================================================

# Eski fonksiyonu yeni versiyona yönlendir
async def run_multi_agent_research_old(query: str, verbose: bool = True) -> str:
    """Eski API - yeni versiyonu çağırır"""
    return await run_multi_agent_research(query, verbose)

