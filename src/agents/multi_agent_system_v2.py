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
if settings.langsmith_api_key:
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_PROJECT", "ai-research-multi-agent-v2")
    os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
    logger.info("[LANGSMITH] Multi-Agent v2 aktif")


# =============================================================================
# SUB-AGENTS (Başlangıçta bir kere oluşturulur)
# =============================================================================

# Global değişkenler - agent'lar ilk çağrıda oluşturulacak
_researcher_agent = None
_coder_agent = None
_writer_agent = None
_mcp_tools = None


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
    
    # Model
    model = get_llm_model()
    
    # 1. RESEARCHER AGENT (DeepAgent with Planning)
    researcher_prompt = """Sen bir Web Araştırma Uzmanısın (DeepAgent).

🛠️ Tool'ların:
- write_todos: Araştırma planı yap
- firecrawl_*: Web scraping
- read_file/write_file: Araştırma notları kaydet
- task: Alt araştırma için subagent spawn et

📋 İş Akışı:
1. write_todos: Araştırma planı yaz
2. Firecrawl ile araştır
3. write_file: Bulguları "research_notes.md" dosyasına kaydet
4. Özet döndür (detaylar dosyada)

⚡ Önemli: Uzun sonuçları dosyaya kaydet, sadece özet döndür."""
    
    search_tools = [t for t in _mcp_tools if 'search' in t.name.lower() or 'scrape' in t.name.lower()][:2]
    _researcher_agent = create_deep_agent(
        model=model,
        tools=search_tools,
        system_prompt=researcher_prompt,
    )
    
    # 2. CODER AGENT (DeepAgent with File System)
    coder_prompt = """Sen bir Kod Uzmanısın (DeepAgent).

🛠️ Tool'ların:
- write_todos: Kod yazma planı
- read_file: Araştırma notlarını oku ("research_notes.md")
- write_file: Kod'u "code_examples.py" dosyasına kaydet
- edit_file: Kodu düzenle
- task: Karmaşık kod için subagent

📋 İş Akışı:
1. write_todos: ["Araştırma oku", "Kod yaz", "Test et"]
2. read_file: "research_notes.md" oku
3. Kod yaz, write_file ile kaydet
4. Kod snippet'i döndür

⚡ Python tercih et. Temiz, çalışan kod."""
    
    _coder_agent = create_deep_agent(
        model=model,
        tools=[],
        system_prompt=coder_prompt,
    )
    
    # 3. WRITER AGENT (DeepAgent with Context Management)
    writer_prompt = """Sen bir Teknik Yazarsın (DeepAgent).

🛠️ Tool'ların:
- write_todos: Yazı planı
- read_file: Araştırma ve kod dosyalarını oku
- write_file: Final raporu "final_report.md" kaydet
- ls: Dosyaları listele
- task: Kompleks editöryal iş için subagent

📋 İş Akışı:
1. write_todos: ["Dosyaları oku", "Rapor yaz", "Kaydet"]
2. ls: Mevcut dosyaları gör
3. read_file: "research_notes.md", "code_examples.py" oku
4. Rapor yaz, write_file ile kaydet
5. Final rapor döndür

📄 Format:
# [Başlık]

## Özet
[2-3 cümle]

## Detaylar
[Madde madde]

## Kod Örnekleri
```python
[Kod]
```

## Kaynaklar
[Linkler]

⚡ Profesyonel, detaylı, yapılandırılmış."""
    
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
        result = await _researcher_agent.ainvoke(
            {"messages": [{"role": "user", "content": query}]},
            config={"recursion_limit": 10}
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
            config={"recursion_limit": 10}
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
            config={"recursion_limit": 10}
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
- researcher: Web araştırması (DeepAgent)
- coder: Kod örnekleri (DeepAgent)
- writer: Final rapor (DeepAgent)

📋 İş Akışı:
1. write_todos: Genel plan yap
   ["Soruyu analiz et", "Researcher çağır", "Coder çağır", "Writer çağır"]
2. researcher çağır → "research_notes.md" dosyasına kaydedecek
3. coder çağır → "code_examples.py" dosyasına kaydedecek
4. writer çağır → Her iki dosyayı okuyup "final_report.md" oluşturacak
5. read_file: "final_report.md" oku ve döndür

⚡ Strateji:
- Bilgi → researcher
- Kod → researcher + coder
- Rapor → writer (her zaman)

💡 Önemli:
- Her agent kendi dosya sistemini kullanır
- Dosyalar context overflow'u önler
- Subagent'lar otomatik planning yapar"""


async def run_multi_agent_research(query: str, verbose: bool = True) -> str:
    """
    LangChain Tool Calling Pattern ile Multi-Agent
    
    Supervisor → tool'ları çağırır → final rapor döner
    """
    await _init_agents()
    
    logger.info(f"[SUPERVISOR] Başlatılıyor: {query[:50]}...")
    
    try:
        # Supervisor Agent (tool'larla birlikte)
        model = get_llm_model()
        supervisor = create_deep_agent(
            model=model,
            tools=[researcher_tool, coder_tool, writer_tool],
            system_prompt=SUPERVISOR_PROMPT,
        )
        
        # Supervisor'ı çalıştır
        result = await supervisor.ainvoke(
            {"messages": [{"role": "user", "content": query}]},
            config={"recursion_limit": 30}  # Supervisor + her tool çağrısı için
        )
        
        # Final response'u al
        final_response = ""
        if "messages" in result:
            for msg in reversed(result["messages"]):
                if hasattr(msg, 'content') and isinstance(msg.content, str) and msg.content.strip():
                    final_response = msg.content.strip()
                    break
        
        if not final_response:
            return f"# {query}\n\nSonuç alınamadı."
        
        logger.info("[OK] Multi-Agent araştırma tamamlandı!")
        return final_response
    
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

