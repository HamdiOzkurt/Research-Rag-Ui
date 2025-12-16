"""
Multi-Agent Tool Wrappers for LangChain Tool Calling Pattern
Each subagent wrapped as a proper @tool for main agent to invoke
"""

import asyncio
from typing import Optional
from langchain_core.tools import tool
import logging

logger = logging.getLogger(__name__)

# Import actual implementations from multi_agent_system_v2
from .multi_agent_system_v2 import (
    firecrawl_search,
    tavily_search,
    call_llm,
    RESEARCHER_PROMPT,
    CODER_PROMPT,
    WRITER_PROMPT
)


@tool("web_research", description="Web'de kapsamlı araştırma yapar. Güncel bilgi, istatistik, versiyon sorguları için kullan. Firecrawl + Tavily ile çoklu kaynak tarar.")
async def web_research_tool(query: str, limit: int = 5) -> str:
    """
    Web araması yapar ve sonuçları döner.
    
    Args:
        query: Arama sorgusu
        limit: Maksimum sonuç sayısı (default: 5)
        
    Returns:
        str: Arama sonuçları metni
    """
    logger.info(f"[TOOL:web_research] Running for: {query[:50]}...")
    
    # Parallel search with Firecrawl + Tavily
    fire_task = firecrawl_search(query, limit)
    tavily_task = tavily_search(query, limit)
    
    fire_result, tavily_result = await asyncio.gather(fire_task, tavily_task)
    
    # Combine results
    output = []
    sources = []
    
    if fire_result.get("text"):
        output.append(f"## Firecrawl Sonuçları\n\n{fire_result['text']}")
        sources.extend(fire_result.get("sources", []))
    
    if tavily_result.get("text"):
        output.append(f"## Tavily Sonuçları\n\n{tavily_result['text']}")
        sources.extend(tavily_result.get("sources", []))
    
    if not output:
        return "Sonuç bulunamadı."
    
    # Add sources at the end
    if sources:
        output.append("\n## Kaynaklar\n")
        for i, src in enumerate(sources[:limit], 1):
            output.append(f"{i}. [{src['title']}]({src['url']})")
    
    return "\n\n".join(output)


@tool("analyze_research", description="Web arama sonuçlarını analiz edip özet rapor oluşturur. Uzman araştırmacı gibi bilgileri sentezler, karşılaştırır ve önemli noktaları vurgular.")
async def analyze_research_tool(search_results: str, original_query: str) -> str:
    """
    Web arama sonuçlarını analiz edip araştırma raporu oluşturur.
    
    Args:
        search_results: Web search sonuçları (web_research_tool output)
        original_query: Orijinal kullanıcı sorusu
        
    Returns:
        str: Analiz edilmiş araştırma raporu
    """
    logger.info(f"[TOOL:analyze_research] Analyzing for: {original_query[:50]}...")
    
    user_prompt = f"""Kullanıcı Sorusu: {original_query}

Web Arama Sonuçları:
{search_results}

Yukarıdaki sonuçları kullanarak kapsamlı bir araştırma raporu hazırla."""
    
    result = await call_llm(
        system_prompt=RESEARCHER_PROMPT,
        user_prompt=user_prompt,
        role="researcher"
    )
    
    return result


@tool("generate_code_examples", description="Konuyla ilgili çalışan kod örnekleri oluşturur. Temel, orta, ileri seviye örnekler içerir. Modern syntax ve best practices kullanır.")
async def generate_code_tool(research_summary: str, topic: str) -> str:
    """
    Araştırma sonuçlarına göre kod örnekleri oluşturur.
    
    Args:
        research_summary: Araştırma özeti (analyze_research_tool output)
        topic: Kod konusu/başlığı
        
    Returns:
        str: Kod örnekleri (Markdown formatında)
    """
    logger.info(f"[TOOL:generate_code] Generating for: {topic[:50]}...")
    
    user_prompt = f"""Konu: {topic}

Araştırma Özeti:
{research_summary}

Yukarıdaki bilgileri kullanarak pratik kod örnekleri oluştur."""
    
    result = await call_llm(
        system_prompt=CODER_PROMPT,
        user_prompt=user_prompt,
        role="coder"
    )
    
    return result


@tool("write_final_article", description="Araştırma ve kod örneklerinden profesyonel makale yazar. ChatGPT tarzı akıcı, kapsamlı, eğitici içerik oluşturur. Minimum 800 kelime.")
async def write_article_tool(
    original_query: str,
    research_summary: str,
    code_examples: Optional[str] = None
) -> str:
    """
    Final makaleyiy azar (synthesis).
    
    Args:
        original_query: Orijinal kullanıcı sorusu
        research_summary: Araştırma özeti
        code_examples: Kod örnekleri (opsiyonel)
        
    Returns:
        str: Profesyonel makale (Markdown)
    """
    logger.info(f"[TOOL:write_article] Writing for: {original_query[:50]}...")
    
    user_prompt = f"""Kullanıcı Sorusu: {original_query}

Araştırma Özeti:
{research_summary}
"""
    
    if code_examples:
        user_prompt += f"\n\nKod Örnekleri:\n{code_examples}"
    
    user_prompt += "\n\nYukarıdaki bilgileri kullanarak profesyonel, kapsamlı bir makale yaz."
    
    result = await call_llm(
        system_prompt=WRITER_PROMPT,
        user_prompt=user_prompt,
        role="synthesis"  # Use Groq for final synthesis
    )
    
    return result


# Export all tools
ALL_MULTI_AGENT_TOOLS = [
    web_research_tool,
    analyze_research_tool,
    generate_code_tool,
    write_article_tool
]


# =============================================================================
# EXAMPLE USAGE FOR MAIN AGENT
# =============================================================================
"""
from langchain.agents import create_react_agent

multi_agent = create_react_agent(
    llm=main_llm,
    tools=ALL_MULTI_AGENT_TOOLS,
    prompt="Sen bir AI Research Coordinator'ısın. Kompleks araştırma görevlerini parçalayıp subagent'lara delege et..."
)

# Agent will decide:
# 1. Call web_research_tool for current info
# 2. Call analyze_research_tool to process results
# 3. Call generate_code_tool if code needed
# 4. Call write_article_tool for final output
"""
