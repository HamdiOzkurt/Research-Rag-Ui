"""
DeepAgents Araştırma Sistemi
Firecrawl MCP + Gemini 2.5 Flash
"""
import asyncio
import os
from typing import Optional

from deepagents import create_deep_agent
from langchain.chat_models import init_chat_model
from langchain_mcp_adapters.client import MultiServerMCPClient

from ..config import settings


# ============ LANGSMITH ============

def setup_langsmith():
    """LangSmith tracing'i etkinleştirir"""
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_PROJECT", "multi-agent-search")
    if settings.langsmith_api_key:
        os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
        print("✅ LangSmith tracing aktif!")
        print("   📊 https://smith.langchain.com")
        return True
    return False


# ============ MODEL ============

def get_llm_model():
    """Kullanılabilir LLM modelini döndürür"""
    model_string = settings.get_available_model()
    provider, model_name = settings.get_model_provider(model_string)
    print(f"🤖 Model: {provider}:{model_name}")
    
    if provider == "google_genai" and settings.google_api_key:
        os.environ["GOOGLE_API_KEY"] = settings.google_api_key
    if provider == "ollama":
        os.environ["OLLAMA_HOST"] = settings.ollama_base_url
        
    return init_chat_model(model_string)


# ============ SYSTEM PROMPT ============

RESEARCH_INSTRUCTIONS = """You are an expert researcher. Your job is to conduct thorough research and then write a polished report in Turkish.

You have access to web search tools as your primary means of gathering information.

## `firecrawl_search`

Use this to run a web search for a given query. Only use the "query" parameter.

Examples:
- General search: firecrawl_search(query="best open source LLM models 2024")
- Reddit search: firecrawl_search(query="site:reddit.com best LLM models")
- GitHub search: firecrawl_search(query="site:github.com LLM benchmarks")

## `firecrawl_scrape`

Use this to scrape content from a specific URL.

## Workflow:

1. Plan your research approach using write_todos
2. Conduct at least 2-3 different searches
3. Analyze the results
4. Write a comprehensive report in Turkish

## Rules:
- Always respond in Turkish
- Include source URLs in your report
- Be thorough and analytical
"""


# ============ SUBAGENTS ============

def get_subagents():
    """Alt ajan konfigürasyonlarını döndürür"""
    
    search_subagent = {
        "name": "search-agent",
        "description": "Searches the web for information using firecrawl_search tool",
        "system_prompt": """You are a search specialist. Your job is to find relevant information using firecrawl_search.
        
        Use the query parameter only. For specific sites, use site: operator in the query.
        
        Examples:
        - firecrawl_search(query="topic keywords")
        - firecrawl_search(query="site:reddit.com topic")
        - firecrawl_search(query="site:github.com topic")
        
        Return all found URLs and key information."""
    }
    
    analysis_subagent = {
        "name": "analysis-agent", 
        "description": "Analyzes and synthesizes research findings",
        "system_prompt": """You are an analysis specialist. Your job is to:
        
        1. Review all gathered information
        2. Identify key themes and patterns
        3. Evaluate source reliability
        4. Synthesize findings into clear insights
        
        Be objective and thorough in your analysis."""
    }
    
    writer_subagent = {
        "name": "writer-agent",
        "description": "Writes professional reports in Turkish",
        "system_prompt": """You are a professional technical writer. Your job is to:
        
        1. Take analyzed research data
        2. Write a clear, well-structured report in Turkish
        3. Include proper citations and source URLs
        4. Use professional but accessible language
        
        Format:
        - Title
        - Executive Summary
        - Detailed Findings
        - Sources/References"""
    }
    
    return [search_subagent, analysis_subagent, writer_subagent]


# ============ AGENT OLUŞTURMA ============

async def create_research_agent():
    """Firecrawl MCP + DeepAgent oluşturur"""
    
    setup_langsmith()
    
    if not settings.firecrawl_api_key:
        raise ValueError("❌ FIRECRAWL_API_KEY gerekli!")
    
    print("\n🔌 Firecrawl MCP bağlanıyor...")
    
    mcp_client = MultiServerMCPClient({
        "firecrawl": {
            "command": settings.firecrawl_mcp_command,
            "args": settings.firecrawl_mcp_args,
            "env": settings.get_firecrawl_env(),
            "transport": "stdio"
        }
    })
    
    mcp_tools = await mcp_client.get_tools()
    print(f"✅ {len(mcp_tools)} Firecrawl tool yüklendi")
    
    model = get_llm_model()
    
    # DeepAgent oluştur - subagents ile
    agent = create_deep_agent(
        model=model,
        system_prompt=RESEARCH_INSTRUCTIONS,
        tools=mcp_tools,
        subagents=get_subagents(),
    )
    
    print("✅ DeepAgent hazır!\n")
    return agent, mcp_client


# ============ ARAŞTIRMA ÇALIŞTIRMA ============

async def run_research(question: str, verbose: bool = True) -> str:
    """Araştırma agent'ını çalıştırır"""
    
    if verbose:
        print("\n" + "=" * 60)
        print("🔬 DeepAgents Araştırma Sistemi")
        print("   (Gemini + Firecrawl MCP)")
        print("=" * 60)
        print(f"\n📝 Soru: {question}\n")
    
    agent = None
    mcp_client = None
    
    try:
        agent, mcp_client = await create_research_agent()
        
        if verbose:
            print("🚀 Araştırma başlatılıyor...\n")
        
        # Agent'ı çalıştır
        result = await agent.ainvoke({
            "messages": [{"role": "user", "content": question}]
        })
        
        # Son mesajı al
        final_response = ""
        if "messages" in result and result["messages"]:
            # En son mesajın içeriğini al
            last_msg = result["messages"][-1]
            if hasattr(last_msg, 'content'):
                content = last_msg.content
                if isinstance(content, str):
                    final_response = content
                elif isinstance(content, list):
                    texts = []
                    for item in content:
                        if isinstance(item, dict) and 'text' in item:
                            texts.append(item['text'])
                        elif isinstance(item, str):
                            texts.append(item)
                    final_response = "\n".join(texts)
        
        if verbose and final_response:
            print("\n" + "=" * 60)
            print("📊 SONUÇ")
            print("=" * 60)
            print(final_response)
        
        return final_response if final_response.strip() else "❌ Boş yanıt alındı."
    
    except Exception as e:
        error_msg = f"❌ Hata: {str(e)}"
        if verbose:
            print(error_msg)
            import traceback
            traceback.print_exc()
        return error_msg
    
    finally:
        if mcp_client and hasattr(mcp_client, 'close'):
            try:
                await mcp_client.close()
            except:
                pass


def run_research_sync(question: str, verbose: bool = True) -> str:
    """Senkron wrapper"""
    return asyncio.run(run_research(question, verbose))


async def interactive_mode():
    """İnteraktif mod"""
    print("\n🔬 İnteraktif Mod - Çıkmak için 'q'\n")
    
    while True:
        try:
            question = input("📝 Soru: ").strip()
            
            if question.lower() in ['q', 'quit', 'exit']:
                print("\n👋 Görüşmek üzere!")
                break
            
            if not question:
                continue
            
            result = await run_research(question)
            print(f"\n{result}\n")
            print("-" * 60)
            
        except KeyboardInterrupt:
            print("\n\n👋 Görüşmek üzere!")
            break
