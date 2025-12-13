"""
DeepAgents Araştırma Sistemi
Firecrawl MCP + Gemini 2.5 Flash
FINAL WORKING VERSION
"""
import asyncio
import os
import time
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
        
    return init_chat_model(model_string, temperature=0.3)  # Daha tutarlı sonuçlar için


# ============ SYSTEM PROMPT ============

RESEARCH_INSTRUCTIONS = """You are a Turkish Research AI with multiple search tools.

YOUR TOOLS:
- firecrawl_search(query: str) - Web scraping (Firecrawl)
- tavily_web_search(query: str, max_results: int) - AI-optimized search (Tavily)
- github_search_repositories(query: str) - Search GitHub repos
- github_get_file_contents(owner: str, repo: str, path: str) - Get code files
- firecrawl_scrape(url: str) - Get full page content

MANDATORY WORKFLOW:
1. SEARCH: Use appropriate tools based on query type
   - General web: tavily_web_search OR firecrawl_search
   - Code/GitHub: github_search_repositories
   - Specific URL: firecrawl_scrape
2. ANALYZE: Review all results
3. WRITE: Turkish report with sources

OUTPUT FORMAT (ALWAYS IN TURKISH):
# [Başlık]

## Özet
[2-3 cümle]

## Detaylı Bulgular
- [Bulgu 1] [1]
- [Bulgu 2] [2]

## Kaynaklar
[1] URL - Açıklama
[2] URL - Açıklama

CRITICAL RULES:
- NEVER respond without searching first
- ALWAYS cite sources [1], [2], etc.
- Write ONLY in Turkish
- Use multiple tools for comprehensive research

EXAMPLE:
User: "Python pandas GitHub projeleri"
You:
1. github_search_repositories(query="pandas python")
2. tavily_web_search(query="pandas tutorial", max_results=3)
3. Write Turkish report with citations
"""


# ============ AGENT OLUŞTURMA ============

def sanitize_tool_schema(tool):
    """MCP tool schema'larını Gemini uyumlu hale getirir"""
    if hasattr(tool, 'args_schema') and tool.args_schema:
        schema = tool.args_schema
        if hasattr(schema, 'schema'):
            schema_dict = schema.schema()
            # Gemini ile uyumsuz alanları kaldır
            schema_dict.pop('$schema', None)
            schema_dict.pop('additionalProperties', None)
            if 'properties' in schema_dict:
                for prop in schema_dict['properties'].values():
                    if isinstance(prop, dict):
                        prop.pop('$schema', None)
                        prop.pop('additionalProperties', None)
    return tool


async def create_research_agent():
    """Firecrawl MCP + DeepAgent oluşturur"""
    
    setup_langsmith()
    
    if not settings.firecrawl_api_key:
        raise ValueError("❌ FIRECRAWL_API_KEY gerekli! .env dosyasını kontrol edin.")
    
    print("\n🔌 MCP Servers bağlanıyor...")
    
    # MCP Client yapılandırması - Firecrawl + Tavily + GitHub
    mcp_servers = {
        "firecrawl": {
            "command": settings.firecrawl_mcp_command,
            "args": settings.firecrawl_mcp_args,
            "env": settings.get_firecrawl_env(),
            "transport": "stdio"
        }
    }
    
    # Tavily MCP ekle (eğer API key varsa)
    if hasattr(settings, 'tavily_api_key') and settings.tavily_api_key:
        mcp_servers["tavily"] = {
            "command": "npx",
            "args": ["-y", "@tavily/mcp-server"],
            "env": {"TAVILY_API_KEY": settings.tavily_api_key},
            "transport": "stdio"
        }
    
    # GitHub MCP ekle (eğer token varsa)
    if hasattr(settings, 'github_token') and settings.github_token:
        mcp_servers["github"] = {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-github"],
            "env": {"GITHUB_PERSONAL_ACCESS_TOKEN": settings.github_token},
            "transport": "stdio"
        }
    
    mcp_client = MultiServerMCPClient(mcp_servers)
    
    # Tool'ları yükle
    mcp_tools = await mcp_client.get_tools()
    print(f"✅ {len(mcp_tools)} MCP tool yüklendi")
    
    # Tool isimlerini göster
    if mcp_tools:
        print(f"   📋 Tools: {', '.join([t.name for t in mcp_tools])}")
    
    # Tool schema'larını Gemini uyumlu hale getir
    for tool in mcp_tools:
        sanitize_tool_schema(tool)
    
    # LLM modelini al
    model = get_llm_model()
    
    # DeepAgent oluştur
    agent = create_deep_agent(
        model=model,
        instructions=RESEARCH_INSTRUCTIONS,
        tools=mcp_tools,  # Firecrawl + Tavily + GitHub MCP tools
    )
    
    print("✅ DeepAgent hazır!\n")
    return agent, mcp_client


# ============ ARAŞTIRMA ÇALIŞTIRMA ============

async def run_research(question: str, verbose: bool = True) -> str:
    """Araştırma agent'ını çalıştırır"""
    
    if verbose:
        print("\n" + "=" * 70)
        print("🔬 DeepAgents Araştırma Sistemi")
        print("   Gemini 2.5 Flash + Firecrawl MCP")
        print("=" * 70)
        print(f"\n📝 Soru: {question}\n")
    
    agent = None
    mcp_client = None
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            # Agent'ı oluştur
            agent, mcp_client = await create_research_agent()
            
            if verbose:
                print("🚀 Araştırma başlatılıyor...\n")
            
            # Rate limit için başlangıç bekleme
            if attempt > 0:
                wait = 10 * attempt
                print(f"⏳ {wait} saniye bekleniyor (rate limit)...")
                await asyncio.sleep(wait)
            
            # Agent'ı çalıştır
            result = await agent.ainvoke({
                "messages": [{"role": "user", "content": question}]
            })
            
            # --- DEBUG LOGGING ---
            if verbose:
                print(f"\n📦 DEBUG INFO:")
                print(f"   Result Keys: {list(result.keys())}")
                
                if "messages" in result:
                    print(f"   Message Count: {len(result['messages'])}")
                    
                    # Her mesajı incele
                    for i, msg in enumerate(result['messages']):
                        msg_type = type(msg).__name__
                        print(f"\n   🔹 Message {i}: {msg_type}")
                        
                        # Content
                        if hasattr(msg, 'content'):
                            content = msg.content
                            content_preview = str(content)[:200]
                            print(f"      Content: {content_preview}...")
                        
                        # Tool calls
                        if hasattr(msg, 'tool_calls') and msg.tool_calls:
                            print(f"      🔧 Tool Calls: {len(msg.tool_calls)}")
                            for tc in msg.tool_calls:
                                print(f"         - {tc.get('name', 'unknown')}({list(tc.get('args', {}).keys())})")
                        
                        # Tool results
                        if hasattr(msg, 'name'):
                            print(f"      Tool Result from: {msg.name}")
            
            print("\n" + "-" * 70)
            
            # Son AI mesajını bul ve döndür
            final_response = ""
            
            if "messages" in result and result["messages"]:
                # Sondan başa doğru git
                for msg in reversed(result["messages"]):
                    # Sadece AI mesajlarını al
                    if type(msg).__name__ not in ['AIMessage', 'AIMessageChunk']:
                        continue
                    
                    if not hasattr(msg, 'content'):
                        continue
                    
                    content = msg.content
                    
                    # Tool call yapıyorsa geç (henüz yanıt hazır değil)
                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                        continue
                    
                    # String ise direkt al
                    if isinstance(content, str) and content.strip():
                        final_response = content.strip()
                        break
                    
                    # List ise text parçalarını birleştir
                    elif isinstance(content, list):
                        texts = []
                        for item in content:
                            if isinstance(item, dict) and 'text' in item:
                                texts.append(item['text'])
                            elif isinstance(item, str):
                                texts.append(item)
                        
                        combined = "\n".join(texts).strip()
                        if combined:
                            final_response = combined
                            break
            
            # Sonucu göster
            if verbose and final_response:
                print("\n" + "=" * 70)
                print("📊 SONUÇ")
                print("=" * 70 + "\n")
                print(final_response)
                print("\n" + "=" * 70)
            
            if not final_response:
                print("\n❌ UYARI: Agent yanıt üretti ama içerik bulunamadı!")
                print("   Yukarıdaki debug loglarını kontrol edin.")
                return "❌ Araştırma tamamlandı ama yanıt formatlanamadı. Debug loglarına bakın."
            
            return final_response
        
        except Exception as e:
            error_msg = str(e)
            
            # Rate limit hatası
            if "429" in error_msg or "Resource exhausted" in error_msg or "quota" in error_msg.lower():
                wait_time = 30 * (attempt + 1)
                print(f"\n⚠️ Rate limit aşıldı (429 Error)")
                print(f"   {wait_time} saniye bekleniyor... (Deneme {attempt+1}/{max_retries})")
                
                # MCP client'ı kapat
                if mcp_client:
                    try:
                        await mcp_client.close()
                    except:
                        pass
                
                await asyncio.sleep(wait_time)
                continue
            
            # Diğer hatalar
            error_msg = f"❌ Hata: {str(e)}"
            if verbose:
                print(f"\n{error_msg}")
                import traceback
                traceback.print_exc()
            
            return error_msg
        
        finally:
            # MCP client'ı her durumda kapat
            if mcp_client and hasattr(mcp_client, 'close'):
                try:
                    await mcp_client.close()
                except Exception as close_error:
                    if verbose:
                        print(f"⚠️ MCP client kapatma hatası: {close_error}")
    
    return "❌ Maksimum deneme sayısına ulaşıldı. Lütfen birkaç dakika sonra tekrar deneyin."


def run_research_sync(question: str, verbose: bool = True) -> str:
    """Senkron wrapper - CLI için"""
    return asyncio.run(run_research(question, verbose))


# ============ İNTERAKTİF MOD ============

async def interactive_mode():
    """İnteraktif mod - Terminal'den sürekli soru sorabilirsiniz"""
    print("\n" + "=" * 70)
    print("🔬 İnteraktif Araştırma Modu")
    print("=" * 70)
    print("\nKomutlar:")
    print("  - Soru yazın ve Enter'a basın")
    print("  - 'q' veya 'quit' -> Çıkış")
    print("  - 'clear' -> Ekranı temizle")
    print("\n" + "=" * 70 + "\n")
    
    while True:
        try:
            question = input("📝 Soru: ").strip()
            
            if question.lower() in ['q', 'quit', 'exit']:
                print("\n👋 Görüşmek üzere!")
                break
            
            if question.lower() == 'clear':
                os.system('cls' if os.name == 'nt' else 'clear')
                continue
            
            if not question:
                continue
            
            # Araştırmayı çalıştır
            result = await run_research(question, verbose=True)
            
            print("\n" + "-" * 70 + "\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 Görüşmek üzere!")
            break
        except Exception as e:
            print(f"\n❌ Beklenmeyen hata: {e}")
            import traceback
            traceback.print_exc()