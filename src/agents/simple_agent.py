"""
Basit Araştırma Agent'ı - Hızlı Mod
Multi API Key Rotation + 429 Protection
"""
import asyncio
from deepagents import create_deep_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from src.config import settings
from src.models import get_llm_model, sanitize_tool_schema, rotate_key_on_error
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SIMPLE_PROMPT = """Sen bir Türkçe Araştırma Asistanısın. 

Görevin:
1. Verilen soruyu tek bir web araması ile araştır
2. Kısa ve öz bir Türkçe rapor hazırla
3. Hemen sonucu döndür

Rapor Formatı:
# 📊 [Başlık]

## 🎯 Özet
[2-3 cümle özet]

## 📖 Detaylar
[Ana bilgiler, madde madde]

## 💡 Önemli Noktalar
- [Nokta 1]
- [Nokta 2]
- [Nokta 3]

## 🔗 Kaynaklar
- [Kaynak linkler]

Kısa, öz ve hızlı yaz!
"""


async def run_simple_research(query: str, verbose: bool = True, max_retries: int = 3) -> str:
    """
    Basit hızlı araştırma - Multi API Key Rotation desteği
    
    Args:
        query: Araştırma sorusu
        verbose: Log göster
        max_retries: 429 hatası için max deneme
    """
    mcp_client = None
    
    for attempt in range(max_retries):
        try:
            if verbose:
                key_info = f"(Key {settings._current_key_index + 1}/{len(settings.google_api_keys)})" if settings.google_api_keys else ""
                logger.info(f"⚡ Araştırma başlatılıyor {key_info}...")
            
            # MCP client
            mcp_servers = {
                "firecrawl": {
                    "command": settings.firecrawl_mcp_command,
                    "args": settings.firecrawl_mcp_args,
                    "env": settings.get_firecrawl_env(),
                    "transport": "stdio"
                }
            }
            
            mcp_client = MultiServerMCPClient(mcp_servers)
            mcp_tools = await mcp_client.get_tools()
            
            for tool in mcp_tools:
                sanitize_tool_schema(tool)
            
            # Agent oluştur
            model = get_llm_model()
            agent = create_deep_agent(
                model=model,
                instructions=SIMPLE_PROMPT,
                tools=mcp_tools[:2]  # Sadece ilk 2 tool (hızlı olması için)
            )
            
            # Çalıştır
            result = await agent.ainvoke(
                {"messages": [{"role": "user", "content": query}]},
                config={"recursion_limit": 5}
            )
            
            # Sonucu çıkar
            final_response = ""
            if "messages" in result:
                for msg in reversed(result["messages"]):
                    if hasattr(msg, 'content') and isinstance(msg.content, str) and msg.content.strip():
                        final_response = msg.content.strip()
                        break
            
            await mcp_client.close()
            
            if final_response:
                logger.info("✅ Araştırma tamamlandı")
                return final_response
            
            return "Araştırma sonucu alınamadı."
        
        except Exception as e:
            error_msg = str(e)
            logger.error(f"❌ Attempt {attempt + 1}/{max_retries}: {error_msg}")
            
            # MCP client'ı kapat
            if mcp_client:
                try:
                    await mcp_client.close()
                except:
                    pass
            
            # 429 veya quota hatası
            if "429" in error_msg or "quota" in error_msg.lower() or "rate" in error_msg.lower():
                if attempt < max_retries - 1:
                    # Key'i rotate et ve tekrar dene
                    rotate_key_on_error()
                    logger.info(f"🔄 Key rotated, retrying in 2 seconds...")
                    await asyncio.sleep(2)
                    continue
                else:
                    return f"⚠️ Tüm API key'ler rate limit'e takıldı. Lütfen biraz bekleyin veya Ollama kullanın."
            
            # Diğer hatalar
            return f"❌ Hata: {error_msg}"
    
    return "❌ Maksimum deneme sayısına ulaşıldı."
