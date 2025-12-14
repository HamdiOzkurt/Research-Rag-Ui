"""
Basit Araştırma Agent'ı - Hızlı Mod
Multi API Key Rotation + 429 Protection + LangSmith Tracing
"""
import asyncio
import os
from deepagents import create_deep_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from src.config import settings
from src.models import get_llm_model, sanitize_tool_schema, rotate_key_on_error
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============ LANGSMITH TRACING ============
def setup_langsmith():
    """LangSmith tracing'i aktifleştir"""
    if settings.langsmith_api_key:
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        os.environ.setdefault("LANGCHAIN_PROJECT", "ai-research-simple")
        os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
        logger.info("[LANGSMITH] Tracing aktif: ai-research-simple")
        return True
    return False

# Başlangıçta çalıştır
_langsmith_enabled = setup_langsmith()

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
                logger.info(f"[FAST] Araştırma başlatılıyor {key_info}...")
            
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
            # deepagents venv sürümü: `system_prompt` kullanır (instructions değil)
            agent = create_deep_agent(
                model=model,
                tools=mcp_tools[:2],  # Sadece ilk 2 tool (hızlı olması için)
                system_prompt=SIMPLE_PROMPT,
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
            
            # MCP client cleanup (close metodu yoksa __aexit__ dene)
            try:
                if hasattr(mcp_client, 'close'):
                    await mcp_client.close()
                elif hasattr(mcp_client, '__aexit__'):
                    await mcp_client.__aexit__(None, None, None)
            except Exception:
                pass  # Kapatma hatalarını yoksay
            
            if final_response:
                logger.info("[OK] Araştırma tamamlandı")
                return final_response
            
            return "Araştırma sonucu alınamadı."
        
        except Exception as e:
            error_msg = str(e)
            logger.error(f"[ERROR] Attempt {attempt + 1}/{max_retries}: {error_msg}")
            
            # MCP client'ı kapat
            if mcp_client:
                try:
                    if hasattr(mcp_client, 'close'):
                        await mcp_client.close()
                    elif hasattr(mcp_client, '__aexit__'):
                        await mcp_client.__aexit__(None, None, None)
                except Exception:
                    pass
            
            # 429 veya quota hatası
            if "429" in error_msg or "quota" in error_msg.lower() or "rate" in error_msg.lower():
                if attempt < max_retries - 1:
                    # Key'i rotate et ve tekrar dene
                    rotate_key_on_error()
                    logger.info(f"[RETRY] Key rotated, retrying in 2 seconds...")
                    await asyncio.sleep(2)
                    continue
                else:
                    return f"⚠️ Tüm API key'ler rate limit'e takıldı. Lütfen biraz bekleyin veya Ollama kullanın."

            # Geçersiz API key (Gemini)
            if (
                "API_KEY_INVALID" in error_msg
                or "API key not valid" in error_msg
                or ("INVALID_ARGUMENT" in error_msg and "API key" in error_msg)
            ):
                # Eğer birden fazla key varsa, sıradakini dene
                if settings.google_api_keys and len(settings.google_api_keys) > 1 and attempt < max_retries - 1:
                    rotate_key_on_error()
                    logger.warning("🔑 Geçersiz API key tespit edildi, sonraki key deneniyor...")
                    await asyncio.sleep(1)
                    continue

                return (
                    "[ERROR] Google Gemini API key geçersiz.\n\n"
                    "[OK] Çözüm:\n"
                    "- `multi_agent_search/.env` içine geçerli key girin:\n"
                    "  `GOOGLE_API_KEYS=AIza...` (virgülle birden fazla da olabilir)\n"
                    "- Backend'i yeniden başlatın.\n\n"
                    "Alternatif: Lokal ve ücretsiz çalıştırmak için `DEFAULT_MODEL=ollama:llama3.2` kullanın."
                )
            
            # Diğer hatalar
            return f"[ERROR] Hata: {error_msg}"
    
    return "[ERROR] Maksimum deneme sayısına ulaşıldı."
