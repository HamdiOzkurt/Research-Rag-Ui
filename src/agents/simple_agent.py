"""
Basit Tek-Agent Sistem - Rate Limit Dostu
Gemini Free Tier için optimize edilmiş (20 istek/gün)
"""

import asyncio
import logging
from typing import Optional
from langchain_mcp_adapters.client import MultiServerMCPClient
from src.config import settings
from src.models import get_llm_model, sanitize_tool_schema
from deepagents import create_deep_agent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# CACHE SİSTEMİ - API ÇAĞRILARINI AZALTIR
# ============================================================================

_search_cache = {}

def cache_search_result(query: str, result: str):
    """Arama sonuçlarını önbelleğe al"""
    _search_cache[query.lower().strip()] = result

def get_cached_search(query: str) -> Optional[str]:
    """Önbellekten arama sonucu getir"""
    return _search_cache.get(query.lower().strip())


# ============================================================================
# RATE LIMIT AWARE AGENT
# ============================================================================

SIMPLE_RESEARCH_PROMPT = """Sen Türkçe araştırma yapan bir AI asistanısın.

🎯 GÖREVİN:
1. **TEK BİR** tool kullan (en fazla 1 arama)
2. Detaylı Türkçe rapor yaz
3. HEMEN DURDUR

🛠️ ARAÇLAR:
- firecrawl_search(query) - Web scraping
- tavily-search(query, max_results=3) - AI search (ÖNER: Daha hızlı)

📋 RAPOR FORMATI:
# 📊 [Başlık]

## 🎯 Özet
[2-3 cümle özet]

## 📖 Detaylı Açıklama
[En az 3 paragraf - Nedir? Nasıl çalışır? Neden önemli?]

## 💻 Kod Örnekleri (Eğer teknik konuysa)
```python
# Örnek 1: Basit kullanım
kod_burada()
```
**Açıklama:** Ne yaptığı

```python
# Örnek 2: Gelişmiş
advanced_kod()
```

## 🎯 Kullanım Alanları
- Alan 1
- Alan 2
- Alan 3

## ✅ Avantajlar & ❌ Dezavantajlar

### ✅ Artıları:
- Artı 1
- Artı 2

### ❌ Eksileri:
- Eksi 1

## 🚀 Hızlı Başlangıç
1. Adım 1
2. Adım 2
3. Adım 3

## 📚 Kaynaklar
- [Kaynak 1](url)
- [Kaynak 2](url)

---
**🔍 Kaynak:** [Tool adı]

⚠️ KURALLAR:
- SADECE 1 TOOL KULLAN (fazlası rate limit!)
- En az 3 paragraf yaz
- Teknik konularda mutlaka kod örnekleri ver
- Türkçe yaz (kod hariç)
- Araştırma sonrası HEMEN DURDUR
"""


async def create_simple_agent():
    """Rate-limit dostu basit agent"""
    
    logger.info("🔌 MCP bağlanıyor...")
    
    # Sadece Tavily kullan (daha az API çağrısı)
    servers = {}
    
    # Tavily varsa onu kullan (Firecrawl'dan daha ekonomik)
    if hasattr(settings, 'tavily_api_key') and settings.tavily_api_key:
        servers["tavily"] = {
            "command": "npx",
            "args": ["-y", "tavily-mcp@latest"],
            "env": {"TAVILY_API_KEY": settings.tavily_api_key},
            "transport": "stdio"
        }
        logger.info("   ✅ Tavily MCP aktif (önerilen)")
    
    # Yoksa Firecrawl
    if not servers and settings.firecrawl_api_key:
        servers["firecrawl"] = {
            "command": settings.firecrawl_mcp_command,
            "args": settings.firecrawl_mcp_args,
            "env": settings.get_firecrawl_env(),
            "transport": "stdio"
        }
        logger.info("   ✅ Firecrawl MCP aktif")
    
    if not servers:
        raise ValueError("❌ En az bir MCP server gerekli (Tavily veya Firecrawl)")
    
    mcp_client = MultiServerMCPClient(servers)
    tools = await mcp_client.get_tools()
    
    for tool in tools:
        sanitize_tool_schema(tool)
    
    logger.info(f"   📋 {len(tools)} tool yüklendi")
    
    # Model
    model = get_llm_model()
    
    # Agent - DÜŞÜK RECURSION!
    agent = create_deep_agent(
        model=model,
        instructions=SIMPLE_RESEARCH_PROMPT,
        tools=tools
    )
    
    logger.info("✅ Simple agent hazır\n")
    return agent, mcp_client


async def run_simple_research(query: str, verbose: bool = True) -> str:
    """Basit araştırma - Rate limit dostu"""
    
    # Cache kontrolü
    cached = get_cached_search(query)
    if cached:
        logger.info("📦 Önbellekten sonuç döndürülüyor")
        if verbose:
            print("\n💾 (Önbellekten)\n")
        return cached
    
    agent = None
    mcp_client = None
    
    try:
        if verbose:
            print("\n" + "="*70)
            print("🔬 Basit Araştırma (Rate-Limit Dostu)")
            print("="*70)
            print(f"📝 Soru: {query}\n")
        
        agent, mcp_client = await create_simple_agent()
        
        logger.info("🚀 Araştırma başlatılıyor...")
        
        # DÜŞÜK RECURSION LIMIT!
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": query}]},
            config={"recursion_limit": 5}  # AZALTILDI: 15 → 5
        )
        
        # Son mesajı bul
        final_response = ""
        if "messages" in result:
            for msg in reversed(result["messages"]):
                if hasattr(msg, 'content') and isinstance(msg.content, str) and msg.content.strip():
                    final_response = msg.content.strip()
                    break
        
        if not final_response:
            final_response = "❌ Yanıt üretilemedi"
        
        # Cache'e kaydet
        cache_search_result(query, final_response)
        
        if verbose:
            print("\n" + "="*70)
            print("📊 SONUÇ")
            print("="*70)
            print(final_response)
            print("="*70)
        
        return final_response
        
    except Exception as e:
        error_msg = str(e)
        
        if "429" in error_msg or "quota" in error_msg.lower():
            logger.error("❌ Rate limit aşıldı! Çözümler:")
            print("\n⚠️ GEMİNİ API LİMİTİ AŞILDI (429)")
            print("\n🔧 ÇÖZÜMLER:")
            print("1. ⏰ 24 saat bekleyin (günlük 20 istek)")
            print("2. 💳 Gemini API'yi ücretli yapın")
            print("3. 🔄 Farklı bir Google hesabı kullanın")
            print("4. 🏠 Ollama ile local model çalıştırın:")
            print("   • ollama pull llama3.2")
            print("   • .env → DEFAULT_MODEL=ollama:llama3.2")
            return "❌ Rate limit aşıldı. Yukarıdaki çözümlere bakın."
        
        logger.error(f"❌ Hata: {error_msg}")
        return f"❌ Hata: {error_msg}"
        
    finally:
        if mcp_client:
            try:
                await mcp_client.close()
            except:
                pass


# ============================================================================
# BATCH ARAŞTIRMA - GÜNLÜK LİMİTİ PLANLAYARAK KULLAN
# ============================================================================

async def run_batch_research(queries: list[str], delay: int = 5) -> dict[str, str]:
    """
    Birden fazla soruyu sırası ile araştırır
    
    Args:
        queries: Sorular listesi
        delay: Her soru arasında bekleme (saniye)
    
    Returns:
        {soru: yanıt} dictionary
    """
    
    results = {}
    
    print(f"\n📦 Batch araştırma: {len(queries)} soru")
    print(f"   ⏱️ Her soru arası {delay}s bekleme")
    print(f"   ⏰ Tahmini süre: {len(queries) * delay // 60} dakika\n")
    
    for i, query in enumerate(queries, 1):
        print(f"\n[{i}/{len(queries)}] {query}")
        
        try:
            result = await run_simple_research(query, verbose=False)
            results[query] = result
            print("   ✅ Tamamlandı")
            
            # Rate limit için bekle
            if i < len(queries):
                print(f"   ⏳ {delay}s bekleniyor...")
                await asyncio.sleep(delay)
                
        except Exception as e:
            results[query] = f"❌ Hata: {str(e)}"
            print(f"   ❌ Başarısız: {str(e)}")
    
    return results
