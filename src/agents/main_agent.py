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
        print("[OK] LangSmith tracing aktif!")
        print("   📊 https://smith.langchain.com")
        return True
    return False


# ============ MODEL ============

def get_llm_model():
    """Kullanılabilir LLM modelini döndürür"""
    model_string = settings.get_available_model()
    provider, model_name = settings.get_model_provider(model_string)
    print(f"[MODEL] Model: {provider}:{model_name}")
    
    if provider == "google_genai" and settings.google_api_key:
        os.environ["GOOGLE_API_KEY"] = settings.google_api_key
    if provider == "ollama":
        os.environ["OLLAMA_HOST"] = settings.ollama_base_url
        
    return init_chat_model(model_string, temperature=0.3)  # Daha tutarlı sonuçlar için


# ============ SYSTEM PROMPT ============

RESEARCH_INSTRUCTIONS = """You are an expert Turkish Research AI (DeepAgent) that creates professional, comprehensive reports.

🎯 YOUR MISSION:
1. PLAN with write_todos
2. SEARCH using tools
3. SAVE context to files (write_file)
4. CREATE final report
5. STOP immediately

🛠️ DEEPAGENT TOOLS (Built-in):
- write_todos: Create task plan
- read_file/write_file/edit_file/ls: File system for context management
- task: Spawn subagent for complex subtasks

🔍 RESEARCH TOOLS:
- firecrawl_search(query) - Deep web scraping
- tavily-search(query, max_results) - AI-powered search
- github_search_repositories(query) - Find code repositories
- firecrawl_scrape(url) - Extract full page content

📋 WORKFLOW:
1. write_todos: ["Analyze query", "Search web", "Save results", "Write report"]
2. Search with research tools
3. write_file: Save long results to "research_data.md" (prevent context overflow)
4. Generate final report
5. Done!

📋 PROFESSIONAL REPORT FORMAT (ALWAYS IN TURKISH):

# 📊 [Başlık - Açıklayıcı ve Profesyonel]

---

## 🎯 Özet
[2-3 cümle ile konunun özü. Net, anlaşılır ve ilgi çekici yazın.]

---

## 📖 Detaylı Açıklama

### [SEARCH] Nedir?
[İlk paragraf: Konunun tanımı, ne olduğu, temel özellikleri]

### 💡 Nasıl Çalışır?
[İkinci paragraf: Çalışma prensibi, arkasındaki mantık]

### ⚡ Neden Önemli?
[Üçüncü paragraf: Avantajları, kullanım nedenleri, faydaları]

---

## 💻 Kod Örnekleri

### Örnek 1: Temel Kullanım
```python
# Basit ve anlaşılır örnek
# Her satırı açıklayın

# Örnek kod buraya
```
**Açıklama:** [Bu kodun ne yaptığını açıklayın]

### Örnek 2: Gelişmiş Kullanım
```python
# Daha karmaşık, gerçek dünya örneği
# Pratik bir senaryo gösterin

# Örnek kod buraya
```
**Açıklama:** [Bu kodun ne yaptığını açıklayın]

### Örnek 3: Best Practices
```python
# En iyi uygulamalar
# Profesyonel kullanım

# Örnek kod buraya
```
**Açıklama:** [Bu kodun ne yaptığını açıklayın]

---

## 🎯 Kullanım Alanları

| Alan | Açıklama |
|------|----------|
| 🔹 **[Alan 1]** | [Kısa açıklama] |
| 🔹 **[Alan 2]** | [Kısa açıklama] |
| 🔹 **[Alan 3]** | [Kısa açıklama] |

---

## [OK] Avantajlar & [ERROR] Dezavantajlar

### [OK] Avantajlar:
- ✓ [Avantaj 1]
- ✓ [Avantaj 2]
- ✓ [Avantaj 3]

### [ERROR] Dezavantajlar:
- ✗ [Dezavantaj 1]
- ✗ [Dezavantaj 2]

---

## [START] Hızlı Başlangıç

1. **Kurulum:**
   ```bash
   # Kurulum komutu
   ```

2. **İlk Adımlar:**
   - [Adım 1]
   - [Adım 2]
   - [Adım 3]

---

## 📚 Kaynaklar

1. 🔗 [Kaynak Başlığı](URL) - Kısa açıklama
2. 🔗 [Kaynak Başlığı](URL) - Kısa açıklama
3. 🔗 [Kaynak Başlığı](URL) - Kısa açıklama

---

## 💡 İpuçları & Notlar

> **💡 İpucu:** [Önemli bir ipucu]

> **[WARN] Dikkat:** [Uyarı veya önemli not]

> **🎓 Öğrenme Kaynağı:** [Ek öğrenme materyali]

---

**📅 Rapor Tarihi:** {bugünün tarihi}  
**[SEARCH] Arama Kaynağı:** [Kullanılan tool]

---

🎯 CRITICAL RULES:
- Search ONLY ONCE with the most relevant tool
- ALWAYS include minimum 3 code examples with explanations
- Write minimum 3 detailed paragraphs in "Detaylı Açıklama"
- Use emojis for better readability (📊 🎯 💻 [OK] etc.)
- Include tables, lists, and formatted sections
- Add practical tips and warnings
- STOP immediately after writing the report
- Write EVERYTHING in Turkish (except code)

📝 EXAMPLE QUERY: "Python pandas nedir?"
[OK] YOU SHOULD:
1. tavily-search(query="Python pandas tutorial examples best practices", max_results=5)
2. Write comprehensive report with:
   - Professional title with emoji
   - 3+ detailed paragraphs
   - 3 code examples with explanations
   - Use cases table
   - Pros & cons
   - Quick start guide
   - Multiple sources
   - Tips & warnings
3. STOP

[ERROR] NEVER:
- Search more than once
- Write short, incomplete reports
- Skip code examples
- Write in English (except code comments)
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
        raise ValueError("[ERROR] FIRECRAWL_API_KEY gerekli! .env dosyasını kontrol edin.")
    
    print("\n🔌 MCP Servers bağlanıyor...")
    
    # Her MCP'yi ayrı ayrı test et
    working_servers = {}
    
    # 1. Firecrawl (zorunlu)
    print("   🔥 Firecrawl test ediliyor...")
    try:
        test_client = MultiServerMCPClient({
            "firecrawl": {
                "command": settings.firecrawl_mcp_command,
                "args": settings.firecrawl_mcp_args,
                "env": settings.get_firecrawl_env(),
                "transport": "stdio"
            }
        })
        test_tools = await test_client.get_tools()
        if test_tools:
            working_servers["firecrawl"] = {
                "command": settings.firecrawl_mcp_command,
                "args": settings.firecrawl_mcp_args,
                "env": settings.get_firecrawl_env(),
                "transport": "stdio"
            }
            print(f"      [OK] Firecrawl OK ({len(test_tools)} tools)")
    except Exception as e:
        print(f"      [ERROR] Firecrawl başarısız: {str(e)[:100]}")
        raise ValueError("Firecrawl MCP zorunlu ama başlatılamadı!")
    
    # 2. Tavily (opsiyonel)
    if hasattr(settings, 'tavily_api_key') and settings.tavily_api_key:
        print("   [SEARCH] Tavily test ediliyor...")
        try:
            test_client = MultiServerMCPClient({
                "tavily": {
                    "command": "npx",
                    "args": ["-y", "tavily-mcp@latest"],
                    "env": {"TAVILY_API_KEY": settings.tavily_api_key},
                    "transport": "stdio"
                }
            })
            test_tools = await test_client.get_tools()
            if test_tools:
                working_servers["tavily"] = {
                    "command": "npx",
                    "args": ["-y", "tavily-mcp@latest"],
                    "env": {"TAVILY_API_KEY": settings.tavily_api_key},
                    "transport": "stdio"
                }
                print(f"      [OK] Tavily OK ({len(test_tools)} tools)")
        except Exception as e:
            print(f"      [WARN] Tavily atlandı: {str(e)[:100]}")
    
    # 3. GitHub (opsiyonel) - Community package via npx
    # Not: GitHub'ın resmi MCP'si Docker gerektirir, bu yüzden community versiyonunu kullanıyoruz
    if hasattr(settings, 'github_token') and settings.github_token:
        print("   💻 GitHub test ediliyor...")
        try:
            test_client = MultiServerMCPClient({
                "github": {
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-github"],
                    "env": {"GITHUB_PERSONAL_ACCESS_TOKEN": settings.github_token},
                    "transport": "stdio"
                }
            })
            test_tools = await test_client.get_tools()
            if test_tools:
                working_servers["github"] = {
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-github"],
                    "env": {"GITHUB_PERSONAL_ACCESS_TOKEN": settings.github_token},
                    "transport": "stdio"
                }
                print(f"      [OK] GitHub OK ({len(test_tools)} tools)")
        except Exception as e:
            print(f"      [WARN] GitHub atlandı: {str(e)[:100]}")
    
    # Final MCP client - sadece çalışan serverlarla
    print(f"\n[OK] {len(working_servers)} MCP server aktif: {', '.join(working_servers.keys())}")
    mcp_client = MultiServerMCPClient(working_servers)
    mcp_tools = await mcp_client.get_tools()
    print(f"   📋 Toplam {len(mcp_tools)} tool yüklendi")
    
    # Tool schema'larını Gemini uyumlu hale getir
    for tool in mcp_tools:
        sanitize_tool_schema(tool)
    
    # LLM modelini al
    model = get_llm_model()
    
    print("[MODEL] Model:", model.model_name if hasattr(model, 'model_name') else "Unknown")
    
    # DeepAgent oluştur
    # deepagents venv sürümü: `system_prompt` kullanır (instructions değil)
    agent = create_deep_agent(
        model=model,
        tools=mcp_tools,
        system_prompt=RESEARCH_INSTRUCTIONS,
    )
    
    print("[OK] DeepAgent hazır!\n")
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
                print("[START] Araştırma başlatılıyor...\n")
            
            # Rate limit için başlangıç bekleme
            if attempt > 0:
                wait = 10 * attempt
                print(f"⏳ {wait} saniye bekleniyor (rate limit)...")
                await asyncio.sleep(wait)
            
            # Agent'ı çalıştır (recursion_limit ile sonsuz döngüyü engelle)
            result = await agent.ainvoke(
                {"messages": [{"role": "user", "content": question}]},
                config={"recursion_limit": 15}  # Artırıldı: 10 → 15
            )
            
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
                        
                        if texts:
                            final_response = ' '.join(texts).strip()
                            break
                        
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
                print("\n[ERROR] UYARI: Agent yanıt üretti ama içerik bulunamadı!")
                print("   Yukarıdaki debug loglarını kontrol edin.")
                return "[ERROR] Araştırma tamamlandı ama yanıt formatlanamadı. Debug loglarına bakın."
            
            return final_response
        
        except Exception as e:
            error_msg = str(e)
            
            # Rate limit hatası
            if "429" in error_msg or "Resource exhausted" in error_msg or "quota" in error_msg.lower():
                wait_time = 30 * (attempt + 1)
                print(f"\n[WARN] Rate limit aşıldı (429 Error)")
                print(f"   {wait_time} saniye bekleniyor... (Deneme {attempt+1}/{max_retries})")
                
                # MCP client'ı kapat
                if mcp_client:
                    try:
                        if hasattr(mcp_client, 'close'):
                            await mcp_client.close()
                        elif hasattr(mcp_client, '__aexit__'):
                            await mcp_client.__aexit__(None, None, None)
                    except Exception:
                        pass
                
                await asyncio.sleep(wait_time)
                continue
            
            # Diğer hatalar
            error_msg = f"[ERROR] Hata: {str(e)}"
            if verbose:
                print(f"\n{error_msg}")
                import traceback
                traceback.print_exc()
            
            return error_msg
        
        finally:
            # MCP client'ı her durumda kapat
            if mcp_client:
                try:
                    if hasattr(mcp_client, 'close'):
                        await mcp_client.close()
                    elif hasattr(mcp_client, '__aexit__'):
                        await mcp_client.__aexit__(None, None, None)
                except Exception as close_error:
                    if verbose:
                        print(f"[WARN] MCP client kapatma hatası: {close_error}")
    
    return "[ERROR] Maksimum deneme sayısına ulaşıldı. Lütfen birkaç dakika sonra tekrar deneyin."


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
            print(f"\n[ERROR] Beklenmeyen hata: {e}")
            import traceback
            traceback.print_exc()