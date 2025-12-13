"""
Multi-Agent Search System
Ana giriş noktası - Sadece DeepAgents

Kullanım:
    python -m src.main "Araştırma sorunuz"
    
Veya interaktif mod:
    python -m src.main
"""
import sys
import asyncio

from .config import settings
from .agents import run_simple_research, interactive_mode


def print_banner():
    """ASCII banner gösterir"""
    banner = """
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║   🔍 Multi-Agent Araştırma Sistemi                           ║
║   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━                           ║
║   DeepAgents + Firecrawl MCP                                 ║
║                                                               ║
║   📊 LangSmith'te akışı izleyebilirsiniz:                    ║
║      https://smith.langchain.com                             ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def check_config() -> bool:
    """Konfigürasyonu kontrol eder"""
    print("\n📋 Konfigürasyon Kontrolü:")
    
    api_status = settings.validate_api_keys()
    
    status_icons = {True: "✅", False: "❌"}
    
    for api, status in api_status.items():
        icon = status_icons[status]
        required = "(zorunlu)" if api == "firecrawl" else "(opsiyonel)"
        print(f"   {icon} {api.upper()}_API_KEY {required}")
    
    # Firecrawl zorunlu
    if not api_status["firecrawl"]:
        print("\n⚠️  FIRECRAWL_API_KEY gerekli!")
        print("   https://www.firecrawl.dev/app/api-keys adresinden alabilirsiniz.")
        return False
    
    # En az bir LLM gerekli (Gemini veya Ollama)
    has_llm = (
        api_status.get("google (gemini)", False) or 
        api_status.get("ollama (local)", False)
    )
    if not has_llm:
        print("\n⚠️  En az bir LLM gerekli (Gemini veya Ollama)!")
        return False
    
    # LangSmith uyarısı
    if not api_status.get("langsmith", False):
        print("\n💡 İpucu: LANGSMITH_API_KEY eklerseniz akışı izleyebilirsiniz!")
        print("   https://smith.langchain.com")
    
    print(f"\n✅ Model: {settings.default_model}")
    return True


async def main_async(question: str = None):
    """Ana async fonksiyon"""
    
    print_banner()
    
    if not check_config():
        return
    
    # Soru verilmediyse interaktif mod
    if question is None:
        await interactive_mode()
        return
    
    # Tek soru modu - Rate limit dostu
    result = await run_simple_research(question, verbose=True)
    print(f"\n{result}")


def main():
    """CLI giriş noktası"""
    
    # Argümanları parse et
    args = sys.argv[1:]
    question = None
    
    for i, arg in enumerate(args):
        if arg in ["--help", "-h"]:
            print("""
🔍 Multi-Agent Araştırma Sistemi

Kullanım:
    python -m src.main [SORU]

Örnekler:
    python -m src.main "Python web scraping nasıl yapılır?"
    python -m src.main  # İnteraktif mod

Gereksinimler:
    - FIRECRAWL_API_KEY (.env dosyasında)
    - OPENAI_API_KEY veya ANTHROPIC_API_KEY
    
Opsiyonel:
    - LANGSMITH_API_KEY (akış izleme için)
            """)
            return
        elif not arg.startswith("-"):
            question = arg
    
    # Async çalıştır
    asyncio.run(main_async(question))


if __name__ == "__main__":
    main()
