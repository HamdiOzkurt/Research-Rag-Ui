"""
Multi-Agent ReAct Mode - LangChain Tool Calling Pattern
Ana agent subagent'ları tool olarak seçer ve çağırır
"""

import os
import logging
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, SystemMessage
from src.config import settings
from src.models import get_llm_model
from .multi_agent_tools import ALL_MULTI_AGENT_TOOLS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_langsmith(project: str = "ai-research-multi-react"):
    """LangSmith tracing setup"""
    if settings.langsmith_api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = project
        os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
        os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
        logger.info(f"[LANGSMITH] {project} aktif")


def _get_multi_model():
    """Multi-Agent için primary model (Groq veya Gemini)"""
    return get_llm_model()  # settings.default_model kullanır


MULTI_REACT_PROMPT = """Sen bir AI Research Coordinator'ısın (Türkçe). Kompleks araştırma görevlerini parçalayıp subagent'lara delege ediyorsun.

🛠️ TOOLS (Subagents):
- **web_research(query, limit)**: Web'de güncel bilgi ara (Firecrawl + Tavily)
- **analyze_research(search_results, original_query)**: Search sonuçlarını analiz et, özet rapor yaz
- **generate_code_examples(research_summary, topic)**: Kod örnekleri oluştur
- **write_final_article(original_query, research_summary, code_examples)**: Final makale yaz

📋 WORKFLOW (Adım adım):
1. Kullanıcı sorusunu analiz et
2. Güncel bilgi gerekiyorsa → `web_research` çağır
3. Search sonuçlarını → `analyze_research` ile işle
4. Kod örnekleri isteniyorsa → `generate_code_examples` çağır
5. Final rapor için → `write_final_article` çağır

⚡ KURALLAR:
- Her tool çağrısını açıkla ("Web'de araştırma yapıyorum...")
- Tool output'u kontrol et, hata varsa tekrar dene
- Basit sorularda web_research gerekmeye bilir (genel bilgi varsa direkt yaz)
- Final çıktı MUTLAKA Markdown formatında olmalı
- Kaynakları URL ile cite et

🎯 ÖRNEK AKIŞ:
User: "Python FastAPI ile JWT authentication nasıl yapılır?"
→ web_research("FastAPI JWT authentication", limit=5)
→ analyze_research(search_results, original_query)
→ generate_code_examples(research_summary, "FastAPI JWT")
→ write_final_article(original_query, research_summary, code_examples)
→ DONE
"""


# LangGraph ReAct agent
setup_langsmith("ai-research-multi-react")
_model = _get_multi_model()

# Tool calling pattern ile multi-agent
multi_react_graph = create_react_agent(
    _model,
    ALL_MULTI_AGENT_TOOLS,
    prompt=MULTI_REACT_PROMPT
)

# Export for LangGraph Studio (if needed)
graph = multi_react_graph
