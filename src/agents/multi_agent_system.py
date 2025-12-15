"""
Multi-Agent Research System - FIXED VERSION
Gerçek çoklu ajan mimarisi - Supervisor + Researcher + Coder + Writer
LangSmith Tracing desteği
"""

import os
from typing import Annotated, Literal, TypedDict, List
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from src.config import settings
from src.models import get_llm_model, sanitize_tool_schema
from deepagents import create_deep_agent
import logging

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============ LANGSMITH TRACING ============
def setup_langsmith():
    """LangSmith tracing'i aktifleştir"""
    if settings.langsmith_api_key:
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        os.environ.setdefault("LANGCHAIN_PROJECT", "ai-research-multi-agent")
        os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
        logger.info("[LANGSMITH] Tracing aktif: ai-research-multi-agent")
        return True
    return False

_langsmith_enabled = setup_langsmith()


# =============================================================================
# PYDANTIC MODELS (STRUCTURED OUTPUT)
# =============================================================================

class SupervisorPlan(BaseModel):
    """Supervisor'ın oluşturduğu plan"""
    agents: List[str] = Field(description="Çalışacak agentların sırası")
    reason: str = Field(description="Bu planın seçilme sebebi")
    estimated_time: str = Field(default="2-3 dakika", description="Tahmini süre")


# =============================================================================
# STATE DEFINITION
# =============================================================================

class AgentState(TypedDict):
    """Tüm agentlar arasında paylaşılan state"""
    messages: Annotated[list, add_messages]
    query: str
    research_results: str
    code_examples: str
    final_report: str
    next_agent: str
    supervisor_plan: str
    supervisor_reason: str


# =============================================================================
# AGENT PROMPTS
# =============================================================================

SUPERVISOR_PROMPT = """Sen bir Araştırma Yöneticisisin. Kullanıcının sorusunu analiz et ve EN UYGUN agent stratejisini belirle.

Kullanılabilir Agentlar:
- researcher: Web araması, genel bilgi toplama (Firecrawl, Tavily)
- coder: Kod örnekleri, teknik implementasyon
- writer: Final rapor yazma

Soru Analizi:
{query}

Soruyu analiz et ve şu kriterlere göre karar ver:

1. **Sadece Bilgi İstiyorsa** → researcher → writer
   Örnek: "Python nedir?", "DeepAgents nedir?"
   
2. **Kod İstiyorsa** → researcher → coder → writer
   Örnek: "Python ile veri analizi nasıl yapılır?", "Kod örnekleri göster"
   
3. **Sadece Kod İstiyorsa** → coder → writer
   Örnek: "Python pandas kod örneği", "FastAPI authentication kodu"

4. **Karmaşık Araştırma** → researcher → coder → writer
   Örnek: "Machine learning projeleri ve implementasyonu"

CEVAP FORMATI (SADECE BU FORMATTA YAZ):
Plan: [agent1] -> [agent2] -> [agent3]
Sebep: [Kısa açıklama]

Örnek:
Plan: researcher -> coder -> writer
Sebep: Soru hem bilgi hem kod gerektiriyor
"""

RESEARCHER_PROMPT = """Sen bir Araştırmacı Ajansın. Görevin:

1. Kullanıcının sorusunu araştır
2. Firecrawl veya Tavily ile web araması yap
3. Bulduğun bilgileri özetle

Soru: {query}

Araştırma yap ve bulgularını Türkçe özetle. Kaynakları belirt.
"""

CODER_PROMPT = """Sen bir Kod Uzmanısın. Görevin:

1. Konuyla ilgili kod örnekleri bul veya oluştur
2. En az 3 farklı örnek hazırla (basit, orta, gelişmiş)
3. Her örneği açıkla

Konu: {query}
Araştırma Sonuçları: {research_results}

3 kod örneği oluştur ve açıkla. Türkçe yaz.
"""

WRITER_PROMPT = """Sen bir Profesyonel Rapor Yazarısın. Görevin:

Aşağıdaki bilgileri kullanarak kapsamlı bir Türkçe rapor yaz:

Soru: {query}
Araştırma: {research_results}
Kod Örnekleri: {code_examples}

Rapor Formatı:
# 📊 [Başlık]

## 🎯 Özet
[2-3 cümle]

## 📖 Detaylı Açıklama
### [SEARCH] Nedir?
[Paragraf]

### 💡 Nasıl Çalışır?
[Paragraf]

### ⚡ Neden Önemli?
[Paragraf]

## 💻 Kod Örnekleri
{code_examples}

## 🎯 Kullanım Alanları
[Tablo]

## [OK] Avantajlar & [ERROR] Dezavantajlar

## [START] Hızlı Başlangıç

## 📚 Kaynaklar

## 💡 İpuçları & Notlar

Profesyonel, detaylı ve görsel bir rapor yaz!
"""


# =============================================================================
# AGENT NODES (DEEPAGENTS POWERED + ERROR HANDLING)
# =============================================================================

async def supervisor_node(state: AgentState) -> AgentState:
    """Yönetici: Hangi agentların çalışacağına karar verir"""
    try:
        logger.info("🎯 Supervisor başladı...")
        model = get_llm_model()
        
        supervisor = create_deep_agent(
            model=model,
            tools=[],
            system_prompt=SUPERVISOR_PROMPT.format(query=state["query"]),
        )
        
        result = await supervisor.ainvoke(
            {"messages": [{"role": "user", "content": state["query"]}]},
            config={"recursion_limit": 15}  # Artırıldı: 3 → 15
        )
        
        response_content = ""
        if "messages" in result:
            for msg in reversed(result["messages"]):
                if hasattr(msg, 'content') and isinstance(msg.content, str):
                    response_content = msg.content
                    break
        
        # Plan'ı parse et
        plan_line = ""
        reason_line = ""
        for line in response_content.split('\n'):
            if line.strip().startswith("Plan:"):
                plan_line = line.replace("Plan:", "").strip()
            elif line.strip().startswith("Sebep:"):
                reason_line = line.replace("Sebep:", "").strip()
        
        agents_order = plan_line.lower() if plan_line else response_content.lower()
        
        # İlk agentı belirle
        if "researcher" in agents_order:
            state["next_agent"] = "researcher"
        elif "coder" in agents_order:
            state["next_agent"] = "coder"
        else:
            # Fallback: Query analizi
            query_lower = state["query"].lower()
            if any(word in query_lower for word in ["kod", "code", "örnek", "example"]):
                state["next_agent"] = "coder"
            else:
                state["next_agent"] = "researcher"
        
        state["supervisor_plan"] = agents_order
        state["supervisor_reason"] = reason_line
        
        state["messages"].append(AIMessage(
            content=f"🎯 Plan: {agents_order}\n💡 Sebep: {reason_line}" if reason_line 
            else f"🎯 Plan: {agents_order}"
        ))
        
        logger.info(f"[OK] Supervisor tamamlandı. Next: {state['next_agent']}")
        
    except Exception as e:
        logger.error(f"[ERROR] Supervisor Error: {str(e)}", exc_info=True)
        state["messages"].append(AIMessage(content=f"⚠️ Supervisor hatası, varsayılan plan"))
        state["next_agent"] = "researcher"
        state["supervisor_plan"] = "researcher -> coder -> writer"
        state["supervisor_reason"] = "Varsayılan plan (hata)"
    
    return state


async def researcher_node(state: AgentState, mcp_tools: list) -> AgentState:
    """Araştırmacı: Web araması yapar"""
    try:
        logger.info("[SEARCH] Researcher başladı...")
        model = get_llm_model()
        
        search_tools = [t for t in mcp_tools if any(
            name in t.name.lower() 
            for name in ['search', 'scrape', 'tavily', 'firecrawl']
        )]
        
        if not search_tools:
            raise ValueError("Arama tool'u bulunamadı")
        
        researcher = create_deep_agent(
            model=model,
            tools=search_tools,
            system_prompt=RESEARCHER_PROMPT.format(query=state["query"]),
        )
        
        result = await researcher.ainvoke(
            {"messages": [{"role": "user", "content": state["query"]}]},
            config={"recursion_limit": 8}  # Artırıldı: 5 → 8
        )
        
        research_results = ""
        if "messages" in result:
            for msg in reversed(result["messages"]):
                if hasattr(msg, 'content') and isinstance(msg.content, str) and msg.content.strip():
                    research_results = msg.content
                    break
        
        state["research_results"] = research_results or "Arama sonucu bulunamadı"
        state["messages"].append(AIMessage(content=f"[SEARCH] Araştırma tamamlandı"))
        logger.info("[OK] Researcher tamamlandı")
        
    except Exception as e:
        logger.error(f"[ERROR] Researcher Error: {str(e)}", exc_info=True)
        state["research_results"] = "Araştırma başarısız"
        state["messages"].append(AIMessage(content=f"⚠️ Araştırma hatası, devam ediliyor"))
    
    # Next agent belirleme
    if "coder" in state.get("supervisor_plan", "").lower():
        state["next_agent"] = "coder"
    else:
        state["next_agent"] = "writer"
    
    return state


async def coder_node(state: AgentState) -> AgentState:
    """Kodcu: Kod örnekleri oluşturur"""
    try:
        logger.info("💻 Coder başladı...")
        model = get_llm_model()
        
        coder = create_deep_agent(
            model=model,
            tools=[],
            system_prompt=CODER_PROMPT.format(
                query=state["query"],
                research_results=state.get("research_results", "")
            ),
        )
        
        result = await coder.ainvoke(
            {"messages": [{"role": "user", "content": "Kod örnekleri oluştur"}]},
            config={"recursion_limit": 20}  # Artırıldı: 2 → 20
        )
        
        code_examples = ""
        if "messages" in result:
            for msg in reversed(result["messages"]):
                if hasattr(msg, 'content') and isinstance(msg.content, str) and msg.content.strip():
                    code_examples = msg.content
                    break
        
        state["code_examples"] = code_examples or "Kod örneği oluşturulamadı"
        state["messages"].append(AIMessage(content=f"💻 Kod örnekleri hazırlandı"))
        logger.info("[OK] Coder tamamlandı")
        
    except Exception as e:
        logger.error(f"[ERROR] Coder Error: {str(e)}", exc_info=True)
        state["code_examples"] = "Kod örnekleri oluşturulamadı"
        state["messages"].append(AIMessage(content=f"⚠️ Kod oluşturma hatası"))
    
    state["next_agent"] = "writer"
    return state


async def writer_node(state: AgentState) -> AgentState:
    """Yazar: Final raporu oluşturur"""
    try:
        logger.info("📝 Writer başladı...")
        model = get_llm_model()
        
        writer = create_deep_agent(
            model=model,
            tools=[],
            system_prompt=WRITER_PROMPT.format(
                query=state["query"],
                research_results=state.get("research_results", ""),
                code_examples=state.get("code_examples", "")
            ),
        )
        
        result = await writer.ainvoke(
            {"messages": [{"role": "user", "content": "Profesyonel rapor yaz"}]},
            config={"recursion_limit": 25}  # Artırıldı: 5 → 25
        )
        
        final_report = ""
        if "messages" in result:
            for msg in reversed(result["messages"]):
                if hasattr(msg, 'content') and isinstance(msg.content, str) and msg.content.strip():
                    final_report = msg.content
                    break
        
        if not final_report:
            # Fallback rapor
            final_report = f"""# {state['query']}

## Araştırma Sonuçları
{state.get('research_results', 'Bilgi bulunamadı')}

## Kod Örnekleri
{state.get('code_examples', 'Örnek bulunamadı')}
"""
        
        state["final_report"] = final_report
        state["messages"].append(AIMessage(content=f"📝 Rapor tamamlandı"))
        logger.info("[OK] Writer tamamlandı")
        
    except Exception as e:
        logger.error(f"[ERROR] Writer Error: {str(e)}", exc_info=True)
        state["final_report"] = f"# Rapor Oluşturulamadı\n\nHata: {str(e)}"
        state["messages"].append(AIMessage(content=f"⚠️ Rapor yazma hatası"))
    
    state["next_agent"] = "END"
    return state


# =============================================================================
# ROUTER
# =============================================================================

def route_agent(state: AgentState) -> Literal["researcher", "coder", "writer", "END"]:
    """Bir sonraki agentı belirler"""
    next_agent = state.get("next_agent", "END")
    
    if next_agent == "END":
        return END
    return next_agent


# =============================================================================
# GRAPH BUILDER
# =============================================================================

async def create_multi_agent_system():
    """Multi-Agent sistemi oluşturur"""
    
    logger.info("🔌 MCP Servers bağlanıyor...")
    
    mcp_servers = {
        "firecrawl": {
            "command": settings.firecrawl_mcp_command,
            "args": settings.firecrawl_mcp_args,
            "env": settings.get_firecrawl_env(),
            "transport": "stdio"
        }
    }
    
    if hasattr(settings, 'tavily_api_key') and settings.tavily_api_key:
        mcp_servers["tavily"] = {
            "command": "npx",
            "args": ["-y", "tavily-mcp@latest"],
            "env": {"TAVILY_API_KEY": settings.tavily_api_key},
            "transport": "stdio"
        }
    
    mcp_client = MultiServerMCPClient(mcp_servers)
    mcp_tools = await mcp_client.get_tools()
    
    for tool in mcp_tools:
        sanitize_tool_schema(tool)
    
    logger.info(f"[OK] {len(mcp_tools)} tool yüklendi")
    
    # Graph oluştur
    workflow = StateGraph(AgentState)
    
    # Node'ları ekle - LAMBDA YERINE DOĞRUDAN ASYNC WRAPPER
    async def researcher_wrapper(state: AgentState) -> AgentState:
        return await researcher_node(state, mcp_tools)
    
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("researcher", researcher_wrapper)
    workflow.add_node("coder", coder_node)
    workflow.add_node("writer", writer_node)
    
    # Edge'leri ekle
    workflow.set_entry_point("supervisor")
    
    # Supervisor'dan routing
    workflow.add_conditional_edges(
        "supervisor",
        route_agent,
        {
            "researcher": "researcher",
            "coder": "coder",
            "writer": "writer",
            "END": END
        }
    )
    
    # Researcher'dan routing
    workflow.add_conditional_edges(
        "researcher",
        route_agent,
        {
            "coder": "coder",
            "writer": "writer",
            "END": END
        }
    )
    
    # Coder'dan routing
    workflow.add_conditional_edges(
        "coder",
        route_agent,
        {
            "writer": "writer",
            "END": END
        }
    )
    
    # Writer'dan END
    workflow.add_edge("writer", END)
    
    # Compile
    app = workflow.compile()
    
    logger.info("[OK] Multi-Agent sistem hazır!")
    return app, mcp_client


# =============================================================================
# RUN FUNCTION
# =============================================================================

async def run_multi_agent_research(query: str, verbose: bool = True) -> str:
    """Multi-agent sistemi çalıştırır"""
    
    app, mcp_client = await create_multi_agent_system()
    
    # Initial state
    initial_state = {
        "messages": [],
        "query": query,
        "research_results": "",
        "code_examples": "",
        "final_report": "",
        "next_agent": "",
        "supervisor_plan": "",
        "supervisor_reason": ""
    }
    
    if verbose:
        print("[START] Multi-Agent araştırma başlatılıyor...\n")
    
    try:
        result = await app.ainvoke(
            initial_state,
            config={"recursion_limit": 15}
        )
        
        final_report = result.get("final_report", "Rapor oluşturulamadı")
        
        if verbose:
            print("\n" + "="*70)
            print("📊 SONUÇ")
            print("="*70)
            print(final_report)
            print("="*70)
        
        return final_report
        
    except Exception as e:
        logger.error(f"[ERROR] Run Error: {str(e)}", exc_info=True)
        return f"Sistem hatası: {str(e)}"
    finally:
        # MCP client'ı temizle
        try:
            await mcp_client.cleanup()
        except:
            pass 