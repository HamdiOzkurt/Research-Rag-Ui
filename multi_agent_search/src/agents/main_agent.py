"""Deep Research mode.

This module powers the **deep** mode used by the backend SSE endpoint.

Implementation goal:
- Match LangChain/LangGraph's proven ReAct-style agent pattern (model <-> tools loop).
- Be compatible with LangGraph Studio by relying on a LangGraph graph.
"""
import asyncio
import os
import json
import logging
import uuid
from typing import Optional, Any, Dict

from langchain_core.messages import AIMessage, HumanMessage

from .deep_graph import graph as deep_graph, setup_langsmith as setup_langsmith_graph

from ..config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============ LANGSMITH ============

def setup_langsmith():
    """LangSmith tracing'i etkinleştirir"""
    tracing_enabled = os.getenv("LANGSMITH_TRACING", "").strip().lower() in {"1", "true", "yes", "on"}
    if not tracing_enabled:
        return False

    if not settings.langsmith_api_key:
        return False

    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "ai-research-deep")
    os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
    os.environ.setdefault("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
    logger.info("[LANGSMITH] Tracing aktif")
    return True


# NOTE: Deep mode is now powered by LangGraph ReAct graph in `deep_graph.py`.


# ============ ARAŞTIRMA ÇALIŞTIRMA ============

async def run_research(question: str, verbose: bool = True):
    """Deep research runner (SSE-friendly).

    Uses LangGraph ReAct agent graph (`deep_graph.py`) so the architecture is:
    request -> model <-> tools -> result.
    """
    setup_langsmith()
    setup_langsmith_graph(project="ai-research-deep")

    run_id = str(uuid.uuid4())
    config: Dict[str, Any] = {
        "recursion_limit": 40,
        "run_id": run_id,
        "run_name": "deep-react",
    }

    yield {"status": "initializing", "message": "🚀 Deep Research başlatılıyor...", "agent": "deep"}
    yield {"status": "planning", "message": "🧠 Plan oluşturuluyor...", "agent": "deep"}

    final_text: Optional[str] = None

    # Stream tool events as status updates (best-effort)
    try:
        async for event in deep_graph.astream_events(
            {"messages": [HumanMessage(content=question)]},
            config=config,
            version="v2",
        ):
            ev = event.get("event")
            
            # Track node/chain execution
            if ev == "on_chain_start":
                node_name = event.get("name", "")
                if "supervisor" in node_name.lower():
                    yield {"status": "planning", "message": "🧠 Yönetici: Araştırma planı hazırlanıyor...", "agent": "deep"}
                elif "researcher" in node_name.lower():
                    yield {"status": "researching", "message": "🔬 Araştırmacı: Veri toplama başladı...", "agent": "deep"}
                elif "clarify" in node_name.lower():
                    yield {"status": "planning", "message": "🤔 Soru netleştiriliyor...", "agent": "deep"}
                elif "compress" in node_name.lower():
                    yield {"status": "writing", "message": "📝 Bulgular özetleniyor...", "agent": "deep"}
                elif "final_report" in node_name.lower():
                    yield {"status": "writing", "message": "📄 Nihai rapor yazılıyor...", "agent": "deep"}
            
            if ev == "on_chain_end":
                node_name = event.get("name", "")
                if "researcher" in node_name.lower():
                    yield {"status": "researching", "message": "✅ Araştırma tamamlandı", "agent": "deep"}

            if ev == "on_tool_start":
                name = (event.get("name") or "tool")
                # More descriptive tool messages
                if "tavily" in name.lower() or "search" in name.lower():
                    yield {"status": "searching", "message": f"🌐 Web araması yapılıyor...", "agent": "deep"}
                elif "think" in name.lower():
                    yield {"status": "planning", "message": "💭 Strateji belirleniyor...", "agent": "deep"}
                else:
                    yield {"status": "searching", "message": f"🛠️ {name} çalıştırılıyor...", "agent": "deep"}

            if ev == "on_tool_end":
                # If tool returned sources, forward to UI as meta.sources
                out = event.get("data", {}).get("output")
                try:
                    if isinstance(out, str):
                        parsed = json.loads(out)
                    else:
                        parsed = out
                    sources = (parsed or {}).get("sources")
                    if isinstance(sources, list) and sources:
                        yield {"meta": {"sources": sources[:20]}, "status": "searching", "message": f"📚 {len(sources)} kaynak bulundu", "agent": "deep"}
                except Exception:
                    pass

        result = await deep_graph.ainvoke({"messages": [HumanMessage(content=question)]}, config=config)
        msgs = result.get("messages", []) if isinstance(result, dict) else []
        for m in reversed(msgs):
            if isinstance(m, AIMessage) and getattr(m, "content", None):
                final_text = m.content
                break

        if not final_text:
            final_text = "(No output)"

        yield {"status": "done", "message": "✅ Tamamlandı", "content": final_text, "agent": "deep"}
        return
    except Exception as e:
        logger.error(f"[DEEP] Error: {e}", exc_info=True)
        yield {"status": "error", "message": f"Hata: {e}", "content": f"[ERROR] {e}", "agent": "deep"}
        return