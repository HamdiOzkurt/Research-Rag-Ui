"""Simplified Deep Research - Single Researcher Version (No Supervisor)"""

from typing import TypedDict, List, Annotated
import operator
from langchain_core.messages import HumanMessage, AIMessage, AnyMessage
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
import os

class ResearchState(TypedDict):
    messages: Annotated[List[AnyMessage], operator.add]
    research_brief: str
    findings: str

async def create_brief(state: ResearchState):
    """Convert user question to research brief."""
    user_question = state["messages"][-1].content if state["messages"] else "No question"
    return {
        "research_brief": f"Research Topic: {user_question}",
        "messages": [AIMessage(content=f"📋 Research brief created: {user_question}")]
    }

async def conduct_research(state: ResearchState):
    """Conduct simple research using Tavily."""
    from ..deep_research.utils import tavily_search
    
    brief = state.get("research_brief", "")
    
    # Simple search
    try:
        search_result = await tavily_search(
            queries=[brief],
            max_results=3,
            config=None
        )
        findings = f"Research completed. Found information:\n\n{search_result[:500]}..."
    except Exception as e:
        findings = f"Research error: {e}"
    
    return {
        "findings": findings,
        "messages": [AIMessage(content="🔬 Research completed")]
    }

async def write_report(state: ResearchState):
    """Write final report."""
    brief = state.get("research_brief", "No brief")
    findings = state.get("findings", "No findings")
    
    # Initialize model
    model = init_chat_model("ollama:qwen2.5:7b", temperature=0.3)
    
    prompt = f"""Write a concise research report in Turkish.

Research Topic: {brief}

Findings: {findings}

Please write a brief, well-structured report with:
- A clear title
- Key findings
- Conclusion

Use Markdown format."""

    response = await model.ainvoke([HumanMessage(content=prompt)])
    
    return {
        "messages": [AIMessage(content=response.content)]
    }

# Build graph
builder = StateGraph(ResearchState)
builder.add_node("create_brief", create_brief)
builder.add_node("research", conduct_research)
builder.add_node("write_report", write_report)

builder.add_edge(START, "create_brief")
builder.add_edge("create_brief", "research")
builder.add_edge("research", "write_report")
builder.add_edge("write_report", END)

# Export
graph = builder.compile()
