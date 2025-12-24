"""Utility functions and helpers for the Deep Research agent."""
import asyncio
import logging
import os
import warnings
from datetime import datetime, timedelta, timezone
from typing import Annotated, Any, Dict, List, Literal, Optional

import aiohttp
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    MessageLikeRepresentation,
    filter_messages,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import (
    BaseTool,
    InjectedToolArg,
    StructuredTool,
    ToolException,
    tool,
)
from tavily import AsyncTavilyClient

from .configuration import Configuration, SearchAPI
from .state import ResearchComplete, Summary
from .prompts import summarize_webpage_prompt

logger = logging.getLogger(__name__)

# =============================================================================
# MODELS / API KEYS UTILS
# =============================================================================

def get_api_key_for_model(model_name: str, config: RunnableConfig):
    """Get API key for specific model."""
    # Simplified logic: In our project we rely on .env usually, but respect config if passed
    if model_name.startswith("openai:"): 
        return os.getenv("OPENAI_API_KEY")
    # For Ollama no key needed usually
    return None

def get_tavily_api_key(config: RunnableConfig):
    return os.getenv("TAVILY_API_KEY")

def get_today_str() -> str:
    now = datetime.now()
    return f"{now:%a} {now:%b} {now.day}, {now:%Y}"

def is_token_limit_exceeded(exception: Exception, model_name: str = None) -> bool:
    """Check if exception is token limit error."""
    # Simplified check
    text = str(exception).lower()
    return "context length" in text or "token limit" in text or "too long" in text

def remove_up_to_last_ai_message(messages: list[MessageLikeRepresentation]) -> list[MessageLikeRepresentation]:
    """Truncate history."""
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], AIMessage):
            return messages[:i]
    return messages

def get_notes_from_tool_calls(messages: list[MessageLikeRepresentation]):
    return [tool_msg.content for tool_msg in filter_messages(messages, include_types="tool")]

def get_model_token_limit(model_string: str):
    # hardcoded fallback
    return 128000

# =============================================================================
# TOOLS
# =============================================================================

@tool(description="Strategic reflection tool for research planning")
def think_tool(reflection: str) -> str:
    """Tool for strategic reflection."""
    return f"Reflection recorded: {reflection}"

async def _tavily_search_async(queries: List[str], max_results: int = 5, config: RunnableConfig = None):
    try:
        api_key = get_tavily_api_key(config)
        client = AsyncTavilyClient(api_key=api_key)
        tasks = [client.search(q, max_results=max_results, include_raw_content=True) for q in queries]
        return await asyncio.gather(*tasks)
    except Exception as e:
        logger.error(f"Tavily search error: {e}")
        return []

async def summarize_webpage(model: BaseChatModel, webpage_content: str) -> str:
    try:
        prompt = summarize_webpage_prompt.format(webpage_content=webpage_content, date=get_today_str())
        # Use simple invoke instead of wait_for to avoid asyncio complexity in some envs
        resp = await model.ainvoke([HumanMessage(content=prompt)])
        return resp.content
    except Exception as e:
        return webpage_content[:2000] # Fallback truncate

@tool
async def tavily_search(
    queries: List[str],
    max_results: Annotated[int, InjectedToolArg] = 5,
    config: RunnableConfig = None
) -> str:
    """Tavily search tool."""
    # 1. Search
    results_list = await _tavily_search_async(queries, max_results, config)
    
    unique_results = {}
    for res in results_list:
        if not isinstance(res, dict): continue
        for r in res.get("results", []):
            unique_results[r['url']] = r
            
    # 2. Summarize (Simplified: Just return snippets if no model passed, or use global model)
    # Note: In a real implementation we would init a model here. For speed/simplicity, we return snippets.
    
    out = "Search Results:\n\n"
    for url, r in unique_results.items():
        out += f"Title: {r.get('title')}\nURL: {url}\nContent: {r.get('content') or r.get('raw_content')[:500]}\n\n"
        
    return out

async def get_all_tools(config: RunnableConfig):
    # Simplified tools getter
    from .state import ResearchComplete
    return [tavily_search, think_tool, tool(ResearchComplete)]
