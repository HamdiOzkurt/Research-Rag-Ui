"""Utility functions and helpers for the Deep Research agent - Adapted for Ollama."""

import asyncio
import logging
import os
from datetime import datetime
from typing import Annotated, Any, List, Literal, Optional

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    MessageLikeRepresentation,
    filter_messages,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from tavily import AsyncTavilyClient

from .configuration import Configuration, SearchAPI
from .prompts import summarize_webpage_prompt
from .state import ResearchComplete, Summary

logger = logging.getLogger(__name__)

##########################
# Tavily Search Tool Utils
##########################

TAVILY_SEARCH_DESCRIPTION = (
    "A search engine optimized for comprehensive, accurate, and trusted results. "
    "Useful for when you need to answer questions about current events."
)

@tool(description=TAVILY_SEARCH_DESCRIPTION)
async def tavily_search(
    queries: List[str],
    config: RunnableConfig = None
) -> str:
    """Fetch and summarize search results from Tavily search API."""
    configurable = Configuration.from_runnable_config(config)
    max_results = 5
    
    # Step 1: Execute search queries asynchronously
    search_results = await tavily_search_async(
        queries,
        max_results=max_results,
        include_raw_content=True,
        config=config
    )
    
    # Step 2: Deduplicate results by URL
    unique_results = {}
    for response in search_results:
        if not isinstance(response, dict):
            continue
        for result in response.get('results', []):
            url = result.get('url')
            if url and url not in unique_results:
                unique_results[url] = {**result, "query": response.get('query', '')}
    
    if not unique_results:
        return "No valid search results found. Please try different search queries."
    
    # Step 3: Format the output (simplified - no summarization for speed)
    formatted_output = "Search results:\n\n"
    for i, (url, result) in enumerate(unique_results.items()):
        content = result.get('content') or result.get('raw_content', '')[:1000]
        formatted_output += f"\n--- SOURCE {i+1}: {result.get('title', 'No title')} ---\n"
        formatted_output += f"URL: {url}\n\n"
        formatted_output += f"CONTENT:\n{content}\n\n"
        formatted_output += "-" * 80 + "\n"
    
    return formatted_output

async def tavily_search_async(
    search_queries, 
    max_results: int = 5, 
    include_raw_content: bool = True, 
    config: RunnableConfig = None
):
    """Execute multiple Tavily search queries asynchronously."""
    try:
        tavily_client = AsyncTavilyClient(api_key=get_tavily_api_key(config))
        
        search_tasks = [
            tavily_client.search(
                query,
                max_results=max_results,
                include_raw_content=include_raw_content
            )
            for query in search_queries
        ]
        
        search_results = await asyncio.gather(*search_tasks, return_exceptions=True)
        # Filter out exceptions
        return [r for r in search_results if isinstance(r, dict)]
    except Exception as e:
        logger.error(f"Tavily search error: {e}")
        return []

async def summarize_webpage(model: BaseChatModel, webpage_content: str) -> str:
    """Summarize webpage content using AI model."""
    try:
        prompt_content = summarize_webpage_prompt.format(
            webpage_content=webpage_content, 
            date=get_today_str()
        )
        
        summary = await asyncio.wait_for(
            model.ainvoke([HumanMessage(content=prompt_content)]),
            timeout=60.0
        )
        
        if hasattr(summary, 'summary'):
            return f"<summary>\n{summary.summary}\n</summary>\n\n<key_excerpts>\n{summary.key_excerpts}\n</key_excerpts>"
        return str(summary.content)
        
    except Exception as e:
        logger.warning(f"Summarization failed: {e}")
        return webpage_content[:2000]

##########################
# Reflection Tool Utils
##########################

@tool(description="Strategic reflection tool for research planning")
def think_tool(reflection: str) -> str:
    """Tool for strategic reflection on research progress."""
    return f"Reflection recorded: {reflection}"

##########################
# Tool Utils
##########################

async def get_search_tool(search_api: SearchAPI):
    """Configure and return search tools based on the specified API provider."""
    if search_api == SearchAPI.TAVILY:
        search_tool = tavily_search
        search_tool.metadata = {
            **(search_tool.metadata or {}), 
            "type": "search", 
            "name": "web_search"
        }
        return [search_tool]
    return []
    
async def get_all_tools(config: RunnableConfig):
    """Assemble complete toolkit including research and search tools."""
    tools = [tool(ResearchComplete), think_tool]
    
    configurable = Configuration.from_runnable_config(config)
    search_api = configurable.search_api
    if isinstance(search_api, str):
        search_api = SearchAPI(search_api)
    
    search_tools = await get_search_tool(search_api)
    tools.extend(search_tools)
    
    return tools

def get_notes_from_tool_calls(messages: list[MessageLikeRepresentation]):
    """Extract notes from tool call messages."""
    return [tool_msg.content for tool_msg in filter_messages(messages, include_types="tool")]

##########################
# Token Limit Utils
##########################

def is_token_limit_exceeded(exception: Exception, model_name: str = None) -> bool:
    """Determine if an exception indicates a token/context limit was exceeded."""
    error_str = str(exception).lower()
    return "context length" in error_str or "token limit" in error_str or "too long" in error_str

def get_model_token_limit(model_string: str):
    """Look up the token limit for a specific model."""
    # Ollama models typically have 32k-128k context
    if "qwen2.5" in model_string.lower():
        return 32000
    if "llama" in model_string.lower():
        return 8192
    return 32000  # Default

def remove_up_to_last_ai_message(messages: list[MessageLikeRepresentation]) -> list[MessageLikeRepresentation]:
    """Truncate message history by removing up to the last AI message."""
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], AIMessage):
            return messages[:i]
    return messages

##########################
# Misc Utils
##########################

def get_today_str() -> str:
    """Get current date formatted for display."""
    now = datetime.now()
    return f"{now:%a} {now:%b} {now.day}, {now:%Y}"

def get_api_key_for_model(model_name: str, config: RunnableConfig):
    """Get API key for a specific model."""
    model_name = model_name.lower()
    if model_name.startswith("openai:"):
        return os.getenv("OPENAI_API_KEY")
    elif model_name.startswith("anthropic:"):
        return os.getenv("ANTHROPIC_API_KEY")
    elif model_name.startswith("google"):
        return os.getenv("GOOGLE_API_KEY")
    # Ollama doesn't need API key
    return None

def get_tavily_api_key(config: RunnableConfig):
    """Get Tavily API key from environment."""
    return os.getenv("TAVILY_API_KEY")
