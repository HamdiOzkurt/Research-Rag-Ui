"""
LangGraph Store Integration - Cross-Thread Persistent Memory
Replaces/extends Supabase memory with LangGraph's Store API
"""

import os
import logging
from typing import Optional, Dict, Any, List
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore  # Development fallback
import json

logger = logging.getLogger(__name__)


class HybridMemoryStore:
    """
    Hybrid memory system:
    - Short-term: In-memory for current conversation
    - Long-term: LangGraph Store for cross-thread persistence
    
    Usage:
        store = HybridMemoryStore()
        await store.save_memory(thread_id="abc", key="user_prefs", value={"theme": "dark"})
        prefs = await store.get_memory(thread_id="abc", key="user_prefs")
    """
    
    def __init__(self, store: Optional[BaseStore] = None):
        """
        Initialize hybrid memory store.
        
        Args:
            store: LangGraph BaseStore instance. If None, uses InMemoryStore for dev.
        """
        if store is None:
            logger.warning("[MEMORY] No store provided, using InMemoryStore (dev only)")
            self.store = InMemoryStore()
        else:
            self.store = store
        
        self._short_term_cache: Dict[str, Dict[str, Any]] = {}
        logger.info("[MEMORY] HybridMemoryStore initialized")
    
    async def save_memory(
        self,
        thread_id: str,
        key: str,
        value: Any,
        namespace: str = "user_memory"
    ):
        """
        Save memory to both short-term and long-term storage.
        
        Args:
            thread_id: Conversation thread ID
            key: Memory key (e.g., "user_preferences", "research_context")
            value: Data to store (will be JSON serialized)
            namespace: Store namespace (default: "user_memory")
        """
        # Short-term cache
        cache_key = f"{thread_id}:{key}"
        self._short_term_cache[cache_key] = value
        
        # Long-term Store
        try:
            item_id = f"{thread_id}::{key}"
            await self.store.aput(
                namespace=(namespace,),
                key=item_id,
                value={"data": value, "thread_id": thread_id, "key": key}
            )
            logger.info(f"[MEMORY] Saved {key} for thread {thread_id[:8]}...")
        except Exception as e:
            logger.error(f"[MEMORY] Save failed: {e}")
    
    async def get_memory(
        self,
        thread_id: str,
        key: str,
        namespace: str = "user_memory"
    ) -> Optional[Any]:
        """
        Retrieve memory (checks cache first, then Store).
        
        Args:
            thread_id: Conversation thread ID
            key: Memory key
            namespace: Store namespace
            
        Returns:
            Stored value or None
        """
        # Check short-term cache first
        cache_key = f"{thread_id}:{key}"
        if cache_key in self._short_term_cache:
            logger.debug(f"[MEMORY] Cache hit for {key}")
            return self._short_term_cache[cache_key]
        
        # Fallback to long-term Store
        try:
            item_id = f"{thread_id}::{key}"
            item = await self.store.aget(namespace=(namespace,), key=item_id)
            
            if item and "value" in item:
                value = item["value"].get("data")
                # Populate cache
                self._short_term_cache[cache_key] = value
                logger.info(f"[MEMORY] Retrieved {key} from Store")
                return value
        except Exception as e:
            logger.error(f"[MEMORY] Get failed: {e}")
        
        return None
    
    async def list_memories(
        self,
        thread_id: str,
        namespace: str = "user_memory"
    ) -> List[Dict[str, Any]]:
        """
        List all memories for a thread.
        
        Args:
            thread_id: Conversation thread ID
            namespace: Store namespace
            
        Returns:
            List of memory items
        """
        try:
            # Search by prefix
            items = await self.store.asearch(
                namespace_prefix=(namespace,),
            )
            
            # Filter by thread_id
            thread_items = []
            for item in items:
                if item.get("value", {}).get("thread_id") == thread_id:
                    thread_items.append({
                        "key": item["value"]["key"],
                        "data": item["value"]["data"]
                    })
            
            logger.info(f"[MEMORY] Found {len(thread_items)} memories for thread {thread_id[:8]}...")
            return thread_items
        except Exception as e:
            logger.error(f"[MEMORY] List failed: {e}")
            return []
    
    async def delete_memory(
        self,
        thread_id: str,
        key: str,
        namespace: str = "user_memory"
    ):
        """
        Delete a specific memory.
        
        Args:
            thread_id: Conversation thread ID
            key: Memory key
            namespace: Store namespace
        """
        # Remove from cache
        cache_key = f"{thread_id}:{key}"
        self._short_term_cache.pop(cache_key, None)
        
        # Remove from Store
        try:
            item_id = f"{thread_id}::{key}"
            await self.store.adelete(namespace=(namespace,), key=item_id)
            logger.info(f"[MEMORY] Deleted {key} for thread {thread_id[:8]}...")
        except Exception as e:
            logger.error(f"[MEMORY] Delete failed: {e}")
    
    async def clear_thread_memories(self, thread_id: str):
        """Delete all memories for a thread."""
        memories = await self.list_memories(thread_id)
        for mem in memories:
            await self.delete_memory(thread_id, mem["key"])
        
        # Clear cache
        self._short_term_cache = {
            k: v for k, v in self._short_term_cache.items()
            if not k.startswith(f"{thread_id}:")
        }
        
        logger.info(f"[MEMORY] Cleared all memories for thread {thread_id[:8]}...")


# =============================================================================
# INTEGRATION WITH AGENTS
# =============================================================================

def create_agent_with_memory(agent_graph, store: Optional[BaseStore] = None):
    """
    Wrap a LangGraph agent with HybridMemoryStore.
    
    Args:
        agent_graph: LangGraph compiled graph
        store: Optional BaseStore instance
        
    Returns:
        Agent with memory capabilities
    
    Usage:
        from langgraph.store.postgres import PostgresStore
        
        store = PostgresStore(conn_string="postgresql://...")
        agent = create_agent_with_memory(deep_graph, store=store)
        
        # Agent can now access store in config:
        config = {"configurable": {"store": store}}
        agent.invoke({"messages": [...]}, config=config)
    """
    memory = HybridMemoryStore(store=store)
    
    # Inject memory into agent config
    original_invoke = agent_graph.invoke
    original_astream = agent_graph.astream
    
    async def invoke_with_memory(input_data, config=None):
        config = config or {}
        config.setdefault("configurable", {})
        config["configurable"]["memory"] = memory
        return await original_invoke(input_data, config)
    
    async def astream_with_memory(input_data, config=None):
        config = config or {}
        config.setdefault("configurable", {})
        config["configurable"]["memory"] = memory
        async for event in original_astream(input_data, config):
            yield event
    
    agent_graph.invoke = invoke_with_memory
    agent_graph.astream = astream_with_memory
    
    logger.info("[MEMORY] Agent wrapped with HybridMemoryStore")
    return agent_graph


# =============================================================================
# EXAMPLE USAGE IN TOOLS
# =============================================================================
"""
@tool("save_research_context")
async def save_context_tool(context: str, runtime: ToolRuntime):
    '''Araştırma context'ini long-term memory'ye kaydet'''
    memory = runtime.config["configurable"]["memory"]
    await memory.save_memory(
        thread_id=runtime.thread_id,
        key="research_context",
        value=context
    )
    return "Context kaydedildi"

@tool("get_past_research")
async def get_past_tool(runtime: ToolRuntime):
    '''Önceki araştırma context'ini getir'''
    memory = runtime.config["configurable"]["memory"]
    context = await memory.get_memory(
        thread_id=runtime.thread_id,
        key="research_context"
    )
    return context or "Geçmiş araştırma bulunamadı"
"""
