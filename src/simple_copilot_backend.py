"""
Smart Backend with Caching & Rate Limiting
Showcase Mode: No Auth, No Billing
"""
from fastapi import FastAPI, HTTPException
from fastapi import Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import asyncio
import hashlib
import json
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict
from collections import OrderedDict
import uuid
import os

from .agents.simple_agent import run_simple_research
from .agents.multi_agent_system_v2 import run_multi_agent_research  # LangChain Tool Calling
from .agents.main_agent import run_research as run_deep_research
from .config import settings
from .memory.supabase_memory import get_memory
from .memory.hitl_approval import get_hitl_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Research API",
    description="Showcase Backend (No Auth)",
    version="2.0.0"
)

# CORS
allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "")
allowed_origins: List[str] = ["http://localhost:3000", "http://localhost:3001"]
if allowed_origins_env.strip():
    allowed_origins = [o.strip() for o in allowed_origins_env.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Showcase Mode: Single Demo User
DEMO_USER_ID = "demo-user-showcase"

# =============================================================================
# SIMPLE CACHE (In-Memory)
# =============================================================================

class SimpleCache:
    """Basit in-memory cache"""
    
    def __init__(self, max_size: int = 100, ttl_minutes: int = 60):
        self.cache: OrderedDict = OrderedDict()
        self.max_size = max_size
        self.ttl = timedelta(minutes=ttl_minutes)
    
    def _hash_query(self, query: str) -> str:
        return hashlib.md5(query.lower().strip().encode()).hexdigest()
    
    def get(self, query: str) -> Optional[str]:
        key = self._hash_query(query)
        if key in self.cache:
            entry = self.cache[key]
            if datetime.now() - entry["time"] < self.ttl:
                logger.info(f"[OK] Cache HIT: {query[:50]}...")
                return entry["response"]
            else:
                del self.cache[key]
        return None
    
    def set(self, query: str, response: str):
        key = self._hash_query(query)
        if len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)
        self.cache[key] = {
            "response": response,
            "time": datetime.now()
        }
        logger.info(f"[CACHE] Cached: {query[:50]}...")
    
    def stats(self) -> dict:
        return {
            "entries": len(self.cache),
            "max_size": self.max_size,
            "ttl_minutes": self.ttl.seconds // 60
        }

cache = SimpleCache(max_size=100, ttl_minutes=60)

# =============================================================================
# MODELS
# =============================================================================

class ChatRequest(BaseModel):
    message: str
    use_cache: bool = True
    mode: str = "simple"  # "simple" | "multi" | "deep"
    thread_id: Optional[str] = None
    options: Optional[Dict] = None  # Shared state: {web_search, need_code, need_long_report}

class ChatResponse(BaseModel):
    response: str
    success: bool = True
    cached: bool = False
    thread_id: Optional[str] = None
    saved: bool = False

class ApprovalSubmission(BaseModel):
    approval_id: str
    approved: bool
    feedback: Optional[str] = None

# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/")
async def root():
    """Health check with stats"""
    provider, model_name = settings.get_model_provider(settings.default_model)
    return {
        "status": "online",
        "mode": "Showcase (No Auth)",
        "model": settings.default_model,
        "available_modes": {
            "simple": "Hizli tek agent (default)",
            "multi": "Multi-Agent: Supervisor + Researcher + Coder + Writer",
            "deep": "Deep Research: MCP + Full pipeline"
        },
        "llm": {
            "provider": provider,
            "model_name": model_name,
        },
        "cache": cache.stats(),
    }

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """Chat endpoint with Streaming (No Auth)"""
    
    query = request.message.strip()
    user_id = DEMO_USER_ID
    
    if not query:
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    memory = get_memory()
    thread_id = request.thread_id or str(uuid.uuid4())
    saved = False

    # 0. History: User message
    saved = memory.save_message(
        role="user",
        content=query,
        metadata={"user_id": user_id, "thread_id": thread_id},
    ) or saved

    # 1. Cache Check
    if request.use_cache:
        cached_response = cache.get(query)
        if cached_response:
            saved = memory.save_message(
                role="assistant",
                content=cached_response,
                metadata={"user_id": user_id, "thread_id": thread_id, "cached": True},
            ) or saved
            # Return as SSE
            async def cached_stream():
                yield f"data: {json.dumps({'status': 'done', 'content': cached_response, 'cached': True, 'thread_id': thread_id})}\n\n"
            return StreamingResponse(cached_stream(), media_type="text/event-stream")
    
    # 2. AI Execution with Streaming
    async def stream_agent_response():
        try:
            mode = request.mode.lower()
            logger.info(f"[QUERY] New query ({mode}): {query[:50]}...")

            run_id = str(uuid.uuid4())
            
            final_content = ""
            
            # Select agent generator
            if mode == "multi":
                agent_gen = run_multi_agent_research(query, verbose=False, options=request.options)
            elif mode == "deep":
                agent_gen = run_deep_research(query, verbose=False)
            else:
                agent_gen = run_simple_research(query, verbose=False)
            
            # Stream status updates
            async for update in agent_gen:
                # Ensure observability identifiers are always present
                if isinstance(update, dict):
                    update.setdefault("thread_id", thread_id)
                    update.setdefault("run_id", run_id)
                # Send status update to frontend
                yield f"data: {json.dumps(update)}\n\n"
                
                # Capture final content
                if update.get("status") == "done" and "content" in update:
                    final_content = update["content"]
            
            # Cache if successful
            if final_content and not final_content.startswith(("[ERROR]", "⚠️")):
                cache.set(query, final_content)
                
                # History: Assistant message
                memory.save_message(
                    role="assistant",
                    content=final_content,
                    metadata={"user_id": user_id, "thread_id": thread_id, "cached": False},
                )
        
        except Exception as e:
            error_msg = str(e)
            logger.error(f"[ERROR] Error: {error_msg}", exc_info=True)
            yield f"data: {json.dumps({'status': 'error', 'message': f'Hata: {error_msg}', 'content': f'[ERROR] {error_msg}', 'thread_id': thread_id, 'run_id': run_id})}\n\n"
    
    return StreamingResponse(stream_agent_response(), media_type="text/event-stream")

@app.get("/threads")
async def list_threads(limit: int = 20):
    """List threads for demo user"""
    memory = get_memory()
    return memory.list_threads(user_id=DEMO_USER_ID, limit=limit)

@app.get("/threads/{thread_id}")
async def get_thread(thread_id: str, limit: int = 200):
    """Get thread messages"""
    memory = get_memory()
    return memory.load_thread(user_id=DEMO_USER_ID, thread_id=thread_id, limit=limit)

@app.delete("/threads/{thread_id}")
async def delete_thread(thread_id: str):
    """Delete thread"""
    memory = get_memory()
    ok = memory.delete_thread(user_id=DEMO_USER_ID, thread_id=thread_id)
    return {"success": ok}

@app.get("/stats")
async def stats():
    """System stats"""
    return {
        "cache": cache.stats(),
        "user": "Demo User (Showcase Mode)"
    }

@app.get("/health")
async def health():
    """Health check"""
    memory = get_memory()
    api_status = settings.validate_api_keys()
    return {
        "status": "ok",
        "apis": api_status,
        "supabase": {
            "enabled": memory.is_enabled(),
            "table": "conversations",
        },
    }

@app.delete("/cache")
async def clear_cache():
    cache.cache.clear()
    return {"message": "Cache cleared", "entries": 0}

@app.post("/copilotkit")
async def copilotkit_endpoint(request: dict):
    """
    CopilotKit uyumlu endpoint
    CopilotKit bu endpoint'e mesaj gönderir, biz de streaming response döneriz
    """
    messages = request.get("messages", [])
    if not messages:
        raise HTTPException(status_code=400, detail="No messages provided")
    
    # Son mesajı al
    last_message = messages[-1]
    query = last_message.get("content", "").strip()
    
    if not query:
        raise HTTPException(status_code=400, detail="Empty message")
    
    # Mode'u context'ten al (CopilotKit readable'dan gelecek)
    context = request.get("context", {})
    mode = context.get("mode", "simple")
    
    user_id = DEMO_USER_ID
    thread_id = str(uuid.uuid4())
    
    async def copilotkit_stream():
        """CopilotKit formatında stream"""
        try:
            # Agent seç
            if mode == "multi":
                agent_gen = run_multi_agent_research(query, verbose=False)
            elif mode == "deep":
                agent_gen = run_deep_research(query, verbose=False)
            else:
                agent_gen = run_simple_research(query, verbose=False)
            
            final_content = ""
            
            # Her status update'i CopilotKit formatında gönder
            async for update in agent_gen:
                if update.get("status") == "done" and "content" in update:
                    final_content = update["content"]
                    # Final mesajı CopilotKit formatında
                    copilot_msg = {
                        "role": "assistant",
                        "content": final_content,
                    }
                    yield f"data: {json.dumps(copilot_msg)}\n\n"
                else:
                    # Intermediate status (CopilotKit'e göndermeyebiliriz ama log için güzel)
                    pass
            
            # Cache
            if final_content:
                cache.set(query, final_content)
                get_memory().save_message(
                    role="assistant",
                    content=final_content,
                    metadata={"user_id": user_id, "thread_id": thread_id},
                )
        
        except Exception as e:
            error_msg = str(e)
            logger.error(f"[ERROR] CopilotKit: {error_msg}")
            error_response = {
                "role": "assistant",
                "content": f"❌ Error: {error_msg}",
            }
            yield f"data: {json.dumps(error_response)}\n\n"
    
    return StreamingResponse(copilotkit_stream(), media_type="text/event-stream")


# =============================================================================
# HITL APPROVAL ENDPOINTS
# =============================================================================

@app.post("/api/approval/submit")
async def submit_approval(submission: ApprovalSubmission):
    """Submit user's approval decision for HITL flow"""
    hitl = get_hitl_manager()
    success = hitl.submit_approval(
        approval_id=submission.approval_id,
        approved=submission.approved,
        feedback=submission.feedback
    )
    
    if not success:
        raise HTTPException(status_code=404, detail="Approval request not found")
    
    logger.info(f"[HITL] Approval {'✅ approved' if submission.approved else '❌ rejected'}: {submission.approval_id[:8]}...")
    return {
        "success": True,
        "message": "Approval submitted",
        "approved": submission.approved
    }


@app.get("/api/approval/pending")
async def get_pending_approvals():
    """Get all pending approval requests (for debugging/monitoring)"""
    hitl = get_hitl_manager()
    pending = hitl.get_pending_approvals()
    
    return {
        "success": True,
        "count": len(pending),
        "approvals": pending
    }


@app.post("/api/approval/{approval_id}/cancel")
async def cancel_approval(approval_id: str):
    """Cancel a pending approval request"""
    hitl = get_hitl_manager()
    hitl.cancel_approval(approval_id)
    
    return {
        "success": True,
        "message": f"Approval {approval_id} cancelled"
    }


if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*70)
    print("[START] AI Research API (Showcase Mode)")
    print("   [OK] No Auth Required")
    print("   [OK] Demo User Active")
    print("="*70)
    
    uvicorn.run(
        "src.simple_copilot_backend:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
