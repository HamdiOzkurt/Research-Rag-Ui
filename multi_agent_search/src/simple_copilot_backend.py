"""
Smart Backend with Caching & Rate Limiting
Showcase Mode: No Auth, No Billing
"""
from fastapi import FastAPI, HTTPException, Request  # ✅ Request eklendi
from fastapi import Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
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
from pathlib import Path

from .agents.simple_agent import run_simple_research
from .agents.rag_agent import graph as rag_graph, _rag_state  # ✅ HITL state
from .agents.main_agent import run_research as run_deep_research
from .config import settings
from .memory.supabase_memory import get_memory
from .memory.hitl_approval import get_hitl_manager

# Inngest for background jobs
import inngest
import inngest.fast_api

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Research API",
    description="Showcase Backend (No Auth)",
    version="2.0.0"
)

# ============ INNGEST CLIENT ============
inngest_client = inngest.Inngest(
    app_id="multi_agent_search",
    is_production=False,  # Dev mode: no signing key required
    logger=logger,
)
import os
os.environ.setdefault("INNGEST_SIGNING_KEY", "signkey_dev_local")

# Ensure uploads directory exists early (used by multiple endpoints and static mount)
Path("./uploads").mkdir(exist_ok=True)

# ============ STARTUP: INGEST EXISTING FILES ============
@app.on_event("startup")
async def startup_event():
    """Ingest existing files in uploads/ directory on startup"""
    try:
        upload_dir = Path("./uploads")
        if not upload_dir.exists():
            return

        logger.info("[STARTUP] Checking for existing files in uploads/...")
        from .agents.rag_agent import ingest_text, load_pdf, load_docx, load_csv, transcribe_audio, analyze_image
        
        count = 0
        for file_path in upload_dir.iterdir():
            if not file_path.is_file():
                continue
            
            # Skip debug files generated during processing
            if "_debug" in file_path.name or "_pypdf_debug" in file_path.name or "_markdown_debug" in file_path.name:
                continue
                
            try:
                ext = file_path.suffix.lower()
                text = ""
                
                if ext == ".pdf":
                    text = load_pdf.invoke(str(file_path))
                elif ext in [".docx", ".doc"]:
                    text = load_docx.invoke(str(file_path))
                elif ext == ".csv":
                    text = load_csv.invoke(str(file_path))
                elif ext in [".mp3", ".wav", ".m4a"]:
                    text = await transcribe_audio.ainvoke(str(file_path))
                elif ext in [".jpg", ".jpeg", ".png", ".gif", ".webp"]:
                    text = await analyze_image.ainvoke(str(file_path))
                elif ext == ".txt":
                    text = file_path.read_text(encoding="utf-8")
                
                if text:
                    ingest_text(text, file_path.name)
                    count += 1
                    logger.info(f"[STARTUP] Ingested: {file_path.name}")
            except Exception as e:
                logger.error(f"[STARTUP] Failed to ingest {file_path.name}: {e}")
        
        logger.info(f"[STARTUP] Completed. Ingested {count} files.")
            
    except Exception as e:
        logger.error(f"[STARTUP] Error during file ingestion: {e}")


# CORS
allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "")
allowed_origins: List[str] = [
    "http://localhost:3000",
    "http://localhost:3001",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:3001",
]
if allowed_origins_env.strip():
    allowed_origins = [o.strip() for o in allowed_origins_env.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount uploads directory for serving images and files
# FIX: Detect correct uploads directory based on CWD
upload_fs_path = Path("multi_agent_search/uploads")
if not upload_fs_path.exists():
    upload_fs_path = Path("uploads")

if not upload_fs_path.exists():
    # Create if missing
    upload_fs_path.mkdir(parents=True, exist_ok=True)

app.mount("/uploads", StaticFiles(directory=str(upload_fs_path)), name="uploads")
print(f"[SERVER] ✅ Serving /uploads from: {upload_fs_path.resolve()}")

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
                # RAG Agent - Document processing & semantic search
                async def rag_stream_wrapper():
                    """Wrap RAG agent stream to match backend format"""
                    try:
                        # LangGraph create_react_agent expects messages format
                        yield {
                            "status": "planning",
                            "message": "RAG: Dokümanlardan ilgili içerik aranıyor...",
                            "agent": "multi",
                        }

                        result = await rag_graph.ainvoke({"messages": [("user", query)]})
                        final_msg = ""
                        images = []
                        
                        if isinstance(result, dict):
                            # Extract last AI message content
                            messages = result.get("messages", [])
                            for msg in reversed(messages):
                                if hasattr(msg, "content") and msg.content:
                                    final_msg = msg.content
                                    break
                            if not final_msg:
                                final_msg = str(result.get("output") or result)
                            
                            # Extract images from metadata if available
                            images = result.get("images", [])
                        else:
                            final_msg = str(result)

                        yield {
                            "status": "done",
                            "message": "Bitti",
                            "agent": "multi",
                            "content": final_msg,
                            "images": images,
                        }
                    except Exception as e:
                        logger.error(f"[RAG] Stream error: {e}", exc_info=True)
                        yield {
                            "status": "error",
                            "message": f"RAG Agent hatası: {str(e)}",
                            "content": f"RAG Agent hatası: {str(e)}",
                            "agent": "multi",
                        }
                
                agent_gen = rag_stream_wrapper()
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
    """Health check (fast, no network calls).

    Frontend uses this endpoint to decide whether the backend is reachable.
    Keep it non-blocking: avoid Supabase/LLM network calls here.
    """
    return {
        "status": "ok",
        "version": app.version,
        "supabase_configured": bool(settings.supabase_url and settings.supabase_key),
        "google_configured": bool(settings.google_api_key),
        "groq_configured": bool(settings.groq_api_key),
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
                # RAG Agent wrapper
                async def rag_stream_wrapper():
                    """Wrap RAG agent stream to match backend format"""
                    try:
                        yield {
                            "status": "planning",
                            "message": "RAG: Dokümanlardan ilgili içerik aranıyor...",
                            "agent": "multi",
                        }

                        result = await rag_graph.ainvoke({"messages": [("user", query)]})
                        final_msg = ""
                        if isinstance(result, dict):
                            messages = result.get("messages", [])
                            for msg in reversed(messages):
                                if hasattr(msg, "content") and msg.content:
                                    final_msg = msg.content
                                    break
                            if not final_msg:
                                final_msg = str(result.get("output") or result)
                        else:
                            final_msg = str(result)

                        yield {
                            "status": "done",
                            "message": "Bitti",
                            "agent": "multi",
                            "content": final_msg,
                        }
                    except Exception as e:
                        logger.error(f"[RAG] Stream error: {e}", exc_info=True)
                        yield {
                            "status": "error",
                            "message": f"RAG Agent hatası: {str(e)}",
                            "content": f"RAG Agent hatası: {str(e)}",
                            "agent": "multi",
                        }
                
                agent_gen = rag_stream_wrapper()
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
# RAG FILE UPLOAD ENDPOINTS
# =============================================================================

from fastapi import File, UploadFile, Form
import shutil
from pathlib import Path

UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

class FileIngestRequest(BaseModel):
    file_path: str
    source_name: Optional[str] = None
    
    class Config:
        extra = "ignore"  # Ignore extra fields instead of erroring

@app.post("/api/rag/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a file for RAG processing"""
    try:
        # Save file
        file_path = UPLOAD_DIR / file.filename
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"[RAG] File uploaded: {file.filename} ({file_path})")
        return {
            "status": "success",
            "filename": file.filename,
            "path": str(file_path),
            "size": file_path.stat().st_size
        }
    except Exception as e:
        logger.error(f"[RAG] Upload error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

class IngestMode(BaseModel):
    """Ingest mode: sync or async"""
    mode: str = "sync"  # "sync" or "async" (async requires Inngest Dev Server)

@app.post("/api/rag/ingest")
async def ingest_file(request: FileIngestRequest, mode: str = "sync"):
    """Process uploaded file and ingest into vector store
    
    Args:
        mode: "sync" (immediate) or "async" (background with Inngest)
    """
    try:
        logger.info(f"[RAG] Ingest request ({mode}): file_path={request.file_path}, source_name={request.source_name}")
        
        file_path = Path(request.file_path)
        source_name = request.source_name or file_path.name
        
        logger.info(f"[RAG] Resolved path: {file_path}, exists: {file_path.exists()}")
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        ext = file_path.suffix.lower()
        
        # ASYNC MODE: Trigger Inngest background job for PDFs (if available)
        if mode == "async" and ext == ".pdf":
            try:
                # Check if Inngest is available
                await inngest_client.send(
                    inngest.Event(
                        name="rag/pdf.uploaded",
                        data={
                            "file_path": str(file_path),
                            "source_name": source_name
                        }
                    )
                )
                
                logger.info(f"[RAG] 🚀 Queued for background processing: {source_name}")
                return {
                    "status": "queued",
                    "source": source_name,
                    "message": "PDF processing started in background",
                    "mode": "async"
                }
            except Exception as inngest_error:
                logger.warning(f"[RAG] Inngest unavailable, falling back to sync mode: {inngest_error}")
                # Fall through to sync processing
        
        # SYNC MODE: Process immediately
        from .agents.rag_agent import ingest_text, load_pdf, load_docx, load_csv, transcribe_audio, analyze_image
        
        if ext == ".pdf":
            text = load_pdf.invoke(str(file_path))
        elif ext in [".docx", ".doc"]:
            text = load_docx.invoke(str(file_path))
        elif ext == ".csv":
            text = load_csv.invoke(str(file_path))
        elif ext in [".mp3", ".wav", ".m4a"]:
            text = await transcribe_audio.ainvoke(str(file_path))
        elif ext in [".jpg", ".jpeg", ".png", ".gif", ".webp"]:
            text = await analyze_image.ainvoke(str(file_path))
        else:
            # Plain text
            text = file_path.read_text(encoding="utf-8")
        
        # Ingest into vector store
        chunks_count = ingest_text(text, source_name)
        
        # Check if it was a duplicate (0 chunks = already ingested)
        was_duplicate = chunks_count == 0
        
        if was_duplicate:
            logger.warning(f"[RAG] ⚠️ Duplicate detected: {source_name}")
        else:
            logger.info(f"[RAG] ✅ Ingested {source_name}: {chunks_count} chunks")
        
        return {
            "status": "success" if not was_duplicate else "duplicate",
            "source": source_name,
            "chunks": chunks_count,
            "text_length": len(text),
            "duplicate": was_duplicate,
            "mode": "sync"
        }
    except Exception as e:
        logger.error(f"[RAG] Ingest error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/rag/documents")
async def list_documents():
    """List all ingested documents from vector store (not filesystem)"""
    try:
        from .agents.rag_agent import get_ingested_sources
        
        # Get sources from vector store
        sources = get_ingested_sources()
        
        # Enrich with file metadata if file exists
        files = []
        for source_name in sources:
            file_path = UPLOAD_DIR / source_name
            
            doc_info = {
                "filename": source_name,
                "path": str(file_path) if file_path.exists() else None,
                "size": file_path.stat().st_size if file_path.exists() else 0,
                "modified": file_path.stat().st_mtime if file_path.exists() else None,
                "ingested": True  # Always true since it's from vector store
            }
            files.append(doc_info)
        
        return {"documents": files}
    except Exception as e:
        logger.error(f"[RAG] List error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/rag/debug/vector-store")
async def debug_vector_store():
    """Debug endpoint: Show what's in the vector store (JSON)"""
    try:
        from .agents.rag_agent import _vector_store
        
        # Get all documents from vector store
        all_docs = _vector_store.similarity_search("", k=100)  # Get all chunks
        
        chunks_by_source = {}
        for doc in all_docs:
            source = doc.metadata.get("source", "unknown")
            if source not in chunks_by_source:
                chunks_by_source[source] = []
            chunks_by_source[source].append({
                "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "full_length": len(doc.page_content),
                "metadata": doc.metadata
            })
        
        return {
            "total_chunks": len(all_docs),
            "sources": list(chunks_by_source.keys()),
            "chunks_by_source": {k: len(v) for k, v in chunks_by_source.items()},
            "sample_chunks": chunks_by_source
        }
    except Exception as e:
        logger.error(f"[RAG] Debug error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/rag/sources")
async def list_sources():
    """List all ingested sources in vector store"""
    try:
        from .agents.rag_agent import get_ingested_sources
        
        sources = get_ingested_sources()
        return {
            "total": len(sources),
            "sources": sources
        }
    except Exception as e:
        logger.error(f"[RAG] Error listing sources: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/rag/clear")
async def clear_rag_store():
    """Clear entire RAG vector store"""
    try:
        from .agents.rag_agent import clear_vector_store
        
        clear_vector_store()
        return {
            "status": "success",
            "message": "Vector store cleared"
        }
    except Exception as e:
        logger.error(f"[RAG] Error clearing store: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/rag/sources/{source_name}")
async def remove_source(source_name: str):
    """Remove a specific source from vector store"""
    try:
        from .agents.rag_agent import remove_source as rag_remove_source
        
        success = rag_remove_source(source_name)
        if not success:
            raise HTTPException(status_code=404, detail="Source not found")
        
        return {
            "status": "success",
            "message": f"Source removed: {source_name}"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[RAG] Error removing source: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/rag/images")
async def list_images():
    """List all extracted images from PDFs"""
    try:
        uploads_dir = Path("./uploads")
        images = []
        
        for img_folder in uploads_dir.glob("*_images"):
            pdf_name = img_folder.name.replace("_images", "")
            for img_file in img_folder.glob("*.png"):
                images.append({
                    "source": pdf_name,
                    "filename": img_file.name,
                    "url": f"/uploads/{img_folder.name}/{img_file.name}",
                    "size": img_file.stat().st_size
                })
        
        return {
            "total_images": len(images),
            "images": images
        }
    except Exception as e:
        logger.error(f"[IMAGES] Error listing images: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/debug/chunks", response_class=HTMLResponse)
async def debug_chunks_html():
    """Web UI: View all chunks in vector store"""
    try:
        from .agents.rag_agent import _vector_store
        
        # Get all documents from vector store
        all_docs = _vector_store.similarity_search("", k=100)
        
        chunks_by_source = {}
        for idx, doc in enumerate(all_docs):
            source = doc.metadata.get("source", "unknown")
            if source not in chunks_by_source:
                chunks_by_source[source] = []
            chunks_by_source[source].append({
                "id": idx + 1,
                "content": doc.page_content,
                "length": len(doc.page_content),
                "metadata": doc.metadata
            })
        
        # Generate HTML
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>RAG Vector Store - Chunks Debug</title>
            <style>
                * {{ margin: 0; padding: 0; box-sizing: border-box; }}
                body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #0d1117; color: #c9d1d9; padding: 20px; }}
                .container {{ max-width: 1400px; margin: 0 auto; }}
                h1 {{ color: #58a6ff; margin-bottom: 20px; font-size: 28px; }}
                .stats {{ background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 20px; margin-bottom: 20px; }}
                .stat-item {{ display: inline-block; margin-right: 30px; }}
                .stat-label {{ color: #8b949e; font-size: 14px; }}
                .stat-value {{ color: #58a6ff; font-size: 24px; font-weight: bold; }}
                .file-section {{ background: #161b22; border: 1px solid #30363d; border-radius: 6px; margin-bottom: 20px; overflow: hidden; }}
                .file-header {{ background: #21262d; padding: 15px 20px; border-bottom: 1px solid #30363d; cursor: pointer; user-select: none; }}
                .file-header:hover {{ background: #2d333b; }}
                .file-name {{ color: #58a6ff; font-size: 18px; font-weight: 600; }}
                .file-meta {{ color: #8b949e; font-size: 14px; margin-left: 15px; }}
                .chunks-container {{ padding: 20px; display: none; }}
                .chunks-container.active {{ display: block; }}
                .chunk {{ background: #0d1117; border: 1px solid #30363d; border-radius: 6px; padding: 15px; margin-bottom: 15px; }}
                .chunk-header {{ display: flex; justify-content: space-between; margin-bottom: 10px; padding-bottom: 8px; border-bottom: 1px solid #30363d; }}
                .chunk-id {{ color: #f85149; font-weight: bold; }}
                .chunk-length {{ color: #8b949e; font-size: 13px; }}
                .chunk-content {{ color: #c9d1d9; line-height: 1.6; white-space: pre-wrap; word-wrap: break-word; font-family: 'Consolas', monospace; font-size: 13px; }}
                .chunk-content img {{ cursor: pointer; transition: transform 0.2s; }}
                .chunk-content img:hover {{ transform: scale(1.02); box-shadow: 0 4px 12px rgba(88, 166, 255, 0.3); }}
                .toggle-icon {{ float: right; transition: transform 0.3s; }}
                .toggle-icon.active {{ transform: rotate(180deg); }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>📦 RAG Vector Store - Chunks Debug</h1>
                <div class="stats">
                    <div class="stat-item">
                        <div class="stat-label">Total Chunks</div>
                        <div class="stat-value">{total_chunks}</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Total Files</div>
                        <div class="stat-value">{total_files}</div>
                    </div>
                </div>
        """.format(
            total_chunks=len(all_docs),
            total_files=len(chunks_by_source)
        )
        
        # Add file sections
        for source, chunks in chunks_by_source.items():
            html += f"""
                <div class="file-section">
                    <div class="file-header" onclick="toggleChunks('{source}')">
                        <span class="file-name">📄 {source}</span>
                        <span class="file-meta">{len(chunks)} chunks</span>
                        <span class="toggle-icon" id="icon-{source}">▼</span>
                    </div>
                    <div class="chunks-container" id="chunks-{source}">
            """
            
            for chunk in chunks:
                # Convert markdown image paths to clickable URLs
                import re
                content = chunk['content']
                
                # Replace ![alt](path) with HTML img tags
                def replace_image(match):
                    alt_text = match.group(1) or "Image"
                    img_path = match.group(2)
                    
                    # Ensure path starts with /
                    if not img_path.startswith('/') and not img_path.startswith('http'):
                        img_path = '/' + img_path
                    
                    return f'''<div style="margin:10px 0; background:#161b22; padding:10px; border-radius:6px;">
                        <a href="{img_path}" target="_blank">
                            <img src="{img_path}" 
                                 style="max-width:100%; max-height:400px; border:1px solid #30363d; border-radius:6px; display:block;" 
                                 alt="{alt_text}"
                                 onerror="this.parentElement.innerHTML='❌ Image not found: {img_path}'"
                            />
                        </a>
                        <small style="color:#8b949e">🖼️ {alt_text} - Click to open full size</small>
                    </div>'''
                
                content = re.sub(
                    r'!\[([^\]]*)\]\(([^)]+)\)',
                    replace_image,
                    content
                )
                
                # ✅ YENİ: Metadata'yı göster (TÜM ALANLAR)
                metadata = chunk.get('metadata', {})
                metadata_items = []
                
                # Core metadata
                metadata_items.append(f"Source: {metadata.get('source', 'N/A')}")
                metadata_items.append(f"Chunk ID: {metadata.get('chunk_id', 'N/A')}")
                metadata_items.append(f"Index: {metadata.get('chunk_index', 'N/A')}")
                
                # Content metadata
                if metadata.get('title'):
                    metadata_items.append(f"Title: {metadata['title'][:80]}")
                if metadata.get('summary'):
                    metadata_items.append(f"Summary: {metadata['summary'][:120]}...")
                
                # Visual metadata
                metadata_items.append(f"Has Images: {'✅' if metadata.get('has_images') else '❌'}")
                if metadata.get('cross_references'):
                    metadata_items.append(f"Cross-Refs: {metadata['cross_references']}")
                if metadata.get('table_count', 0) > 0:
                    metadata_items.append(f"Tables: {metadata['table_count']}")
                
                # Section metadata
                if metadata.get('section_h1'):
                    metadata_items.append(f"Section H1: {metadata['section_h1']}")
                if metadata.get('section_h2'):
                    metadata_items.append(f"Section H2: {metadata['section_h2']}")
                
                metadata_html = f"""
                    <div style="background:#0d1117; border:1px solid #30363d; border-radius:4px; padding:8px; margin:10px 0; font-size:12px;">
                        <strong style="color:#58a6ff;">📋 Metadata:</strong><br>
                        <span style="color:#8b949e;">{'<br>• '.join([''] + metadata_items)}</span>
                    </div>
                """ if metadata_items else ""
                
                html += f"""
                        <div class="chunk">
                            <div class="chunk-header">
                                <span class="chunk-id">Chunk #{chunk['id']}</span>
                                <span class="chunk-length">{chunk['length']} chars</span>
                                
                            </div>
                            {metadata_html}
                            <div class="chunk-content">{content}</div>
                        </div>
                """
            
            html += """
                    </div>
                </div>
            """
        
        html += """
            </div>
            <script>
                function toggleChunks(sourceId) {
                    const container = document.getElementById('chunks-' + sourceId);
                    const icon = document.getElementById('icon-' + sourceId);
                    container.classList.toggle('active');
                    icon.classList.toggle('active');
                }
            </script>
        </body>
        </html>
        """
        
        return HTMLResponse(content=html)
        
    except Exception as e:
        logger.error(f"[DEBUG] Error generating chunks HTML: {e}", exc_info=True)
        return HTMLResponse(content=f"<h1>Error: {str(e)}</h1>", status_code=500)


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


# ============ INNGEST FUNCTIONS ============

@inngest_client.create_function(
    fn_id="process_pdf_async",
    trigger=inngest.TriggerEvent(event="rag/pdf.uploaded"),
)
async def process_pdf_async(ctx: inngest.Context) -> dict:
    """Process PDF file in background with vision analysis"""
    try:
        file_path = ctx.event.data["file_path"]
        source_name = ctx.event.data["source_name"]
        
        ctx.logger.info(f"[INNGEST] Processing PDF: {source_name}")
        
        # Import here to avoid circular imports
        from .agents.rag_agent import load_pdf, ingest_text
        
        # Load PDF with images (slow operation)
        text = load_pdf.invoke(file_path)
        
        # Ingest into vector store (with agentic chunking + vision)
        chunk_count = ingest_text(text, source_name)
        
        ctx.logger.info(f"[INNGEST] ✅ Processed {source_name}: {chunk_count} chunks")
        
        return {
            "success": True,
            "source": source_name,
            "chunks": chunk_count
        }
    except Exception as e:
        ctx.logger.error(f"[INNGEST] Error processing PDF: {e}")
        return {"success": False, "error": str(e)}


@inngest_client.create_function(
    fn_id="analyze_image_batch",
    trigger=inngest.TriggerEvent(event="rag/images.batch"),
)
async def analyze_image_batch(ctx: inngest.Context) -> dict:
    """Analyze multiple images in background (slow llava operations)"""
    try:
        image_paths = ctx.event.data["image_paths"]
        
        ctx.logger.info(f"[INNGEST] Analyzing {len(image_paths)} images...")
        
        results = []
        for img_path in image_paths:
            try:
                from .agents.rag_agent import analyze_image
                analysis = await analyze_image.ainvoke(img_path)
                results.append({"path": img_path, "analysis": analysis, "success": True})
            except Exception as e:
                results.append({"path": img_path, "error": str(e), "success": False})
        
        success_count = sum(1 for r in results if r["success"])
        ctx.logger.info(f"[INNGEST] ✅ Analyzed {success_count}/{len(image_paths)} images")
        
        return {"success": True, "results": results}
    except Exception as e:
        ctx.logger.error(f"[INNGEST] Batch image analysis failed: {e}")
        return {"success": False, "error": str(e)}


@inngest_client.create_function(
    fn_id="cleanup_old_files",
    trigger=inngest.TriggerCron(cron="0 2 * * *"),  # Daily at 2 AM
)
async def cleanup_old_files(ctx: inngest.Context) -> dict:
    """Clean up old uploaded files (scheduled task)"""
    try:
        upload_dir = Path("./uploads")
        cutoff_date = datetime.now() - timedelta(days=30)  # 30 days old
        
        deleted_count = 0
        for file_path in upload_dir.iterdir():
            if file_path.is_file():
                file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                if file_time < cutoff_date:
                    file_path.unlink()
                    deleted_count += 1
        
        ctx.logger.info(f"[INNGEST] 🗑️ Cleaned up {deleted_count} old files")
        return {"success": True, "deleted": deleted_count}
    except Exception as e:
        ctx.logger.error(f"[INNGEST] Cleanup failed: {e}")
        return {"success": False, "error": str(e)}


# ============ SERVE INNGEST ENDPOINT ============
# Dev mode: Works without signing key when using Inngest Dev Server
inngest.fast_api.serve(
    app, 
    inngest_client, 
    [process_pdf_async, analyze_image_batch, cleanup_old_files]
)
logger.info("[INNGEST] ✅ Endpoint registered at /api/inngest")
logger.info("[INNGEST] Start dev server: npx inngest-cli@latest dev -u http://127.0.0.1:8000/api/inngest")


# ==================== HITL RAG STATE ENDPOINTS ====================

@app.get("/api/rag/state")
async def get_rag_state():
    """
    Get current RAG state for frontend HITL workflow.
    
    Returns RAGState with:
    - retrieved_chunks: Chunks waiting for approval
    - approved_chunk_ids: User-approved chunk IDs
    - awaiting_approval: True if waiting for user
    - is_synthesizing: True if generating answer
    - current_query: User's query
    """
    try:
        state_dict = _rag_state.model_dump()
        logger.info(f"[HITL API] State requested - awaiting_approval={_rag_state.awaiting_approval}, chunks={len(_rag_state.retrieved_chunks)}")
        return state_dict
    except Exception as e:
        logger.error(f"[HITL API] Error getting state: {e}")
        return {"error": str(e)}


@app.post("/api/rag/approve")
async def approve_chunks(request: Request):
    """
    User approved chunks - trigger synthesis.
    
    Body: { "approved_chunk_ids": ["chunk1", "chunk2", ...] }
    """
    try:
        data = await request.json()
        approved_ids = data.get("approved_chunk_ids", [])
        
        if not approved_ids:
            return {"error": "No chunks selected", "status": "error"}
        
        logger.info(f"[HITL API] User approved {len(approved_ids)} chunks")
        
        # Update state
        _rag_state.approved_chunk_ids = approved_ids
        _rag_state.awaiting_approval = False
        _rag_state.is_synthesizing = True
        
        # Call synthesize_answer tool
        from .agents.rag_agent import synthesize_answer
        
        answer = await synthesize_answer(approved_ids)
        
        return {
            "status": "success",
            "count": len(approved_ids),
            "message": "Chunks approved and answer generated",
            "answer": answer
        }
        
    except Exception as e:
        logger.error(f"[HITL API] Error approving chunks: {e}")
        return {"error": str(e), "status": "error"}


@app.post("/api/rag/reset")
async def reset_rag_state():
    """Reset RAG state to initial condition"""
    try:
        global _rag_state
        from .models.rag_models import RAGState
        _rag_state = RAGState()  # Fresh state
        
        logger.info("[HITL API] State reset")
        return {"status": "success", "message": "RAG state reset"}
    except Exception as e:
        logger.error(f"[HITL API] Error resetting state: {e}")
        return {"error": str(e), "status": "error"}


@app.get("/api/rag/chunks")
async def get_all_chunks(query: Optional[str] = None):
    """
    Get ALL chunks from vector store with full metadata.
    If query is provided, calculates relevance scores and sorts by relevance.
    Used for Chunk Explorer UI.
    """
    try:
        from .agents.rag_agent import _vector_store
        
        # 1. Get ALL raw data first to populate the list
        all_data = _vector_store.get()
        
        if not all_data or 'documents' not in all_data:
            return []
            
        # 2. If query is provided, perform semantic search to get scores
        score_map = {}
        if query and query.strip():
            try:
                # Search for relevance scores (k=len(ids) to score potentially all)
                # Note: similarity_search_with_relevance_scores converts distance to 0-1 score
                search_results = _vector_store.similarity_search_with_relevance_scores(
                    query, 
                    k=len(all_data['ids']) 
                )
                
                # Map chunk_id (or index) to score
                for doc, score in search_results:
                    # Try to use chunk_id from metadata, fallback to matching content/source
                    c_id = doc.metadata.get("chunk_id")
                    if c_id:
                        score_map[c_id] = score
                        
            except Exception as e:
                logger.warning(f"[CHUNKS API] Semantic search failed: {e}")

        chunks = []
        for i, (content, metadata, doc_id) in enumerate(zip(all_data['documents'], all_data['metadatas'], all_data['ids'])):
            
            # Metadata chunk_id vs Chroma ID sync
            c_id = metadata.get("chunk_id", doc_id)
            
            # Extract images - 1. Method: Markdown
            import re
            from pathlib import Path
            image_paths = re.findall(r'!\[.*?\]\((.*?)\)', content)
            
            # 2. Method: Filesystem fallback
            if not image_paths and metadata.get("has_images", False):
                source_file = metadata.get("source", "")
                if source_file:
                    base_name = Path(source_file).stem
                    # Check uploads directories
                    upload_dir = Path("uploads")
                    found_images = []
                    
                    # A. Subdirectory check: uploads/DocName_images/
                    specific_dir = upload_dir / f"{base_name}_images"
                    if specific_dir.exists():
                        found_images.extend(specific_dir.glob("*.png"))
                        found_images.extend(specific_dir.glob("*.jpg"))
                        found_images.extend(specific_dir.glob("*.jpeg"))
                    
                    # B. Root uploads check
                    if not found_images:
                        found_images.extend(upload_dir.glob(f"{base_name}-*.png"))
                        found_images.extend(upload_dir.glob(f"{base_name}-*.jpg"))
                    
                    # Convert to web-accessible paths (relative to root, e.g. "uploads/folder/img.png")
                    image_paths = [str(img).replace("\\", "/") for img in found_images]
            
            # Determine Relevance Score
            # If query exists: use score from vector search (default 0.0 if not in top results)
            # If no query: default to 0.0
            relevance = 0.0
            if query and query.strip():
                relevance = score_map.get(c_id, 0.0) 
            
            # ✅ Rich metadata
            chunk_data = {
                "chunk_id": c_id,
                "title": metadata.get("title", "Başlıksız"),
                "summary": metadata.get("summary", content[:200] + "..."),
                "content": content,
                "source": metadata.get("source", "unknown"),
                "metadata": {
                    "chunk_index": metadata.get("chunk_index", i),
                    "section_h1": metadata.get("section_h1"),
                    "section_h2": metadata.get("section_h2"),
                    "section_h3": metadata.get("section_h3"),
                    "has_images": metadata.get("has_images", False),
                    "cross_references": metadata.get("cross_references"),
                    "table_count": metadata.get("table_count", 0),
                    "page": metadata.get("page"),
                    "paragraph_index": metadata.get("paragraph_index"),
                    "created_at": metadata.get("created_at"),
                    "updated_at": metadata.get("updated_at"),
                    "chunk_type": metadata.get("chunk_type"),
                    "importance_score": metadata.get("importance_score"),
                    "raw": metadata
                },
                "images": image_paths,
                "relevance": relevance,
                "tags": metadata.get("tags", [])
            }
            chunks.append(chunk_data)
            
        # 4. Sort by relevance if query exists, else by index
        if query and query.strip():
            chunks.sort(key=lambda x: x['relevance'], reverse=True)
        else:
            chunks.sort(key=lambda x: (x['source'], x['metadata'].get('chunk_index', 0)))
            
        return {
            "chunks": chunks,
            "total": len(chunks),
            "query": query
        }

    except Exception as e:
        logger.error(f"[CHUNKS API] Error: {e}", exc_info=True)
        return {"chunks": [], "error": str(e)}
        



logger.info("[HITL] ✅ RAG State endpoints registered:")
logger.info("  - GET  /api/rag/state (Get current state)")
logger.info("  - POST /api/rag/approve (Approve chunks)")
logger.info("  - POST /api/rag/reset (Reset state)")


if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*70)
    print("[START] AI Research API (Showcase Mode)")
    print("   [OK] No Auth Required")
    print("   [OK] Inngest: /api/inngest")
    print("   [OK] Demo User Active")
    print("="*70)
    
    uvicorn.run(
        "src.simple_copilot_backend:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
