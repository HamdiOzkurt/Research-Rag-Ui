"""
Smart Backend with Caching & Rate Limiting
429 hatasını önlemek için
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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
from .config import settings
from .memory.supabase_memory import get_memory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Research API",
    description="Smart backend with caching",
    version="2.0.0"
)

# CORS (deploy için ALLOWED_ORIGINS ile override edilebilir)
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


# =============================================================================
# SIMPLE CACHE (In-Memory)
# =============================================================================

class SimpleCache:
    """Basit in-memory cache - 429 hatasını önler"""
    
    def __init__(self, max_size: int = 100, ttl_minutes: int = 60):
        self.cache: OrderedDict = OrderedDict()
        self.max_size = max_size
        self.ttl = timedelta(minutes=ttl_minutes)
    
    def _hash_query(self, query: str) -> str:
        """Query'yi hash'le"""
        return hashlib.md5(query.lower().strip().encode()).hexdigest()
    
    def get(self, query: str) -> Optional[str]:
        """Cache'den al"""
        key = self._hash_query(query)
        if key in self.cache:
            entry = self.cache[key]
            if datetime.now() - entry["time"] < self.ttl:
                logger.info(f"✅ Cache HIT: {query[:50]}...")
                return entry["response"]
            else:
                del self.cache[key]
        return None
    
    def set(self, query: str, response: str):
        """Cache'e kaydet"""
        key = self._hash_query(query)
        
        # Max size kontrolü
        if len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)
        
        self.cache[key] = {
            "response": response,
            "time": datetime.now()
        }
        logger.info(f"💾 Cached: {query[:50]}...")
    
    def stats(self) -> dict:
        """Cache istatistikleri"""
        return {
            "entries": len(self.cache),
            "max_size": self.max_size,
            "ttl_minutes": self.ttl.seconds // 60
        }


# Global cache
cache = SimpleCache(max_size=100, ttl_minutes=60)


# =============================================================================
# RATE LIMITER
# =============================================================================

class RateLimiter:
    """Basit rate limiter"""
    
    def __init__(self, max_requests: int = 10, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window = timedelta(seconds=window_seconds)
        self.requests: list = []
    
    def is_allowed(self) -> bool:
        """İstek yapılabilir mi?"""
        now = datetime.now()
        
        # Eski istekleri temizle
        self.requests = [r for r in self.requests if now - r < self.window]
        
        if len(self.requests) < self.max_requests:
            self.requests.append(now)
            return True
        
        return False
    
    def remaining(self) -> int:
        """Kalan istek sayısı"""
        now = datetime.now()
        self.requests = [r for r in self.requests if now - r < self.window]
        return max(0, self.max_requests - len(self.requests))
    
    def reset_time(self) -> int:
        """Sıfırlanmaya kalan saniye"""
        if not self.requests:
            return 0
        oldest = min(self.requests)
        reset = oldest + self.window - datetime.now()
        return max(0, int(reset.total_seconds()))


# Gemini free: 15 req/min, güvenli limit: 10
rate_limiter = RateLimiter(max_requests=10, window_seconds=60)


# =============================================================================
# MODELS
# =============================================================================

class ChatRequest(BaseModel):
    message: str
    use_cache: bool = True  # Cache kullan mı?
    # SaaS: kullanıcı bazlı history
    user_id: Optional[str] = None  # Clerk user id (frontend gönderir)
    thread_id: Optional[str] = None  # conversation/thread id


class ChatResponse(BaseModel):
    response: str
    success: bool = True
    cached: bool = False
    remaining_requests: int = 10
    thread_id: Optional[str] = None
    saved: bool = False


# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/")
async def root():
    """Health check with stats"""
    return {
        "status": "online",
        "service": "AI Research API v2",
        "version": "2.0.0",
        "model": settings.default_model,
        "cache": cache.stats(),
        "rate_limit": {
            "remaining": rate_limiter.remaining(),
            "reset_in_seconds": rate_limiter.reset_time()
        }
    }


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Smart chat endpoint with caching"""
    
    query = request.message.strip()
    
    if not query:
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    memory = get_memory()
    thread_id = request.thread_id or str(uuid.uuid4())
    saved = False

    # 0. History: kullanıcı varsa user mesajını kaydet (cache olsa bile)
    if request.user_id:
        saved = memory.save_message(
            role="user",
            content=query,
            metadata={"user_id": request.user_id, "thread_id": thread_id},
        ) or saved

    # 1. Cache kontrol
    if request.use_cache:
        cached_response = cache.get(query)
        if cached_response:
            # cache response'u da history'e kaydet
            if request.user_id:
                saved = memory.save_message(
                    role="assistant",
                    content=cached_response,
                    metadata={"user_id": request.user_id, "thread_id": thread_id, "cached": True},
                ) or saved
            return ChatResponse(
                response=cached_response,
                success=True,
                cached=True,
                remaining_requests=rate_limiter.remaining(),
                thread_id=thread_id,
                saved=saved,
            )
    
    # 2. Rate limit kontrol
    if not rate_limiter.is_allowed():
        return ChatResponse(
            response=f"⚠️ Rate limit! Lütfen {rate_limiter.reset_time()} saniye bekleyin.\n\n💡 İpucu: Aynı soruyu tekrar sorarsanız cache'den gelir.",
            success=False,
            cached=False,
            remaining_requests=0,
            thread_id=thread_id,
            saved=saved,
        )
    
    # 3. AI'dan yanıt al
    try:
        logger.info(f"🔍 New query: {query[:50]}...")
        
        result = await run_simple_research(query, verbose=False)
        
        # Cache'e kaydet
        if result and not result.startswith("❌"):
            cache.set(query, result)

        # History: assistant mesajını kaydet
        if request.user_id:
            saved = memory.save_message(
                role="assistant",
                content=result,
                metadata={"user_id": request.user_id, "thread_id": thread_id, "cached": False},
            ) or saved
        
        return ChatResponse(
            response=result,
            success=True,
            cached=False,
            remaining_requests=rate_limiter.remaining(),
            thread_id=thread_id,
            saved=saved,
        )
    
    except Exception as e:
        error_msg = str(e)
        
        # 429 hatası özel handling
        if "429" in error_msg or "quota" in error_msg.lower():
            return ChatResponse(
                response="⚠️ API limiti doldu! Lütfen 1 dakika bekleyin veya Ollama kullanın.",
                success=False,
                cached=False,
                remaining_requests=0,
                thread_id=thread_id,
                saved=saved,
            )
        
        logger.error(f"❌ Error: {error_msg}", exc_info=True)
        return ChatResponse(
            response=f"❌ Hata: {error_msg}",
            success=False,
            cached=False,
            remaining_requests=rate_limiter.remaining(),
            thread_id=thread_id,
            saved=saved,
        )


@app.get("/threads")
async def list_threads(user_id: str, limit: int = 20):
    """
    Kullanıcıya ait thread listesi.
    Not: Şimdilik user_id frontend'den geliyor. Production'da Clerk JWT verify eklenmeli.
    """
    memory = get_memory()
    return memory.list_threads(user_id=user_id, limit=limit)


@app.get("/threads/{thread_id}")
async def get_thread(thread_id: str, user_id: str, limit: int = 200):
    """Thread mesajlarını döndürür."""
    memory = get_memory()
    return memory.load_thread(user_id=user_id, thread_id=thread_id, limit=limit)


@app.delete("/threads/{thread_id}")
async def delete_thread(thread_id: str, user_id: str):
    """Thread'i sil."""
    memory = get_memory()
    ok = memory.delete_thread(user_id=user_id, thread_id=thread_id)
    return {"success": ok}


@app.get("/stats")
async def stats():
    """Cache ve rate limit istatistikleri"""
    return {
        "cache": cache.stats(),
        "rate_limit": {
            "max_requests_per_minute": rate_limiter.max_requests,
            "remaining": rate_limiter.remaining(),
            "reset_in_seconds": rate_limiter.reset_time()
        }
    }


@app.get("/health")
async def health():
    """Env + Supabase bağlantı durumu (debug için)"""
    memory = get_memory()
    api_status = settings.validate_api_keys()
    return {
        "status": "ok",
        "model": settings.default_model,
        "apis": api_status,
        "supabase": {
            "enabled": memory.is_enabled(),
            "has_url": bool(os.getenv("SUPABASE_URL")),
            "has_key": bool(os.getenv("SUPABASE_KEY")),
            "table": "conversations",
        },
    }


@app.delete("/cache")
async def clear_cache():
    """Cache'i temizle"""
    cache.cache.clear()
    return {"message": "Cache cleared", "entries": 0}


if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*70)
    print("🚀 AI Research API v2.0")
    print("   ✅ Response Caching")
    print("   ✅ Rate Limiting")
    print("   ✅ 429 Protection")
    print("="*70)
    
    uvicorn.run(
        "src.simple_copilot_backend:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
