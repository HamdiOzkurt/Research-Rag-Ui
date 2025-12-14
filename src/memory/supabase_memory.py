"""
Supabase Memory Manager
Conversation history'yi Supabase'de saklar
"""
try:
    from supabase import create_client, Client
except Exception:  # pragma: no cover
    create_client = None
    Client = object  # type: ignore
from typing import List, Dict, Optional, Any
from datetime import datetime
import json
import uuid
import os
from collections import defaultdict


class SupabaseMemory:
    """Supabase ile conversation memory yönetimi"""
    
    def __init__(self):
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_KEY")
        self.session_id = os.getenv("SESSION_ID", str(uuid.uuid4()))
        # UI tarafında thread_id yoksa session_id fallback olarak kullanılabilir
        
        if not create_client:
            self.client = None
            print("⚠️ supabase paketi kurulu değil - memory devre dışı")
        elif not self.supabase_url or not self.supabase_key:
            self.client = None
            print("⚠️ Supabase credentials bulunamadı - memory devre dışı")
        else:
            try:
                self.client: Client = create_client(self.supabase_url, self.supabase_key)
                print("✅ Supabase memory aktif")
            except Exception as e:
                self.client = None
                print(f"⚠️ Supabase bağlantı hatası: {e}")

    def is_enabled(self) -> bool:
        """Supabase memory aktif mi?"""
        return bool(self.client)
    
    def save_message(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict] = None
    ) -> bool:
        """Mesajı Supabase'e kaydet"""
        if not self.client:
            return False
        
        try:
            # Yeni schema önerisi: ai_threads + ai_messages.
            # Şimdilik backward compatible: tek tabloya yazıyoruz.
            md = metadata or {}
            user_id = md.get("user_id")
            thread_id = md.get("thread_id") or self.session_id

            data = {
                "session_id": str(thread_id),
                "message_role": role,
                "message_content": content,
                "metadata": json.dumps(md) if md else None,
                "created_at": datetime.utcnow().isoformat()
            }
            
            self.client.table("conversations").insert(data).execute()
            return True
            
        except Exception as e:
            print(f"⚠️ Supabase kayıt hatası: {e}")
            return False
    
    def load_history(
        self,
        n_messages: int = 20,
        session_id: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """Conversation history'yi yükle"""
        if not self.client:
            return []
        
        try:
            sid = session_id or self.session_id
            
            response = (
                self.client.table("conversations")
                .select("message_role, message_content, created_at")
                .eq("session_id", sid)
                .order("created_at", desc=False)
                .limit(n_messages)
                .execute()
            )
            
            messages = []
            for row in response.data:
                messages.append({
                    "role": row["message_role"],
                    "content": row["message_content"]
                })
            
            return messages
            
        except Exception as e:
            print(f"⚠️ Supabase yükleme hatası: {e}")
            return []

    # =============================================================================
    # SaaS helpers (user_id + thread_id)
    # =============================================================================

    def list_threads(self, user_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        conversations tablosundan user_id bazlı thread listesi çıkarır.
        (metadata.user_id ve session_id=thread_id kullanıyoruz)
        """
        if not self.client:
            return []
        try:
            # Supabase JSON query her projede farklı olabiliyor; burada basitçe son mesajları çekip python'da gruplayalım
            resp = (
                self.client.table("conversations")
                .select("session_id, message_role, message_content, created_at, metadata")
                .order("created_at", desc=True)
                .limit(500)
                .execute()
            )
            threads: Dict[str, Dict[str, Any]] = {}
            for row in resp.data or []:
                md = {}
                if row.get("metadata"):
                    try:
                        md = json.loads(row["metadata"])
                    except Exception:
                        md = {}
                if md.get("user_id") != user_id:
                    continue
                tid = row.get("session_id")
                if tid not in threads:
                    threads[tid] = {
                        "thread_id": tid,
                        "last_message_at": row.get("created_at"),
                        "preview": row.get("message_content", "")[:120],
                    }
            return list(threads.values())[:limit]
        except Exception as e:
            print(f"⚠️ Supabase threads hatası: {e}")
            return []

    def load_thread(self, user_id: str, thread_id: str, limit: int = 200) -> List[Dict[str, Any]]:
        """Thread mesajlarını döndürür."""
        if not self.client:
            return []
        try:
            resp = (
                self.client.table("conversations")
                .select("message_role, message_content, created_at, metadata, session_id")
                .eq("session_id", thread_id)
                .order("created_at", desc=False)
                .limit(limit)
                .execute()
            )
            out: List[Dict[str, Any]] = []
            for row in resp.data or []:
                md = {}
                if row.get("metadata"):
                    try:
                        md = json.loads(row["metadata"])
                    except Exception:
                        md = {}
                if md.get("user_id") != user_id:
                    continue
                out.append(
                    {
                        "role": row.get("message_role"),
                        "content": row.get("message_content"),
                        "created_at": row.get("created_at"),
                        "metadata": md,
                    }
                )
            return out
        except Exception as e:
            print(f"⚠️ Supabase load_thread hatası: {e}")
            return []

    def delete_thread(self, user_id: str, thread_id: str) -> bool:
        """Thread'i sil."""
        if not self.client:
            return False
        try:
            # güvenlik: önce rowları çekip user_id eşleşiyor mu kontrol et
            msgs = self.load_thread(user_id=user_id, thread_id=thread_id, limit=1)
            if not msgs:
                return False
            self.client.table("conversations").delete().eq("session_id", thread_id).execute()
            return True
        except Exception as e:
            print(f"⚠️ Supabase delete_thread hatası: {e}")
            return False
    
    def clear_session(self, session_id: Optional[str] = None):
        """Belirli bir session'ı temizle"""
        if not self.client:
            return False
        
        try:
            sid = session_id or self.session_id
            self.client.table("conversations").delete().eq("session_id", sid).execute()
            return True
        except Exception as e:
            print(f"⚠️ Supabase silme hatası: {e}")
            return False


# Global instance
_memory_instance = None


def get_memory() -> SupabaseMemory:
    """Singleton memory instance"""
    global _memory_instance
    if _memory_instance is None:
        _memory_instance = SupabaseMemory()
    return _memory_instance


def save_to_supabase(role: str, content: str, metadata: Optional[Dict] = None) -> bool:
    """Kısa yol: Mesaj kaydet"""
    return get_memory().save_message(role, content, metadata)


def load_conversation_history(n_messages: int = 20) -> List[Dict[str, str]]:
    """Kısa yol: History yükle"""
    return get_memory().load_history(n_messages)


def get_or_create_session_id() -> str:
    """Session ID al veya oluştur"""
    return get_memory().session_id
