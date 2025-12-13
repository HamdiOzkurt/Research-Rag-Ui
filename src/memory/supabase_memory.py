"""
Supabase Memory Manager
Conversation history'yi Supabase'de saklar
"""
from supabase import create_client, Client
from typing import List, Dict, Optional
from datetime import datetime
import json
import uuid
import os


class SupabaseMemory:
    """Supabase ile conversation memory yönetimi"""
    
    def __init__(self):
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_KEY")
        self.session_id = os.getenv("SESSION_ID", str(uuid.uuid4()))
        
        if not self.supabase_url or not self.supabase_key:
            self.client = None
            print("⚠️ Supabase credentials bulunamadı - memory devre dışı")
        else:
            try:
                self.client: Client = create_client(self.supabase_url, self.supabase_key)
                print("✅ Supabase memory aktif")
            except Exception as e:
                self.client = None
                print(f"⚠️ Supabase bağlantı hatası: {e}")
    
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
            data = {
                "session_id": self.session_id,
                "message_role": role,
                "message_content": content,
                "metadata": json.dumps(metadata) if metadata else None,
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
