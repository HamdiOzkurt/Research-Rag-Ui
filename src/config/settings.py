"""
Proje Konfigürasyonu
Multi API Key Rotation + Ollama desteği
"""
import os
import random
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)

# .env dosyasını yükle
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path)


class Settings(BaseModel):
    """Uygulama ayarları - Multi API Key desteği"""
    
    # ============ MULTI API KEYS ============
    
    # Birden fazla Gemini API key (virgülle ayrılmış)
    # .env: GOOGLE_API_KEYS=key1,key2,key3
    google_api_keys: List[str] = Field(default_factory=lambda: Settings._parse_api_keys())
    
    # Mevcut aktif key index
    _current_key_index: int = 0
    _failed_keys: set = set()  # Hata veren key'ler
    
    @staticmethod
    def _parse_api_keys() -> List[str]:
        """API key'leri parse et"""
        # Önce GOOGLE_API_KEYS dene (çoklu)
        keys_str = os.getenv("GOOGLE_API_KEYS", "")
        if keys_str:
            keys = [k.strip() for k in keys_str.split(",") if k.strip()]
            if keys:
                return keys
        
        # Yoksa GOOGLE_API_KEY dene (tekli)
        single_key = os.getenv("GOOGLE_API_KEY", "")
        if single_key:
            return [single_key]
        
        return []
    
    @property
    def google_api_key(self) -> Optional[str]:
        """Aktif API key'i döndür (rotation ile)"""
        if not self.google_api_keys:
            return None
        
        available_keys = [k for i, k in enumerate(self.google_api_keys) 
                         if i not in self._failed_keys]
        
        if not available_keys:
            # Tüm key'ler failed, reset yap
            self._failed_keys.clear()
            available_keys = self.google_api_keys
        
        return available_keys[self._current_key_index % len(available_keys)]
    
    def rotate_api_key(self, mark_failed: bool = False):
        """Sonraki API key'e geç"""
        if mark_failed and self.google_api_keys:
            self._failed_keys.add(self._current_key_index)
            logger.warning(f"[WARN] API Key {self._current_key_index + 1} failed, rotating...")
        
        self._current_key_index = (self._current_key_index + 1) % max(1, len(self.google_api_keys))
        logger.info(f"[ROTATE] Rotated to API Key {self._current_key_index + 1}/{len(self.google_api_keys)}")
    
    def get_random_api_key(self) -> Optional[str]:
        """Rastgele bir API key döndür"""
        available_keys = [k for i, k in enumerate(self.google_api_keys) 
                         if i not in self._failed_keys]
        
        if not available_keys:
            self._failed_keys.clear()
            available_keys = self.google_api_keys
        
        return random.choice(available_keys) if available_keys else None
    
    # ============ Ollama (Local) ============
    ollama_base_url: str = Field(
        default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )
    
    # ============ MCP API Keys ============
    firecrawl_api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("FIRECRAWL_API_KEY")
    )
    tavily_api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("TAVILY_API_KEY")
    )
    
    # ============ Database ============
    supabase_url: Optional[str] = Field(
        default_factory=lambda: os.getenv("SUPABASE_URL")
    )
    supabase_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("SUPABASE_KEY")
    )
    
    # ============ İzleme ============
    langsmith_api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("LANGSMITH_API_KEY")
    )
    
    # ============ Model Seçimi ============
    default_model: str = Field(
        default_factory=lambda: os.getenv("DEFAULT_MODEL", "google_genai:gemini-2.0-flash-exp")
    )
    secondary_model: str = Field(
        default_factory=lambda: os.getenv("SECONDARY_MODEL", "ollama:llama3.2")
    )
    
    # ============ Firecrawl MCP ============
    firecrawl_mcp_command: str = "npx"
    firecrawl_mcp_args: list = ["-y", "firecrawl-mcp"]
    
    # ============ Yollar ============
    project_root: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent.parent
    )
    
    def get_firecrawl_env(self) -> dict:
        """Firecrawl MCP için environment variables"""
        return {"FIRECRAWL_API_KEY": self.firecrawl_api_key or ""}
    
    def get_model_provider(self, model_string: str) -> tuple:
        """Model string'inden provider ve model adını çıkarır"""
        if ":" in model_string:
            provider, model = model_string.split(":", 1)
            return provider, model
        return "google_genai", model_string
    
    def validate_api_keys(self) -> dict:
        """API key'lerin durumunu kontrol eder"""
        return {
            "google_gemini": {
                "available": bool(self.google_api_keys),
                "count": len(self.google_api_keys),
                "active_index": self._current_key_index + 1
            },
            "ollama_local": self._check_ollama(),
            "firecrawl": bool(self.firecrawl_api_key),
            "tavily": bool(self.tavily_api_key),
            "langsmith": bool(self.langsmith_api_key),
        }
    
    def _check_ollama(self) -> bool:
        """Ollama'nın çalışıp çalışmadığını kontrol eder"""
        try:
            import httpx
            response = httpx.get(f"{self.ollama_base_url}/api/tags", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def get_available_model(self) -> str:
        """Kullanılabilir ilk modeli döndürür"""
        provider, _ = self.get_model_provider(self.default_model)
        
        # Gemini kontrolü
        if provider == "google_genai" and self.google_api_key:
            return self.default_model
        
        # Ollama kontrolü
        if provider == "ollama" and self._check_ollama():
            return self.default_model
        
        # Secondary model dene
        sec_provider, _ = self.get_model_provider(self.secondary_model)
        
        if sec_provider == "google_genai" and self.google_api_key:
            return self.secondary_model
        
        if sec_provider == "ollama" and self._check_ollama():
            return self.secondary_model
        
        raise ValueError("Hiçbir LLM kullanılamıyor! GOOGLE_API_KEY(S) veya Ollama kurulumu gerekli.")


# Global settings instance
settings = Settings()
