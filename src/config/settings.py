"""
Proje Konfigürasyonu
Ollama (local) + Gemini 2.5 Flash desteği
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Optional

# .env dosyasını yükle
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path)


class Settings(BaseModel):
    """Uygulama ayarları"""
    
    # ============ LLM API Keys ============
    
    # Gemini (Google AI)
    google_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("GOOGLE_API_KEY"))
    
    # Ollama (Local)
    ollama_base_url: str = Field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    

    
    # ============ MCP API Keys ============
    
    # Firecrawl (web scraping)
    firecrawl_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("FIRECRAWL_API_KEY"))
    
    # Tavily (web search)
    tavily_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("TAVILY_API_KEY"))
    
    # ============ Database ============
    
    # Supabase (conversation memory)
    supabase_url: Optional[str] = Field(default_factory=lambda: os.getenv("SUPABASE_URL"))
    supabase_key: Optional[str] = Field(default_factory=lambda: os.getenv("SUPABASE_KEY"))
    
    # ============ İzleme ============
    
    # LangSmith
    langsmith_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("LANGSMITH_API_KEY"))
    
    # ============ Model Seçimi ============
    
    # Varsayılan: Gemini 2.5 Flash
    default_model: str = Field(
        default_factory=lambda: os.getenv("DEFAULT_MODEL", "google_genai:gemini-2.5-flash-preview-05-20")
    )
    
    # İkincil model (Ollama local)
    secondary_model: str = Field(
        default_factory=lambda: os.getenv("SECONDARY_MODEL", "ollama:llama3.2")
    )
    
    # ============ Firecrawl MCP ============
    
    firecrawl_mcp_command: str = "npx"
    firecrawl_mcp_args: list = ["-y", "firecrawl-mcp"]
    
    # ============ Yollar ============
    
    project_root: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent)
    
    def get_firecrawl_env(self) -> dict:
        """Firecrawl MCP için environment variables"""
        return {
            "FIRECRAWL_API_KEY": self.firecrawl_api_key or ""
        }
    
    def get_model_provider(self, model_string: str) -> tuple:
        """Model string'inden provider ve model adını çıkarır"""
        if ":" in model_string:
            provider, model = model_string.split(":", 1)
            return provider, model
        return "google_genai", model_string
    
    def validate_api_keys(self) -> dict:
        """API key'lerin durumunu kontrol eder"""
        return {
            "google (gemini)": bool(self.google_api_key),
            "ollama (local)": self._check_ollama(),
            "firecrawl": bool(self.firecrawl_api_key),
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
        
        raise ValueError("Hiçbir LLM kullanılamıyor! GOOGLE_API_KEY veya Ollama kurulumu gerekli.")


# Global settings instance
settings = Settings()
