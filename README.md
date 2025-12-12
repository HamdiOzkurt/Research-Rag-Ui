# 🔬 DeepAgents Multi-Agent Research System

> DeepAgents + Firecrawl MCP + Gemini/Ollama ile güçlü multi-agent araştırma sistemi

## ✨ Özellikler

- 🤖 **DeepAgents Framework** - LangChain ekosistemi üzerine kurulu gelişmiş ajan sistemi
- 🔍 **Firecrawl MCP** - Web scraping ve arama için güçlü araçlar
- 🧠 **Gemini 2.5 Flash** - Google'ın en hızlı ve yetenekli modeli
- 🏠 **Ollama Desteği** - Yerel LLM'ler için (privacy-first)
- 📊 **LangSmith Entegrasyonu** - Ajan akışlarını izleme ve debug

## 🏗️ Mimari

```
┌─────────────────────────────────────────────────────────────┐
│                    🧠 ANA AJAN (Orchestrator)               │
│                     Gemini 2.5 Flash                        │
└─────────────────────┬───────────────────────────────────────┘
                      │
          ┌───────────┼───────────┐
          │           │           │
          ▼           ▼           ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│ 🔍 search   │ │ 📊 analysis │ │ ✍️ writer   │
│   -agent    │ │   -agent    │ │   -agent    │
├─────────────┤ ├─────────────┤ ├─────────────┤
│ Web araması │ │ Veri analizi│ │ Türkçe rapor│
│ firecrawl   │ │ Güvenilirlik│ │ yazımı      │
│ kullanır    │ │ kontrolü    │ │             │
└──────┬──────┘ └─────────────┘ └─────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│                    🔧 MCP ARAÇLARI                          │
│  firecrawl_search | firecrawl_scrape | firecrawl_map        │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Hızlı Başlangıç

### 1. Repo'yu Klonla

```bash
git clone https://github.com/HamdiOzkurt/Deepagents_Multi.git
cd Deepagents_Multi
```

### 2. Virtual Environment Oluştur

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# veya
.\venv\Scripts\Activate.ps1  # Windows PowerShell
```

### 3. Bağımlılıkları Yükle

```bash
pip install -r requirements.txt
```

### 4. API Key'leri Ayarla

```bash
cp .env.example .env
# .env dosyasını düzenle ve API key'leri ekle
```

**Gerekli API Key'ler:**
- `GOOGLE_API_KEY` - [Google AI Studio](https://aistudio.google.com/app/apikey)
- `FIRECRAWL_API_KEY` - [Firecrawl](https://www.firecrawl.dev/app/api-keys)
- `LANGSMITH_API_KEY` (opsiyonel) - [LangSmith](https://smith.langchain.com)

### 5. Çalıştır

```bash
python -m src.main "En iyi açık kaynak LLM modelleri hangileri?"
```

## 📁 Proje Yapısı

```
Deepagents_Multi/
├── src/
│   ├── agents/
│   │   ├── __init__.py
│   │   └── main_agent.py      # DeepAgent ve subagent'lar
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py        # Konfigürasyon yönetimi
│   ├── __init__.py
│   ├── main.py                # CLI giriş noktası
│   └── models.py              # Pydantic modelleri
├── .env.example               # Örnek environment dosyası
├── .gitignore
├── README.md
└── requirements.txt
```

## 🧩 Subagent'lar

| Agent | Görevi |
|-------|--------|
| `search-agent` | Firecrawl ile web araması yapar |
| `analysis-agent` | Toplanan verileri analiz eder |
| `writer-agent` | Türkçe profesyonel rapor yazar |

## 🔧 Konfigürasyon

### Model Seçimi

`.env` dosyasında:

```bash
# Gemini (önerilen)
DEFAULT_MODEL=google_genai:gemini-2.5-flash

# Ollama (yerel)
DEFAULT_MODEL=ollama:llama3.2:8b
```

### LangSmith İzleme

LangSmith ile ajan akışlarını izleyebilirsiniz:

1. [smith.langchain.com](https://smith.langchain.com) adresinden kayıt ol
2. API key al
3. `.env` dosyasına `LANGSMITH_API_KEY` ekle

## 📖 Kullanım Örnekleri

```bash
# Tek soru
python -m src.main "React vs Vue karşılaştırması"

# İnteraktif mod
python -m src.main --interactive
```

## 🛣️ Roadmap

- [ ] Supabase entegrasyonu (araştırma hafızası)
- [ ] GitHub MCP (kod araştırması)
- [ ] Streamlit UI
- [ ] Docker desteği
- [ ] API endpoint'leri

## 📄 Lisans

MIT License

## 🤝 Katkıda Bulunma

Pull request'ler kabul edilir! Büyük değişiklikler için önce bir issue açın.
