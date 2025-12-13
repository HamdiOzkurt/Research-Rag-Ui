# 🔬 DeepAgents Multi-Agent Research System

**Gemini 2.5 Flash + Firecrawl + Tavily + GitHub + Supabase**

Gelişmiş web araştırması yapan, kod arama, Türkçe raporlar üreten, conversation history'yi saklayan AI araştırma asistanı.

---

## ✨ Özellikler

- 🤖 **DeepAgents Framework** - LangGraph tabanlı agent orchestration
- 🔍 **Çoklu Arama Kaynakları**:
  - Firecrawl MCP (web scraping)
  - Tavily MCP (AI-optimized search - 1000 arama/ay ücretsiz)
  - GitHub MCP (code & repo search - ücretsiz)
- 🧠 **Gemini 2.5 Flash** - Google'ın en hızlı modeli
- 💾 **Supabase Memory** - Conversation history persistence
- 📊 **LangSmith Tracing** - Agent akışlarını izleme
- 🎨 **Modern Streamlit UI** - Kullanıcı dostu arayüz
- 🇹🇷 **Türkçe Raporlar** - Kaynaklarla desteklenmiş detaylı analiz
- ⚡ **Paralel Tool Execution** - Birden fazla arama aynı anda

---

## 🚀 Hızlı Başlangıç

### 1. Kurulum

```bash
# Repo'yu klonla
git clone <repo-url>
cd multi_agent_search

# Virtual environment oluştur
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Bağımlılıkları yükle
pip install -r requirements.txt
```

### 2. API Keylerini Ayarla

`.env` dosyası oluştur:

```bash
# Zorunlu
GOOGLE_API_KEY=your_gemini_api_key
FIRECRAWL_API_KEY=your_firecrawl_api_key

# Opsiyonel (Daha fazla arama kaynağı için)
TAVILY_API_KEY=your_tavily_api_key
GITHUB_TOKEN=ghp_your_github_token
LANGSMITH_API_KEY=your_langsmith_api_key

# Supabase (Memory için)
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your_supabase_anon_key

# Model seçimi
DEFAULT_MODEL=google_genai:gemini-2.5-flash
```

### 3. Supabase Tablosunu Oluştur (Opsiyonel)

Eğer conversation memory istiyorsan:

1. [Supabase](https://supabase.com) hesabı aç
2. Yeni proje oluştur
3. SQL Editor'da `supabase_migration.sql` dosyasını çalıştır

### 4. Çalıştır

**Streamlit UI:**
```bash
streamlit run src/app.py
```

**CLI (Terminal):**
```bash
python -m src.main "Python ile veri manipülasyonu nedir?"
```

**İnteraktif Mod:**
```bash
python -m src.main --interactive
```

---

## 📁 Proje Yapısı

```
multi_agent_search/
├── src/
│   ├── agents/
│   │   └── main_agent.py      # DeepAgent orchestration
│   ├── config/
│   │   └── settings.py        # Ayarlar ve API keys
│   ├── memory/
│   │   └── supabase_memory.py # Conversation persistence
│   ├── app.py                 # Streamlit UI
│   └── main.py                # CLI entry point
├── .env                       # API keys (gitignore'da)
├── .env.example               # Örnek config
├── requirements.txt           # Python dependencies
├── supabase_migration.sql     # Database schema
└── README.md
```

---

## 🔧 Kullanım

### Streamlit UI

1. `streamlit run src/app.py`
2. Tarayıcıda `http://localhost:8501` aç
3. Soru sor ve bekle!

**Özellikler:**
- 💬 Chat interface
- 💾 Otomatik conversation history (Supabase aktifse)
- 🔄 Real-time status updates
- 📋 Kaynak atıfları

### CLI

**Tek Soru:**
```bash
python -m src.main "Python pandas nedir?"
```

**İnteraktif Mod:**
```bash
python -m src.main --interactive
```

---

## 🛠️ Yapılandırma

### Model Seçimi

`.env` dosyasında:

```bash
# Gemini (Önerilen)
DEFAULT_MODEL=google_genai:gemini-2.5-flash

# Ollama (Local)
DEFAULT_MODEL=ollama:llama3.1:8b
```

### MCP Serverlar

`main_agent.py` içinde otomatik olarak şunlar yüklenir:
- **Firecrawl** - Her zaman aktif (web scraping)
- **Tavily** - Eğer `TAVILY_API_KEY` varsa (AI search - 1000/ay ücretsiz)
- **GitHub** - Eğer `GITHUB_TOKEN` varsa (code search - ücretsiz)

### Memory (Supabase)

Supabase credentials yoksa memory devre dışı kalır, uygulama normal çalışır.

---

## 📊 LangSmith Tracing

Agent akışlarını izlemek için:

1. [LangSmith](https://smith.langchain.com) hesabı aç
2. API key al
3. `.env`'ye ekle:
```bash
LANGSMITH_API_KEY=your_key
```

4. https://smith.langchain.com adresinde trace'leri gör

---

## 🧪 Örnek Sorular

**Genel Araştırma:**
- "Python ile veri manipülasyonu nasıl yapılır?"
- "LangChain ve LangGraph arasındaki farklar nelerdir?"
- "2024'te en popüler açık kaynak LLM'ler hangileri?"

**Kod/GitHub Araması:**
- "Python pandas için en iyi GitHub projeleri"
- "LangChain ile agent nasıl yapılır? GitHub örnekleri"
- "FastAPI authentication örnekleri"

**Database:**
- "Supabase ile PostgreSQL nasıl kullanılır?"

---

## 🤝 Katkıda Bulunma

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Commit yapın (`git commit -m 'Add amazing feature'`)
4. Push yapın (`git push origin feature/amazing-feature`)
5. Pull Request açın

---

## 📝 Lisans

MIT License

---

## 🙏 Teşekkürler

- [DeepAgents](https://github.com/langchain-ai/deepagents) - Agent framework
- [LangChain](https://langchain.com) - LLM orchestration
- [Firecrawl](https://firecrawl.dev) - Web scraping
- [Tavily](https://tavily.com) - AI search
- [GitHub](https://github.com) - Code search & repositories
- [Supabase](https://supabase.com) - Database
- [Streamlit](https://streamlit.io) - UI framework

---

## 📞 İletişim

Sorularınız için issue açın veya PR gönderin!

**Made with ❤️ using DeepAgents**
