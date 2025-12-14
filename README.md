# 🔍 AI Research Assistant

Modern AI araştırma asistanı - DeepAgents + LangGraph + Next.js

## ⚡ Hızlı Başlangıç

```powershell
cd multi_agent_search
.\start.ps1
```

Tarayıcıda aç: http://localhost:3000

---

## 🔑 API Key Ayarları

### Multi API Key (429 Hatası Çözümü!)

`.env` dosyasına birden fazla Gemini key ekleyebilirsiniz:

```env
# Çoklu key (virgülle ayrılmış) - ÖNERILEN!
GOOGLE_API_KEYS=AIzaSy-key1,AIzaSy-key2,AIzaSy-key3

# Firecrawl (zorunlu)
FIRECRAWL_API_KEY=fc-your-key

# Model
DEFAULT_MODEL=google_genai:gemini-2.0-flash-exp
```

**Nasıl çalışır?**
1. İlk key rate limit'e takılırsa
2. Otomatik olarak ikinci key'e geçer
3. Tüm key'ler kullanıldıysa başa döner

### Ollama (Sınırsız, Ücretsiz)

```bash
# Kur
winget install Ollama.Ollama

# Model indir
ollama pull llama3.2

# .env'de değiştir
DEFAULT_MODEL=ollama:llama3.2
```

---

## 📁 Klasör Yapısı

```
multi_agent_search/
├── src/
│   ├── simple_copilot_backend.py  # FastAPI backend
│   ├── agents/
│   │   ├── simple_agent.py        # Hızlı mod
│   │   ├── main_agent.py          # Standart mod
│   │   └── multi_agent_system.py  # Derin araştırma
│   ├── config/
│   │   └── settings.py            # Multi API key desteği
│   └── models.py                  # LLM helpers
├── copilotkit-ui/                 # Next.js frontend
│   └── app/
│       ├── page.tsx               # Ana sayfa
│       └── components/
│           ├── ChatInterface.tsx      # Full screen chat
│           ├── SidebarInterface.tsx   # Sidebar chat
│           └── PopupInterface.tsx     # Popup chat
├── start.ps1                      # PowerShell starter
└── requirements.txt
```

---

## 🎨 UI Modları

| Mod | Açıklama |
|-----|----------|
| 💬 **CopilotChat** | Full screen chat |
| 📋 **CopilotSidebar** | Dashboard + Chat sidebar |
| 💭 **CopilotPopup** | Floating popup chat |

---

## 🛡️ 429 Rate Limit Koruması

### Özellikler
- ✅ **Multi API Key Rotation**: Birden fazla key arasında döner
- ✅ **Response Caching**: Aynı sorulara cache'den yanıt
- ✅ **Rate Limiting**: Dakikada 10 istek limiti
- ✅ **Auto Retry**: 429 hatası alınırsa otomatik key değiştirir

### Cache İstatistikleri
```
GET http://localhost:8000/stats
```

---

## 📊 API Endpoints

| Endpoint | Method | Açıklama |
|----------|--------|----------|
| `/` | GET | Health check |
| `/chat` | POST | Chat endpoint |
| `/health` | GET | System health |
| `/stats` | GET | Cache & rate limit stats |
| `/cache` | DELETE | Cache temizle |

### Örnek İstek
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Python pandas nedir?"}'
```

---

## 🚀 Geliştirme

### Backend
```bash
cd multi_agent_search
python -m uvicorn src.simple_copilot_backend:app --reload --port 8000
```

### Frontend
```bash
cd copilotkit-ui
npm run dev
```

---

## 📦 Gereksinimler

### Python
```
deepagents
langgraph
langchain
langchain-mcp-adapters
langchain-google-genai
langchain-ollama
fastapi
uvicorn
```

### Node.js
```
next
react
tailwindcss
```

---

## 🎯 Yol Haritası

- [x] Multi API Key Rotation
- [x] Response Caching
- [x] Rate Limiting
- [x] 3 UI Modu
- [ ] Auth (Clerk)
- [ ] Database (Supabase)
- [ ] Billing (Stripe)
- [ ] Deploy (Vercel + Railway)

---

**Made with ❤️ using DeepAgents, LangGraph & Next.js**
