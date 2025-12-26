# 🔍 AI Research Assistant & RAG System

Modern, çok ajanlı (multi-agent) araştırma asistanı ve RAG (Retrieval-Augmented Generation) sistemi. **DeepAgents**, **LangGraph** ve **Next.js** teknolojileri ile güçlendirilmiştir.

---

## ⚡ Hızlı Başlangıç

Projeyi çalıştırmak için backend ve frontend'i ayrı ayrı başlatmanız gerekmektedir.

### 1. Backend'i Başlat (Python)
Ana dizinde (`multi_agent_search/`):

```powershell
# Sanal ortamı aktif et (varsa)
.\venv\Scripts\activate

# Backend sunucusunu başlat
python -m uvicorn src.simple_copilot_backend:app --reload --port 8000
```

### 2. Frontend'i Başlat (Next.js)
Yeni bir terminal açın ve:

```powershell
cd copilotkit-ui
npm run dev
```

Tarayıcıda aç: [http://localhost:3000](http://localhost:3000)

---

## 🏗️ Mimari & Özellikler

Bu proje iki ana yapay zeka mimarisini barındırır. Detaylı şemalar için **[ARCHITECTURE.md](ARCHITECTURE.md)** dosyasına bakınız.

### 1. Deep Research (Derin Araştırma)
Karmaşık soruları analiz eden, planlayan ve internetten güncel veri toplayarak kapsamlı raporlar oluşturan ajan yapısı.
- **Supervisor-Worker Modeli:** Görevleri yöneten ve dağıtan hiyerarşik yapı.
- **Hybrid LLM:** Groq (Hızlı) ve Ollama (Lokal/Sınırsız) modellerini hibrit kullanabilme yeteneği.

### 2. RAG (Dokümanla Sohbet)
PDF, DOCX vb. belgelerinizle konuşmanızı sağlayan sistem.
- **Akıllı Parçalama (Chunking):** Metinleri ve görselleri anlamsal bütünlüğe göre böler.
- **Hybrid Search & Re-ranking:** En alakalı cevapları bulmak için gelişmiş vektör ve anahtar kelime araması.

---

## 🔑 API Key Ayarları

### Multi API Key (429 Hatası Çözümü!)

`.env` dosyasında birden fazla Gemini key tanımlayarak rate limit hatalarını aşabilirsiniz. Sistem otomatik olarak key değiştirir (rotation).

```env
# Çoklu key (virgülle ayrılmış) - ÖNERILEN!
GOOGLE_API_KEYS=AIzaSy-key1,AIzaSy-key2,AIzaSy-key3

# Firecrawl (Web Arama için zorunlu)
FIRECRAWL_API_KEY=fc-your-key

# Varsayılan Model
DEFAULT_MODEL=google_genai:gemini-2.0-flash-exp
```

### Ollama (Lokal/Sınırsız)

```bash
# Modeli indir
ollama pull llama3.2

# .env ayarı
DEFAULT_MODEL=ollama:llama3.2
```

---

## 📁 Güncel Klasör Yapısı

```
multi_agent_search/
├── src/
│   ├── simple_copilot_backend.py      # FastAPI backend girişi
│   ├── agents/
│   │   ├── deep_research/             # Derin Araştırma Ajanı (Modüler)
│   │   │   ├── configuration.py       # Ayarlar ve Promptlar
│   │   │   └── graph.py               # LangGraph akışı
│   │   ├── rag_agent.py               # RAG (Doküman) Ajanı
│   │   ├── agentic_chunker.py         # Akıllı Doküman Parçalayıcı
│   │   └── simple_agent.py            # Basit Chat Ajanı
│   ├── config/
│   │   └── settings.py
│   └── models.py
├── copilotkit-ui/                     # Next.js Frontend
│   └── app/
│       ├── components/                # UI Bileşenleri (Chat, Sidebar, Popup)
│       └── page.tsx
├── ARCHITECTURE.md                    # Mimari Şemalar ve Diyagramlar
└── requirements.txt
```

---

## 🎨 UI Modları

| Mod | Açıklama |
|-----|----------|
| 💬 **CopilotChat** | Tam ekran chat deneyimi |
| 📋 **CopilotSidebar** | Yanda açılan asistan paneli |
| 💭 **CopilotPopup** | Sağ alt köşede yüzen chat balonu |

---

## 🛡️ Performans ve Güvenlik

- **Rate Limit Koruması:** Dakikada belirli istek sayısı ile API güvenliği.
- **Otomatik Key Rotasyonu:** 429 hatalarında bir sonraki API anahtarına geçiş.
- **Response Caching:** Sık sorulan sorular için önbellekten hızlı yanıt.

### İstatistikleri Görüntüle
Cache ve rate limit durumunu görmek için:
`GET http://localhost:8000/stats`

---

## 🚀 Geliştirme Notları

Dokümantasyon veya mimari değişiklikleri için `ARCHITECTURE.md` dosyasını güncellemeyi unutmayın.

**Made with ❤️ using DeepAgents, LangGraph & Next.js**
