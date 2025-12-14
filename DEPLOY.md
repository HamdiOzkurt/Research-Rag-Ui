# 🚀 Deploy Guide (Vercel + Railway/Render) — SaaS-ready

Bu doküman **Frontend’i Vercel’e**, **Backend’i Railway/Render’a** deploy etmek için.

## 1) Supabase — Conversation History Table

Supabase projenizde SQL editor’a gidip şu dosyadaki SQL’i çalıştırın:
- `supabase_conversations.sql`

> Backend’in Supabase’a yazabilmesi için genelde `SUPABASE_KEY` olarak **service_role** key kullanılır.

## 2) Backend Deploy (Railway/Render/Fly)

Bu repo backend için Docker ile hazır:
- `Dockerfile`
- `.dockerignore`

### Gerekli Environment Variables (Backend)
- **LLM**
  - `GOOGLE_API_KEYS` (virgülle ayrılmış çoklu key) **veya** `GOOGLE_API_KEY`
  - `DEFAULT_MODEL` (ör: `google_genai:gemini-2.0-flash-exp`)
  - (opsiyonel) `SECONDARY_MODEL`
- **Tools**
  - `FIRECRAWL_API_KEY`
  - (opsiyonel) `TAVILY_API_KEY`
- **Supabase**
  - `SUPABASE_URL`
  - `SUPABASE_KEY`
- **CORS**
  - `ALLOWED_ORIGINS` (örn: `https://your-vercel-app.vercel.app,http://localhost:3000`)

### Start Command
Docker CMD zaten hazır:
`python -m uvicorn src.simple_copilot_backend:app --host 0.0.0.0 --port 8000`

> Deploy sonrası backend URL’niz ör: `https://your-backend.up.railway.app`

## 3) Frontend Deploy (Vercel)

Vercel’de yeni proje oluştur:
- **Root Directory**: `multi_agent_search/copilotkit-ui`

### Gerekli Environment Variables (Frontend / Vercel)
- **Clerk**
  - `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY`
  - `CLERK_SECRET_KEY`
  - `NEXT_PUBLIC_CLERK_SIGN_IN_URL=/sign-in`
  - `NEXT_PUBLIC_CLERK_SIGN_UP_URL=/sign-up`
  - `NEXT_PUBLIC_CLERK_AFTER_SIGN_IN_URL=/`
  - `NEXT_PUBLIC_CLERK_AFTER_SIGN_UP_URL=/`
- **Backend URL**
  - `NEXT_PUBLIC_BACKEND_URL=https://your-backend-domain`

> Frontend bu env ile backend’e gider. Local’de default `http://localhost:8000`.

## 4) Local → Production farkı (Önemli)

Şu anda history **frontend’in gönderdiği `userId`** ile tutuluyor (Clerk `user.id`).  
Production’da güvenlik için backend’e **Clerk JWT verification** eklemeliyiz (sonraki adım).

## 5) Hızlı Kontrol Listesi

- [ ] Supabase’da `conversations` tablosu oluşturuldu
- [ ] Backend env’ler girildi (özellikle `GOOGLE_API_KEYS`, `FIRECRAWL_API_KEY`)
- [ ] `ALLOWED_ORIGINS` içine Vercel domain’i eklendi
- [ ] Frontend env’ler girildi (`NEXT_PUBLIC_BACKEND_URL`, Clerk keys)
- [ ] Login sonrası UI açılıyor
- [ ] Chat atınca `/chat` response geliyor ve `/threads?user_id=...` listeliyor


