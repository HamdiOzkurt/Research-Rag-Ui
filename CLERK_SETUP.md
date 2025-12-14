# 🔐 Clerk Authentication Setup

## 1️⃣ Clerk Hesabı Oluşturun

1. https://clerk.com adresine gidin
2. Sign up yapın (ücretsiz)
3. Yeni bir uygulama oluşturun: "AI Research Assistant"

## 2️⃣ API Keys Alın

Dashboard'da:
- **Publishable Key** (pk_test_...)
- **Secret Key** (sk_test_...)

## 3️⃣ .env.local Oluşturun

`multi_agent_search/copilotkit-ui/.env.local` dosyası oluşturun (örnek değişkenler `multi_agent_search/copilotkit-ui/env.example` dosyasında var):

```env
# Clerk Keys
NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=pk_test_your-key-here
CLERK_SECRET_KEY=sk_test_your-key-here

# Backend URL (optional)
NEXT_PUBLIC_BACKEND_URL=http://localhost:8000

# Clerk URLs (optional)
NEXT_PUBLIC_CLERK_SIGN_IN_URL=/sign-in
NEXT_PUBLIC_CLERK_SIGN_UP_URL=/sign-up
NEXT_PUBLIC_CLERK_AFTER_SIGN_IN_URL=/
NEXT_PUBLIC_CLERK_AFTER_SIGN_UP_URL=/
```

## 4️⃣ Frontend'i Yeniden Başlatın

```powershell
# Terminal'i durdurun (Ctrl+C)
# Sonra tekrar başlatın
cd copilotkit-ui
npm run dev
```

## 5️⃣ Test Edin!

http://localhost:3000 açın:
- ✅ Login ekranını göreceksiniz
- ✅ Sign up yapın
- ✅ Dashboard'a erişin!

---

## 🎨 Eklenen Özellikler

### 1. Protected Routes
- Tüm sayfalar artık login gerektiriyor
- `/sign-in` ve `/sign-up` public

### 2. User Button
- Sağ üstte kullanıcı avatarı
- Profile, settings, sign out

### 3. Middleware
- Otomatik auth kontrolü
- Redirect to sign-in

---

## 🔧 Troubleshooting

### "CLERK_PUBLISHABLE_KEY is missing"
`.env.local` dosyasını kontrol edin.

### Redirect loop
Middleware'deki route'ları kontrol edin.

---

**Auth hazır! Test edin!** 🚀

