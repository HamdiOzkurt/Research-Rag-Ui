# Human-in-the-Loop RAG Entegrasyon Planı

Bu belge, mevcut RAG sistemini, kullanıcının bulunan döküman parçalarını (chunks) inceleyip onayladığı ve sadece onaylanan kaynaklarla cevap üretildiği "Human-in-the-Loop" (İnsan Döngüde) mimarisine dönüştürmek için gereken adımları içerir.

**Hedef:** Hallucination'ı önlemek ve kullanıcıya cevap oluşturulurken kullanılan kaynaklar üzerinde tam kontrol vermek.

---

## 🏗️ Mimari Bakış

Mevcut akış:
`Soru -> Ara -> Chunk Bul -> Hemen Cevap Üret -> Kullanıcıya Dön`

Yeni akış:
`Soru -> Ara -> Chunk Bul -> ⏸️ DUR (Kullanıcıya Göster) -> 👤 Kullanıcı Seçer/Onaylar -> ✅ Devam Et -> Sadece Seçilenlerle Cevap Üret -> Kullanıcıya Dön`

---

## 📅 Uygulama Adımları

### FAZ 1: Backend State Yönetimi (src/models)

State yapımızı, onay sürecini destekleyecek şekilde güncellemeliyiz.

1.  **`src/models/rag_models.py` Güncellemesi:**
    *   `RAGState` sınıfı oluştur (veya güncelle).
    *   Şu alanları ekle:
        ```python
        class RAGState(BaseModel):
            ...
            retrieved_chunks: List[RetrievedChunk] = []  # Bulunan ham chunklar
            approved_chunk_ids: List[str] = []           # Kullanıcının seçtikleri
            awaiting_approval: bool = False              # UI'ın onay beklemesi için flag
            is_synthesizing: bool = False                # Cevap üretiliyor mu?
            current_query: Optional[str] = None
        ```

### FAZ 2: Agent Yetenekleri (src/agents/rag_agent.py)

Agent'ın araçlarını (tools) ikiye bölmeliyiz: Arama ve Sentezleme.

2.  **`search_knowledge_base` Tool Güncellemesi:**
    *   Artık direkt cevap d *dönmemeli*.
    *   Chunkları bulup `state.retrieved_chunks` içine kaydetmeli.
    *   `state.awaiting_approval = True` yapmalı.
    *   LLM'e "Kullanıcıya kaynakları incelemesini söyle" mesajı dönmeli.

3.  **`synthesize_with_sources` Tool Eklemesi:**
    *   Yeni bir tool.
    *   Sadece `state.approved_chunk_ids` boş değilse çalışmalı.
    *   Seçilen chunkları prompt'a ekleyip nihai cevabı üretmeli.

### FAZ 3: Frontend Arayüzü (Premium Source Inspector)

Videodaki yapıyı **ücretsiz** ve **manuel** olarak kuracağız. AGUI yerine Custom React Components kullanacağız.

4.  **Yeni Bileşen: `SourceInspectorPanel.tsx` (Sağ Panel)**
    *   **Tasarım:** Glassmorphism etkili, sağdan kayarak açılan (slide-over) modern bir panel.
    *   **Özellikler:**
        *   📊 **Confidence Score Bar:** Her chunk'ın ne kadar alakalı olduğunu renkli bar ile göster.
        *   👁️ **Quick Preview:** Karta tıklayınca içeriği genişlet.
        *   ✨ **Smart Selection:** Yüksek puanlıları otomatik öner.
    *   **Tech Stack:** Tailwind CSS + Framer Motion (varsa) veya CSS Transitions.

5.  **Entegrasyon:**
    *   `DashboardPage.tsx` içine eklenecek.
    *   `useCopilotReadable` hook'u ile backend'deki `RAGState` dinlenecek.
    *   `awaiting_approval` True olduğunda panel otomatik açılacak.
    *   "Onayla" butonu `useCopilotAction` ile backend'e `approve_sources` çağrısı yapacak.

### FAZ 4: Backend-Frontend Bağlantısı (src/simple_copilot_backend.py)

6.  **Endpoint Güncellemeleri:**
    *   CopilotKit state senkronizasyonunun doğru çalıştığından emin olunması.
    *   Onay işlemi için bir `action` tanımlanması (Örn: `approve_sources`).

---

## 🛠️ Teknik Detaylar & Kurallar

*   **Pydantic AI:** State validation için aktif olarak kullanılacak.
*   **Approval Flow:** Kullanıcı hiçbir şey seçmezse, varsayılan davranış (örneğin en iyi 3 chunk) veya uyarı mesajı belirlenmeli.
*   **UI/UX:** Onay ekranı kullanıcıyı yormamalı. Chunklar net, kısa özetlerle gösterilmeli.

## ✅ Başarı Kriterleri

1.  Kullanıcı soru sorduğunda, kaynaklar sol panelde listeleniyor mu?
2.  Chat botu, kullanıcı onay verene kadar bekliyor mu?
3.  Kullanıcı chunkları seçip onayladığında, bot **sadece** o chunkları kullanarak cevap veriyor mu?
4.  Sistem hallucination yapmadan dökümana sadık kalıyor mu?
