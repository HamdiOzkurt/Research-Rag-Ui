# Proje Mimarisi ve Akış Diyagramları

Bu dosyada, projenizde kullanılan iki ana mimari (**Deep Research Agent** ve **RAG**) görselleştirilmiş ve açıklanmıştır. Bu şemaları animasyonlu akışlarınızı oluştururken referans olarak kullanabilirsiniz.

## 1. Deep Research (Derin Araştırma) Mimarisi

Bu yapı, "Supervisor-Worker" (Yönetici-Çalışan) modeline dayalı hiyerarşik bir **LangGraph** yapısıdır.

### Akış Senaryosu:
1.  **Giriş:** Kullanıcı soruyu sorar.
2.  **Netleştirme (Clarify):** Agent soruyu analiz eder, gerekiyorsa kullanıcıdan detay ister.
3.  **Planlama (Brief):** Soru, yapılandırılmış bir araştırma planına (Brief) dönüştürülür.
4.  **Yönetim (Supervisor):** Yönetici ajan, planı inceler ve alt görevler oluşturur.
5.  **Dağıtım (Worker):** "Researcher" (Araştırmacı) ajanları internete (Tavily) gidip veri toplar.
6.  **Sentez (Report):** Toplanan tüm veriler birleştirilip nihai rapor yazılır.

```mermaid
graph TD
    %% Ana Graph Stilleri
    classDef mainNode fill:#2d3748,stroke:#4fd1c5,stroke-width:2px,color:#fff;
    classDef subNode fill:#4a5568,stroke:#cbd5e0,stroke-width:1px,color:#fff;
    classDef startEnd fill:#fc8181,stroke:#c53030,stroke-width:2px,color:#fff;

    %% Giriş ve Bitiş
    START((Başlangıç)) --> CLARIFY[Soruyu Netleştir\n(Clarify with User)]
    CLARIFY -- Soru Belirsizse --> ASK_USER[Kullanıcıya Soru Sor]
    ASK_USER --> CLARIFY
    CLARIFY -- Soru Netse --> BRIEF[Araştırma Planı Özeti\n(Write Research Brief)]
    
    %% Yönetici Katmanı
    BRIEF --> SUPERVISOR_SUBGRAPH{{"Yönetici Alt Sistemi\n(Supervisor Subgraph)"}}
    
    subgraph "Supervisor Subgraph (Yönetim Katmanı)"
        SUPERVISOR[Yönetici Ajan\n(Supervisor)]
        TOOLS[Araçlar\n(Tools & Strategy)]
        
        SUPERVISOR --> |Düşünür & Karar Verir| TOOLS
        TOOLS --> |Sonuçları İletir| SUPERVISOR
    end
    
    SUPERVISOR_SUBGRAPH --> |Araştırma Görevi| RESEARCHER_SUBGRAPH{{"Araştırmacı Alt Sistemi\n(Researcher Subgraph)"}}
    
    %% Araştırmacı Katmanı
    subgraph "Researcher Subgraph (Çalışan Katmanı)"
        RESEARCHER[Araştırmacı Ajan\n(Researcher)]
        EXECUTE[Araç Çalıştır\n(Exe Tool)]
        COMPRESS[Veriyi Özetle\n(Compress)]
        WEB[(Tavily Web Search)]
        
        RESEARCHER --> |Araştırma Yapar| EXECUTE
        EXECUTE --> |Sorgu| WEB
        WEB --> |Ham Veri| EXECUTE
        EXECUTE --> |Sonuç| RESEARCHER
        RESEARCHER --> |Yeterli Veri?| COMPRESS
    end
    
    RESEARCHER_SUBGRAPH --> |Özetlenmiş Bilgi| SUPERVISOR_SUBGRAPH
    
    %% Raporlama
    SUPERVISOR_SUBGRAPH -- Araştırma Tamam --> REPORT[Nihai Rapor Oluştur\n(Final Report Generation)]
    REPORT --> END((Bitiş))

    class START,END startEnd;
    class SUPERVISOR,RESEARCHER,BRIEF,CLARIFY,REPORT mainNode;
    class TOOLS,EXECUTE,COMPRESS subNode;
```

---

## 2. RAG (Retrieval-Augmented Generation) Mimarisi

Bu yapı, belgelerinizden (PDF, DOCX, vb.) bilgi çekip cevap üreten sistemdir. **Hybrid Search** (Vektör + Anahtar Kelime) ve **Re-ranking** (Yeniden Sıralama) mekanizmalarını içerir.

### Akış Senaryosu:
1.  **Yükleme (Ingestion):** PDF/DOCX dosyaları yüklenir, görseller ve metinler ayrıştırılır.
2.  **Parçalama (Chunking):** Metinler akıllıca (başlık ve görsel bazlı) parçalara bölünür.
3.  **Vektörleme:** `OllamaEmbeddings` kullanılarak metinler sayısal vektörlere çevrilir ve `ChromaDB`ye kaydedilir.
4.  **Sorgulama (Retrieval):** Kullanıcı sorusu vektöre çevrilir ve veritabanında en yakın parçalar bulunur.
5.  **Sıralama (Reranking):** Bulunan parçalar `Cross-Encoder` ile en alakalıdan en aza doğru tekrar sıralanır (Hassas Ayar).
6.  **Cevaplama (Generation):** En iyi parçalar LLM'e verilir ve cevap üretilir.

```mermaid
graph LR
    %% Stiller
    classDef process fill:#3182ce,stroke:#2b6cb0,stroke-width:2px,color:#fff;
    classDef db fill:#d69e2e,stroke:#b7791f,stroke-width:2px,color:#fff;
    classDef model fill:#805ad5,stroke:#6b46c1,stroke-width:2px,color:#fff;

    %% Sol Taraf: Doküman İşleme (Offline)
    subgraph "Doküman İşleme Hattı (Ingestion)"
        DOC[Doküman\n(PDF/DOCX)] --> EXTRACT[İçerik Ayıklama\n(PyMuPDF / python-docx)]
        EXTRACT --> CHUNK[Akıllı Parçalama\n(Heading & Image Splitter)]
        CHUNK --> EMBED_MODEL[["Embedding Modeli\n(Ollama)"]]
        EMBED_MODEL --> VECTOR_DB[("Vektör Veritabanı\n(ChromaDB)")]
    end

    %% Sağ Taraf: Sorgu ve Cevap (Online)
    subgraph "Sorgu ve Cevap Hattı (RAG)"
        USER((Kullanıcı)) --> QUERY[Soru]
        QUERY --> EMBED_QUERY[["Sorgu Embedding"]]
        EMBED_QUERY --> SEARCH[Benzerlik Araması\n(Similarity Search)]
        
        VECTOR_DB -.-> SEARCH
        
        SEARCH --> RESULTS[Ham Sonuçlar]
        RESULTS --> RERANK[["Cross-Encoder\n(Yeniden Sıralama)"]]
        
        RERANK --> TOP_K[En İyi K Parça]
        TOP_K --> CONTEXT[Bağlam (Context)]
        
        CONTEXT --> LLM[["LLM Modeli\n(Gemini/Ollama)"]]
        QUERY --> LLM
        
        LLM --> ANSWER[Cevap]
    end

    class DOC,EXTRACT,CHUNK,SEARCH,RERANK,CONTEXT process;
    class VECTOR_DB,RESULTS,TOP_K db;
    class EMBED_MODEL,EMBED_QUERY,LLM model;
```
