=== DÖKÜMAN GÖRSELLERİ (Vision Model Analysis) ===
🖼️ IMAGE BLOCK
Source: YAPAY ZEKA FİNAL- AHMET HAMDİ ÖZKURT.pdf
File: YAPAY-ZEKA-FİNAL--AHMET-HAMDİ-ÖZKURT.pdf-3-0.png
Markdown: ![Chart/Graph](http://127.0.0.1:8000/uploads/YAPAY ZEKA FİNAL- AHMET HAMDİ ÖZKURT_images/YAPAY-ZEKA-FİNAL--AHMET-HAMDİ-ÖZKURT.pdf-3-0.png)
Vision analysis...
[...end of context example]
INFO:httpx:HTTP Request: POST https://api.groq.com/openai/v1/chat/completions "HTTP/1.1 200 OK"
INFO:httpx:HTTP Request: POST http://localhost:11434/api/embed "HTTP/1.1 200 OK"
INFO:src.agents.rag_agent:[RAG ENSEMBLE] Top chunk scores (BM25+Semantic): [(np.float64(0.737), 'AI Methodology Summary')]
INFO:src.agents.rag_agent:[RAG QUALITY] Filtered to 1 chunks (threshold: 0.516)
INFO:src.agents.rag_agent:[RAG] 📕 Text-only query detected
INFO:src.agents.rag_agent:[RAG] 🔍 Scanning ALL chunks for images (smart multi-chunk analysis)
INFO:src.agents.rag_agent:[RAG SCORER] Image: YAPAY-ZEKA-FİNAL--AHMET-HAMDİ-ÖZKURT.pdf-3-0.png | Score: 3 | Chunk: #0
INFO:src.agents.rag_agent:[RAG] ✅ Selected 1 best images from 1 candidates
INFO:src.agents.rag_agent:[RAG] 🖼️ Analyzing BEST image: YAPAY-ZEKA-FİNAL--AHMET-HAMDİ-ÖZKURT.pdf-3-0.png (Score: 3)
INFO:src.agents.rag_agent:[RAG] Trying vision model: moondream
INFO:httpx:HTTP Request: POST http://localhost:11434/api/chat "HTTP/1.1 404 Not Found"
INFO:src.agents.rag_agent:[RAG] Trying vision model: llava:latest
INFO:     127.0.0.1:61065 - "GET /health HTTP/1.1" 200 OK
INFO:     127.0.0.1:61065 - "GET /stats HTTP/1.1" 200 OK
INFO:httpx:HTTP Request: POST http://localhost:11434/api/chat "HTTP/1.1 200 OK"
INFO:src.agents.rag_agent:[RAG] ✅ Vision SUCCESS with llava:latest (1336 chars)
INFO:src.agents.rag_agent:[RAG] ✅ Vision SUCCESS
INFO:src.agents.rag_agent:[RAG FILTER] Keeping 1 analyzed images despite text query
INFO:src.agents.rag_agent:[RAG] Retrieved 1 chunks + 1 image blocks
INFO:src.agents.rag_agent:[RAG DEBUG] FINAL CONTEXT sent to LLM (first 500 chars):
⚠️ BU DÖKÜMAN İÇERİĞİ - SADECE BUNU KULLAN ⚠️

=== DÖKÜMAN METNİ (Retrieved from Documents) ===
📄 Source: YAPAY ZEKA FİNAL- AHMET HAMDİ ÖZKURT.pdf
## **YÖNTEM**





=== DÖKÜMAN GÖRSELLERİ (Vision Model Analysis) ===
🖼️ IMAGE BLOCK
Source: YAPAY ZEKA FİNAL- AHMET HAMDİ ÖZKURT.pdf
File: YAPAY-ZEKA-FİNAL--AHMET-HAMDİ-ÖZKURT.pdf-3-0.png
Markdown: ![Chart/Graph](http://127.0.0.1:8000/uploads/YAPAY ZEKA FİNAL- AHMET HAMDİ ÖZKURT_images/YAPAY-ZEKA-FİNAL--AHMET-HAMDİ-ÖZKURT.pdf-3-0.png)
Vision analysis...
[...end of context example]
INFO:httpx:HTTP Request: POST https://api.groq.com/openai/v1/chat/completions "HTTP/1.1 200 OK"
INFO:src.simple_copilot_backend:[CACHE] Cached: yöntemden bahseder misin bana pdfdeki...
INFO:httpx:HTTP Request: POST https://gdumibpautfnztbprjjw.supabase.co/rest/v1/conversations "HTTP/2 201 Created"
INFO:     127.0.0.1:56865 - "GET /uploads/YAPAY%20ZEKA%20F%C4%B0NAL-%20AHMET%20HAMD%C4%B0%20%C3%96ZKURT_images/YAPAY-ZEKA-F%C4%B0NAL--AHMET-HAMD%C4%B0-%C3%96ZKURT.pdf-3-0.png HTTP/1.1" 200 OK
INFO:     127.0.0.1:65233 - "GET /health HTTP/1.1" 200 OK
INFO:     127.0.0.1:65233 - "GET /stats HTTP/1.1" 200 OK
INFO:     127.0.0.1:52348 - "GET /health HTTP/1.1" 200 OK
INFO:     127.0.0.1:52348 - "GET /stats HTTP/1.1" 200 OK
INFO:     127.0.0.1:64861 - "GET /health HTTP/1.1" 200 OK
INFO:     127.0.0.1:64861 - "GET /stats HTTP/1.1" 200 OK
INFO:     127.0.0.1:54195 - "GET /health HTTP/1.1" 200 OK
INFO:     127.0.0.1:54195 - "GET /stats HTTP/1.1" 200 OK
INFO:     127.0.0.1:54885 - "GET /health HTTP/1.1" 200 OK
INFO:     127.0.0.1:54885 - "GET /stats HTTP/1.1" 200 OK
INFO:     127.0.0.1:60942 - "GET /health HTTP/1.1" 200 OK
INFO:     127.0.0.1:60942 - "GET /stats HTTP/1.1" 200 OK
INFO:     127.0.0.1:52877 - "GET /health HTTP/1.1" 200 OK
INFO:     127.0.0.1:52877 - "GET /stats HTTP/1.1" 200 OK
INFO:     127.0.0.1:62916 - "GET /health HTTP/1.1" 200 OK
INFO:     127.0.0.1:62916 - "GET /stats HTTP/1.1" 200 OK
INFO:     127.0.0.1:62916 - "GET /health HTTP/1.1" 200 OK
INFO:     127.0.0.1:62916 - "GET /stats HTTP/1.1" 200 OK
INFO:     127.0.0.1:54413 - "GET /health HTTP/1.1" 200 OK
INFO:     127.0.0.1:54413 - "GET /stats HTTP/1.1" 200 OK
INFO:     127.0.0.1:52232 - "GET /health HTTP/1.1" 200 OK
INFO:     127.0.0.1:52232 - "GET /stats HTTP/1.1" 200 OK
INFO:     127.0.0.1:50738 - "GET /health HTTP/1.1" 200 OK
INFO:     127.0.0.1:50738 - "GET /stats HTTP/1.1" 200 OK
INFO:     127.0.0.1:62026 - "GET /health HTTP/1.1" 200 OK
INFO:     127.0.0.1:62026 - "GET /stats HTTP/1.1" 200 OK
INFO:     127.0.0.1:49770 - "GET /health HTTP/1.1" 200 OK
INFO:     127.0.0.1:49770 - "GET /stats HTTP/1.1" 200 OK
INFO:     127.0.0.1:54964 - "GET /health HTTP/1.1" 200 OK
INFO:     127.0.0.1:54964 - "GET /stats HTTP/1.1" 200 OK
INFO:     127.0.0.1:63897 - "GET /health HTTP/1.1" 200 OK
INFO:     127.0.0.1:63897 - "GET /stats HTTP/1.1" 200 OK
INFO:     127.0.0.1:52488 - "GET /health HTTP/1.1" 200 OK
INFO:     127.0.0.1:52488 - "GET /stats HTTP/1.1" 200 OK
INFO:     127.0.0.1:54530 - "GET /health HTTP/1.1" 200 OK
INFO:     127.0.0.1:54530 - "GET /stats HTTP/1.1" 200 OK
INFO:     127.0.0.1:64412 - "GET /uploads/YAPAY%20ZEKA%20F%C4%B0NAL-%20AHMET%20HAMD%C4%B0%20%C3%96ZKURT_images/YAPAY-ZEKA-F%C4%B0NAL--AHMET-HAMD%C4%B0-%C3%96ZKURT.pdf-3-0.png HTTP/1.1" 200 OK
INFO:     127.0.0.1:58482 - "GET /health HTTP/1.1" 200 OK
INFO:     127.0.0.1:58482 - "GET /stats HTTP/1.1" 200 OK
INFO:     127.0.0.1:49977 - "GET /health HTTP/1.1" 200 OK
INFO:     127.0.0.1:49977 - "GET /stats HTTP/1.1" 200 OK
INFO:     127.0.0.1:51731 - "GET /health HTTP/1.1" 200 OK
INFO:     127.0.0.1:51731 - "GET /stats HTTP/1.1" 200 OK
INFO:     127.0.0.1:63225 - "GET /health HTTP/1.1" 200 OK
INFO:     127.0.0.1:63225 - "GET /stats HTTP/1.1" 200 OK
INFO:     127.0.0.1:51185 - "GET /health HTTP/1.1" 200 OK
INFO:     127.0.0.1:51185 - "GET /stats HTTP/1.1" 200 OK
INFO:     127.0.0.1:56959 - "GET /health HTTP/1.1" 200 OK
INFO:     127.0.0.1:56959 - "GET /stats HTTP/1.1" 200 OK
INFO:     127.0.0.1:52547 - "GET /health HTTP/1.1" 200 OK
INFO:     127.0.0.1:52547 - "GET /stats HTTP/1.1" 200 OK
INFO:     127.0.0.1:61204 - "GET /health HTTP/1.1" 200 OK
INFO:     127.0.0.1:61204 - "GET /stats HTTP/1.1" 200 OK
INFO:     127.0.0.1:59607 - "GET /health HTTP/1.1" 200 OK
INFO:     127.0.0.1:59607 - "GET /stats HTTP/1.1" 200 OK
INFO:     127.0.0.1:59607 - "GET /health HTTP/1.1" 200 OK
INFO:     127.0.0.1:59607 - "GET /stats HTTP/1.1" 200 OK
INFO:     127.0.0.1:52327 - "GET /health HTTP/1.1" 200 OK
INFO:     127.0.0.1:52327 - "GET /stats HTTP/1.1" 200 OK
INFO:     127.0.0.1:54055 - "GET /health HTTP/1.1" 200 OK
INFO:     127.0.0.1:54055 - "GET /stats HTTP/1.1" 200 OK
INFO:     127.0.0.1:65170 - "GET /health HTTP/1.1" 200 OK
INFO:     127.0.0.1:65170 - "GET /stats HTTP/1.1" 200 OK
INFO:     127.0.0.1:54610 - "GET /health HTTP/1.1" 200 OK
INFO:     127.0.0.1:54610 - "GET /stats HTTP/1.1" 200 OK
INFO:     127.0.0.1:55479 - "GET /health HTTP/1.1" 200 OK
INFO:     127.0.0.1:55479 - "GET /stats HTTP/1.1" 200 OK
INFO:     127.0.0.1:60746 - "GET /health HTTP/1.1" 200 OK
INFO:     127.0.0.1:60746 - "GET /stats HTTP/1.1" 200 OK
INFO:     127.0.0.1:53246 - "GET /health HTTP/1.1" 200 OK
INFO:     127.0.0.1:53246 - "GET /stats HTTP/1.1" 200 OK
INFO:     127.0.0.1:51274 - "GET /health HTTP/1.1" 200 OK
INFO:     127.0.0.1:51274 - "GET /stats HTTP/1.1" 200 OK
INFO:     127.0.0.1:62413 - "GET /health HTTP/1.1" 200 OK
INFO:     127.0.0.1:62413 - "GET /stats HTTP/1.1" 200 OK
INFO:     127.0.0.1:49845 - "GET /health HTTP/1.1" 200 OK
INFO:     127.0.0.1:49845 - "GET /stats HTTP/1.1" 200 OK
INFO:     127.0.0.1:60920 - "GET /health HTTP/1.1" 200 OK
INFO:     127.0.0.1:60920 - "GET /stats HTTP/1.1" 200 OK
INFO:     127.0.0.1:64738 - "GET /health HTTP/1.1" 200 OK
INFO:     127.0.0.1:64738 - "GET /stats HTTP/1.1" 200 OK
