📋 Word (DOCX) Desteği - Yapılacaklar Listesi
🎯 Hedef
Word dosyalarının PDF ile AYNI kalitede işlenmesi:

Başlıklar (H1-H6) korunmalı
Görseller extract edilmeli ve Markdown'a dönmeli
Tablolar düzgün parse edilmeli
Vision model görselleri analiz edebilmeli


✅ Adım 1: DOCX → Markdown Dönüşümünü İyileştir (KRİTİK)
1.1. Mevcut load_docx Fonksiyonunu Güçlendir
Dosya: src/agents/rag_agent.py → load_docx() fonksiyonu
Değişiklik:
python@tool
def load_docx(file_path: str) -> str:
    """Load DOCX with images - PDF kalitesinde Markdown"""
    from docx import Document as DocxDocument
    from docx.oxml.table import CT_Tbl
    from docx.oxml.text.paragraph import CT_P
    from docx.table import Table
    from docx.text.paragraph import Paragraph
    
    doc = DocxDocument(file_path)
    images_folder = Path(file_path).parent / f"{Path(file_path).stem}_images"
    images_folder.mkdir(exist_ok=True)
    
    markdown_parts = []
    image_counter = 0
    image_map = {}  # rel_id -> saved_path
    
    # 1. Extract ALL images from document
    for rel_id, rel in doc.part.rels.items():
        if "image" in rel.target_ref:
            image_data = rel.target_part.blob
            ext = rel.target_part.content_type.split('/')[-1]
            
            image_filename = f"{Path(file_path).stem}-{image_counter}.{ext}"
            image_path = images_folder / image_filename
            image_path.write_bytes(image_data)
            
            # Store RELATIVE path (like PDF does)
            relative_path = f"{images_folder.name}/{image_filename}"
            image_map[rel_id] = relative_path
            image_counter += 1
    
    # 2. Process paragraphs + tables in DOCUMENT ORDER
    for element in doc.element.body:
        if isinstance(element, CT_P):
            para = Paragraph(element, doc)
            text = para.text.strip()
            style_name = para.style.name if para.style else ""
            
            # Convert headings to Markdown (same as PDF)
            if "Heading 1" in style_name:
                markdown_parts.append(f"# {text}")
            elif "Heading 2" in style_name:
                markdown_parts.append(f"## {text}")
            elif "Heading 3" in style_name:
                markdown_parts.append(f"### {text}")
            elif "Heading 4" in style_name:
                markdown_parts.append(f"#### {text}")
            elif "Heading 5" in style_name:
                markdown_parts.append(f"##### {text}")
            elif "Heading 6" in style_name:
                markdown_parts.append(f"###### {text}")
            elif text:
                markdown_parts.append(text)
            
            # Check for inline images (CRITICAL!)
            for run in para.runs:
                for drawing in run.element.xpath('.//w:drawing'):
                    blip = drawing.xpath('.//a:blip/@r:embed', 
                        namespaces={'a': '...', 'r': '...'})[0]
                    if blip in image_map:
                        markdown_parts.append(f"![Image]({image_map[blip]})")
        
        elif isinstance(element, CT_Tbl):
            # Convert table to Markdown
            table = Table(element, doc)
            markdown_parts.append("\n")
            for i, row in enumerate(table.rows):
                cells = [cell.text.strip() for cell in row.cells]
                markdown_parts.append("| " + " | ".join(cells) + " |")
                if i == 0:
                    markdown_parts.append("| " + " | ".join(["---"] * len(cells)) + " |")
            markdown_parts.append("\n")
    
    md_text = "\n\n".join(markdown_parts)
    
    # 3. Save debug file (same as PDF)
    debug_path = Path(file_path).parent / f"{Path(file_path).stem}_docx_debug.md"
    debug_path.write_text(md_text, encoding="utf-8")
    
    logger.info(f"[DOCX] ✅ Converted {len(md_text)} chars with {image_counter} images")
    return md_text
Neden Gerekli:

✅ PDF ile aynı Markdown formatı üretir
✅ Görseller extract edilir ve relative path ile referans edilir
✅ H5/H6 başlıklar korunur (Agentic Chunker için kritik)


✅ Adım 2: Görsel Yollarını Test Et
2.1. Word Dosyasını Yükle ve Debug Dosyasını Kontrol Et
Komut:
bash# 1. Word dosyasını yükle (API veya frontend üzerinden)
# 2. uploads/ klasöründe şu dosyayı kontrol et:
uploads/DOSYA_ADI_docx_debug.md
Kontrol Edilecekler:
markdown# Başlık 1
## Başlık 2
### Alt Başlık

Normal paragraf metni.

##### Algoritma Başlığı

Algoritma açıklaması...

![Image](DOSYA_ADI_images/DOSYA_ADI-0.png)

#### Tablo Başlığı

| Sütun 1 | Sütun 2 |
| --- | --- |
| Değer 1 | Değer 2 |
Beklenen:

✅ Başlıklar # ile başlıyor
✅ Görseller ![Image](uploads/...) formatında
✅ Tablolar Markdown table formatında


✅ Adım 3: Chunking Stratejisini Doğrula
3.1. Agentic Chunker'ın Word için Çalıştığını Test Et
Test Kodu:
python# Test et:
from src.agents.agentic_chunker import agentic_chunk_text

md_text = load_docx("uploads/test.docx")
chunks = agentic_chunk_text(md_text, "test.docx")

# Debug:
for chunk in chunks[:3]:
    print(f"Chunk başlığı: {chunk.metadata['title']}")
    print(f"Has images: {chunk.metadata['has_images']}")
    print(f"İlk 200 char: {chunk.page_content[:200]}")
    print("---")
Beklenen:

✅ H5/H6 başlıklar AYNI chunk'ta (örn: "Ridge Classifier" + açıklama + görsel)
✅ H4 başlıklar FARKLI chunk'larda (örn: "Naive Bayes" ≠ "Ridge Classifier")
✅ Görseller metin ile birlikte


✅ Adım 4: Vision Model ile Entegrasyonu Test Et
4.1. Word'den Gelen Görsellerin Analiz Edildiğini Doğrula
Test Senaryosu:
Kullanıcı Sorusu: "Word dökümanında Naive Bayes algoritmasının görselini açıkla"
Beklenen Loglar:
[RAG] 🖼️ Visual query detected: {'görsel', 'açıkla'}
[RAG] 🔍 Scanning ALL chunks for images
[RAG SCORER] Image: test-0.png | Score: 13 | Chunk: #1
[RAG] Trying vision model: llava
[VISION RESPONSE] Status: SUCCESS
[RAG] ✅ Context ready: 1 chunks + 1 images
Doğrulama:

✅ Vision model görseli analiz etmiş
✅ Cevap hem metin hem görsel analizini içeriyor

🎯 Özet: Ne Yapmalısın?
✅ MUTLAKA YAP (Word için)

☑️ Adım 1: load_docx fonksiyonunu iyileştir (yukarıdaki kodu uygula)
☑️ Adım 2: Word dosyası yükle ve _docx_debug.md kontrol et
☑️ Adım 3: Chunking'i test et - H5/H6 aynı chunk'ta mı?
☑️ Adım 4: Vision model test et - Görseller analiz ediliyor mu?

📊 Beklenen Sonuç
ÖNCESİ (PDF):
✅ PDF: PyMuPDF4LLM → Markdown → H5/H6 korunuyor → Vision çalışıyor
❌ Word: python-docx → Zayıf Markdown → H5/H6 kaybolabilir
SONRASI (Adım 1-4 sonrası):
✅ PDF: PyMuPDF4LLM → Markdown → H5/H6 korunuyor → Vision çalışıyor
✅ Word: İyileştirilmiş docx → Markdown → H5/H6 korunuyor → Vision çalışıyor