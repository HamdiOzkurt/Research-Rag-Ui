🔧 Chunk İyileştirmeleri - Öncelikli Aksiyonlar
🔴 KRİTİK: Görsel Extraction Hatası (EN ÖNCELİKLİ)
Sorun
Word dökümanından görseller extract edilmemiş:
❌ Image not found: /Feb6-Earthquake-Sentiment-Analysis-Research-Paper_images/...
Kök Neden
load_docx() fonksiyonunda image extraction mantığı çalışmıyor olabilir:

Word'de görseller media/ klasöründe (örn: media/image1.png)
Ama kodda /Feb6-Earthquake-Sentiment-Analysis-Research-Paper_images/ klasörü aranıyor
Path uyumsuzluğu → görseller bulunamıyor

Çözüm: load_docx() Debug & Fix
Dosya: src/agents/rag_agent.py → load_docx()
1. Önce Debug: Görsel extraction çalışıyor mu?
python@tool
def load_docx(file_path: str) -> str:
    from docx import Document as DocxDocument
    from pathlib import Path
    
    logger.info(f"[DOCX DEBUG] Processing {Path(file_path).name}...")
    
    doc = DocxDocument(file_path)
    images_folder = Path(file_path).parent / f"{Path(file_path).stem}_images"
    images_folder.mkdir(exist_ok=True)
    
    # ✅ DEBUG: Kaç görsel var?
    image_count = 0
    for rel_id, rel in doc.part.rels.items():
        if "image" in rel.target_ref:
            image_count += 1
            logger.info(f"[DOCX DEBUG] Found image: {rel.target_ref}")
    
    logger.info(f"[DOCX DEBUG] Total images in document: {image_count}")
    
    if image_count == 0:
        logger.warning("[DOCX DEBUG] ⚠️ NO IMAGES FOUND - Check if Word file has embedded images")
    
    # ... rest of code
Çalıştır ve loglara bak:

Total images in document: 0 → Word'de görsel yok (beklenmez)
Total images in document: 8 → Görseller var AMA extract edilmiyor


2. Görsel Path Standardizasyonu
Sorun: Word'den extract edilen görseller farklı path'le kaydediliyor
python# MEVCUT KOD (load_docx içinde):
image_filename = f"{Path(file_path).stem}-{image_counter}.{ext}"
image_path = images_folder / image_filename

# Store RELATIVE path
relative_path = f"{images_folder.name}/{image_filename}"
image_map[rel_id] = relative_path

# ✅ SORUN: Markdown'a yazılan path
markdown_parts.append(f"![Image]({relative_path})")
Bu üretir: ![Image](Feb6-Earthquake-Sentiment-Analysis-Research-Paper_images/Feb6-Earthquake-Sentiment-Analysis-Research-Paper-0.png)
AMA döküman diyor ki: ![Image](/Feb6-Earthquake-Sentiment-Analysis-Research-Paper_images/Feb6-Earthquake-Sentiment-Analysis-Research-Paper-5.png) (başında / var)
FIX:
python# ✅ Path'i normalize et (başta / olmasın)
relative_path = f"{images_folder.name}/{image_filename}"
markdown_parts.append(f"![Image]({relative_path})")  # ✅ Doğru

# ❌ YANLIŞ (başta / varsa)
markdown_parts.append(f"![Image](/{relative_path})")  # Bu dökümanındaki hata

3. Görsel Metadata'sını Ekle
python# load_docx() sonunda:
logger.info(f"[DOCX] ✅ Converted {len(md_text)} chars with {image_counter} images")
logger.info(f"[DOCX] 🖼️ Images saved to: {images_folder}")

# ✅ Debug dosyasına yaz
debug_path = Path(file_path).parent / f"{Path(file_path).stem}_docx_debug.md"
debug_path.write_text(md_text, encoding="utf-8")

# ✅ Image list'i de yaz
image_list_path = Path(file_path).parent / f"{Path(file_path).stem}_image_list.txt"
image_list = [
    f"{idx}: {img_map[rel_id]}"
    for idx, (rel_id, img_map) in enumerate(image_map.items())
]
image_list_path.write_text("\n".join(image_list), encoding="utf-8")
logger.info(f"[DOCX] 📝 Image list saved to: {image_list_path}")
Sonra kontrol et:

uploads/Feb6-Earthquake-Sentiment-Analysis-Research-Paper_docx_debug.md → Görsel referansları doğru mu?
uploads/Feb6-Earthquake-Sentiment-Analysis-Research-Paper_image_list.txt → Hangi görseller kaydedilmiş?
uploads/Feb6-Earthquake-Sentiment-Analysis-Research-Paper_images/ → Klasörde dosyalar var mı?


🟡 ÖNEMLI: Metadata Zenginleştirme
Sorun
Chunk metadata'sı çok basit:
python{
    "source": "Feb6-Earthquake-Sentiment-Analysis-Research-Paper.docx",
    "has_images": False  # ❌ Yanlış
}
Çözüm: Hierarchical Metadata
Dosya: src/agents/agentic_chunker.py → extract_propositions_from_markdown()
1. Bölüm Başlıklarını Takip Et
pythondef extract_propositions_from_markdown(markdown_text: str) -> List[str]:
    """Extract propositions with section tracking"""
    if not markdown_text:
        return []
    
    text = markdown_text.replace("\r\n", "\n").replace("\r", "\n")
    
    # ✅ YENİ: Track current section (H1/H2)
    current_h1 = None
    current_h2 = None
    
    # Regex patterns
    h1_pattern = re.compile(r'^#\s+(.+)$', re.MULTILINE)
    h2_pattern = re.compile(r'^##\s+(.+)$', re.MULTILINE)
    major_header_pattern = re.compile(r'^(#{1,4})\s+(.+)$', re.MULTILINE)
    
    # Extract all H1/H2 headers for context
    h1_headers = {m.start(): m.group(1) for m in h1_pattern.finditer(text)}
    h2_headers = {m.start(): m.group(1) for m in h2_pattern.finditer(text)}
    
    # ... rest of existing code
    
    # When creating propositions, add section metadata:
    propositions_with_metadata = []
    for prop in propositions:
        # Find which H1/H2 this belongs to
        prop_start = text.find(prop)
        
        # Find nearest H1 before this position
        section_h1 = None
        for pos, title in sorted(h1_headers.items(), reverse=True):
            if pos < prop_start:
                section_h1 = title
                break
        
        # Find nearest H2 before this position
        section_h2 = None
        for pos, title in sorted(h2_headers.items(), reverse=True):
            if pos < prop_start:
                section_h2 = title
                break
        
        # Store as tuple: (proposition, metadata)
        propositions_with_metadata.append({
            'text': prop,
            'section_h1': section_h1,
            'section_h2': section_h2
        })
    
    return propositions_with_metadata

2. AgenticChunker'a Metadata Ekle
Dosya: src/agents/agentic_chunker.py → agentic_chunk_text()
pythondef agentic_chunk_text(text: str, source_name: str) -> List[Document]:
    """Main entry point with rich metadata"""
    
    # Extract propositions WITH metadata
    propositions_with_meta = extract_propositions_from_markdown(text)
    
    # Use agentic chunker
    chunker = AgenticChunker(source_name=source_name)
    
    # ✅ YENİ: Pass metadata to chunker
    for prop_data in propositions_with_meta:
        chunker.add_proposition(
            prop_data['text'],
            section_h1=prop_data['section_h1'],
            section_h2=prop_data['section_h2']
        )
    
    documents = chunker.get_documents()
    
    # ✅ YENİ: Enrich document metadata
    for doc in documents:
        # Add hierarchical context
        doc.metadata['section_h1'] = doc.metadata.get('section_h1', 'Unknown')
        doc.metadata['section_h2'] = doc.metadata.get('section_h2', None)
        
        # Calculate approximate position (0.0-1.0)
        doc.metadata['position'] = doc.metadata['chunk_index'] / len(documents)
    
    return documents
Sonuç:
python# ÖNCESİ
{
    "source": "paper.docx",
    "has_images": False
}

# SONRASI
{
    "source": "paper.docx",
    "section_h1": "Yöntem",
    "section_h2": "Makine Öğrenmesi",
    "chunk_index": 15,
    "position": 0.45,  # Dökümanın %45'inde
    "has_images": True,
    "title": "Naive Bayes Algoritması",
    "summary": "Bayes teoremine dayanan sınıflandırma yöntemi"
}

🟢 BONUS: Tablo ve Formül İyileştirme (Opsiyonel)
Sorun 1: Tablo Başlıkları Eksik
Chunk #10:
| Kategoriler | Tarih | Şubat | Mart | ...
Ama tablo başlığı ("Tablo 5. Kategori Bazında...") chunk'ta değil.
Çözüm: Tablo tespit algoritması
pythondef _enhance_table_context(markdown_text: str) -> str:
    """Add table titles if missing"""
    lines = markdown_text.split("\n")
    enhanced = []
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Tablo başlangıcı mı?
        if line.startswith("|") and i > 0:
            # Önceki 3 satıra bak - "Tablo X" var mı?
            has_title = False
            for j in range(max(0, i-3), i):
                if "Tablo" in lines[j] or "Table" in lines[j]:
                    has_title = True
                    break
            
            # Başlık yoksa ekle
            if not has_title:
                enhanced.append(f"\n**Tablo** (Başlık tespit edilemedi)\n")
        
        enhanced.append(line)
        i += 1
    
    return "\n".join(enhanced)

Sorun 2: Formüller Kayıp
Word'de formüller genelde MathML veya görsel olarak gelir. python-docx bunları okuyamaz.
Alternatif:

Kullanıcıya uyar: "Bu dökümanda formüller olabilir, PDF versiyonunu yükleyin"
Placeholder ekle: Formül yerine [Formula: Kesinlik = TP/(TP+FP)]

python# load_docx() içinde:
# Formül tespit (heuristic - "=" var ama tablo değil)
if "=" in para.text and "|" not in para.text:
    markdown_parts.append(f"[Formula: {para.text.strip()}]")

📋 Uygulama Sırası
Adım 1: Görsel Debug (5 dakika)
bash# Word dosyasını yeniden yükle
# Loglara bak:
# - Total images in document: ?
# - Images saved to: ?
Adım 2: Görsel Path Fix (10 dakika)

load_docx() içinde relative path'i düzelt
_docx_debug.md kontrol et
Vision model test et

Adım 3: Metadata Zenginleştirme (20 dakika)

extract_propositions_from_markdown() → section tracking ekle
agentic_chunk_text() → metadata propagation
Test query: "Yöntem bölümünde hangi algoritmalar kullanılmış?"

Adım 4: Tablo İyileştirme (Opsiyonel, 10 dakika)

_enhance_table_context() fonksiyonu ekle
Tablo başlıklarını otomatik ekle


🎯 Beklenen İyileşme
Öncesi
Chunk #11:
"Naive Bayes algoritmasının temel çalışma mantığını görselleştirmektedir..."

Metadata: {
    "source": "paper.docx",
    "has_images": False  # ❌ Yanlış
}

❌ Image not found: /paper_images/image2.png
Sonrası
Chunk #11:
"##### Naive Bayes

Naive Bayes algoritmasının temel çalışma mantığını görselleştirmektedir...

![Naive Bayes Diyagramı](paper_images/paper-2.png)"

Metadata: {
    "source": "paper.docx",
    "section_h1": "Yöntem",
    "section_h2": "Makine Öğrenmesi",
    "title": "Naive Bayes Algoritması",
    "has_images": True,
    "position": 0.42,
    "chunk_index": 11
}

✅ Vision Model: "Görsel bir funnel (huni) diyagramı göstermektedir..."

💡 Hızlı Test
python# Test script: test_chunks.py
from src.agents.rag_agent import load_docx, ingest_text

# 1. Word'ü yükle
md_text = load_docx("uploads/paper.docx")

# 2. Markdown kontrol et
with open("test_output.md", "w", encoding="utf-8") as f:
    f.write(md_text)

print("✅ Markdown kaydedildi: test_output.md")
print(f"- Uzunluk: {len(md_text)} chars")
print(f"- Görsel sayısı: {md_text.count('![')}")

# 3. Chunk'la
chunk_count = ingest_text(md_text, "test_paper.docx")
print(f"✅ {chunk_count} chunk oluşturuldu")

# 4. Query test
from src.agents.rag_agent import retrieve_context
context, docs = retrieve_context("Naive Bayes nasıl çalışır?", top_k="3")

print(f"\n📊 Retrieval Sonuçları:")
print(f"- Bulunan döküman: {len(docs)}")
print(f"- İlk chunk metadata: {docs[0].metadata if docs else 'N/A'}")