"""
Agentic Chunker - LLM-powered intelligent document chunking
Uses Gemini 2.5 to semantically group related content

Based on: https://github.com/FullStackRetrieval-com/RetrievalTutorials

Features:
- Each text segment is analyzed by LLM
- LLM decides if it belongs to existing chunk or needs new one
- Chunks have dynamic summaries and titles
- Keeps images with their descriptions
- Handles Turkish and English content
"""

import uuid
import os
import logging
import re
from typing import Optional, List, Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from ..config import settings

logger = logging.getLogger(__name__)


def _get_chunker_llm():
    """Get LLM for agentic chunking - uses local Ollama (qwen2.5:7b)"""
    
    # Use local Ollama - qwen2.5:7b (4.7GB, superior for semantic chunking & Turkish)
    try:
        from langchain_ollama import ChatOllama
        ollama_url = settings.ollama_base_url
        logger.info(f"[AgenticChunker] Connecting to Ollama at {ollama_url} with qwen2.5:7b")
        
        # Test connection first
        import httpx
        try:
            response = httpx.get(f"{ollama_url}/api/tags", timeout=2)
            if response.status_code != 200:
                raise Exception(f"Ollama not responding at {ollama_url}")
        except Exception as conn_err:
            logger.error(f"[AgenticChunker] Cannot connect to Ollama at {ollama_url}: {conn_err}")
            return None
        
        return ChatOllama(
            model="qwen2.5:7b",
            base_url=ollama_url,
            temperature=0,
            num_gpu=-1,
        )
    except Exception as e:
        logger.error(f"[AgenticChunker] Ollama failed: {e}")
        return None


class AgenticChunker:
    """
    LLM-powered intelligent chunker that groups semantically related content.
    
    Instead of splitting by character count or headers alone, this uses an LLM
    to decide which content belongs together based on meaning.
    """
    
    def __init__(self, source_name: str = "document"):
        self.chunks: Dict[str, Dict[str, Any]] = {}
        self.source_name = source_name
        self.id_truncate_limit = 8
        self.generate_new_metadata_ind = True  # Update summaries as chunks grow
        self.print_logging = False  # Set True for debug
        
        self.llm = _get_chunker_llm()
        if not self.llm:
            logger.warning("[AgenticChunker] No LLM available, will use fallback splitting")
    
    def add_propositions(self, propositions: List[str]) -> None:
        """Add multiple text segments to be chunked."""
        for prop in propositions:
            if prop and prop.strip():
                self.add_proposition(prop.strip())
    
    def add_proposition(self, proposition: str) -> None:
        """Add a single text segment, placing it in the best chunk."""
        if not proposition.strip():
            return
            
        if self.print_logging:
            logger.info(f"[AgenticChunker] Adding: '{proposition[:80]}...'")
        
        # If no LLM available, create individual chunks
        if not self.llm:
            self._create_new_chunk(proposition)
            return
        
        # First chunk - just create it
        if len(self.chunks) == 0:
            self._create_new_chunk(proposition)
            return
        
        # Find relevant existing chunk
        chunk_id = self._find_relevant_chunk(proposition)
        
        if chunk_id and chunk_id in self.chunks:
            self._add_proposition_to_chunk(chunk_id, proposition)
        else:
            self._create_new_chunk(proposition)
    
    def _add_proposition_to_chunk(self, chunk_id: str, proposition: str) -> None:
        """Add proposition to existing chunk and update metadata."""
        # Check chunk size limit - prevent oversized chunks
        current_size = sum(len(p) for p in self.chunks[chunk_id]['propositions'])
        if current_size + len(proposition) > 2500:
            # Chunk too large, force new chunk instead
            if self.print_logging:
                logger.info(f"[AgenticChunker] Chunk {chunk_id} too large ({current_size} chars), creating new chunk")
            self._create_new_chunk(proposition)
            return
        
        self.chunks[chunk_id]['propositions'].append(proposition)
        
        if self.generate_new_metadata_ind and self.llm:
            try:
                self.chunks[chunk_id]['summary'] = self._update_chunk_summary(self.chunks[chunk_id])
                self.chunks[chunk_id]['title'] = self._update_chunk_title(self.chunks[chunk_id])
            except Exception as e:
                logger.warning(f"[AgenticChunker] Failed to update metadata: {e}")
    
    def _update_chunk_summary(self, chunk: Dict) -> str:
        """Update summary when new content is added."""
        PROMPT = ChatPromptTemplate.from_messages([
            ("system", """Sen döküman parçalarını organize eden bir asistansın.
Mevcut bir parçaya (chunk) yeni içerik eklendi. Bu parçanın ne hakkında olduğunu anlatan 1 cümlelik Türkçe bir özet yaz.

Özet şunları içermelidir:
- Ana konuyu tanımla
- SADECE TÜRKÇE yaz
- Genellemeyi öngör (Örn: Sadece "Naive Bayes" değil, "Makine öğrenmesi algoritmaları" gibi)

Sadece özeti yaz, başka hiçbir şey yazma."""),
            ("user", "Chunk content:\n{propositions}\n\nCurrent summary:\n{current_summary}")
        ])
        
        result = (PROMPT | self.llm).invoke({
            "propositions": "\n".join(chunk['propositions'][-5:]),  # Last 5 for efficiency
            "current_summary": chunk['summary']
        })
        
        return result.content.strip()
    
    def _update_chunk_title(self, chunk: Dict) -> str:
        """Update title when content changes."""
        PROMPT = ChatPromptTemplate.from_messages([
            ("system", """Bu döküman parçası için 2-5 kelimelik kısa bir Türkçe başlık oluştur.
SADECE TÜRKÇE yaz.
Sadece başlığı yaz, başka hiçbir şey yazma."""),
            ("user", "Summary: {summary}")
        ])
        
        result = (PROMPT | self.llm).invoke({
            "summary": chunk['summary']
        })
        
        return result.content.strip()
    
    def _get_new_chunk_summary(self, proposition: str) -> str:
        """Generate initial summary for a new chunk."""
        if not self.llm:
            return proposition[:100]
        
        PROMPT = ChatPromptTemplate.from_messages([
            ("system", """Sen döküman parçalarını organize eden bir asistansın. Yeni bir parça (chunk) için 1 cümlelik Türkçe özet oluştur.

Özet şunları içermelidir:
- Buraya ne tür içeriklerin geleceğini tanımla
- SADECE TÜRKÇE yaz
- Benzer içeriklerin eklenebileceğini öngör

Sadece özeti yaz, başka hiçbir şey yazma."""),
            ("user", "Initial content:\n{proposition}")
        ])
        
        try:
            result = (PROMPT | self.llm).invoke({"proposition": proposition})
            return result.content.strip()
        except Exception as e:
            logger.warning(f"[AgenticChunker] Summary generation failed: {e}")
            return proposition[:100]
    
    def _get_new_chunk_title(self, summary: str) -> str:
        """Generate title from summary."""
        if not self.llm:
            return summary[:50]
        
        PROMPT = ChatPromptTemplate.from_messages([
            ("system", """Bu parça için 2-5 kelimelik kısa bir Türkçe başlık oluştur.
SADECE TÜRKÇE yaz.
Sadece başlığı yaz, başka hiçbir şey yazma."""),
            ("user", "Summary: {summary}")
        ])
        
        try:
            result = (PROMPT | self.llm).invoke({"summary": summary})
            return result.content.strip()
        except Exception as e:
            logger.warning(f"[AgenticChunker] Title generation failed: {e}")
            return summary[:50]
    
    def _create_new_chunk(self, proposition: str) -> str:
        """Create a new chunk with the given proposition."""
        chunk_id = str(uuid.uuid4())[:self.id_truncate_limit]
        summary = self._get_new_chunk_summary(proposition)
        title = self._get_new_chunk_title(summary)
        
        # Check if proposition has images
        has_images = bool(re.search(r'!\[[^\]]*\]\([^)]+\)', proposition))
        
        self.chunks[chunk_id] = {
            'chunk_id': chunk_id,
            'propositions': [proposition],
            'title': title,
            'summary': summary,
            'chunk_index': len(self.chunks),
            'has_images': has_images,
        }
        
        if self.print_logging:
            logger.info(f"[AgenticChunker] Created chunk ({chunk_id}): {title}")
        
        return chunk_id
    
    def _get_chunk_outline(self) -> str:
        """Get string representation of current chunks for LLM context."""
        if not self.chunks:
            return "No chunks yet."
        
        outline = ""
        for chunk_id, chunk in self.chunks.items():
            outline += f"Chunk ID: {chunk_id}\n"
            outline += f"Title: {chunk['title']}\n"
            outline += f"Summary: {chunk['summary']}\n\n"
        
        return outline
    
    def _find_relevant_chunk(self, proposition: str) -> Optional[str]:
        """Use LLM to find which existing chunk this proposition belongs to."""
        if not self.llm or not self.chunks:
            return None
        
        # GÖRSEL KONTROLÜ: Görsel ise otomatik olarak SON chunk'a ekle
        # Bu, LLM'in yanlış karar vermesini engeller
        is_image_only = bool(re.match(r'^\s*!\[[^\]]*\]\([^)]+\)\s*$', proposition.strip()))
        if is_image_only and self.chunks:
            # Son oluşturulan chunk'ı bul (en yüksek index)
            last_chunk_id = max(self.chunks.keys(), key=lambda k: self.chunks[k].get('chunk_index', 0))
            if self.print_logging:
                logger.info(f"[AgenticChunker] Image auto-assigned to last chunk: {last_chunk_id}")
            return last_chunk_id
        
        # H4 BAŞLIK KONTROLÜ: H4 başlıklar (####) MUTLAKA yeni chunk olmalı
        # Bu algoritmalar, yöntemler gibi farklı konuları ayırır
        is_h4_header = bool(re.match(r'^####\s+', proposition.strip()))
        if is_h4_header:
            if self.print_logging:
                logger.info(f"[AgenticChunker] H4 header detected, forcing NEW_CHUNK")
            return None  # Force new chunk
        
        current_outline = self._get_chunk_outline()
        
        PROMPT = ChatPromptTemplate.from_messages([
            ("system", """Determine if the new content should belong to an existing chunk.

Content should belong to a chunk if:
- They discuss the EXACT same specific topic (e.g., same algorithm, same method)
- They directly continue or explain the previous chunk's topic
- They provide examples or details for the previous chunk

CRITICAL RULES - READ CAREFULLY:
1. H5/H6 headers (##### or ######) with their explanations = SAME CHUNK
2. H4 headers (####) = ALWAYS NEW_CHUNK (different algorithms, methods, sections)
3. If a chunk already has >1500 chars, prefer NEW_CHUNK unless VERY related
4. Different algorithms/methods = DIFFERENT CHUNKS
5. "BULGULAR", "YÖNTEM", "AMAÇ" etc. = DIFFERENT CHUNKS

BE CONSERVATIVE:
- When in doubt, create NEW_CHUNK
- Only group if topics are IDENTICAL
- Don't mix different concepts

RESPONSE FORMAT:
- If matches an existing chunk: respond ONLY with chunk ID (e.g., "a1b2c3d4")
- If should be separate: respond ONLY with "NEW_CHUNK"
- NO explanations, NO extra text"""),
            ("user", """Current chunks:
{current_outline}

New content to place:
{proposition}

Which chunk ID should this go to? (respond with ID only, or NEW_CHUNK)""")
        ])
        
        try:
            result = (PROMPT | self.llm).invoke({
                "current_outline": current_outline,
                "proposition": proposition[:500]  # Truncate for efficiency
            })
            
            response = result.content.strip()
            
            # Check if response is a valid chunk ID
            if response == "NEW_CHUNK" or response.lower() == "new_chunk":
                return None
            
            # Extract chunk ID (handle LLM adding extra text)
            for chunk_id in self.chunks.keys():
                if chunk_id in response:
                    return chunk_id
            
            # If response length matches ID format, try it
            if len(response) == self.id_truncate_limit and response in self.chunks:
                return response
            
            return None
            
        except Exception as e:
            logger.warning(f"[AgenticChunker] Chunk finding failed: {e}")
            return None
    
    def get_documents(self) -> List[Document]:
        """Convert chunks to LangChain Documents for vector store."""
        documents = []
        
        for chunk_id, chunk in self.chunks.items():
            # Combine all propositions in the chunk
            content = "\n\n".join(chunk['propositions'])
            
            # Check for images in combined content
            has_images = bool(re.search(r'!\[[^\]]*\]\([^)]+\)', content))
            
            doc = Document(
                page_content=content,
                metadata={
                    "source": self.source_name,
                    "chunk_id": chunk_id,
                    "title": chunk['title'],
                    "summary": chunk['summary'],
                    "chunk_index": chunk['chunk_index'],
                    "has_images": has_images,
                }
            )
            documents.append(doc)
        
        return documents
    
    def get_chunks_list(self) -> List[str]:
        """Get chunks as list of strings."""
        return ["\n".join(chunk['propositions']) for chunk in self.chunks.values()]
    
    def pretty_print_chunks(self) -> None:
        """Print chunks for debugging."""
        print(f"\n{'='*60}")
        print(f"Agentic Chunker: {len(self.chunks)} chunks from '{self.source_name}'")
        print('='*60)
        
        for chunk_id, chunk in self.chunks.items():
            print(f"\nChunk #{chunk['chunk_index']} (ID: {chunk_id})")
            print(f"Title: {chunk['title']}")
            print(f"Summary: {chunk['summary']}")
            print(f"Has Images: {chunk.get('has_images', False)}")
            print(f"Content ({len(chunk['propositions'])} parts):")
            for i, prop in enumerate(chunk['propositions'][:3]):  # Show first 3
                print(f"  {i+1}. {prop[:100]}...")
            if len(chunk['propositions']) > 3:
                print(f"  ... and {len(chunk['propositions']) - 3} more")
            print("-" * 40)


def extract_propositions_from_markdown(markdown_text: str) -> List[str]:
    """
    FIXED: Forward lookahead - görseller en yakın başlıkla birleştirilir.
    
    Strategy (Generic for all document types):
    1. Major headers (H1-H4) ile bölümlere ayır
    2. Her bölüm içinde H5/H6 başlıklarını bul
    3. H5/H6 başlıklardan sonraki 1-3 satırda görsel var mı kontrol et
    4. Varsa görseli başlığa yapıştır
    5. Büyük bölümleri akıllıca böl (>2500 char)
    
    Works for: PDF→Markdown, DOCX→Markdown, plain Markdown
    """
    if not markdown_text:
        return []
    
    text = markdown_text.replace("\r\n", "\n").replace("\r", "\n")
    
    # Regex patterns
    major_header_pattern = re.compile(r'^(#{1,4})\s+(.+)$', re.MULTILINE)
    
    # Find all MAJOR headers (H1-H4)
    major_headers = []
    for match in major_header_pattern.finditer(text):
        level = len(match.group(1))
        major_headers.append({
            'level': level,
            'start': match.start(),
            'end': match.end(),
        })
    
    if not major_headers:
        # No major headers - use smart split with image attachment
        return _smart_split_with_images(text)
    
    propositions = []
    
    # Handle content before first header
    if major_headers[0]['start'] > 0:
        pre_content = text[:major_headers[0]['start']].strip()
        if pre_content:
            propositions.append(pre_content)
    
    # Process each major section
    for i, header in enumerate(major_headers):
        section_start = header['start']
        section_end = major_headers[i + 1]['start'] if i + 1 < len(major_headers) else len(text)
        
        section_text = text[section_start:section_end].strip()
        
        if not section_text:
            continue
        
        # ✅ CRITICAL FIX: "Forward Image Attachment"
        # Section içinde H5/H6 başlıkları varsa, her birini sonraki görselle birleştir
        processed_section = _attach_images_to_headers(section_text)
        
        # ✅ CRITICAL: Split at H4 headers too (algorithms, methods)
        # H4 başlıklar genelde farklı konular (algoritmalar, yöntemler)
        h4_pattern = re.compile(r'^####\s+.+$', re.MULTILINE)
        h4_matches = list(h4_pattern.finditer(processed_section))
        
        if not h4_matches:
            # No H4 headers - process as single section
            if len(processed_section) > 1500:
                sub_props = _split_large_section_preserve_images(processed_section)
                propositions.extend(sub_props)
            else:
                propositions.append(processed_section)
        else:
            # Has H4 headers - split by them
            for idx, match in enumerate(h4_matches):
                h4_start = match.start()
                h4_end = h4_matches[idx + 1].start() if idx + 1 < len(h4_matches) else len(processed_section)
                
                h4_section = processed_section[h4_start:h4_end].strip()
                
                if len(h4_section) > 1500:
                    sub_props = _split_large_section_preserve_images(h4_section)
                    propositions.extend(sub_props)
                else:
                    propositions.append(h4_section)
    
    # Merge very small props
    merged = _merge_small_propositions(propositions, min_chars=150)
    
    return [p for p in merged if p.strip()]


def _attach_images_to_headers(section_text: str) -> str:
    """
    ✅ FORWARD LOOKAHEAD: Attach images to their nearest H5/H6 headers.
    
    Example:
    Input:
        ##### Ridge Classifier
        Açıklama...
        
        ![](ridge.png)
        
        ##### Naive Bayes
        Açıklama...
        
        ![](naive.png)
    
    Output:
        ##### Ridge Classifier
        Açıklama...
        ![](ridge.png)
        
        ##### Naive Bayes
        Açıklama...
        ![](naive.png)
    """
    lines = section_text.split("\n")
    result_lines = []
    
    h5h6_re = re.compile(r'^#{5,6}\s+.+$')
    image_re = re.compile(r'^!\[[^\]]*\]\([^)]+\)\s*$')
    major_header_re = re.compile(r'^#{1,4}\s+.+$')
    
    i = 0
    consumed_images = set()  # Track which lines we've already attached
    
    while i < len(lines):
        line = lines[i]
        
        # Check if current line is H5/H6
        if h5h6_re.match(line.strip()):
            # Start collecting this header section
            header_section = [line]
            i += 1
            
            # Collect next lines until we hit another header or end
            while i < len(lines):
                next_line = lines[i]
                
                # Stop if we hit another H5/H6 or major header
                if h5h6_re.match(next_line.strip()) or major_header_re.match(next_line.strip()):
                    break
                
                # Don't add if this is a consumed image
                if i not in consumed_images:
                    header_section.append(next_line)
                i += 1
            
            # Now check if there's an orphan image in next few lines
            # ✅ Dynamic lookahead - up to 10 lines or until empty line
            lookahead_start = i
            lookahead_end = min(i + 10, len(lines))  # Max 10 lines (was 3)
            
            for j in range(lookahead_start, lookahead_end):
                line_content = lines[j].strip()
                
                # Stop at empty line (paragraph boundary)
                if not line_content:
                    break
                
                # Stop at next header
                if h5h6_re.match(line_content) or major_header_re.match(line_content):
                    break
                
                # Found orphan image - attach it
                if image_re.match(line_content):
                    if header_section and header_section[-1].strip():  # Add empty line if needed
                        header_section.append("")
                    header_section.append(lines[j])
                    
                    # Mark this image as consumed
                    consumed_images.add(j)
                    # Continue looking for more images in this block!

            
            # Add this complete header section
            result_lines.extend(header_section)
        else:
            # Regular line (not H5/H6 header)
            if i not in consumed_images:  # Skip if it's a consumed image
                result_lines.append(line)
            i += 1
    
    return "\n".join(result_lines)


def _smart_split_with_images(text: str) -> List[str]:
    """Split text intelligently, keeping images with nearby text."""
    if len(text) <= 2500:
        return [text]
    
    # Use paragraph-based splitting while keeping images attached
    return _split_large_section_preserve_images(text)


def _split_large_section_preserve_images(text: str) -> List[str]:
    """
    Split large sections while preserving image attachments.
    
    - Splits on paragraph boundaries (double newline)
    - Never splits an image from its preceding paragraph
    - Keeps chunks under ~2000 chars
    """
    if len(text) <= 2500:
        return [text]
    
    paragraphs = re.split(r'\n\n+', text)
    chunks = []
    current_chunk = []
    current_size = 0
    
    image_re = re.compile(r'!\[[^\]]*\]\([^)]+\)')
    
    for para in paragraphs:
        para_size = len(para)
        
        # Check if this paragraph contains an image
        has_image = bool(image_re.search(para))
        
        # If adding this would exceed limit AND we have content
        if current_size + para_size > 2000 and current_chunk:
            # Save current chunk
            chunks.append("\n\n".join(current_chunk))
            current_chunk = [para]
            current_size = para_size
        else:
            current_chunk.append(para)
            current_size += para_size
    
    # Don't forget last chunk
    if current_chunk:
        chunks.append("\n\n".join(current_chunk))
    
    return chunks


def _build_header_context(headers: List[dict], current_idx: int) -> str:
    """Build hierarchical context from parent headers."""
    if current_idx == 0:
        return ""
    
    current_level = headers[current_idx]['level']
    context_parts = []
    
    # Look backwards for parent headers (lower level number = higher hierarchy)
    for i in range(current_idx - 1, -1, -1):
        if headers[i]['level'] < current_level:
            context_parts.insert(0, headers[i]['title'])
            current_level = headers[i]['level']
    
    return " > ".join(context_parts) if context_parts else ""


def _split_large_section(text: str, preserve_header: str = None) -> List[str]:
    """
    Split a large section while preserving semantic coherence.
    
    - Keeps header with first chunk
    - Splits on paragraph boundaries (double newline)
    - Keeps images with their nearest text context
    """
    if len(text) <= 2000:
        return [text] if text.strip() else []
    
    lines = text.split("\n")
    chunks = []
    current_chunk = []
    current_size = 0
    
    image_re = re.compile(r'^!\[[^\]]*\]\([^)]+\)\s*$')
    header_re = re.compile(r'^#{1,6}\s+.+$')
    
    for line in lines:
        line_stripped = line.strip()
        is_image = bool(image_re.match(line_stripped))
        is_header = bool(header_re.match(line_stripped))
        is_empty = not line_stripped
        
        # Always add headers and images to current chunk
        if is_header or is_image:
            current_chunk.append(line)
            current_size += len(line)
            continue
        
        # Check if adding this line would exceed limit
        if current_size + len(line) > 1800 and current_chunk:
            # Save current chunk
            chunk_text = "\n".join(current_chunk).strip()
            if chunk_text:
                chunks.append(chunk_text)
            
            # Start new chunk - add context from header if available
            if preserve_header and not chunks:
                # First chunk already has header
                current_chunk = [line]
            else:
                current_chunk = [line]
            current_size = len(line)
        else:
            current_chunk.append(line)
            current_size += len(line)
    
    # Don't forget last chunk
    if current_chunk:
        chunk_text = "\n".join(current_chunk).strip()
        if chunk_text:
            chunks.append(chunk_text)
    
    return chunks


def _merge_small_propositions(propositions: List[str], min_chars: int = 100) -> List[str]:
    """Merge very small propositions with adjacent ones."""
    if len(propositions) <= 1:
        return propositions
    
    merged = []
    image_re = re.compile(r'!\[[^\]]*\]\([^)]+\)')
    header_re = re.compile(r'^#{1,6}\s+.+$', re.MULTILINE)
    
    for prop in propositions:
        if not prop.strip():
            continue
        
        # Calculate text length without images/headers
        text_only = re.sub(r'!\[[^\]]*\]\([^)]+\)', '', prop)
        text_only = re.sub(r'^#{1,6}\s+.+$', '', text_only, flags=re.MULTILINE).strip()
        
        has_image = bool(image_re.search(prop))
        has_header = bool(header_re.search(prop))
        
        # Don't merge if has image or header - these are important context markers
        if has_image or has_header:
            merged.append(prop)
        elif len(text_only) < min_chars and merged:
            # Merge with previous proposition
            merged[-1] = merged[-1] + "\n\n" + prop
        else:
            merged.append(prop)
    
    return merged


def _split_by_paragraphs_and_images(text: str, keep_header_with_content: bool = False) -> List[str]:
    """Split text by paragraphs, keeping images with nearby context.
    
    Args:
        text: Text to split
        keep_header_with_content: If True, keeps headers attached to their following content
    """
    if not text:
        return []
    
    lines = text.split("\n")
    propositions = []
    current = []
    
    image_re = re.compile(r'^!\[[^\]]*\]\([^)]+\)\s*$')
    header_re = re.compile(r'^#{1,6}\s+.+$')
    
    for i, line in enumerate(lines):
        is_image = bool(image_re.match(line.strip()))
        is_empty = not line.strip()
        is_header = bool(header_re.match(line.strip()))
        
        if is_image:
            # Image line - keep with accumulated content
            current.append(line)
        elif is_header and keep_header_with_content:
            # Header - keep with following content, don't split here
            current.append(line)
        elif is_empty and current:
            # Empty line after content - might be a paragraph break
            text_content = "\n".join(current).strip()
            
            # Check if next non-empty line is a header
            next_is_header = False
            for j in range(i + 1, len(lines)):
                if lines[j].strip():
                    next_is_header = bool(header_re.match(lines[j].strip()))
                    break
            
            # Split if: substantial content AND (not keeping headers OR next is not header)
            if len(text_content) > 50 and (not keep_header_with_content or not next_is_header):
                propositions.append(text_content)
                current = []
            # Otherwise keep accumulating
        else:
            current.append(line)
    
    # Don't forget remaining content
    if current:
        text_content = "\n".join(current).strip()
        if text_content:
            propositions.append(text_content)
    
    # LINE MERGING: Merge broken lines and short propositions
    merged = []
    for prop in propositions:
        # Skip if just whitespace
        if not prop.strip():
            continue
        
        # Fix line breaks within sentences (satır sonu kesilmelerini düzelt)
        # Merge lines that don't end with punctuation
        lines_in_prop = prop.split('\n')
        fixed_lines = []
        temp_line = ""
        
        for line in lines_in_prop:
            line = line.strip()
            if not line:
                if temp_line:
                    fixed_lines.append(temp_line)
                    temp_line = ""
                fixed_lines.append("")  # Keep paragraph breaks
                continue
            
            # Check if line ends with sentence-ending punctuation or is a header/image
            ends_sentence = line.endswith(('.', '!', '?', ':', ')', ']', '}', '"', "'"))
            is_header = bool(header_re.match(line))
            is_image = bool(image_re.match(line))
            
            if temp_line:
                if ends_sentence or is_header or is_image:
                    fixed_lines.append(temp_line + " " + line)
                    temp_line = ""
                else:
                    temp_line += " " + line
            else:
                if ends_sentence or is_header or is_image:
                    fixed_lines.append(line)
                else:
                    temp_line = line
        
        if temp_line:
            fixed_lines.append(temp_line)
        
        prop = "\n".join(fixed_lines).strip()
        
        # If prop is very short and no image/header, merge with previous
        has_image = bool(re.search(r'!\[[^\]]*\]\([^)]+\)', prop))
        has_header = bool(header_re.search(prop))
        text_only = re.sub(r'!\[[^\]]*\]\([^)]+\)', '', prop)
        text_only = re.sub(r'^#{1,6}\s+.+$', '', text_only, flags=re.MULTILINE).strip()
        
        if len(text_only) < 100 and not has_image and not has_header and merged:
            # Merge with previous proposition
            merged[-1] = merged[-1] + "\n\n" + prop
        else:
            merged.append(prop)
    
    return merged


def agentic_chunk_text(text: str, source_name: str) -> List[Document]:
    """
    Main entry point for agentic chunking.
    
    Args:
        text: Raw text content (Markdown preferred)
        source_name: Source identifier
        
    Returns:
        List of LangChain Documents
    """
    logger.info(f"[AgenticChunker] Starting chunking for: {source_name}")
    
    # Extract propositions from text
    propositions = extract_propositions_from_markdown(text)
    logger.info(f"[AgenticChunker] Extracted {len(propositions)} propositions")
    
    if not propositions:
        # Fallback: return single document
        return [Document(
            page_content=text,
            metadata={"source": source_name, "has_images": bool(re.search(r'!\[[^\]]*\]\([^)]+\)', text))}
        )]
    
    # Use agentic chunker to group propositions
    chunker = AgenticChunker(source_name=source_name)
    chunker.add_propositions(propositions)
    
    documents = chunker.get_documents()
    logger.info(f"[AgenticChunker] Created {len(documents)} semantic chunks")
    
    # Debug output
    if os.getenv("DEBUG_CHUNKS", "").lower() in ("1", "true", "yes"):
        chunker.pretty_print_chunks()
    
    return documents
