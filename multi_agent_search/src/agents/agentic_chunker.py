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

# ✅ PYDANTIC AI: Type-safe LLM interactions
try:
    from ..models.rag_models import ChunkDecision
    from pydantic_ai import Agent
    from pydantic_ai.providers.ollama import OllamaProvider  # ✅ Yeni API
    from pydantic import ValidationError
    import asyncio
    import concurrent.futures
    import httpx
    PYDANTIC_AI_AVAILABLE = True
    logger.info("[AgenticChunker] Pydantic AI loaded successfully")
except ImportError as e:
    PYDANTIC_AI_AVAILABLE = False
    logger.warning(f"[AgenticChunker] Pydantic AI not available: {e}")



def _get_chunker_llm():
    """Get LLM for agentic chunking - uses local Ollama (qwen2.5:3b)"""
    
    # Use local Ollama - qwen2.5:3b 
    try:
        from langchain_ollama import ChatOllama
        ollama_url = settings.ollama_base_url
        logger.info(f"[AgenticChunker] Connecting to Ollama at {ollama_url} with qwen2.5:3b")
        
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
            model="qwen2.5:3b",
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
        self._chunk_counter = 0  # ✅ FIX: Stable chunk indexing
        
        self.llm = _get_chunker_llm()
        if not self.llm:
            logger.warning("[AgenticChunker] No LLM available, will use fallback splitting")
    
    def add_propositions(self, propositions: List[str]) -> None:
        """Add multiple text segments to be chunked."""
        for prop in propositions:
            if prop and prop.strip():
                self.add_proposition(prop.strip())
    
    def add_proposition(self, proposition: str, **kwargs) -> None:
        """Add a single text segment, placing it in the best chunk."""
        if not proposition.strip():
            return
            
        if self.print_logging:
            logger.info(f"[AgenticChunker] Adding: '{proposition[:80]}...'")
        
        # If no LLM available, create individual chunks
        if not self.llm:
            self._create_new_chunk(proposition, **kwargs)
            return
        
        # First chunk - check size before creating
        if len(self.chunks) == 0:
            # ✅ FIX: Split oversized first proposition
            if len(proposition) > 1800:
                if self.print_logging:
                    logger.info(f"[AgenticChunker] First proposition too large ({len(proposition)} chars), splitting...")
                # Split into smaller parts (simple paragraph split)
                parts = proposition.split('\n\n')
                current_part = []
                current_size = 0
                
                for para in parts:
                    if current_size + len(para) > 1800 and current_part:
                        # Create chunk with accumulated parts
                        self._create_new_chunk('\n\n'.join(current_part), **kwargs)
                        current_part = [para]
                        current_size = len(para)
                    else:
                        current_part.append(para)
                        current_size += len(para)
                
                # Create final chunk
                if current_part:
                    self._create_new_chunk('\n\n'.join(current_part), **kwargs)
            else:
                self._create_new_chunk(proposition, **kwargs)
            return
        
        # Find relevant existing chunk
        chunk_id = self._find_relevant_chunk(proposition)
        
        if chunk_id and chunk_id in self.chunks:
            self._add_proposition_to_chunk(chunk_id, proposition)
        else:
            self._create_new_chunk(proposition, **kwargs)
    
    def _add_proposition_to_chunk(self, chunk_id: str, proposition: str) -> None:
        """Add proposition to existing chunk and update metadata."""
        # ✅ GELİŞME 1: HARD LIMIT (2000 Karakter)
        current_size = sum(len(p) for p in self.chunks[chunk_id]['propositions'])
        if current_size + len(proposition) > 2000:
            if self.print_logging:
                logger.info(f"[AgenticChunker] Chunk {chunk_id} limit (2000) aşıldı, yeni chunk açılıyor.")
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
            # ✅ FIX: Use first 2 + last 3 propositions to prevent "title drift"
            "propositions": "\n".join(
                chunk['propositions'][:2] + chunk['propositions'][-3:]
                if len(chunk['propositions']) > 5
                else chunk['propositions']
            ),
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
    
    def _create_new_chunk(self, proposition: str, **kwargs) -> str:
        """Create a new chunk with the given proposition."""
        # ✅ FIX: Use hex for clean alphanumeric IDs (no dashes)
        chunk_id = uuid.uuid4().hex[:self.id_truncate_limit]
        summary = self._get_new_chunk_summary(proposition)
        title = self._get_new_chunk_title(summary)
        
        # Check if proposition has images
        has_images = bool(re.search(r'!\[[^\]]*\]\([^)]+\)', proposition))
        
        # ✅ FIX: Use counter for stable indexing
        self._chunk_counter += 1
        
        self.chunks[chunk_id] = {
            'chunk_id': chunk_id,
            'propositions': [proposition],
            'title': title,
            'summary': summary,
            'chunk_index': self._chunk_counter,  # ✅ FIX: Stable index
            'has_images': has_images,
            'section_h1': kwargs.get('section_h1'),
            'section_h2': kwargs.get('section_h2')
        }
        
        if self.print_logging:
            logger.info(f"[AgenticChunker] Created chunk ({chunk_id}): {title}")
        
        return chunk_id
    
    def _get_chunk_outline(self, chunks_dict: Optional[Dict[str, Any]] = None) -> str:
        """Get string representation of chunks for LLM context.
        
        Args:
            chunks_dict: Optional filtered chunks dict. If None, uses all chunks.
        """
        target_chunks = chunks_dict if chunks_dict is not None else self.chunks
        
        if not target_chunks:
            return "No chunks yet."
        
        outline = ""
        for chunk_id, chunk in target_chunks.items():
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
            last_chunk = self.chunks[last_chunk_id]
            
            # ✅ FIX: Section kontrolü - Görsel farklı section'daysa yeni chunk aç
            # Bu, PDF'den gelen görsellerin yanlış chunk'a eklenmesini engeller
            current_size = sum(len(p) for p in last_chunk['propositions'])
            if current_size + len(proposition) <= 1800:  # Size limit
                if self.print_logging:
                    logger.info(f"[AgenticChunker] Image auto-assigned to last chunk: {last_chunk_id}")
                return last_chunk_id
            else:
                if self.print_logging:
                    logger.info(f"[AgenticChunker] Image would exceed size, creating new chunk")
                return None
        
        # ✅ SIZE-BASED SAFETY: Pre-filter oversized chunks
        if self.chunks:
            available_chunks = {}
            for chunk_id, chunk in self.chunks.items():
                chunk_size = sum(len(p) for p in chunk['propositions'])
                if chunk_size + len(proposition) > 1800:
                    if self.print_logging:
                        logger.info(f"[AgenticChunker] Chunk {chunk_id} ({chunk_size} chars) would exceed 1800, skipping")
                else:
                    available_chunks[chunk_id] = chunk
            
            # If NO chunks are available (all oversized), force new chunk
            if not available_chunks:
                if self.print_logging:
                    logger.info(f"[AgenticChunker] All chunks oversized, forcing NEW_CHUNK")
                return None
        else:
            available_chunks = {}
        
        # H4 BAŞLIK KONTROLÜ: H4 başlıklar (####) MUTLAKA yeni chunk olmalı
        # Bu algoritmalar, yöntemler gibi farklı konuları ayırır
        is_h4_header = bool(re.match(r'^####\s+', proposition.strip()))
        if is_h4_header:
            if self.print_logging:
                logger.info(f"[AgenticChunker] H4 header detected, forcing NEW_CHUNK")
            return None  # Force new chunk
        
        # ✅ Sadece available_chunks için outline göster (oversized olanlar hariç)
        current_outline = self._get_chunk_outline(available_chunks)
        
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
        
        # ✅ PYDANTIC AI: Try structured output first
        if PYDANTIC_AI_AVAILABLE:
            try:
                # 1. Define the async function
                async def run_pydantic_agent():
                    provider = OllamaProvider(base_url=settings.ollama_base_url)
                    agent = Agent(
                        model="ollama:qwen2.5:3b",
                        result_type=ChunkDecision,
                        system_prompt="""You are a document chunking expert.
Decide if content belongs to existing chunk or needs new one.
RULES:
1. Group semantically related content
2. Different topics = NEW_CHUNK
3. H4/H5 headers = NEW_CHUNK
4. Max 1800 chars per chunk
Respond with action, chunk_id (if existing), confidence, reasoning.""",
                        retries=2,
                    )
                    return await agent.run(
                        f"""Chunks:\n{current_outline}\n\nNew content:\n{proposition[:500]}\n\nWhich chunk?"""
                    )

                # 2. Robust Execution Logic (Fixes RuntimeWarning)
                def _run_in_thread():
                    """Helper to run async code in a clean thread with asyncio.run"""
                    return asyncio.run(run_pydantic_agent())

                try:
                    # Check if we are already in an async loop
                    current_loop = asyncio.get_running_loop()
                except RuntimeError:
                    current_loop = None

                if current_loop and current_loop.is_running():
                    # ASYNC CONTEXT: Run in thread pool to avoid blocking/nesting errors
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(_run_in_thread)
                        result = future.result(timeout=20.0)
                else:
                    # SYNC CONTEXT: Run directly
                    result = asyncio.run(run_pydantic_agent())
                
                # 3. Process Result
                decision: ChunkDecision = result.data
                
                if self.print_logging:
                    logger.info(
                        f"[PydanticAI] {decision.action} "
                        f"(conf: {decision.confidence:.2f}) - {decision.reasoning[:50] if decision.reasoning else 'N/A'}"
                    )
                
                if decision.action == "NEW_CHUNK":
                    return None
                
                if decision.chunk_id and decision.chunk_id in available_chunks:
                    return decision.chunk_id
                
                return None
            
            except Exception as e:
                # Silent fail to fallback
                if self.print_logging:
                    logger.debug(f"[PydanticAI] Fallback triggered: {e}")
        
        # ⚠️ FALLBACK: Original string parsing (if Pydantic AI unavailable or fails)
        try:
            result = (PROMPT | self.llm).invoke({
                "current_outline": current_outline,
                "proposition": proposition[:500]
            })
            
            response = result.content.strip()
            
            if response == "NEW_CHUNK" or response.lower() == "new_chunk":
                return None
            
            for chunk_id in available_chunks.keys():
                if chunk_id in response:
                    return chunk_id
            
            if len(response) == self.id_truncate_limit and response in available_chunks:
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
            raw_content = "\n\n".join(chunk['propositions'])
            
            # ✅ YENİ: Header Injection - Chunk başına metadata bilgisi ekle
            header_parts = []
            if chunk.get('section_h1'):
                header_parts.append(f"Bölüm: {chunk['section_h1']}")
            if chunk.get('section_h2'):
                header_parts.append(f"Alt Bölüm: {chunk['section_h2']}")
            
            # Eğer header varsa, içeriğin başına ekle
            content = raw_content
            if header_parts:
                header_text = " | ".join(header_parts)
                content = f"[{header_text}]\n\n{raw_content}"
            
            # Check for images in combined content
            has_images = bool(re.search(r'!\[[^\]]*\]\([^)]+\)', content))
            
            # ✅ YENİ: Cross-references ve table count
            cross_refs = self._find_cross_references(content)
            table_count = self._count_tables(content)
            
            # ✅ YENİ - Daha detaylı log
            first_line = content.split('\n')[0][:60] if content else ''
            refs_str = f" | Refs: {cross_refs}" if cross_refs else ""
            tables_str = f" | Tables: {table_count}" if table_count > 0 else ""
            logger.info(
                f"[CHUNK {chunk_id}] "
                f"Title: '{chunk['title'][:50]}' | "  # ✅ FIX: Increased from 40 to 50
                f"Size: {len(content)} chars | "
                f"Images: {has_images}{refs_str}{tables_str} | "
                f"First: '{first_line}...'"
            )
            
            doc = Document(
                page_content=content,  # ✅ Artık header içeriyor
                metadata={
                    "source": self.source_name,
                    "chunk_id": chunk_id,
                    "title": chunk['title'],
                    "summary": chunk['summary'],
                    "chunk_index": chunk['chunk_index'],
                    "has_images": has_images,
                    "section_h1": chunk.get('section_h1'),
                    "section_h2": chunk.get('section_h2'),
                    # ✅ FIX: ChromaDB doesn't support list - convert to comma-separated string
                    "cross_references": ", ".join(cross_refs) if cross_refs else "",
                    "table_count": table_count,
                }
            )
            documents.append(doc)
        
        return documents
    
    def _find_cross_references(self, content: str) -> List[str]:
        """Find figure/table references in content (TR/EN support)."""
        if not content:
            return []
        try:
            patterns = [
                r'Şekil\s*\d+', r'Tablo\s*\d+',
                r'Figure\s*\d+', r'Table\s*\d+',
                r'Grafik\s*\d+', r'Chart\s*\d+',
            ]
            refs = []
            for pattern in patterns:
                refs.extend(re.findall(pattern, content, re.IGNORECASE))
            # Deduplicate
            seen = set()
            unique = []
            for ref in refs:
                key = ref.strip().lower()
                if key not in seen:
                    seen.add(key)
                    unique.append(ref.strip().title())
            return unique
        except Exception:
            return []
    
    def _count_tables(self, content: str) -> int:
        """Count markdown tables in content."""
        if not content:
            return 0
        try:
            # Count separator rows (|---|---|)
            return len(re.findall(r'^\s*\|[\s\-:]+\|', content, re.MULTILINE))
        except Exception:
            return 0
    
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


# ==========================================
# ✅ MAIN EXTRACTION FUNCTION - FIXED
# ==========================================

def extract_propositions_from_markdown(markdown_text: str) -> List[Dict[str, Any]]:
    """
    Extract propositions with section tracking (Hierarchical Metadata).
    
    Returns list of dicts: {'text': str, 'section_h1': str, 'section_h2': str}
    """
    if not markdown_text:
        return []
    
    text = markdown_text.replace("\r\n", "\n").replace("\r", "\n")
    
    # Track current section headers
    h1_pattern = re.compile(r'^#\s+(.+)$', re.MULTILINE)
    h2_pattern = re.compile(r'^##\s+(.+)$', re.MULTILINE)
    
    # Extract all header positions
    h1_headers = {m.start(): m.group(1).strip() for m in h1_pattern.finditer(text)}
    h2_headers = {m.start(): m.group(1).strip() for m in h2_pattern.finditer(text)}
    
    # ✅ FIX: Use proper extraction (not simple paragraph split)
    raw_props = _extract_propositions_with_headers(text)
    
    propositions_with_metadata = []
    
    current_pos = 0
    for prop in raw_props:
        clean_prop = prop.strip()
        if not clean_prop:
            continue
            
        found_pos = text.find(clean_prop[:50], current_pos)
        if found_pos == -1:
            found_pos = current_pos
        else:
            current_pos = found_pos
        
        # Find nearest H1 before this position
        section_h1 = None
        best_h1_pos = -1
        for pos, title in h1_headers.items():
            if pos < found_pos and pos > best_h1_pos:
                section_h1 = title
                best_h1_pos = pos
        
        # Find nearest H2 before this position
        section_h2 = None
        best_h2_pos = -1
        for pos, title in h2_headers.items():
            if pos < found_pos and pos > best_h2_pos:
                section_h2 = title
                best_h2_pos = pos
        
        # H2 is only valid if it's after the current H1 (hierarchical)
        if best_h2_pos < best_h1_pos:
            section_h2 = None
            
        propositions_with_metadata.append({
            'text': prop,
            'section_h1': section_h1,
            'section_h2': section_h2
        })
        
        current_pos += len(clean_prop)
    
    return propositions_with_metadata


# ==========================================
# ✅ NEW HELPER FUNCTIONS - PROPER LOGIC
# ==========================================

def _extract_propositions_with_headers(text: str) -> List[str]:
    """
    Extract propositions using major header splitting + image attachment.
    
    This is the PROPER extraction logic (was unreachable before).
    """
    # Find major headers (H1-H3)
    major_header_pattern = re.compile(r'^(#{1,3})\s+.+', re.MULTILINE)
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
        
        # Attach images to their H5/H6 headers
        processed_section = _attach_images_to_headers(section_text)
        
        # Split at H4 headers (algorithms, methods)
        h4_pattern = re.compile(r'^####\s+.+', re.MULTILINE)
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
    """
    lines = section_text.split("\n")
    result_lines = []
    
    h5h6_re = re.compile(r'^#{5,6}\s+.+$')
    image_re = re.compile(r'^!\[[^\]]*\]\([^)]+\)\s*$')
    major_header_re = re.compile(r'^#{1,4}\s+.+$')
    
    i = 0
    consumed_images = set()
    
    while i < len(lines):
        line = lines[i]
        
        if h5h6_re.match(line.strip()):
            header_section = [line]
            i += 1
            
            while i < len(lines):
                next_line = lines[i]
                
                if h5h6_re.match(next_line.strip()) or major_header_re.match(next_line.strip()):
                    break
                
                if i not in consumed_images:
                    header_section.append(next_line)
                i += 1
            
            # ✅ Dynamic lookahead - up to 10 lines
            lookahead_start = i
            lookahead_end = min(i + 10, len(lines))
            
            for j in range(lookahead_start, lookahead_end):
                line_content = lines[j].strip()
                
                if not line_content:
                    break
                
                if h5h6_re.match(line_content) or major_header_re.match(line_content):
                    break
                
                if image_re.match(line_content):
                    if header_section and header_section[-1].strip():
                        header_section.append("")
                    header_section.append(lines[j])
                    consumed_images.add(j)
            
            result_lines.extend(header_section)
        else:
            if i not in consumed_images:
                result_lines.append(line)
            i += 1
    
    return "\n".join(result_lines)


def _smart_split_with_images(text: str) -> List[str]:
    """Split text intelligently, keeping images with nearby text."""
    if len(text) <= 2500:
        return [text]
    
    return _split_large_section_preserve_images(text)


def _split_large_section_preserve_images(text: str) -> List[str]:
    """
    Split large sections while preserving image attachments.
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
        
        if current_size + para_size > 2000 and current_chunk:
            chunks.append("\n\n".join(current_chunk))
            current_chunk = [para]
            current_size = para_size
        else:
            current_chunk.append(para)
            current_size += para_size
    
    if current_chunk:
        chunks.append("\n\n".join(current_chunk))
    
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
        
        text_only = re.sub(r'!\[[^\]]*\]\([^)]+\)', '', prop)
        text_only = re.sub(r'^#{1,6}\s+.+$', '', text_only, flags=re.MULTILINE).strip()
        
        has_image = bool(image_re.search(prop))
        has_header = bool(header_re.search(prop))
        
        if has_image or has_header:
            merged.append(prop)
        elif len(text_only) < min_chars and merged:
            merged[-1] = merged[-1] + "\n\n" + prop
        else:
            merged.append(prop)
    
    return merged


# ==========================================
# MAIN ENTRY POINT
# ==========================================

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
    
    # Extract propositions from text (now returns dicts with metadata)
    propositions_data = extract_propositions_from_markdown(text)
    logger.info(f"[AgenticChunker] Extracted {len(propositions_data)} propositions")
    
    if not propositions_data:
        # Fallback: return single document
        return [Document(
            page_content=text,
            metadata={"source": source_name, "has_images": bool(re.search(r'!\[[^\]]*\]\([^)]+\)', text))}
        )]
    
    # Use agentic chunker to group propositions
    chunker = AgenticChunker(source_name=source_name)
    
    for prop_item in propositions_data:
        # Pass metadata to add_proposition
        chunker.add_proposition(
            prop_item['text'],
            section_h1=prop_item['section_h1'],
            section_h2=prop_item['section_h2']
        )
    
    documents = chunker.get_documents()
    logger.info(f"[AgenticChunker] Created {len(documents)} semantic chunks")
    
    # Debug output
    if os.getenv("DEBUG_CHUNKS", "").lower() in ("1", "true", "yes"):
        chunker.pretty_print_chunks()
    
    return documents