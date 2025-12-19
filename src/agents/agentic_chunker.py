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
    """Get LLM for agentic chunking - uses local Ollama (llama3.2:3b)"""
    
    # Use local Ollama - llama3.2:3b (2GB, optimal for semantic chunking)
    try:
        from langchain_ollama import ChatOllama
        logger.info("[AgenticChunker] Using Ollama llama3.2:3b (local)")
        return ChatOllama(
            model="llama3.2:3b",
            base_url=settings.ollama_base_url,
            temperature=0,
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
            ("system", """You are organizing document chunks. A new piece of content was added to a chunk.
Generate a brief 1-sentence summary of what this chunk is about.

The summary should:
- Describe the main topic
- Be in the same language as the content (Turkish or English)
- Anticipate generalization (e.g., "makine öğrenmesi algoritmaları" not just "Naive Bayes")

Only respond with the summary, nothing else."""),
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
            ("system", """Generate a brief 2-5 word title for this document chunk.
Be in the same language as the content.
Only respond with the title, nothing else."""),
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
            ("system", """You are organizing document chunks. Create a brief 1-sentence summary for a new chunk.

The summary should:
- Describe what kind of content belongs here
- Be in the same language as the content
- Anticipate similar content that might be added

Only respond with the summary, nothing else."""),
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
            ("system", """Generate a brief 2-5 word title for this chunk.
Be in the same language as the content.
Only respond with the title, nothing else."""),
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
        
        current_outline = self._get_chunk_outline()
        
        PROMPT = ChatPromptTemplate.from_messages([
            ("system", """Determine if the new content should belong to an existing chunk.

Content should belong to a chunk if:
- They discuss the same topic or concept
- They are semantically related
- One provides context for the other

IMPORTANT for images: If the new content is an image (starts with ![), it should go with:
- The chunk that describes what the image shows
- The chunk about the same algorithm/concept shown in the image

If you find a matching chunk, respond with ONLY the chunk ID (e.g., "a1b2c3d4").
If no chunk matches, respond with "NEW_CHUNK"."""),
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
    Extract semantic propositions from markdown text.
    
    Splits by:
    1. Major headers (H1-H4)
    2. Images with surrounding context
    3. Paragraphs that are semantically complete
    """
    if not markdown_text:
        return []
    
    text = markdown_text.replace("\r\n", "\n").replace("\r", "\n")
    
    propositions = []
    
    # Split by major headers first
    header_pattern = re.compile(r'^(#{1,4})\s+(.+)$', re.MULTILINE)
    
    # Find all header positions
    headers = list(header_pattern.finditer(text))
    
    if not headers:
        # No headers - split by paragraphs and images
        return _split_by_paragraphs_and_images(text)
    
    # Process each section
    for i, match in enumerate(headers):
        start = match.start()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        
        section = text[start:end].strip()
        if section:
            # Further split section by images
            section_parts = _split_by_paragraphs_and_images(section)
            propositions.extend(section_parts)
    
    # Handle content before first header
    if headers and headers[0].start() > 0:
        pre_content = text[:headers[0].start()].strip()
        if pre_content:
            propositions.insert(0, pre_content)
    
    return [p for p in propositions if p.strip()]


def _split_by_paragraphs_and_images(text: str) -> List[str]:
    """Split text by paragraphs, keeping images with nearby context."""
    if not text:
        return []
    
    lines = text.split("\n")
    propositions = []
    current = []
    
    image_re = re.compile(r'^!\[[^\]]*\]\([^)]+\)\s*$')
    
    for line in lines:
        is_image = bool(image_re.match(line.strip()))
        is_empty = not line.strip()
        
        if is_image:
            # Image line - if we have accumulated content, add the image to it
            current.append(line)
        elif is_empty and current:
            # Empty line after content - might be a paragraph break
            # Check if current has substantial content
            text_content = "\n".join(current).strip()
            if len(text_content) > 50:  # Substantial content
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
    
    # Merge very short propositions with adjacent ones
    merged = []
    for prop in propositions:
        # Skip if just whitespace
        if not prop.strip():
            continue
        
        # If prop is very short and no image, merge with previous
        has_image = bool(re.search(r'!\[[^\]]*\]\([^)]+\)', prop))
        text_only = re.sub(r'!\[[^\]]*\]\([^)]+\)', '', prop).strip()
        
        if len(text_only) < 100 and not has_image and merged:
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
