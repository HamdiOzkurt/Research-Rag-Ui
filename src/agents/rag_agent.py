"""
RAG Agent with LangChain - Document Processing & Semantic Search
Based on: https://docs.langchain.com/oss/python/langchain/rag

Features:
- PDF, DOCX, Audio, Image processing
- Vector store (Supabase pgvector)
- Semantic search with embeddings
- LangGraph agent with retrieval tool
"""

import os
import logging
from typing import List, Optional, Literal, Union
from pathlib import Path

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langgraph.prebuilt import create_react_agent
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
try:
    from langchain_community.document_loaders import (
        PyPDFLoader,
        Docx2txtLoader,
        CSVLoader,
    )
except ImportError:
    # Fallback: manual loaders
    PyPDFLoader = None
    Docx2txtLoader = None
    CSVLoader = None

import pymupdf4llm  # High-quality PDF to Markdown

from ..config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============ LANGSMITH ============
def setup_langsmith():
    """LangSmith tracing for RAG agent"""
    tracing_enabled = os.getenv("LANGSMITH_TRACING", "").strip().lower() in {"1", "true", "yes", "on"}
    if not tracing_enabled:
        return False
    if not settings.langsmith_api_key:
        return False

    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key

    # Standard LangChain vars
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "ai-research-rag-agent")

    logger.info("[LANGSMITH] Tracing aktif")
    return True


# ============ VECTOR STORE SETUP ============

# Initialize embeddings (Ollama - Local & Free)
# Override via env: OLLAMA_EMBED_MODEL
_embeddings = OllamaEmbeddings(
    model=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text:latest"),
    base_url=settings.ollama_base_url,
)

# Initialize vector store (in-memory for now, Supabase later)
_vector_store = InMemoryVectorStore(_embeddings)

# Text splitter for chunking
_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True,
)


# ============ DOCUMENT PROCESSING TOOLS ============

@tool
def load_pdf(file_path: str) -> str:
    """
    Load and extract text from a PDF file with images using PyMuPDF4LLM.
    Extracts text, tables, and saves page images.
    
    Args:
        file_path: Absolute path to PDF file
        
    Returns:
        Extracted text content (Markdown)
    """
    # 1. Try PyMuPDF4LLM (Fast, extracts images, good Markdown)
    try:
        import pymupdf4llm
        
        logger.info(f"[PDF] Processing {Path(file_path).name} with PyMuPDF4LLM...")
        
        # Create images folder
        images_folder = Path(file_path).parent / f"{Path(file_path).stem}_images"
        images_folder.mkdir(exist_ok=True)
        
        # Convert PDF to Markdown with images
        md_text = pymupdf4llm.to_markdown(
            file_path,
            write_images=True,
            image_path=str(images_folder),
            image_format="png"
        )
        
        logger.info(f"[PDF] ✅ Converted {len(md_text)} chars with images (PyMuPDF4LLM)")
        
        # Save debug file
        debug_path = Path(file_path).parent / f"{Path(file_path).stem}_pymupdf_debug.md"
        debug_path.write_text(md_text, encoding="utf-8")
        logger.info(f"[PDF] 📝 Markdown saved to: {debug_path}")
        logger.info(f"[PDF] 🖼️ Images saved to: {images_folder}")
        
        if len(md_text) >= 100:
            return md_text
            
    except Exception as e:
        logger.warning(f"[PDF] PyMuPDF4LLM failed ({e}), falling back to Unstructured...")
    
    # 2. Fallback: Unstructured
    try:
        from unstructured.partition.pdf import partition_pdf
        
        logger.info(f"[PDF] Processing {Path(file_path).name} with Unstructured (fallback)...")
        
        elements = partition_pdf(filename=file_path, strategy="fast")
        
        text_parts = []
        for el in elements:
            element_text = str(el)
            element_type = el.category if hasattr(el, 'category') else 'Unknown'
            
            if element_type == 'Title':
                text_parts.append(f"# {element_text}")
            elif element_type == 'Header':
                text_parts.append(f"## {element_text}")
            elif element_type == 'Table':
                text_parts.append(f"\n{element_text}\n")
            else:
                text_parts.append(element_text)
        
        text = "\n\n".join(text_parts)
        logger.info(f"[PDF] ✅ Converted {len(text)} chars from {len(elements)} elements (Unstructured OCR)")
        
        debug_path = Path(file_path).parent / f"{Path(file_path).stem}_unstructured_debug.md"
        debug_path.write_text(text, encoding="utf-8")
        logger.info(f"[PDF] 📝 Unstructured output saved to: {debug_path}")
        
        if len(text) >= 50:
            return text
            
    except Exception as e2:
        logger.warning(f"[PDF] Unstructured failed ({e2}), falling back to PyPDF...")

    # 3. Last Fallback: PyPDF (Simple text extraction)
    try:
        from pypdf import PdfReader
        reader = PdfReader(file_path)
        text = "\n\n".join([page.extract_text() for page in reader.pages])
        logger.info(f"[PDF] Loaded {len(reader.pages)} pages, {len(text)} chars (PyPDF - last resort)")
        
        debug_path = Path(file_path).parent / f"{Path(file_path).stem}_pypdf_debug.txt"
        debug_path.write_text(text, encoding="utf-8")
        logger.info(f"[PDF] 📝 PyPDF output saved to: {debug_path}")
        
        return text
    except Exception as e3:
        logger.error(f"[PDF] All methods failed: {e3}")
        return f"Error loading PDF: {str(e3)}"


@tool
def load_docx(file_path: str) -> str:
    """
    Load and extract text from a DOCX file using Unstructured.
    
    Args:
        file_path: Absolute path to DOCX file
        
    Returns:
        Extracted text content (Markdown)
    """
    try:
        from unstructured.partition.docx import partition_docx
        
        logger.info(f"[DOCX] Processing {Path(file_path).name} with Unstructured...")
        
        elements = partition_docx(filename=file_path)
        text = "\n\n".join([str(el) for el in elements])
        
        logger.info(f"[DOCX] ✅ Converted {len(text)} chars (Unstructured)")
        
        # DEBUG: Save to file to inspect
        debug_path = Path(file_path).parent / f"{Path(file_path).stem}_markdown_debug.md"
        debug_path.write_text(text, encoding="utf-8")
        logger.info(f"[DOCX] 📝 Markdown saved to: {debug_path}")
        
        return text
        
    except Exception as e:
        logger.warning(f"[DOCX] Unstructured failed ({e}), falling back to Docx2txt...")
        
        # Fallback: Docx2txtLoader
        try:
            if Docx2txtLoader:
                loader = Docx2txtLoader(file_path)
                docs = loader.load()
                text = "\n\n".join([doc.page_content for doc in docs])
                return text
            else:
                import docx2txt
                return docx2txt.process(file_path)
        except Exception as e2:
            logger.error(f"[DOCX] Error loading {file_path}: {e2}")
            return f"Error loading DOCX: {str(e)}"


@tool
def load_csv(file_path: str) -> str:
    """
    Load and extract data from a CSV file.
    
    Args:
        file_path: Absolute path to CSV file
        
    Returns:
        CSV data as formatted text
    """
    try:
        if CSVLoader:
            loader = CSVLoader(file_path)
            docs = loader.load()
            text = "\n\n".join([doc.page_content for doc in docs])
            logger.info(f"[CSV] Loaded {len(docs)} rows from {Path(file_path).name}")
            return text
        else:
            # Fallback: csv module
            import csv
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = [str(dict(row)) for row in reader]
                text = "\n\n".join(rows)
                logger.info(f"[CSV] Loaded {len(rows)} rows from {Path(file_path).name}")
                return text
    except Exception as e:
        logger.error(f"[CSV] Error loading {file_path}: {e}")
        return f"Error loading CSV: {str(e)}"


@tool
async def transcribe_audio(file_path: str, language: str = "tr") -> str:
    """
    Transcribe audio file locally.
    
    Args:
        file_path: Absolute path to audio file (mp3, wav, m4a)
        language: Language code (default: tr for Turkish)
        
    Returns:
        Transcribed text
    """
    try:
        # Local STT recommendation: faster-whisper
        # Install (optional): pip install faster-whisper ffmpeg-python
        try:
            from faster_whisper import WhisperModel  # type: ignore
        except Exception:
            return (
                "Local STT için `faster-whisper` önerilir.\n"
                "Kurulum: `pip install faster-whisper ffmpeg-python` ve sistemde ffmpeg olmalı.\n"
                "Alternatif: whisper.cpp (CLI)."
            )

        model_name = os.getenv("WHISPER_MODEL", "small")
        device = os.getenv("WHISPER_DEVICE", "cpu")
        compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "int8")

        whisper_model = WhisperModel(model_name, device=device, compute_type=compute_type)
        segments, info = whisper_model.transcribe(file_path, language=language)
        text = "".join(seg.text for seg in segments).strip()
        logger.info(f"[AUDIO] Transcribed {Path(file_path).name} ({info.language}) - {len(text)} chars")
        return text or "(Boş transkripsiyon)"
    except Exception as e:
        logger.error(f"[AUDIO] Error transcribing {file_path}: {e}")
        return f"Error transcribing audio: {str(e)}"


@tool
async def analyze_image(file_path: str, prompt: str = "Bu görseli detaylı Türkçe açıkla") -> str:
    """
    Analyze image locally via Ollama vision model.
    
    Args:
        file_path: Absolute path to image file
        prompt: Analysis prompt (default: Turkish description)
        
    Returns:
        Image analysis text
    """
    try:
        import base64
        import httpx

        with open(file_path, "rb") as image_file:
            image_b64 = base64.b64encode(image_file.read()).decode("utf-8")

        vision_model = os.getenv("OLLAMA_VISION_MODEL", "llava:latest")

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{settings.ollama_base_url}/api/chat",
                json={
                    "model": vision_model,
                    "messages": [
                        {"role": "user", "content": prompt, "images": [image_b64]}
                    ],
                    "stream": False,
                },
            )

        if response.status_code != 200:
            return f"Ollama Vision error: HTTP {response.status_code} - {response.text}"

        data = response.json()
        text = (data.get("message", {}) or {}).get("content", "")
        logger.info(f"[IMAGE] Analyzed {Path(file_path).name} with {vision_model}")
        return text or "(Boş analiz)"
    except Exception as e:
        logger.error(f"[IMAGE] Error analyzing {file_path}: {e}")
        return f"Error analyzing image: {str(e)}"


# ============ RAG CORE FUNCTIONS ============

def ingest_documents(documents: List[Document], source_name: str = "upload") -> int:
    """
    Ingest documents into vector store.
    
    Args:
        documents: List of Document objects
        source_name: Source identifier for metadata
        
    Returns:
        Number of chunks added
    """
    # Split documents into chunks
    all_splits = _text_splitter.split_documents(documents)
    
    # Add source metadata
    for split in all_splits:
        split.metadata["source"] = source_name
    
    # Add to vector store
    document_ids = _vector_store.add_documents(documents=all_splits)
    
    logger.info(f"[RAG] Ingested {len(all_splits)} chunks from {source_name}")
    return len(all_splits)


def ingest_text(text: str, source_name: str = "upload") -> int:
    """
    Ingest raw text into vector store with Markdown-aware splitting.
    
    Args:
        text: Raw text content (Markdown preferred)
        source_name: Source identifier
        
    Returns:
        Number of chunks added
    """
    # 1. Markdown Header Splitting (Structure aware)
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    md_header_splits = markdown_splitter.split_text(text)

    # 2. Recursive Character Splitting (Size aware)
    # Apply recursive splitter on the header splits to ensure chunk size limits
    final_splits = _text_splitter.split_documents(md_header_splits)

    # Add source metadata
    for split in final_splits:
        split.metadata["source"] = source_name

    # Add to vector store
    if final_splits:
        _vector_store.add_documents(documents=final_splits)
    
    logger.info(f"[RAG] Ingested {len(final_splits)} chunks from {source_name} (Markdown optimized)")
    return len(final_splits)


# ============ RETRIEVAL TOOL ============

from pydantic import BaseModel, Field

class RetrieveContextInput(BaseModel):
    """Input schema for retrieve_context tool"""
    query: str = Field(description="Search query to find relevant document chunks")
    # Groq API strict validation fix: Accept string, convert to int internally
    top_k: str = Field(default="3", description="Number of top results to return (e.g. '3')")

@tool(args_schema=RetrieveContextInput, response_format="content_and_artifact")
def retrieve_context(query: str, top_k: str = "3"):
    """
    Retrieve relevant context from ingested documents using semantic search.
    AUTOMATICALLY analyzes images found in the context using vision models.
    
    Args:
        query: Search query
        top_k: Number of top results to return (default: "3")
        
    Returns:
        Retrieved context as formatted text with source metadata AND image analysis
    """
    # Ensure top_k is an integer
    try:
        k_val = int(top_k)
    except (ValueError, TypeError):
        k_val = 3

    retrieved_docs = _vector_store.similarity_search(query, k=k_val)
    
    if not retrieved_docs:
        return "No relevant documents found.", []
    
    # Extract image paths from markdown content
    import re
    image_analyses = []
    
    for doc in retrieved_docs:
        # Find all markdown image references: ![alt](path)
        img_matches = re.findall(r'!\[([^\]]*)\]\(([^)]+)\)', doc.page_content)
        
        for alt_text, img_path in img_matches:
            # Convert relative path to absolute
            full_path = Path("uploads") / img_path if not Path(img_path).is_absolute() else Path(img_path)
            
            if full_path.exists():
                try:
                    logger.info(f"[RAG] 🖼️ Analyzing image: {full_path.name}")
                    
                    # Analyze image with Groq vision
                    analysis = analyze_image.invoke(str(full_path))
                    
                    # Create image URL for frontend
                    image_url = f"http://127.0.0.1:8000/{img_path}"
                    
                    image_analyses.append(f"""
🖼️ IMAGE FOUND: {full_path.name}
**Image URL for display**: {image_url}
**Markdown to include in response**: ![{alt_text or 'Chart/Graph'}]({image_url})

**Vision Analysis**:
{analysis}
---
""")
                except Exception as e:
                    logger.error(f"[RAG] Failed to analyze image {full_path}: {e}")
    
    # Replace image paths in chunks with full URLs BEFORE sending to LLM
    processed_docs = []
    for doc in retrieved_docs:
        content = doc.page_content
        # Replace all image markdown with full URLs
        content = re.sub(
            r'!\[([^\]]*)\]\(([^)]+)\)',
            lambda m: f'![{m.group(1)}](http://127.0.0.1:8000/{m.group(2)})',
            content
        )
        processed_docs.append((doc.metadata.get('source', 'unknown'), content))
    
    # Combine text context with image analyses
    text_context = "\n\n".join(
        (f"📄 Source: {source}\n{content}")
        for source, content in processed_docs
    )
    
    if image_analyses:
        combined_context = text_context + "\n\n" + "\n".join(image_analyses)
        logger.info(f"[RAG] Retrieved {len(retrieved_docs)} chunks + {len(image_analyses)} image analyses")
    else:
        combined_context = text_context
        logger.info(f"[RAG] Retrieved {len(retrieved_docs)} chunks (no images found)")
    
    return combined_context, retrieved_docs


# ============ RAG AGENT ============

RAG_SYSTEM_PROMPT = """You are a RAG agent with vision. Documents are pre-loaded.

WORKFLOW:
1. Call retrieve_context with user's query keywords
2. Analyze returned text + image analyses  
3. Generate answer using retrieved information

IMAGE HANDLING:
- When you see "Markdown to include in response" in context, COPY that exact markdown line into your answer
- Example: If context says `![Chart](http://://url)`, you MUST include that in your response
- Always show images when discussing graphs/charts/figures

TOOL: retrieve_context(query, top_k="3")
- Use for all document questions
- Automatically analyzes images in chunks
- Returns text + vision analysis

RESPONSE RULES:
- Use markdown formatting
- When context includes "Markdown to include in response: ![...](...)", copy that line into your answer
- Show images when discussing graphs/charts
"""


def _get_rag_model():
    """Get LLM model for RAG agent - use Groq for critical RAG tasks"""
    # RAG kritik bir görev -> Groq kullan (güvenilir, hosted)
    if settings.groq_api_key:
        from langchain_groq import ChatGroq
        logger.info("[RAG] Using Groq (llama-3.3-70b-versatile) for critical RAG tasks")
        return ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=settings.groq_api_key,
            temperature=0
        )
    # Fallback: settings default
    logger.info(f"[RAG] Fallback to default model: {settings.default_model}")
    provider, model_name = settings.get_model_provider(settings.default_model)
    return init_chat_model(model=model_name, model_provider=provider, temperature=0)


# Create RAG agent
setup_langsmith()

_rag_tools = [
    load_pdf,
    load_docx,
    load_csv,
    transcribe_audio,
    analyze_image,
    retrieve_context
]

# Lazy init graph - model created on demand
_graph_instance = None

def _get_graph():
    """Lazy load RAG graph with fresh model"""
    global _graph_instance
    if _graph_instance is not None:
        return _graph_instance
    
    model = _get_rag_model()
    logger.info(f"[RAG] Using model: {type(model).__name__}")
    
    # Use prompt parameter for system message (LangGraph 0.2+)
    _graph_instance = create_react_agent(
        model, 
        _rag_tools,
        prompt=RAG_SYSTEM_PROMPT
    )
    
    return _graph_instance

# Export graph via property with image extraction
class _GraphProxy:
    def __getattr__(self, name):
        # Intercept ainvoke to extract images
        if name == 'ainvoke':
            async def ainvoke_with_images(input_data, *args, **kwargs):
                result = await _get_graph().ainvoke(input_data, *args, **kwargs)
                
                # Extract images from the conversation
                images = []
                if isinstance(result, dict) and "messages" in result:
                    for msg in result["messages"]:
                        if hasattr(msg, "content") and isinstance(msg.content, str):
                            # Find image references
                            import re
                            img_matches = re.findall(
                                r'!\[([^\]]*)\]\((http://127\.0\.0\.1:8000/[^)]+)\)', 
                                msg.content
                            )
                            for alt, url in img_matches:
                                if not any(img['url'] == url for img in images):
                                    images.append({
                                        "url": url,
                                        "alt": alt or "Chart/Graph",
                                    })
                
                # Add images to result
                if isinstance(result, dict):
                    result["images"] = images
                
                return result
            return ainvoke_with_images
        
        return getattr(_get_graph(), name)

graph = _GraphProxy()
