"""
Type-safe models for RAG system using Pydantic.

These models provide:
- Runtime validation
- Type safety
- Auto-completion in IDEs
- JSON schema generation
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, Literal
from datetime import datetime


# ==================== CHUNK MODELS ====================

class ChunkMetadata(BaseModel):
    """Metadata for a single chunk - Type-safe!"""
    
    chunk_id: str = Field(..., description="Unique chunk identifier")
    title: str = Field(..., min_length=5, max_length=200, description="Chunk title")
    summary: str = Field(..., description="Brief summary of chunk content")
    chunk_index: int = Field(..., ge=0, description="Sequential chunk number")
    
    # Content metadata
    has_images: bool = Field(default=False, description="Contains images")
    cross_references: list[str] = Field(default_factory=list, description="Figure/Table references")
    table_count: int = Field(default=0, ge=0, description="Number of tables")
    
    # Section metadata
    section_h1: Optional[str] = Field(None, description="Main section heading")
    section_h2: Optional[str] = Field(None, description="Sub-section heading")
    
    # Source metadata
    source: str = Field(..., description="Source document name")
    
    class Config:
        frozen = False  # Allow updates during processing
        extra = "forbid"  # Reject unknown fields


class ChunkContent(BaseModel):
    """Full chunk with content and metadata"""
    
    metadata: ChunkMetadata
    content: str = Field(..., min_length=1, description="Chunk text content")
    propositions: list[str] = Field(default_factory=list, description="Individual propositions")
    
    @field_validator('content')
    @classmethod
    def content_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Content cannot be empty or whitespace")
        return v


# ==================== AGENTIC CHUNKER MODELS ====================

class ChunkDecision(BaseModel):
    """LLM decision for chunk placement - Structured output!"""
    
    action: Literal["NEW_CHUNK", "EXISTING_CHUNK"] = Field(
        ..., 
        description="Whether to create new chunk or use existing"
    )
    chunk_id: Optional[str] = Field(
        None, 
        description="ID of existing chunk (if action=EXISTING_CHUNK)"
    )
    confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Confidence in decision (0-1)"
    )
    reasoning: str = Field(
        default="",
        description="Brief explanation of decision"
    )
    
    @field_validator('chunk_id')
    @classmethod
    def validate_chunk_id(cls, v: Optional[str], info) -> Optional[str]:
        """Ensure chunk_id is provided when action is EXISTING_CHUNK"""
        if info.data.get('action') == 'EXISTING_CHUNK' and not v:
            raise ValueError("chunk_id required when action=EXISTING_CHUNK")
        return v


class ChunkSummary(BaseModel):
    """LLM-generated chunk summary"""
    
    summary: str = Field(..., min_length=10, max_length=500)
    key_topics: list[str] = Field(default_factory=list, max_length=5)


class ChunkTitle(BaseModel):
    """LLM-generated chunk title"""
    
    title: str = Field(..., min_length=5, max_length=100)


# ==================== RAG RETRIEVAL MODELS ====================

class SearchParams(BaseModel):
    """Type-safe search parameters"""
    
    query: str = Field(..., min_length=3, max_length=1000)
    top_k: int = Field(default=3, ge=1, le=20, description="Number of results")
    similarity_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score"
    )
    search_type: Literal["semantic", "hybrid", "bm25"] = Field(
        default="hybrid",
        description="Search algorithm"
    )


class RetrievedChunk(BaseModel):
    """Single retrieved chunk with similarity score"""
    
    chunk_id: str
    content: str
    similarity: float = Field(..., ge=0.0, le=1.0)
    metadata: dict = Field(default_factory=dict)
    
    # Optional enriched fields
    document_title: Optional[str] = None
    document_source: Optional[str] = None
    chunk_index: Optional[int] = None


class RetrievalResult(BaseModel):
    """Complete retrieval result - Type-safe!"""
    
    query: str
    chunks: list[RetrievedChunk]
    context_text: str = Field(..., description="Formatted context for LLM")
    images: list[str] = Field(default_factory=list, description="Image paths")
    
    # Stats
    total_chunks_found: int = Field(..., ge=0)
    search_type: str
    timestamp: datetime = Field(default_factory=datetime.now)


# ==================== VISION MODELS ====================

class VisionAnalysis(BaseModel):
    """Vision model analysis result"""
    
    image_path: str
    description: str = Field(..., min_length=10)
    model_used: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    detected_elements: list[str] = Field(default_factory=list)


# ==================== ERROR MODELS ====================

class RAGError(BaseModel):
    """Structured error information"""
    
    error_type: str
    message: str
    details: Optional[dict] = None
    timestamp: datetime = Field(default_factory=datetime.now)
