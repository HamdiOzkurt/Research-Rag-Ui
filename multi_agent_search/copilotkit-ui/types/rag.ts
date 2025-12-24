/**
 * TypeScript types for HITL RAG workflow
 */

export interface RetrievedChunk {
    chunk_id: string;
    content: string;
    title: string;
    summary: string;
    confidence: number;
    source: string;
    has_images: boolean;
    section_h1?: string;
    section_h2?: string;
}

export interface RAGState {
    retrieved_chunks: RetrievedChunk[];
    approved_chunk_ids: string[];
    awaiting_approval: boolean;
    is_synthesizing: boolean;
    current_query?: string;
}
