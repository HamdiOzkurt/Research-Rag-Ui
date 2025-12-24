"use client";

import { useState, useEffect, useCallback } from "react";
import ChunkCard from "./ChunkCard";
import type { RAGState } from "@/types/rag";

interface Chunk {
    chunk_id: string;
    title: string;
    summary: string;
    content: string;
    source: string;
    metadata: {
        chunk_index?: number;
        section_h1?: string;
        section_h2?: string;
        section_h3?: string;
        has_images?: boolean;
        cross_references?: string;
        table_count?: number;
        page?: number;
        paragraph_index?: number;
        created_at?: string;
        updated_at?: string;
        chunk_type?: string;
        importance_score?: number;
        raw?: any;
    };
    images?: string[];
    relevance: number;
    tags: string[];
}

export default function ChunkExplorer({ ragState }: { ragState: RAGState | null }) {
    const [chunks, setChunks] = useState<Chunk[]>([]);
    const [selectedIds, setSelectedIds] = useState<string[]>([]);
    const [approvedIds, setApprovedIds] = useState<string[]>([]);
    const [searchQuery, setSearchQuery] = useState("");
    const [isLoading, setIsLoading] = useState(true);
    const [isApproving, setIsApproving] = useState(false); // ✅ Button loading state
    const [currentQuery, setCurrentQuery] = useState<string | null>(null);

    // ✅ Sync with RAG State (Agent Results)
    useEffect(() => {
        if (ragState?.retrieved_chunks && ragState.retrieved_chunks.length > 0) {
            console.log("🔄 Syncing UI with RAG Agent results:", ragState.retrieved_chunks.length);

            const agentChunks: Chunk[] = ragState.retrieved_chunks.map((rc: any) => ({
                chunk_id: rc.chunk_id,
                title: rc.title || "Başlıksız Bölüm",
                summary: rc.summary || rc.content.slice(0, 150) + "...",
                content: rc.content,
                source: rc.source,
                metadata: {
                    section_h1: rc.section_h1,
                    section_h2: rc.section_h2,
                    has_images: rc.has_images,
                    page: rc.metadata?.page,
                },
                images: rc.images || [],
                relevance: rc.confidence || 0.9,
                tags: ["Agent Found"]
            }));

            setChunks(prevAll => {
                const agentIds = new Set(agentChunks.map(c => c.chunk_id));
                const others = prevAll.filter(c => !agentIds.has(c.chunk_id));
                const cleanedOthers = others.map(c => ({ ...c, relevance: 0, tags: [] }));
                return [...agentChunks, ...cleanedOthers];
            });

        } else if (ragState && (!ragState.retrieved_chunks || ragState.retrieved_chunks.length === 0)) {
            // Case 2: Clean up
            setChunks(prevAll => {
                const hasAgentArtifacts = prevAll.some(c => c.relevance > 0);
                if (!hasAgentArtifacts) return prevAll;
                return prevAll.map(c => ({ ...c, relevance: 0, tags: [] }));
            });
            setApprovedIds([]);
            if (selectedIds.length > 0) setSelectedIds([]);
        }
    }, [ragState]);

    // Fetch chunks
    const fetchChunks = useCallback(async () => {
        try {
            const url = searchQuery
                ? `http://localhost:8000/api/rag/chunks?query=${encodeURIComponent(searchQuery)}`
                : "http://localhost:8000/api/rag/chunks";

            const res = await fetch(url);
            const data = await res.json();

            if (data.chunks) {
                setChunks(data.chunks);
                setCurrentQuery(data.current_query || (searchQuery ? `"${searchQuery}"` : null));
            }
        } catch (error) {
            console.error("Failed to fetch chunks:", error);
        } finally {
            setIsLoading(false);
        }
    }, [searchQuery]);

    // Initial load
    useEffect(() => {
        fetchChunks();
    }, []);

    // Search trigger
    useEffect(() => {
        if (!searchQuery) return;
        const timer = setTimeout(() => {
            fetchChunks();
        }, 500);
        return () => clearTimeout(timer);
    }, [searchQuery, fetchChunks]);

    // Filtering & Sorting
    const filteredChunks = chunks;
    const sortedChunks = [...filteredChunks].sort((a, b) => {
        // Selected first
        const aSelected = selectedIds.includes(a.chunk_id);
        const bSelected = selectedIds.includes(b.chunk_id);
        if (aSelected && !bSelected) return -1;
        if (!aSelected && bSelected) return 1;
        // Relevance
        if (a.relevance !== b.relevance) return b.relevance - a.relevance;
        // Title
        return a.title.localeCompare(b.title);
    });

    const relevantChunks = sortedChunks.filter(c => c.relevance >= 0.4);
    const otherChunks = sortedChunks.filter(c => c.relevance < 0.4);

    // Helpers
    const toggleSelect = (id: string) => {
        setSelectedIds(prev =>
            prev.includes(id) ? prev.filter(x => x !== id) : [...prev, id]
        );
    };
    const selectAll = () => setSelectedIds(chunks.map(c => c.chunk_id));
    const selectNone = () => setSelectedIds([]);
    const selectTop3 = () => setSelectedIds(sortedChunks.slice(0, 3).map(c => c.chunk_id));

    // Approve
    const handleApprove = async () => {
        if (selectedIds.length === 0) return;

        setIsApproving(true);
        try {
            const res = await fetch("http://localhost:8000/api/rag/approve", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ approved_chunk_ids: selectedIds })
            });

            const data = await res.json();
            if (data.status === "success") {
                // Use document for more reliable event propagation
                const answerEvent = new CustomEvent("hitl-answer", {
                    bubbles: true, // Allow blobbing up
                    detail: { answer: data.answer }
                });
                document.dispatchEvent(answerEvent);

                setApprovedIds(prev => [...new Set([...prev, ...selectedIds])]);
                setSelectedIds([]);
            } else {
                alert("Onay başarısız: " + data.error);
            }
        } catch (error) {
            console.error("Approval error:", error);
            alert("Bağlantı hatası!");
        } finally {
            setIsApproving(false);
        }
    };

    return (
        <div className="h-screen flex flex-col bg-gradient-to-br from-gray-900/95 to-gray-800/95 backdrop-blur-xl border-l border-white/10">
            {/* Header Area */}
            <div className="p-6 border-b border-white/10 flex flex-col gap-4 bg-black/20">
                <div className="flex justify-between items-center">
                    <div>
                        <h2 className="text-xl font-bold text-white flex items-center gap-2">
                            📚 Knowledge Base
                        </h2>
                        <p className="text-xs text-gray-400 mt-1 font-mono">
                            {chunks.length} chunks • <span className="text-green-400">{approvedIds.length} approved</span> • {selectedIds.length} selected
                        </p>
                    </div>
                    {approvedIds.length > 0 && (
                        <button onClick={() => setApprovedIds([])} className="text-xs text-gray-500 hover:text-white transition" title="Clear Approved">
                            ↺
                        </button>
                    )}
                </div>

                {currentQuery && (
                    <div className="p-3 bg-blue-500/10 border border-blue-500/20 rounded-lg text-sm text-blue-300 flex items-start gap-2">
                        <span className="mt-0.5">🎯</span>
                        <span className="italic line-clamp-2">Query: "{currentQuery}"</span>
                    </div>
                )}

                <div className="space-y-3">
                    <input
                        type="text"
                        placeholder="🔍 Search chunks..."
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        className="w-full px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-blue-500 text-sm"
                    />
                    <div className="flex gap-2">
                        <button onClick={selectAll} className="px-3 py-1.5 text-xs bg-blue-600/10 hover:bg-blue-600/20 text-blue-400 border border-blue-500/20 rounded transition flex-1">
                            Select All
                        </button>
                        <button onClick={selectTop3} className="px-3 py-1.5 text-xs bg-emerald-600/10 hover:bg-emerald-600/20 text-emerald-400 border border-emerald-500/20 rounded transition flex-1">
                            Select Top 3
                        </button>
                        <button onClick={selectNone} className="px-3 py-1.5 text-xs bg-red-600/10 hover:bg-red-600/20 text-red-400 border border-red-500/20 rounded transition flex-1">
                            Clear
                        </button>
                    </div>
                </div>
            </div>

            {/* List Area */}
            <div className="flex-1 overflow-y-auto p-4 space-y-6 custom-scrollbar">
                {isLoading ? (
                    <div className="text-center text-gray-500 py-12 flex flex-col items-center">
                        <div className="w-6 h-6 border-2 border-blue-500 border-t-transparent rounded-full animate-spin mb-3"></div>
                        <span className="text-sm">Loading chunks...</span>
                    </div>
                ) : filteredChunks.length === 0 ? (
                    <div className="text-center text-gray-500 py-12 text-sm italic">
                        No chunks found matching criteria.
                    </div>
                ) : (
                    <>
                        {relevantChunks.length > 0 && (
                            <div className="mb-2">
                                <div className="flex items-center gap-2 mb-3 px-1">
                                    <span className="text-amber-500 text-xs font-bold tracking-wider uppercase">High Relevance ({relevantChunks.length})</span>
                                    <div className="h-px bg-gradient-to-r from-amber-500/20 to-transparent flex-1"></div>
                                </div>
                                {relevantChunks.map(chunk => (
                                    <ChunkCard
                                        key={chunk.chunk_id}
                                        chunk={chunk}
                                        isSelected={selectedIds.includes(chunk.chunk_id)}
                                        // @ts-ignore
                                        isApproved={approvedIds.includes(chunk.chunk_id)}
                                        onToggleSelect={() => toggleSelect(chunk.chunk_id)}
                                    />
                                ))}
                            </div>
                        )}

                        {otherChunks.length > 0 && (
                            <div>
                                <div className="flex items-center gap-2 mb-3 mt-6 px-1">
                                    <span className="text-slate-500 text-xs font-bold tracking-wider uppercase">Other Sources ({otherChunks.length})</span>
                                    <div className="h-px bg-slate-800 flex-1"></div>
                                </div>
                                {otherChunks.map(chunk => (
                                    <ChunkCard
                                        key={chunk.chunk_id}
                                        chunk={chunk}
                                        isSelected={selectedIds.includes(chunk.chunk_id)}
                                        // @ts-ignore
                                        isApproved={approvedIds.includes(chunk.chunk_id)}
                                        onToggleSelect={() => toggleSelect(chunk.chunk_id)}
                                    />
                                ))}
                            </div>
                        )}
                    </>
                )}
            </div>

            {/* Footer */}
            <div className="p-4 border-t border-white/10 bg-black/40 backdrop-blur-md">
                <button
                    onClick={handleApprove}
                    disabled={selectedIds.length === 0 || isApproving}
                    className={`
                        w-full py-3 px-4 rounded-lg text-sm font-bold tracking-wide transition-all shadow-lg transform duration-200
                        ${selectedIds.length > 0
                            ? isApproving
                                ? 'bg-gray-700 text-gray-300 cursor-wait border border-white/10 scale-[0.98]'
                                : 'bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 hover:shadow-blue-500/40 hover:-translate-y-0.5 active:scale-95 text-white shadow-blue-500/25'
                            : 'bg-white/5 text-gray-500 cursor-not-allowed'}
                    `}
                >
                    {isApproving ? (
                        <div className="flex items-center justify-center gap-2">
                            <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                            <span>Analiz Ediliyor...</span>
                        </div>
                    ) : (
                        selectedIds.length > 0
                            ? `Approve ${selectedIds.length} Sources for Assistant`
                            : "Select Sources to Approve"
                    )}
                </button>
            </div>
        </div>
    );
}
