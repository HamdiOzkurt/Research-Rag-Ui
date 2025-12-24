"use client";

import { useState } from "react";
import type { RetrievedChunk } from "@/types/rag";

interface Props {
    chunks: RetrievedChunk[];
    onApprove: (chunkIds: string[]) => void;
    isLoading?: boolean;
}

export default function SourceInspector({ chunks, onApprove, isLoading }: Props) {
    const [selected, setSelected] = useState<string[]>([]);

    const toggleChunk = (id: string) => {
        setSelected(prev =>
            prev.includes(id)
                ? prev.filter(x => x !== id)
                : [...prev, id]
        );
    };

    const selectAll = () => {
        setSelected(chunks.map(c => c.chunk_id));
    };

    const selectTopN = (n: number) => {
        setSelected(chunks.slice(0, n).map(c => c.chunk_id));
    };

    return (
        <div className="fixed right-0 top-0 h-screen w-96 bg-gradient-to-br from-gray-900/95 to-gray-800/95 backdrop-blur-xl border-l border-white/10 shadow-2xl overflow-y-auto z-50">
            {/* Header */}
            <div className="sticky top-0 bg-gray-900/80 backdrop-blur p-6 border-b border-white/10">
                <h2 className="text-2xl font-bold text-white mb-2">
                    📚 Kaynaklar
                </h2>
                <p className="text-sm text-gray-400">
                    {chunks.length} kaynak bulundu - Hangilerini kullanayım?
                </p>

                {/* Quick Actions */}
                <div className="flex gap-2 mt-4">
                    <button
                        onClick={selectAll}
                        className="px-3 py-1 text-xs bg-blue-600/20 hover:bg-blue-600/30 text-blue-400 rounded-md transition"
                    >
                        ✓ Tümü
                    </button>
                    <button
                        onClick={() => selectTopN(3)}
                        className="px-3 py-1 text-xs bg-green-600/20 hover:bg-green-600/30 text-green-400 rounded-md transition"
                    >
                        ⚡ En İyi 3
                    </button>
                </div>
            </div>

            {/* Chunks List */}
            <div className="p-4 space-y-3">
                {chunks.map((chunk, idx) => {
                    const isSelected = selected.includes(chunk.chunk_id);

                    return (
                        <div
                            key={chunk.chunk_id}
                            onClick={() => toggleChunk(chunk.chunk_id)}
                            className={`
                p-4 rounded-lg border-2 transition-all cursor-pointer
                ${isSelected
                                    ? 'border-blue-500 bg-blue-500/10'
                                    : 'border-white/10 bg-white/5 hover:bg-white/10'
                                }
              `}
                        >
                            {/* Confidence Bar */}
                            <div className="h-1.5 bg-gray-700 rounded-full mb-3 overflow-hidden">
                                <div
                                    className={`h-full rounded-full transition-all ${chunk.confidence > 0.7 ? 'bg-green-500' :
                                        chunk.confidence > 0.5 ? 'bg-yellow-500' : 'bg-red-500'
                                        }`}
                                    style={{ width: `${chunk.confidence * 100}%` }}
                                />
                            </div>

                            {/* Title & Metadata */}
                            <div className="flex items-start justify-between mb-2">
                                <div className="flex-1">
                                    <h3 className="font-semibold text-white text-sm mb-1">
                                        {idx + 1}. {chunk.title}
                                    </h3>
                                    <p className="text-xs text-gray-400">
                                        {chunk.source}
                                        {chunk.has_images && " 🖼️"}
                                    </p>
                                </div>
                                <div className={`
                  w-5 h-5 rounded border-2 flex items-center justify-center
                  ${isSelected ? 'border-blue-500 bg-blue-500' : 'border-gray-600'}
                `}>
                                    {isSelected && <span className="text-white text-xs">✓</span>}
                                </div>
                            </div>

                            {/* Section Info */}
                            {(chunk.section_h1 || chunk.section_h2) && (
                                <div className="text-xs text-gray-500 mb-2">
                                    {chunk.section_h1 && `📖 ${chunk.section_h1}`}
                                    {chunk.section_h2 && ` > ${chunk.section_h2}`}
                                </div>
                            )}

                            {/* Summary */}
                            <p className="text-xs text-gray-300 line-clamp-2">
                                {chunk.summary}
                            </p>

                            {/* Confidence Score */}
                            <div className="mt-2 text-xs font-mono text-gray-500">
                                Alakalılık: {(chunk.confidence * 100).toFixed(0)}%
                            </div>
                        </div>
                    );
                })}
            </div>

            {/* Approve Button */}
            <div className="sticky bottom-0 p-4 bg-gray-900/90 backdrop-blur border-t border-white/10">
                <button
                    onClick={() => onApprove(selected)}
                    disabled={selected.length === 0 || isLoading}
                    className={`
            w-full py-3 rounded-lg font-semibold transition-all duration-200 transform
            ${selected.length === 0
                            ? 'bg-gray-700 text-gray-500 cursor-not-allowed'
                            : isLoading
                                ? 'bg-gray-800 text-white/50 cursor-wait border border-white/10'
                                : 'bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-500 hover:to-purple-500 hover:shadow-xl hover:-translate-y-0.5 active:translate-y-0 active:scale-95 text-white shadow-lg shadow-blue-500/20'
                        }
          `}
                >
                    {isLoading ? (
                        <div className="flex items-center justify-center gap-2">
                            <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                            <span>Analiz Başlatılıyor...</span>
                        </div>
                    ) : (
                        <>✅ {selected.length} Kaynağı Onayla</>
                    )}
                </button>
            </div>
        </div>
    );
}
