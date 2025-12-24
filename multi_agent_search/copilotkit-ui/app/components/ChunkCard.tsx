"use client";

import { useState, useEffect } from "react";
// Feather Icons veya Lucide kullanılabilirdi ama inline SVG en kolayı
function CheckIcon() {
    return (
        <svg className="w-3.5 h-3.5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
        </svg>
    );
}

function ChevronDown({ className }: { className?: string }) {
    return (
        <svg className={`w-4 h-4 ${className}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
    );
}

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
    };
    images?: string[];
    relevance: number;
    tags: string[];
}

interface ChunkCardProps {
    chunk: Chunk;
    isSelected: boolean;
    isApproved?: boolean;
    onToggleSelect: () => void;
}

export default function ChunkCard({ chunk, isSelected, isApproved = false, onToggleSelect }: ChunkCardProps) {
    const [isExpanded, setIsExpanded] = useState(false);
    const [selectedImage, setSelectedImage] = useState<string | null>(null);

    // Kapatma tuşu (ESC)
    useEffect(() => {
        const handleEsc = (e: KeyboardEvent) => {
            if (e.key === 'Escape') setSelectedImage(null);
        };
        window.addEventListener('keydown', handleEsc);
        return () => window.removeEventListener('keydown', handleEsc);
    }, [selectedImage]);

    // Renk skalası
    const relevancePercent = Math.round(chunk.relevance * 100);
    const getRelevanceColor = () => {
        if (relevancePercent >= 70) return "bg-emerald-500/10 text-emerald-400 border-emerald-500/20";
        if (relevancePercent >= 40) return "bg-amber-500/10 text-amber-400 border-amber-500/20";
        return "bg-slate-500/10 text-slate-400 border-slate-500/20";
    };
    const badgeClass = getRelevanceColor();

    // Kart Stili
    const cardStyle = isApproved
        ? "border-emerald-500/50 bg-emerald-900/10 shadow-[0_0_15px_-3px_rgba(16,185,129,0.25)]"
        : isSelected
            ? "border-blue-500/50 bg-blue-900/10 shadow-[0_0_15px_-3px_rgba(59,130,246,0.25)]"
            : "border-white/5 bg-[#161b22] hover:border-white/10 hover:bg-[#1c2128]";

    // Image URL Helper
    const getImageUrl = (path: string) => {
        if (!path) return "";
        if (path.startsWith("http")) return path;

        const cleanPath = path.replace(/\\/g, "/");
        // Backend serves 'uploads' directory at root
        if (cleanPath.startsWith("uploads/")) {
            return `http://localhost:8000/${cleanPath}`;
        }
        // Fallback or absolute path cleanup
        const fileName = cleanPath.split("/").pop();
        // Bazı durumlarda klasör ismi de gerekebilir, en güvenlisi uploads altında recursive aramak ama 
        // şimdilik basit varsayım:
        if (chunk.source) {
            // "Source.docx" -> "Source_images/file.png" pattern
            const docName = chunk.source.split(".").shift(); // ext hariç
            // Try to reconstruct likely path if only filename is given
            if (!cleanPath.includes("/")) {
                return `http://localhost:8000/uploads/${docName}_images/${fileName}`;
            }
        }

        return `http://localhost:8000/uploads/${fileName}`;
    };

    const openImage = (index: number) => {
        if (chunk.images && chunk.images[index]) {
            setSelectedImage(getImageUrl(chunk.images[index]));
        }
    };


    // Helper to render content with inline images
    const renderContentWithImages = (text: string) => {
        // Split by markdown image syntax: ![alt](url)
        const parts = text.split(/(!\[.*?\]\(.*?\))/g);

        return parts.map((part, idx) => {
            const match = part.match(/!\[(.*?)\]\((.*?)\)/);
            if (match) {
                const alt = match[1];
                const path = match[2];
                const src = getImageUrl(path);

                return (
                    <div key={idx} className="my-3 rounded-lg overflow-hidden border border-white/10 bg-black/20">
                        <img
                            src={src}
                            alt={alt || "Embedded Image"}
                            className="w-full h-auto max-h-[300px] object-contain mx-auto opacity-90 hover:opacity-100 transition-opacity"
                            onError={(e) => {
                                e.currentTarget.style.display = 'none'; // Kırık linkse gizle
                            }}
                        />
                        {alt && <div className="text-[10px] text-center text-gray-500 py-1 bg-black/40 italic">{alt}</div>}
                    </div>
                );
            }
            return <span key={idx}>{part}</span>;
        });
    };

    return (
        <>
            <div className={`relative group rounded-xl border transition-all duration-200 mb-3 overflow-hidden ${cardStyle}`}>
                {/* Header (Clickable) */}
                <div className="p-4 flex gap-3 cursor-pointer select-none" onClick={() => setIsExpanded(!isExpanded)}>

                    {/* Checkbox (Left) */}
                    <div className="pt-0.5 shrink-0" onClick={(e) => e.stopPropagation()}>
                        <div
                            onClick={!isApproved ? onToggleSelect : undefined}
                            className={`
                                w-8 h-8 -ml-1.5 flex items-center justify-center rounded-full transition-all 
                                ${!isApproved ? "cursor-pointer hover:bg-white/10" : "cursor-default"}
                            `}
                        >
                            <div className={`
                                w-5 h-5 rounded border flex items-center justify-center transition-all
                                ${isSelected || isApproved
                                    ? (isApproved ? "bg-emerald-600 border-emerald-600 opacity-60" : "bg-blue-600 border-blue-600")
                                    : "border-white/20 bg-white/5 group-hover:border-blue-500"}
                            `}>
                                {(isSelected || isApproved) && <CheckIcon />}
                            </div>
                        </div>
                    </div>

                    {/* Content */}
                    <div className="flex-1 min-w-0">
                        {/* Badges Row */}
                        <div className="flex items-center gap-2 mb-1.5 flex-wrap">
                            <span className={`text-[10px] font-bold px-2 py-0.5 rounded border ${badgeClass}`}>
                                {relevancePercent}% Match
                            </span>

                            {isApproved && (
                                <span className="text-[10px] font-bold px-2 py-0.5 rounded border bg-emerald-500/20 text-emerald-400 border-emerald-500/30 flex items-center gap-1">
                                    ✓ Approved
                                </span>
                            )}

                            {/* Image Badge */}
                            {chunk.metadata.has_images && (
                                <span className="text-[10px] font-medium bg-purple-500/10 text-purple-400 px-1.5 py-0.5 rounded border border-purple-500/20">
                                    🖼️ Image
                                </span>
                            )}
                            {/* Table Badge */}
                            {chunk.metadata.table_count && chunk.metadata.table_count > 0 && (
                                <span className="text-[10px] font-medium bg-orange-500/10 text-orange-400 px-1.5 py-0.5 rounded border border-orange-500/20">
                                    📊 Table
                                </span>
                            )}
                        </div>

                        {/* Title */}
                        <h4 className="text-sm font-semibold text-gray-200 leading-snug group-hover:text-blue-400 transition-colors mb-0.5">
                            {chunk.title}
                        </h4>

                        {/* Metadata Line */}
                        <div className="flex items-center gap-2 text-[10px] text-gray-500 font-mono mt-0.5">
                            <span className="truncate max-w-[150px]">{chunk.source}</span>
                            {chunk.metadata.page && <span>• P.{chunk.metadata.page}</span>}
                        </div>

                        {/* Summary */}
                        {!isExpanded && (
                            <p className="text-xs text-slate-400 mt-2 line-clamp-2 leading-relaxed">
                                {chunk.summary}
                            </p>
                        )}
                    </div>

                    {/* Arrow Icon */}
                    <div className="shrink-0 text-gray-600 group-hover:text-gray-400 transition-colors pt-1">
                        <ChevronDown className={`transform transition-transform duration-200 ${isExpanded ? 'rotate-180' : ''}`} />
                    </div>
                </div>

                {/* Expanded Content */}
                {isExpanded && (
                    <div className="px-4 pb-4 pl-12 border-t border-white/5 pt-3 mt-1 bg-black/20">
                        {/* ✅ ESKİ METADATA GÖRÜNÜMÜ (GERİ GELDİ) */}
                        <div className="bg-black/40 p-3 rounded-lg border border-white/10 mb-4 font-mono text-xs">
                            <div className="flex items-center gap-2 mb-2 text-blue-400 font-bold border-b border-white/10 pb-1">
                                <span>📄 Metadata:</span>
                            </div>
                            <ul className="space-y-1.5 text-gray-400">
                                <li className="flex gap-2">
                                    <span className="w-20 shrink-0 text-gray-500">• Source:</span>
                                    <span className="text-gray-300 break-all">{chunk.source}</span>
                                </li>
                                <li className="flex gap-2">
                                    <span className="w-20 shrink-0 text-gray-500">• Chunk ID:</span>
                                    <span className="text-gray-300">{chunk.chunk_id.slice(0, 8)}...</span>
                                </li>
                                {chunk.metadata.chunk_index !== undefined && (
                                    <li className="flex gap-2">
                                        <span className="w-20 shrink-0 text-gray-500">• Index:</span>
                                        <span className="text-gray-300">{chunk.metadata.chunk_index}</span>
                                    </li>
                                )}
                                <li className="flex gap-2">
                                    <span className="w-20 shrink-0 text-gray-500">• Section H1:</span>
                                    <span className="text-gray-300 text-amber-500">{chunk.metadata.section_h1 || "N/A"}</span>
                                </li>
                                <li className="flex gap-2">
                                    <span className="w-20 shrink-0 text-gray-500">• Section H2:</span>
                                    <span className="text-gray-300">{chunk.metadata.section_h2 || "N/A"}</span>
                                </li>
                                <li className="flex gap-2">
                                    <span className="w-20 shrink-0 text-gray-500">• Has Images:</span>
                                    <span className={chunk.metadata.has_images ? "text-green-400 font-bold" : "text-red-400 font-bold"}>
                                        {chunk.metadata.has_images ? "✓ YES" : "✕ NO"}
                                    </span>
                                </li>
                            </ul>
                        </div>

                        {/* Full Content with Inline Images */}
                        <div className="prose prose-invert prose-sm max-w-none text-slate-300 text-xs leading-relaxed whitespace-pre-wrap">
                            <span className="text-blue-400 font-bold block mb-1"># İçerik:</span>
                            {renderContentWithImages(chunk.content)}
                        </div>

                        {/* Image Gallery - Metadata'ya bakmaksızın resim varsa göster */}
                        {chunk.images && chunk.images.length > 0 && (
                            <div className="border-t border-white/10 mt-4 pt-4">
                                <span className="text-[10px] font-bold text-purple-400 uppercase tracking-wider mb-2 block">
                                    Attached Images ({chunk.images.length})
                                </span>
                                <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
                                    {chunk.images.map((img, idx) => {
                                        const src = getImageUrl(img);
                                        return (
                                            <div
                                                key={idx}
                                                className="aspect-video bg-black/40 rounded border border-white/10 overflow-hidden cursor-zoom-in group/img"
                                                onClick={() => openImage(idx)}
                                            >
                                                <img
                                                    src={src}
                                                    alt={`Figure ${idx + 1}`}
                                                    className="w-full h-full object-cover opacity-80 group-hover/img:opacity-100 transition-opacity"
                                                />
                                            </div>
                                        )
                                    })}
                                </div>
                            </div>
                        )}
                    </div>
                )}
            </div>


            {/* Image Modal */}
            {selectedImage && (
                <div
                    className="fixed inset-0 z-50 bg-black/90 backdrop-blur-sm flex items-center justify-center p-4 animate-in fade-in duration-200"
                    onClick={() => setSelectedImage(null)}
                >
                    <button
                        className="absolute top-4 right-4 text-white/70 hover:text-white p-2"
                        onClick={() => setSelectedImage(null)}
                    >
                        <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                        </svg>
                    </button>
                    <img
                        src={selectedImage}
                        alt="Zoomed view"
                        className="max-w-full max-h-[90vh] object-contain rounded shadow-2xl"
                        onClick={(e) => e.stopPropagation()}
                    />
                </div>
            )}
        </>
    );
}
