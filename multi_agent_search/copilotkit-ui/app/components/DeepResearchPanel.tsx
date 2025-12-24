"use client";

import { Brain, FileText, Globe, Layers, Sparkles } from "lucide-react";
import type { AgentStatus } from "../types/agent";

interface DeepResearchPanelProps {
    status: AgentStatus | null;
    logs: any[]; // We will pass activity logs here
}

export default function DeepResearchPanel({ status, logs }: DeepResearchPanelProps) {
    // Filter logs for deep research specific steps
    const researchLogs = logs.filter(l =>
        ["planning", "researching", "searching", "writing"].includes(l.status)
    );

    return (
        <div className="h-screen flex flex-col bg-gradient-to-br from-slate-900 to-slate-950 text-white border-l border-white/10">

            {/* Header */}
            <div className="p-6 border-b border-white/10 bg-black/20 backdrop-blur-md">
                <div className="flex items-center gap-3">
                    <div className="p-2 bg-indigo-500/20 rounded-lg border border-indigo-500/30">
                        <Globe className="w-5 h-5 text-indigo-400" />
                    </div>
                    <div>
                        <h2 className="text-lg font-bold">Deep Research Center</h2>
                        <div className="flex items-center gap-2 text-xs text-slate-400">
                            <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
                            {status ? status.message : "Idle"}
                        </div>
                    </div>
                </div>
            </div>

            {/* Main Content Area */}
            <div className="flex-1 overflow-y-auto p-4 space-y-6 custom-scrollbar">

                {/* 1. Project Plan Section */}
                <div className="space-y-3">
                    <div className="flex items-center gap-2 text-xs uppercase tracking-wider text-slate-500 font-semibold px-2">
                        <Layers className="w-4 h-4" />
                        Research Plan
                    </div>
                    <div className="bg-white/5 border border-white/10 rounded-xl p-4">
                        {researchLogs.length === 0 ? (
                            <div className="text-sm text-slate-500 italic text-center py-4">
                                Waiting for research to start...
                            </div>
                        ) : (
                            <div className="space-y-4">
                                {/* Simulation of a structured plan based on logs */}
                                {researchLogs.map((log, idx) => (
                                    <div key={idx} className="flex gap-3 relative">
                                        {/* Vertical Line */}
                                        {idx < researchLogs.length - 1 && (
                                            <div className="absolute left-[11px] top-8 bottom-[-16px] w-px bg-white/10"></div>
                                        )}

                                        <div className={`w-6 h-6 rounded-full flex items-center justify-center shrink-0 z-10 
                                    ${log.status === 'done' ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30' :
                                                log.status === 'error' ? 'bg-red-500/20 text-red-400 border border-red-500/30' :
                                                    'bg-blue-500/20 text-blue-400 border border-blue-500/30'}`}>
                                            {idx + 1}
                                        </div>
                                        <div className="flex-1">
                                            <div className="text-sm font-medium text-slate-200">
                                                {log.status.charAt(0).toUpperCase() + log.status.slice(1)}
                                            </div>
                                            <div className="text-xs text-slate-400 mt-1">
                                                {log.message}
                                            </div>
                                            {/* Tool Outputs / Metadata */}
                                            {log.meta && (
                                                <div className="mt-2 text-xs bg-black/30 p-2 rounded border border-white/5 font-mono text-slate-500">
                                                    {JSON.stringify(log.meta).slice(0, 100)}...
                                                </div>
                                            )}
                                        </div>
                                    </div>
                                ))}
                            </div>
                        )}
                    </div>
                </div>

                {/* 2. Key Findings / Live Notes */}
                <div className="space-y-3">
                    <div className="flex items-center gap-2 text-xs uppercase tracking-wider text-slate-500 font-semibold px-2">
                        <Sparkles className="w-4 h-4" />
                        Live Findings
                    </div>

                    <div className="grid grid-cols-1 gap-3">
                        {/* This would be populated dynamically from the 'notes' in the backend later */}
                        <div className="bg-emerald-900/10 border border-emerald-500/20 rounded-lg p-3">
                            <div className="text-xs text-emerald-400 mb-1 font-mono">LATEST INSIGHT</div>
                            <p className="text-sm text-slate-300">
                                Agent is actively synthesizing data. Notes will appear here once available.
                            </p>
                        </div>
                    </div>
                </div>

            </div>

            {/* Footer Status */}
            <div className="p-4 bg-black/40 border-t border-white/10 backdrop-blur">
                <div className="flex justify-between items-center text-xs text-slate-400 font-mono">
                    <span>RAM: SYSTEM_OPTIMAL</span>
                    <span>AGENTS: ACTIVE</span>
                </div>
            </div>

        </div>
    );
}
