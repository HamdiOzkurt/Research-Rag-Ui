'use client'

import { useState, useEffect, useCallback } from "react";
import { useSearchParams } from "next/navigation";
import CopilotChatInterface from "./components/CopilotChatInterface";
import DashboardPage from "./components/DashboardPage";
import ChunkExplorer from "./components/ChunkExplorer";
import DeepResearchPanel from "./components/DeepResearchPanel";
import { health, stats as fetchStats, AgentStatus } from "@/lib/api";
import AppShell, { UIMode } from "./components/AppShell";
import type { RAGState } from "@/types/rag";
import type { AgentMode } from "@/lib/api"; // ✅ Import AgentMode

export default function Home() {
  const [mode, setMode] = useState<UIMode>("chat");
  const [backendStatus, setBackendStatus] = useState<"checking" | "online" | "offline">("checking");
  const [stats, setStats] = useState<any>(null);
  const searchParams = useSearchParams();

  // HITL State polls
  const [ragState, setRagState] = useState<RAGState | null>(null);

  // ✅ Agent Mode State (Lifted from ChatInterface)
  const [agentMode, setAgentMode] = useState<AgentMode>("simple");

  // ✅ Lifted State for Deep Research Panel
  const [agentStatus, setAgentStatus] = useState<AgentStatus | null>(null);
  const [activity, setActivity] = useState<any[]>([]);

  // ✅ RESIZABLE PANEL STATE
  const [sidebarWidth, setSidebarWidth] = useState(600); // Default width
  const [isResizing, setIsResizing] = useState(false);

  // ✅ Show Explorer Logic
  const showRightPanel = mode === "chat" && (agentMode === "multi" || agentMode === "deep");
  const showExplorer = mode === "chat" && agentMode === "multi"; // Legacy use for resizer

  // Resize Handlers
  const startResizing = useCallback(() => setIsResizing(true), []);
  const stopResizing = useCallback(() => setIsResizing(false), []);

  // Resize Effect
  useEffect(() => {
    const resize = (mouseMoveEvent: MouseEvent) => {
      if (isResizing) {
        const newWidth = window.innerWidth - mouseMoveEvent.clientX;
        // Constraints
        if (newWidth > 320 && newWidth < window.innerWidth - 450) {
          setSidebarWidth(newWidth);
        }
      }
    };

    if (isResizing) {
      window.addEventListener("mousemove", resize);
      window.addEventListener("mouseup", stopResizing);
      // UX improvements while dragging
      document.body.style.userSelect = "none";
      document.body.style.cursor = "col-resize";
    }

    return () => {
      window.removeEventListener("mousemove", resize);
      window.removeEventListener("mouseup", stopResizing);
      document.body.style.userSelect = "";
      document.body.style.cursor = "";
    };
  }, [isResizing, stopResizing]);

  // Initial Logic & Query Params
  useEffect(() => {
    health()
      .then(() => setBackendStatus("online"))
      .catch(() => setBackendStatus("offline"));

    const checkStats = async () => {
      const data = await fetchStats();
      if (data) setStats(data);
    };
    checkStats();

    const modeParam = searchParams.get("mode");
    if (modeParam && ["dashboard", "chat"].includes(modeParam)) {
      setMode(modeParam as UIMode);
    }
  }, [searchParams]);

  // Polling for RAG State
  useEffect(() => {
    if (!showExplorer) return;
    const pollState = async () => {
      try {
        const res = await fetch("http://localhost:8000/api/rag/state");
        const data = await res.json();
        if (data) setRagState(data);
      } catch (e) {
        // silent fail
      }
    };
    const interval = setInterval(pollState, 2000);
    return () => clearInterval(interval);
  }, [showExplorer]);


  return (
    <div className="flex h-screen w-full bg-background overflow-hidden relative">

      {/* LEFT PANEL: Chat & AppShell */}
      <div
        className="h-full flex-col flex transition-none"
        style={{
          width: showRightPanel ? `calc(100% - ${showExplorer ? sidebarWidth : 450}px)` : '100%',
          flex: 'none'
        }}
      >
        <AppShell
          mode={mode}
          setMode={setMode}
          backendStatus={backendStatus}
          stats={stats}
          userName="Demo User"
        >
          {mode === "dashboard" && (
            <DashboardPage
              stats={stats}
              backendStatus={backendStatus}
              currentMode={mode}
              onNavigate={setMode}
              userId="demo-user"
            />
          )}
          {mode === "chat" && (
            <CopilotChatInterface
              agentMode={agentMode}
              onModeChange={setAgentMode}
              onStatusChange={setAgentStatus}
              onActivityChange={setActivity}
            />
          )}
        </AppShell>
      </div>

      {/* RESIZER HANDLE (Only when Explorer Visible) */}
      {showExplorer && (
        <div
          onMouseDown={startResizing}
          className={`w-1.5 h-full cursor-col-resize z-50 transition-colors flex items-center justify-center border-l border-white/5
            ${isResizing ? 'bg-blue-600' : 'bg-[#0d1117] hover:bg-blue-500'}`}
        >
          <div className={`h-8 w-0.5 rounded-full transition-colors ${isResizing ? 'bg-white' : 'bg-white/20'
            }`} />
        </div>
      )}

      {/* RIGHT PANEL: Chunk Explorer (Only when Visible) */}
      {showExplorer && (
        <div
          className="h-full bg-[#0d1117] flex flex-col"
          style={{ width: `${sidebarWidth}px`, flex: 'none' }}
        >
          <ChunkExplorer ragState={ragState} />
        </div>
      )}
    </div>
  );
}
