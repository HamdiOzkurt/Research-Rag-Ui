'use client'

import { useState, useEffect } from "react";
import { useSearchParams } from "next/navigation";
import CopilotChatInterface from "./components/CopilotChatInterface";
import DashboardPage from "./components/DashboardPage";
import { health, stats as fetchStats } from "@/lib/api";
import AppShell, { UIMode } from "./components/AppShell";

export default function Home() {
  const [mode, setMode] = useState<UIMode>("chat"); // Default to chat (CopilotKit)
  const [backendStatus, setBackendStatus] = useState<"checking" | "online" | "offline">("checking");
  const [stats, setStats] = useState<any>(null);
  const searchParams = useSearchParams();

  useEffect(() => {
    const requested = (searchParams.get("mode") || "").toLowerCase();
    if (requested === "dashboard" || requested === "chat") {
      setMode(requested as UIMode);
    }
  }, [searchParams]);

  useEffect(() => {
    const checkBackend = async () => {
      try {
        await health();
        setBackendStatus("online");
        setStats(await fetchStats());
      } catch {
        setBackendStatus("offline");
      }
    };

    checkBackend();
    const interval = setInterval(checkBackend, 10000);
    return () => clearInterval(interval);
  }, []);

  return (
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
      {mode === "chat" && <CopilotChatInterface />}
    </AppShell>
  );
}
