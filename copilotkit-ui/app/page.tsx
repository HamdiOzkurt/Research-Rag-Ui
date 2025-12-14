'use client'

import { useState, useEffect } from "react";
import { useSearchParams } from "next/navigation";
import { useAuth, useUser } from "@clerk/nextjs";
import ChatInterface from "./components/ChatInterface";
import SidebarInterface from "./components/SidebarInterface";
import PopupInterface from "./components/PopupInterface";
import DashboardPage from "./components/DashboardPage";
import { health, stats as fetchStats } from "@/lib/api";
import AppShell, { UIMode } from "./components/AppShell";

export default function Home() {
  const [mode, setMode] = useState<UIMode>("dashboard");
  const [backendStatus, setBackendStatus] = useState<"checking" | "online" | "offline">("checking");
  const [stats, setStats] = useState<any>(null);
  const { user, isLoaded } = useUser();
  const { getToken } = useAuth();
  const searchParams = useSearchParams();

  useEffect(() => {
    const requested = (searchParams.get("mode") || "").toLowerCase();
    if (requested === "dashboard" || requested === "chat" || requested === "sidebar" || requested === "popup") {
      setMode(requested as UIMode);
    }
  }, [searchParams]);

  useEffect(() => {
    const checkBackend = async () => {
      try {
        await health();
        setBackendStatus("online");
        const token = await getToken();
        setStats(await fetchStats(token));
      } catch {
        setBackendStatus("offline");
      }
    };

    checkBackend();
    const interval = setInterval(checkBackend, 10000);
    return () => clearInterval(interval);
  }, []);

  if (!isLoaded) {
    return (
      <div className="flex items-center justify-center h-screen bg-background">
        <div className="text-center">
          <div className="w-10 h-10 border-2 border-slate-300 border-t-slate-900 rounded-full animate-spin mx-auto mb-3"></div>
          <p className="text-sm text-muted-foreground">Loading...</p>
        </div>
      </div>
    );
  }

  return (
    <AppShell
      mode={mode}
      setMode={setMode}
      backendStatus={backendStatus}
      stats={stats}
      userName={(user?.firstName || user?.username) ?? null}
    >
        {mode === "dashboard" && (
          <DashboardPage
            stats={stats}
            backendStatus={backendStatus}
            currentMode={mode}
            onNavigate={setMode}
            userId={user?.id}
          />
        )}
        {mode === "chat" && <ChatInterface />}
        {mode === "sidebar" && <SidebarInterface />}
        {mode === "popup" && <PopupInterface />}
    </AppShell>
  );
}
