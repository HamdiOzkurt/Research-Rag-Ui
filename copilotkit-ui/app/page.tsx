'use client'

import { useState, useEffect } from "react";
import { UserButton, useUser } from "@clerk/nextjs";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import ChatInterface from "./components/ChatInterface";
import SidebarInterface from "./components/SidebarInterface";
import PopupInterface from "./components/PopupInterface";
import DashboardPage from "./components/DashboardPage";
import { API_BASE, health, stats as fetchStats } from "@/lib/api";

type UIMode = "chat" | "sidebar" | "popup" | "dashboard";

export default function Home() {
  const [mode, setMode] = useState<UIMode>("dashboard");
  const [backendStatus, setBackendStatus] = useState<"checking" | "online" | "offline">("checking");
  const [stats, setStats] = useState<any>(null);
  const { user, isLoaded } = useUser();

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

  if (!isLoaded) {
    return (
      <div className="flex items-center justify-center h-screen bg-gradient-to-br from-slate-50 to-slate-100">
        <div className="text-center">
          <div className="w-16 h-16 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
          <p className="text-slate-600">Loading...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="h-screen bg-gradient-to-br from-slate-50 to-slate-100">
      {/* Top Bar */}
      <div className="bg-white/80 backdrop-blur-lg border-b border-slate-200 px-6 py-3 sticky top-0 z-50 shadow-sm">
        <div className="flex items-center justify-between">
          {/* Logo & Status */}
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-gradient-to-br from-blue-500 via-purple-500 to-indigo-600 rounded-xl flex items-center justify-center shadow-lg">
                <span className="text-white text-xl">🔍</span>
              </div>
              <div>
                <h1 className="text-lg font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                  AI Research Assistant
                </h1>
                <div className="flex items-center space-x-2">
                  <div className={`w-2 h-2 rounded-full animate-pulse ${
                    backendStatus === "online" ? "bg-green-500" :
                    backendStatus === "offline" ? "bg-red-500" : "bg-yellow-500"
                  }`}></div>
                  <span className="text-xs text-slate-600">
                    {backendStatus === "online" 
                      ? `Connected • ${stats?.cache?.entries || 0} cached`
                      : backendStatus === "offline" ? "Offline" : "Checking..."}
                  </span>
                </div>
              </div>
            </div>
          </div>

          <div className="flex items-center space-x-3">
            {/* UI Mode Selector */}
            <div className="hidden md:flex items-center space-x-2">
              <Button
                onClick={() => setMode("dashboard")}
                variant={mode === "dashboard" ? "default" : "ghost"}
                size="sm"
                className="gap-2"
              >
                <span>📊</span>
                <span>Dashboard</span>
              </Button>
              <Button
                onClick={() => setMode("chat")}
                variant={mode === "chat" ? "default" : "ghost"}
                size="sm"
                className="gap-2"
              >
                <span>💬</span>
                <span>Chat</span>
              </Button>
              <Button
                onClick={() => setMode("sidebar")}
                variant={mode === "sidebar" ? "default" : "ghost"}
                size="sm"
                className="gap-2"
              >
                <span>📋</span>
                <span>Sidebar</span>
              </Button>
              <Button
                onClick={() => setMode("popup")}
                variant={mode === "popup" ? "default" : "ghost"}
                size="sm"
                className="gap-2"
              >
                <span>💭</span>
                <span>Popup</span>
              </Button>
            </div>
            
            {/* Rate Limit Badge */}
            {stats?.rate_limit && (
              <Badge variant={stats.rate_limit.remaining > 5 ? "default" : "destructive"}>
                {stats.rate_limit.remaining}/{stats.rate_limit.max_requests_per_minute}
              </Badge>
            )}

            {/* User Button */}
            <div className="flex items-center space-x-3">
              {user && (
                <span className="text-sm text-slate-600 hidden md:inline">
                  {user.firstName || user.username || "User"}
                </span>
              )}
              <UserButton 
                afterSignOutUrl="/sign-in"
                appearance={{
                  elements: {
                    avatarBox: "w-10 h-10"
                  }
                }}
              />
            </div>
          </div>
        </div>
      </div>

      {/* Offline Warning */}
      {backendStatus === "offline" && (
        <div className="bg-red-50 border-b border-red-200 px-6 py-3">
          <div className="flex items-center justify-between max-w-7xl mx-auto">
            <div className="flex items-center space-x-3">
              <div className="w-8 h-8 bg-red-100 rounded-lg flex items-center justify-center">
                <span className="text-red-600">⚠️</span>
              </div>
              <div>
                <p className="text-sm font-medium text-red-900">Backend Offline</p>
                <p className="text-xs text-red-700">
                  Start backend: <code className="bg-red-100 px-2 py-0.5 rounded">.\start.ps1</code>
                </p>
              </div>
            </div>
            <Button onClick={() => window.location.reload()} variant="outline" size="sm">
              Refresh
            </Button>
          </div>
        </div>
      )}

      {/* Content */}
      <div className="h-[calc(100vh-64px)]">
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
      </div>
    </div>
  );
}
