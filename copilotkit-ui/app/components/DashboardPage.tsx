'use client'

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { useEffect, useState } from "react";
import { useAuth } from "@clerk/nextjs";
import { listThreads } from "@/lib/api";

type DashboardProps = {
  stats: any;
  backendStatus: string;
  currentMode: "dashboard" | "chat" | "sidebar" | "popup" | "billing";
  onNavigate: (mode: "dashboard" | "chat" | "sidebar" | "popup" | "billing") => void;
  userId?: string;
};

export default function DashboardPage({ stats, backendStatus, currentMode, onNavigate, userId }: DashboardProps) {
  const [threads, setThreads] = useState<Array<{ thread_id: string; preview: string; last_message_at?: string }>>([]);
  const { getToken } = useAuth();

  useEffect(() => {
    const run = async () => {
      if (!userId) return;
      try {
        const token = await getToken();
        const res = await listThreads(20, token);
        setThreads(Array.isArray(res) ? res : []);
      } catch {
        // ignore
      }
    };
    run();
  }, [userId]);

  return (
    <div className="h-full overflow-y-auto bg-gradient-to-br from-slate-50 via-white to-slate-50">
      <div className="max-w-7xl mx-auto p-8 space-y-8">
        {/* Modern Header */}
        <div className="space-y-3">
          <div className="flex items-center gap-3">
            <div className="h-12 w-12 rounded-2xl bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center shadow-lg">
              <span className="text-2xl">🚀</span>
            </div>
            <div>
              <h1 className="text-3xl font-bold bg-gradient-to-r from-slate-900 via-slate-700 to-slate-900 bg-clip-text text-transparent">
                AI Research Assistant
              </h1>
              <p className="text-sm text-slate-500 mt-1">
                Powered by DeepAgents • Multi-Agent System
              </p>
            </div>
          </div>
        </div>

        {/* Modern Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {/* Backend Status */}
          <div className="group relative overflow-hidden rounded-3xl bg-white/80 backdrop-blur-xl border border-slate-200/50 shadow-xl hover:shadow-2xl transition-all duration-300 hover:-translate-y-1">
            <div className="absolute inset-0 bg-gradient-to-br from-emerald-500/10 to-teal-500/10 opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
            <div className="relative p-6">
              <div className="flex items-center justify-between mb-4">
                <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider">Status</span>
                <div className={`w-3 h-3 rounded-full ${
                  backendStatus === "online" ? "bg-emerald-500 animate-pulse" : "bg-rose-500"
                } shadow-lg`}></div>
              </div>
              <div className="text-3xl font-bold bg-gradient-to-br from-slate-900 to-slate-600 bg-clip-text text-transparent mb-2">
                {backendStatus === "online" ? "Online" : "Offline"}
              </div>
              <p className="text-sm text-slate-500">
                {backendStatus === "online" ? "All systems operational" : "Backend not responding"}
              </p>
            </div>
          </div>

          {/* Cache Entries */}
          <div className="group relative overflow-hidden rounded-3xl bg-white/80 backdrop-blur-xl border border-slate-200/50 shadow-xl hover:shadow-2xl transition-all duration-300 hover:-translate-y-1">
            <div className="absolute inset-0 bg-gradient-to-br from-purple-500/10 to-pink-500/10 opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
            <div className="relative p-6">
              <div className="flex items-center justify-between mb-4">
                <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider">Cache</span>
                <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-purple-500 to-pink-500 flex items-center justify-center text-white">
                  💾
                </div>
              </div>
              <div className="text-3xl font-bold bg-gradient-to-br from-purple-600 to-pink-600 bg-clip-text text-transparent mb-2">
                {stats?.cache?.entries || 0}
              </div>
              <p className="text-sm text-slate-500">
                of {stats?.cache?.max_size || 100} entries
              </p>
            </div>
          </div>

          {/* Rate Limit */}
          <div className="group relative overflow-hidden rounded-3xl bg-white/80 backdrop-blur-xl border border-slate-200/50 shadow-xl hover:shadow-2xl transition-all duration-300 hover:-translate-y-1">
            <div className="absolute inset-0 bg-gradient-to-br from-blue-500/10 to-cyan-500/10 opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
            <div className="relative p-6">
              <div className="flex items-center justify-between mb-4">
                <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider">Requests</span>
                <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-500 to-cyan-500 flex items-center justify-center text-white">
                  ⚡
                </div>
              </div>
              <div className="text-3xl font-bold bg-gradient-to-br from-blue-600 to-cyan-600 bg-clip-text text-transparent mb-2">
                {stats?.rate_limit?.remaining || 0}
              </div>
              <p className="text-sm text-slate-500">
                of {stats?.rate_limit?.max_requests_per_minute || 10} • Resets {stats?.rate_limit?.reset_in_seconds || 0}s
              </p>
            </div>
          </div>

          {/* API Keys */}
          <div className="group relative overflow-hidden rounded-3xl bg-white/80 backdrop-blur-xl border border-slate-200/50 shadow-xl hover:shadow-2xl transition-all duration-300 hover:-translate-y-1">
            <div className="absolute inset-0 bg-gradient-to-br from-amber-500/10 to-orange-500/10 opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
            <div className="relative p-6">
              <div className="flex items-center justify-between mb-4">
                <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider">API</span>
                <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-amber-500 to-orange-500 flex items-center justify-center text-white">
                  🔑
                </div>
              </div>
              <div className="text-3xl font-bold bg-gradient-to-br from-amber-600 to-orange-600 bg-clip-text text-transparent mb-2">
                Multi-Key
              </div>
              <p className="text-sm text-slate-500">
                Auto rotation enabled
              </p>
            </div>
          </div>
        </div>

        {/* Features Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* AI Agents */}
          <Card>
            <CardHeader>
              <CardTitle>Agents</CardTitle>
              <CardDescription>Available agent modes</CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="flex items-start space-x-3">
                <div className="w-10 h-10 bg-blue-100 rounded-lg flex items-center justify-center flex-shrink-0">
                  <span className="text-sm font-semibold text-blue-700">S</span>
                </div>
                <div className="flex-1">
                  <h4 className="font-medium text-sm">Simple Agent</h4>
                  <p className="text-xs text-slate-600">Fast research • 30-60s</p>
                </div>
                <Badge variant="secondary">Active</Badge>
              </div>
              
              <Separator />
              
              <div className="flex items-start space-x-3">
                <div className="w-10 h-10 bg-purple-100 rounded-lg flex items-center justify-center flex-shrink-0">
                  <span className="text-sm font-semibold text-purple-700">D</span>
                </div>
                <div className="flex-1">
                  <h4 className="font-medium text-sm">DeepAgent</h4>
                  <p className="text-xs text-slate-600">Detailed research • 1-2 min</p>
                </div>
                <Badge variant="outline">Available</Badge>
              </div>
              
              <Separator />
              
              <div className="flex items-start space-x-3">
                <div className="w-10 h-10 bg-indigo-100 rounded-lg flex items-center justify-center flex-shrink-0">
                  <span className="text-sm font-semibold text-indigo-700">M</span>
                </div>
                <div className="flex-1">
                  <h4 className="font-medium text-sm">Multi-Agent System</h4>
                  <p className="text-xs text-slate-600">Comprehensive • 3-5 min</p>
                </div>
                <Badge variant="outline">Available</Badge>
              </div>
            </CardContent>
          </Card>

          {/* UI Modes */}
          <Card>
            <CardHeader>
              <CardTitle>Interface</CardTitle>
              <CardDescription>Choose your workspace</CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              <Button
                onClick={() => onNavigate("chat")}
                variant={currentMode === "chat" ? "default" : "outline"}
                className="w-full justify-start gap-3 h-auto py-3"
              >
                <div className="text-left flex-1">
                  <div className="font-medium">Chat</div>
                  <div className="text-xs opacity-70">Full screen</div>
                </div>
              </Button>
              
              <Button
                onClick={() => onNavigate("sidebar")}
                variant={currentMode === "sidebar" ? "default" : "outline"}
                className="w-full justify-start gap-3 h-auto py-3"
              >
                <div className="text-left flex-1">
                  <div className="font-medium">Sidebar</div>
                  <div className="text-xs opacity-70">Dashboard + assistant</div>
                </div>
              </Button>
              
              <Button
                onClick={() => onNavigate("popup")}
                variant={currentMode === "popup" ? "default" : "outline"}
                className="w-full justify-start gap-3 h-auto py-3"
              >
                <div className="text-left flex-1">
                  <div className="font-medium">Popup</div>
                  <div className="text-xs opacity-70">Floating assistant</div>
                </div>
              </Button>
            </CardContent>
          </Card>
        </div>

        {/* Features */}
        <Card>
          <CardHeader>
            <CardTitle>Features</CardTitle>
            <CardDescription>Core capabilities</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              <div className="flex items-start space-x-3">
                <div className="w-8 h-8 bg-green-100 rounded-lg flex items-center justify-center flex-shrink-0">
                  <span className="text-xs font-semibold text-green-700">K</span>
                </div>
                <div>
                  <h4 className="font-medium text-sm">Multi API Keys</h4>
                  <p className="text-xs text-slate-600">Auto rotation on rate limit</p>
                </div>
              </div>
              
              <div className="flex items-start space-x-3">
                <div className="w-8 h-8 bg-blue-100 rounded-lg flex items-center justify-center flex-shrink-0">
                  <span className="text-xs font-semibold text-blue-700">C</span>
                </div>
                <div>
                  <h4 className="font-medium text-sm">Response Caching</h4>
                  <p className="text-xs text-slate-600">60 min TTL, 100 entries</p>
                </div>
              </div>
              
              <div className="flex items-start space-x-3">
                <div className="w-8 h-8 bg-purple-100 rounded-lg flex items-center justify-center flex-shrink-0">
                  <span className="text-xs font-semibold text-purple-700">W</span>
                </div>
                <div>
                  <h4 className="font-medium text-sm">Web Research</h4>
                  <p className="text-xs text-slate-600">Firecrawl + Tavily MCP</p>
                </div>
              </div>
              
              <div className="flex items-start space-x-3">
                <div className="w-8 h-8 bg-indigo-100 rounded-lg flex items-center justify-center flex-shrink-0">
                  <span className="text-xs font-semibold text-indigo-700">X</span>
                </div>
                <div>
                  <h4 className="font-medium text-sm">Code Examples</h4>
                  <p className="text-xs text-slate-600">AI-generated code samples</p>
                </div>
              </div>
              
              <div className="flex items-start space-x-3">
                <div className="w-8 h-8 bg-pink-100 rounded-lg flex items-center justify-center flex-shrink-0">
                  <span className="text-xs font-semibold text-pink-700">T</span>
                </div>
                <div>
                  <h4 className="font-medium text-sm">Turkish Reports</h4>
                  <p className="text-xs text-slate-600">Professional Turkish output</p>
                </div>
              </div>
              
              <div className="flex items-start space-x-3">
                <div className="w-8 h-8 bg-orange-100 rounded-lg flex items-center justify-center flex-shrink-0">
                  <span className="text-xs font-semibold text-orange-700">R</span>
                </div>
                <div>
                  <h4 className="font-medium text-sm">Rate Protection</h4>
                  <p className="text-xs text-slate-600">Never hit 429 errors</p>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Quick Actions */}
        <Card>
          <CardHeader>
            <CardTitle>🎯 Quick Actions</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            <Button
              onClick={() => onNavigate("chat")}
              variant="outline"
              className="w-full justify-start"
            >
              💬 Start a new chat
            </Button>
            <Button
              onClick={() => window.open(`${process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000"}/docs`, "_blank")}
              variant="outline"
              className="w-full justify-start"
            >
              📖 View API documentation
            </Button>
            <Button
              onClick={async () => {
                if (confirm("Are you sure you want to clear the cache?")) {
                  await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000"}/cache`, { method: "DELETE" });
                  window.location.reload();
                }
              }}
              variant="outline"
              className="w-full justify-start text-red-600 hover:text-red-700"
            >
              🗑️ Clear cache
            </Button>
          </CardContent>
        </Card>

        {/* Conversation History */}
        <Card>
          <CardHeader>
            <CardTitle>💾 Conversation History</CardTitle>
            <CardDescription>Saved per user (Supabase)</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            {!userId && (
              <p className="text-sm text-slate-600">
                Sign in to enable history.
              </p>
            )}
            {userId && threads.length === 0 && (
              <p className="text-sm text-slate-600">
                No saved conversations yet. Ask a question in Chat.
              </p>
            )}
            {userId && threads.length > 0 && (
              <div className="space-y-2">
                {threads.map((t) => (
                  <div key={t.thread_id} className="flex items-center justify-between rounded-lg border p-3">
                    <div className="min-w-0 pr-3">
                      <div className="text-xs text-slate-500 truncate">{t.thread_id}</div>
                      <div className="text-sm text-slate-700 truncate">{t.preview}</div>
                    </div>
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => {
                        // chat ekranında thread id ile devam et
                        try {
                          localStorage.setItem("threadId_chat", t.thread_id);
                        } catch {}
                        onNavigate("chat");
                      }}
                    >
                      Open
                    </Button>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

