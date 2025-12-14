'use client'

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { useEffect, useState } from "react";
import { listThreads } from "@/lib/api";

type DashboardProps = {
  stats: any;
  backendStatus: string;
  currentMode: "dashboard" | "chat" | "sidebar" | "popup";
  onNavigate: (mode: "dashboard" | "chat" | "sidebar" | "popup") => void;
  userId?: string;
};

export default function DashboardPage({ stats, backendStatus, currentMode, onNavigate, userId }: DashboardProps) {
  const [threads, setThreads] = useState<Array<{ thread_id: string; preview: string; last_message_at?: string }>>([]);

  useEffect(() => {
    const run = async () => {
      if (!userId) return;
      try {
        const res = await listThreads(userId, 20);
        setThreads(Array.isArray(res) ? res : []);
      } catch {
        // ignore
      }
    };
    run();
  }, [userId]);

  return (
    <div className="h-full overflow-y-auto">
      <div className="max-w-7xl mx-auto p-6 space-y-6">
        {/* Hero Section */}
        <div className="relative overflow-hidden rounded-2xl bg-gradient-to-br from-blue-500 via-purple-500 to-indigo-600 p-8 text-white shadow-2xl">
          <div className="relative z-10">
            <h1 className="text-4xl font-bold mb-2">Welcome to AI Research</h1>
            <p className="text-blue-100 text-lg">
              Powerful AI-powered research assistant with multi-agent system
            </p>
            <div className="flex items-center space-x-4 mt-6">
              <Badge variant="secondary" className="bg-white/20 text-white border-white/30">
                DeepAgents
              </Badge>
              <Badge variant="secondary" className="bg-white/20 text-white border-white/30">
                LangGraph
              </Badge>
              <Badge variant="secondary" className="bg-white/20 text-white border-white/30">
                Next.js 14
              </Badge>
            </div>
          </div>
          <div className="absolute top-0 right-0 w-64 h-64 bg-white/10 rounded-full blur-3xl"></div>
          <div className="absolute bottom-0 left-0 w-48 h-48 bg-white/10 rounded-full blur-3xl"></div>
        </div>

        {/* Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {/* Backend Status */}
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium text-slate-600">Backend Status</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex items-center space-x-2">
                <div className={`w-3 h-3 rounded-full ${
                  backendStatus === "online" ? "bg-green-500" : "bg-red-500"
                }`}></div>
                <span className="text-2xl font-bold">
                  {backendStatus === "online" ? "Online" : "Offline"}
                </span>
              </div>
              <p className="text-xs text-slate-500 mt-2">
                {backendStatus === "online" ? "All systems operational" : "Backend not responding"}
              </p>
            </CardContent>
          </Card>

          {/* Cache Entries */}
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium text-slate-600">Cached Responses</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{stats?.cache?.entries || 0}</div>
              <p className="text-xs text-slate-500 mt-2">
                Max: {stats?.cache?.max_size || 100} entries
              </p>
            </CardContent>
          </Card>

          {/* Rate Limit */}
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium text-slate-600">Remaining Requests</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-blue-600">
                {stats?.rate_limit?.remaining || 0} / {stats?.rate_limit?.max_requests_per_minute || 10}
              </div>
              <p className="text-xs text-slate-500 mt-2">
                Resets in {stats?.rate_limit?.reset_in_seconds || 0}s
              </p>
            </CardContent>
          </Card>

          {/* API Keys */}
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium text-slate-600">API Keys</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">Multi-Key</div>
              <p className="text-xs text-slate-500 mt-2">
                Rotation enabled
              </p>
            </CardContent>
          </Card>
        </div>

        {/* Features Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* AI Agents */}
          <Card>
            <CardHeader>
              <CardTitle>🤖 AI Agents</CardTitle>
              <CardDescription>Multi-agent research system</CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="flex items-start space-x-3">
                <div className="w-10 h-10 bg-blue-100 rounded-lg flex items-center justify-center flex-shrink-0">
                  <span className="text-xl">⚡</span>
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
                  <span className="text-xl">🔍</span>
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
                  <span className="text-xl">🧠</span>
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
              <CardTitle>🎨 UI Modes</CardTitle>
              <CardDescription>Choose your preferred interface</CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              <Button
                onClick={() => onNavigate("chat")}
                variant={currentMode === "chat" ? "default" : "outline"}
                className="w-full justify-start gap-3 h-auto py-3"
              >
                <span className="text-2xl">💬</span>
                <div className="text-left flex-1">
                  <div className="font-medium">CopilotChat</div>
                  <div className="text-xs opacity-70">Full screen chat interface</div>
                </div>
              </Button>
              
              <Button
                onClick={() => onNavigate("sidebar")}
                variant={currentMode === "sidebar" ? "default" : "outline"}
                className="w-full justify-start gap-3 h-auto py-3"
              >
                <span className="text-2xl">📋</span>
                <div className="text-left flex-1">
                  <div className="font-medium">CopilotSidebar</div>
                  <div className="text-xs opacity-70">Dashboard with collapsible chat</div>
                </div>
              </Button>
              
              <Button
                onClick={() => onNavigate("popup")}
                variant={currentMode === "popup" ? "default" : "outline"}
                className="w-full justify-start gap-3 h-auto py-3"
              >
                <span className="text-2xl">💭</span>
                <div className="text-left flex-1">
                  <div className="font-medium">CopilotPopup</div>
                  <div className="text-xs opacity-70">Floating chat bubble</div>
                </div>
              </Button>
            </CardContent>
          </Card>
        </div>

        {/* Features */}
        <Card>
          <CardHeader>
            <CardTitle>✨ Features</CardTitle>
            <CardDescription>What makes this assistant powerful</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              <div className="flex items-start space-x-3">
                <div className="w-8 h-8 bg-green-100 rounded-lg flex items-center justify-center flex-shrink-0">
                  <span>🔄</span>
                </div>
                <div>
                  <h4 className="font-medium text-sm">Multi API Keys</h4>
                  <p className="text-xs text-slate-600">Auto rotation on rate limit</p>
                </div>
              </div>
              
              <div className="flex items-start space-x-3">
                <div className="w-8 h-8 bg-blue-100 rounded-lg flex items-center justify-center flex-shrink-0">
                  <span>💾</span>
                </div>
                <div>
                  <h4 className="font-medium text-sm">Response Caching</h4>
                  <p className="text-xs text-slate-600">60 min TTL, 100 entries</p>
                </div>
              </div>
              
              <div className="flex items-start space-x-3">
                <div className="w-8 h-8 bg-purple-100 rounded-lg flex items-center justify-center flex-shrink-0">
                  <span>🚀</span>
                </div>
                <div>
                  <h4 className="font-medium text-sm">Web Research</h4>
                  <p className="text-xs text-slate-600">Firecrawl + Tavily MCP</p>
                </div>
              </div>
              
              <div className="flex items-start space-x-3">
                <div className="w-8 h-8 bg-indigo-100 rounded-lg flex items-center justify-center flex-shrink-0">
                  <span>💻</span>
                </div>
                <div>
                  <h4 className="font-medium text-sm">Code Examples</h4>
                  <p className="text-xs text-slate-600">AI-generated code samples</p>
                </div>
              </div>
              
              <div className="flex items-start space-x-3">
                <div className="w-8 h-8 bg-pink-100 rounded-lg flex items-center justify-center flex-shrink-0">
                  <span>📝</span>
                </div>
                <div>
                  <h4 className="font-medium text-sm">Turkish Reports</h4>
                  <p className="text-xs text-slate-600">Professional Turkish output</p>
                </div>
              </div>
              
              <div className="flex items-start space-x-3">
                <div className="w-8 h-8 bg-orange-100 rounded-lg flex items-center justify-center flex-shrink-0">
                  <span>🛡️</span>
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

