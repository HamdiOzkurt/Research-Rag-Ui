'use client'

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { Zap, Users, Brain, MessageSquare, Sparkles, TrendingUp } from "lucide-react";

type DashboardProps = {
  stats: any;
  backendStatus: string;
  currentMode: "dashboard" | "chat";
  onNavigate: (mode: "dashboard" | "chat") => void;
  userId?: string;
};

export default function DashboardPage({ stats, backendStatus, currentMode, onNavigate, userId }: DashboardProps) {
  const features = [
    {
      icon: <Zap className="w-6 h-6" />,
      title: "Simple Mode",
      desc: "Fast single-agent responses",
      color: "from-yellow-500 to-orange-500",
    },
    {
      icon: <Users className="w-6 h-6" />,
      title: "Multi-Agent",
      desc: "Supervisor + 3 specialized agents",
      color: "from-blue-500 to-indigo-500",
    },
    {
      icon: <Brain className="w-6 h-6" />,
      title: "Deep Research",
      desc: "MCP + comprehensive analysis",
      color: "from-purple-500 to-pink-500",
    },
  ];

  return (
    <div className="h-full overflow-y-auto bg-gradient-to-br from-slate-50 via-white to-slate-50">
      <div className="max-w-7xl mx-auto p-8 space-y-8">
        {/* Modern Header */}
        <div className="space-y-3">
          <div className="flex items-center gap-3">
            <div className="h-12 w-12 rounded-2xl bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center shadow-lg">
              <Sparkles className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-3xl font-bold bg-gradient-to-r from-slate-900 via-slate-700 to-slate-900 bg-clip-text text-transparent">
                AI Research Assistant
              </h1>
              <p className="text-sm text-slate-500 mt-1">
                Powered by CopilotKit, LangGraph & Multi-Agent System
              </p>
            </div>
          </div>

          {/* Status Badge */}
          <div className="flex items-center gap-2">
            <Badge 
              variant="outline" 
              className={
                backendStatus === "online" 
                  ? "bg-emerald-500/10 text-emerald-600 border-emerald-500/20" 
                  : "bg-red-500/10 text-red-600 border-red-500/20"
              }
            >
              {backendStatus === "online" ? "🟢 Backend Online" : "🔴 Backend Offline"}
            </Badge>
            {stats?.cache && (
              <Badge variant="secondary" className="text-xs">
                💾 {stats.cache.entries} cached responses
              </Badge>
            )}
          </div>
        </div>

        <Separator />

        {/* Features Grid */}
        <div className="grid md:grid-cols-3 gap-6">
          {features.map((feature, idx) => (
            <Card key={idx} className="border-2 hover:shadow-lg transition-all cursor-pointer group">
              <CardHeader>
                <div className={`w-12 h-12 rounded-xl bg-gradient-to-br ${feature.color} flex items-center justify-center text-white mb-3 group-hover:scale-110 transition-transform`}>
                  {feature.icon}
                </div>
                <CardTitle className="text-lg">{feature.title}</CardTitle>
                <CardDescription>{feature.desc}</CardDescription>
              </CardHeader>
            </Card>
          ))}
        </div>

        {/* CTA Section */}
        <Card className="border-2 border-dashed bg-gradient-to-br from-blue-50 to-indigo-50">
          <CardContent className="pt-6">
            <div className="flex flex-col md:flex-row items-center justify-between gap-4">
              <div className="flex items-center gap-4">
                <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-blue-500 to-indigo-600 flex items-center justify-center text-white shadow-xl">
                  <MessageSquare className="w-8 h-8" />
                </div>
                <div>
                  <h3 className="text-xl font-bold text-slate-900">
                    Start Your Research
                  </h3>
                  <p className="text-sm text-slate-600 mt-1">
                    Try: "Python FastAPI nedir?" or "Web scraping best practices"
                  </p>
                </div>
              </div>
              <Button 
                size="lg"
                onClick={() => onNavigate("chat")}
                className="bg-gradient-to-r from-blue-500 to-indigo-600 hover:from-blue-600 hover:to-indigo-700 text-white shadow-lg"
              >
                Open Chat
                <MessageSquare className="w-4 h-4 ml-2" />
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* Stats Grid */}
        <div className="grid md:grid-cols-4 gap-4">
          <Card>
            <CardHeader className="pb-3">
              <CardDescription className="text-xs">Cache Hit Rate</CardDescription>
              <CardTitle className="text-2xl font-bold">
                {stats?.cache ? Math.round((stats.cache.entries / (stats.cache.max_size || 1)) * 100) : 0}%
              </CardTitle>
            </CardHeader>
          </Card>
          
          <Card>
            <CardHeader className="pb-3">
              <CardDescription className="text-xs">Agent Modes</CardDescription>
              <CardTitle className="text-2xl font-bold">3</CardTitle>
            </CardHeader>
          </Card>

          <Card>
            <CardHeader className="pb-3">
              <CardDescription className="text-xs">Status</CardDescription>
              <CardTitle className="text-2xl font-bold">
                {backendStatus === "online" ? "✅" : "❌"}
              </CardTitle>
            </CardHeader>
          </Card>

          <Card>
            <CardHeader className="pb-3">
              <CardDescription className="text-xs">CopilotKit</CardDescription>
              <CardTitle className="text-2xl font-bold text-emerald-600">Active</CardTitle>
            </CardHeader>
          </Card>
        </div>

        {/* Info Cards */}
        <div className="grid md:grid-cols-2 gap-6">
          <Card className="border-l-4 border-l-blue-500">
            <CardHeader>
              <CardTitle className="text-lg flex items-center gap-2">
                <Sparkles className="w-5 h-5 text-blue-500" />
                CopilotKit Features
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-2 text-sm text-slate-600">
              <div className="flex items-center gap-2">
                <Badge variant="secondary" className="text-xs">✨</Badge>
                <span>Natural language commands: "Switch to multi-agent mode"</span>
              </div>
              <div className="flex items-center gap-2">
                <Badge variant="secondary" className="text-xs">🤝</Badge>
                <span>Human-in-the-loop: Review agent plans before execution</span>
              </div>
              <div className="flex items-center gap-2">
                <Badge variant="secondary" className="text-xs">📊</Badge>
                <span>Real-time agent visualization</span>
              </div>
            </CardContent>
          </Card>

          <Card className="border-l-4 border-l-purple-500">
            <CardHeader>
              <CardTitle className="text-lg flex items-center gap-2">
                <TrendingUp className="w-5 h-5 text-purple-500" />
                Tech Stack
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-2 text-sm text-slate-600">
              <div>• <span className="font-medium">Frontend:</span> Next.js 14 + CopilotKit</div>
              <div>• <span className="font-medium">Backend:</span> FastAPI + Streaming SSE</div>
              <div>• <span className="font-medium">AI:</span> LangGraph + DeepAgents + MCP</div>
              <div>• <span className="font-medium">Memory:</span> Supabase</div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
