'use client'

import { useState, useRef, useEffect } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Zap, Users, Brain, Loader2, Send, Sparkles, Settings, FileText } from "lucide-react";
import { chatStream, AgentMode, type AgentStatus, getThread } from "@/lib/api";
import AgentStatusBar from "./AgentStatusBar";
import MarkdownRenderer from "./MarkdownRenderer";
import ThreadsList from "./ThreadsList";
import RAGFileUpload from "./RAGFileUpload";

type ActivityItem = {
  id: string;
  at: number;
  status: AgentStatus["status"];
  message: string;
  meta?: Record<string, unknown>;
  runId?: string;
};

function stageLabel(status: AgentStatus["status"]) {
  switch (status) {
    case "initializing":
      return "Başlatılıyor";
    case "planning":
      return "Plan";
    case "searching":
      return "Arama";
    case "researching":
      return "Araştırma";
    case "coding":
      return "Kod";
    case "writing":
      return "Yazım";
    case "done":
      return "Bitti";
    case "error":
      return "Hata";
    default:
      return "Durum";
  }
}

function formatElapsed(ms: number) {
  const totalSeconds = Math.max(0, Math.floor(ms / 1000));
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}:${seconds.toString().padStart(2, "0")}`;
}

const AGENT_MODES: { value: AgentMode; label: string; icon: React.ReactNode; desc: string }[] = [
  { value: "simple", label: "Hızlı", icon: <Zap className="w-4 h-4" />, desc: "Tek agent, hızlı yanıt" },
  { value: "multi", label: "RAG Agent", icon: <FileText className="w-4 h-4" />, desc: "Doküman Analizi & Q/A" },
  { value: "deep", label: "Deep Research", icon: <Brain className="w-4 h-4" />, desc: "Web Search & Planning" },
];

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  images?: Array<{ url: string; alt: string }>;
}

type ThreadMessage = {
  role: "user" | "assistant";
  content: string;
  created_at?: string;
};

export default function CopilotChatInterface() {
  const [mode, setMode] = useState<AgentMode>("simple");
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [threadId, setThreadId] = useState<string | null>(null);
  const [agentStatus, setAgentStatus] = useState<AgentStatus | null>(null);
  const [activity, setActivity] = useState<ActivityItem[]>([]);
  const [latestSources, setLatestSources] = useState<Array<{ title?: string; url: string; provider?: string }>>([]);
  const [showOptions, setShowOptions] = useState(false);
  const [showThreads, setShowThreads] = useState(false);
  const [options, setOptions] = useState<{ web_search: "none" | "tavily" | "both"; need_code: boolean; need_long_report: boolean }>({
    web_search: "both",
    need_code: true,
    need_long_report: true,
  });
  const runStartRef = useRef<number | null>(null);
  const currentRunIdRef = useRef<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleResumeThread = async (threadId: string) => {
    setShowThreads(false);
    try {
      const data = await getThread(threadId, 200);
      const list: ThreadMessage[] = Array.isArray(data) ? (data as ThreadMessage[]) : [];
      const nextMessages: Message[] = list
        .filter((m) => m && (m.role === "user" || m.role === "assistant") && typeof m.content === "string")
        .map((m, idx) => ({
          id: `${threadId}-${idx}-${m.created_at || ""}`,
          role: m.role,
          content: m.content,
        }));

      setThreadId(threadId);
      setMessages(nextMessages);
      setAgentStatus(null);
      setActivity([]);
      setLatestSources([]);
    } catch (err) {
      console.error(err);
      setMessages((prev) => [
        ...prev,
        {
          id: (Date.now() + 1).toString(),
          role: "assistant",
          content: "Error: Failed to load thread history.",
        },
      ]);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: input.trim(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);
    setAgentStatus(null);
    setActivity([]);
    setLatestSources([]);
    runStartRef.current = Date.now();
    currentRunIdRef.current = null;

    try {
      await chatStream({
        message: userMessage.content,
        threadId: threadId,
        mode: mode,
        options: mode === "multi" ? options : undefined,
        onStatus: (status: AgentStatus) => {
          setAgentStatus(status);

          if (status.thread_id) {
            setThreadId(status.thread_id);
          }

          const sources = (status.meta as any)?.sources;
          if (Array.isArray(sources) && sources.length > 0) {
            setLatestSources(
              sources
                .filter((s: any) => s && typeof s.url === "string")
                .map((s: any) => ({ title: s.title, url: s.url, provider: s.provider }))
            );
          }

          if (status.run_id && !currentRunIdRef.current) {
            currentRunIdRef.current = status.run_id;
          }

          const startedAt = runStartRef.current ?? Date.now();
          const at = Date.now();
          const trimmed = (status.message || "").trim();
          const label = stageLabel(status.status);
          const message = trimmed || label;

          setActivity((prev) => {
            const last = prev[prev.length - 1];
            // Avoid noisy duplicates
            if (last && last.status === status.status && last.message === message) {
              return prev;
            }
            const next: ActivityItem = {
              id: `${at}-${Math.random().toString(16).slice(2)}`,
              at,
              status: status.status,
              message,
              meta: status.meta,
              runId: status.run_id,
            };
            // keep last 12 events max
            return [...prev, next].slice(-12);
          });
        },
        onDone: (finalContent: string, metadata?: any) => {
          const assistantMessage: Message = {
            id: (Date.now() + 1).toString(),
            role: "assistant",
            content: finalContent,
            images: metadata?.images || [],
          };
          setMessages(prev => [...prev, assistantMessage]);
          setIsLoading(false);
          setAgentStatus(null);
        },
        onError: (error: string) => {
          const errorMessage: Message = {
            id: (Date.now() + 1).toString(),
            role: "assistant",
            content: `Error: ${error}`,
          };
          setMessages(prev => [...prev, errorMessage]);
          setIsLoading(false);
          setAgentStatus(null);
        }
      });
    } catch (err) {
      setIsLoading(false);
      setAgentStatus(null);
    }
  };

  return (
    <div className="flex h-full bg-gradient-to-b from-background to-muted/20">
      {/* Threads Sidebar */}
      {showThreads && (
        <div className="w-80 border-r bg-background/80 backdrop-blur overflow-y-auto">
          <div className="sticky top-0 bg-background border-b p-4 flex items-center justify-between">
            <h2 className="font-semibold">Threads</h2>
            <button
              onClick={() => setShowThreads(false)}
              className="text-xs text-muted-foreground hover:text-foreground"
            >
              Close
            </button>
          </div>
          <ThreadsList onResumeThread={handleResumeThread} />
        </div>
      )}
      
      <div className="flex-1 flex flex-col max-w-4xl mx-auto w-full">
        {/* Agent Status Bar */}
        {agentStatus && <AgentStatusBar status={agentStatus} mode={mode} />}
        
        {/* Header with Mode Selector */}
        <div className="p-5 border-b bg-background/80 backdrop-blur">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-3">
              <div className="h-10 w-10 rounded-xl bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center">
                <Sparkles className="w-5 h-5 text-white" />
              </div>
              <div>
                <h2 className="text-lg font-semibold">AI Research Assistant</h2>
                <p className="text-xs text-muted-foreground">
                  Multi-Agent System • LangGraph + DeepAgents
                </p>
              </div>
            </div>
            
            <div className="flex items-center gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => setShowThreads(!showThreads)}
                className="text-xs"
              >
                📜 Threads
              </Button>
              <Badge variant="outline" className="bg-emerald-500/10 text-emerald-600 border-emerald-500/20">
                ✨ Streaming Enabled
              </Badge>
            </div>
          </div>

          {/* Mode Selector */}
          <div className="flex items-center gap-2 bg-muted/50 rounded-lg p-1">
            {AGENT_MODES.map((m) => (
              <button
                key={m.value}
                onClick={() => setMode(m.value)}
                title={m.desc}
                disabled={isLoading}
                className={`flex items-center gap-1.5 px-4 py-2 text-sm rounded-md transition-all ${
                  mode === m.value
                    ? "bg-background text-foreground shadow-sm font-medium"
                    : "text-muted-foreground hover:text-foreground"
                } ${isLoading ? "opacity-50 cursor-not-allowed" : ""}`}
              >
                {m.icon}
                <span>{m.label}</span>
              </button>
            ))}
          </div>
          {/* Options Panel (Multi-Agent only) */}
          {mode === "multi" && (
            <div className="mt-3">
              <button
                onClick={() => setShowOptions(!showOptions)}
                className="flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors"
              >
                <Settings className="w-4 h-4" />
                <span>{showOptions ? "Hide" : "Show"} Options</span>
              </button>
              {showOptions && (
                <div className="mt-3 p-3 rounded-lg border bg-muted/30 space-y-2">
                  <div className="flex items-center gap-3">
                    <label className="text-xs text-muted-foreground w-24">Web Search:</label>
                    <select
                      value={options.web_search}
                      onChange={(e) => setOptions({ ...options, web_search: e.target.value as any })}
                      className="text-xs border rounded px-2 py-1 bg-background"
                    >
                      <option value="both">Both (Firecrawl + Tavily)</option>
                      <option value="tavily">Tavily only</option>
                      <option value="none">None</option>
                    </select>
                  </div>
                  <div className="flex items-center gap-3">
                    <label className="text-xs text-muted-foreground w-24">Code Examples:</label>
                    <input
                      type="checkbox"
                      checked={options.need_code}
                      onChange={(e) => setOptions({ ...options, need_code: e.target.checked })}
                      className="w-4 h-4"
                    />
                  </div>
                  <div className="flex items-center gap-3">
                    <label className="text-xs text-muted-foreground w-24">Long Report:</label>
                    <input
                      type="checkbox"
                      checked={options.need_long_report}
                      onChange={(e) => setOptions({ ...options, need_long_report: e.target.checked })}
                      className="w-4 h-4"
                    />
                  </div>
                </div>
              )}
            </div>
          )}
          
          {/* RAG File Upload - Only in Multi Mode */}
          {mode === "multi" && <RAGFileUpload />}
        </div>

        {/* Messages Area */}
        <div className="flex-1 overflow-y-auto p-6">
          {/* Activity / Observability */}
          {isLoading && activity.length > 0 && (
            <div className="mb-4 rounded-xl border bg-card">
              <div className="px-4 py-3 border-b flex items-center justify-between">
                <div className="text-sm font-medium">Çalışma Akışı</div>
                <div className="text-xs text-muted-foreground tabular-nums">
                  {formatElapsed(Date.now() - (runStartRef.current ?? Date.now()))}
                </div>
              </div>
              <div className="px-4 py-3 space-y-2">
                {activity.slice(-6).map((item) => {
                  const t0 = runStartRef.current ?? item.at;
                  const rel = formatElapsed(item.at - t0);
                  return (
                    <div key={item.id} className="flex items-start justify-between gap-4">
                      <div className="min-w-0">
                        <div className="text-xs text-muted-foreground">{stageLabel(item.status)}</div>
                        <div className="text-sm text-foreground truncate">{item.message}</div>
                      </div>
                      <div className="text-xs text-muted-foreground tabular-nums shrink-0">{rel}</div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Tool-call style card: Web sources */}
          {(isLoading || messages.length > 0) && latestSources.length > 0 && (
            <div className="mb-4 rounded-xl border bg-card">
              <div className="px-4 py-3 border-b flex items-center justify-between">
                <div className="text-sm font-medium">Kaynaklar</div>
                <div className="text-xs text-muted-foreground">{latestSources.length}</div>
              </div>
              <div className="px-4 py-3 space-y-2">
                {latestSources.slice(0, 6).map((s) => (
                  <div key={s.url} className="flex items-start justify-between gap-3">
                    <div className="min-w-0">
                      <div className="text-sm text-foreground truncate">
                        {s.title || s.url}
                      </div>
                      <a
                        href={s.url}
                        target="_blank"
                        rel="noreferrer"
                        className="text-xs text-muted-foreground hover:underline truncate block"
                      >
                        {s.url}
                      </a>
                    </div>
                    {s.provider && (
                      <Badge variant="outline" className="text-xs shrink-0">
                        {s.provider}
                      </Badge>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {messages.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full text-center">
              <div className="mb-6">
                <div className="w-20 h-20 rounded-2xl bg-gradient-to-br from-blue-500/20 to-purple-500/20 flex items-center justify-center mb-4 mx-auto">
                  <Brain className="w-10 h-10 text-blue-500" />
                </div>
                <h3 className="text-xl font-semibold mb-2">
                  Multi-Agent Research System
                </h3>
                <p className="text-sm text-muted-foreground max-w-md">
                  Ask anything and watch the agents work in real-time.
                </p>
              </div>
              
              {/* Example prompts */}
              <div className="grid gap-2 max-w-lg w-full">
                {[
                  "Python FastAPI nedir ve nasıl kullanılır?",
                  "Web scraping best practices nelerdir?",
                  "React vs Vue karşılaştırması yap",
                ].map((prompt, i) => (
                  <button
                    key={i}
                    onClick={() => setInput(prompt)}
                    className="text-left px-4 py-3 rounded-xl border bg-card hover:bg-muted/50 transition-colors text-sm"
                  >
                    <span className="text-muted-foreground mr-2">→</span>
                    {prompt}
                  </button>
                ))}
              </div>
              
              <div className="flex gap-2 mt-6">
                <Badge variant="secondary" className="text-xs">
                  🔍 Web Research
                </Badge>
                <Badge variant="secondary" className="text-xs">
                  💻 Code Generation
                </Badge>
                <Badge variant="secondary" className="text-xs">
                  📊 Report Writing
                </Badge>
              </div>
            </div>
          ) : (
            <div className="space-y-4">
              {messages.map((msg) => (
                <div
                  key={msg.id}
                  className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
                >
                  <div
                    className={`max-w-[80%] rounded-2xl px-4 py-3 ${
                      msg.role === "user"
                        ? "bg-slate-900 text-white"
                        : "bg-card border"
                    }`}
                  >
                    {msg.role === "assistant" ? (
                      <>
                        <MarkdownRenderer content={msg.content} />
                        {msg.images && msg.images.length > 0 && (
                          <div className="mt-4 grid grid-cols-2 gap-2">
                            {msg.images.map((img, idx) => (
                              <div key={idx} className="relative group cursor-pointer" onClick={() => window.open(img.url, '_blank')}>
                                <img 
                                  src={img.url} 
                                  alt={img.alt} 
                                  className="w-full h-auto rounded border border-gray-300 dark:border-gray-700 hover:opacity-90 transition-opacity"
                                />
                                <div className="absolute inset-0 bg-black bg-opacity-0 group-hover:bg-opacity-10 transition-all rounded flex items-center justify-center">
                                  <span className="opacity-0 group-hover:opacity-100 text-white text-sm bg-black bg-opacity-70 px-2 py-1 rounded">Click to enlarge</span>
                                </div>
                                {img.alt && <p className="text-xs text-gray-500 mt-1">{img.alt}</p>}
                              </div>
                            ))}
                          </div>
                        )}
                      </>
                    ) : (
                      <div className="prose prose-sm dark:prose-invert max-w-none whitespace-pre-wrap">
                        {msg.content}
                      </div>
                    )}
                  </div>
                </div>
              ))}
              
              {isLoading && !agentStatus && (
                <div className="flex items-center gap-2 text-sm text-muted-foreground">
                  <Loader2 className="w-4 h-4 animate-spin" />
                  <span>Connecting to agents...</span>
                </div>
              )}
              
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        {/* Input Area */}
        <div className="p-4 border-t bg-background">
          <form onSubmit={handleSubmit} className="flex gap-3">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask anything..."
              disabled={isLoading}
              className="flex-1 px-4 py-3 rounded-xl border bg-muted/50 focus:outline-none focus:ring-2 focus:ring-blue-500/50 disabled:opacity-50"
            />
            <Button
              type="submit"
              disabled={isLoading || !input.trim()}
              className="px-6 rounded-xl bg-gradient-to-r from-blue-500 to-indigo-600 hover:from-blue-600 hover:to-indigo-700"
            >
              {isLoading ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <Send className="w-4 h-4" />
              )}
            </Button>
          </form>
          <p className="text-xs text-center text-muted-foreground mt-2">
            💡 Select agent mode above: <span className="font-medium">Hızlı</span> for quick answers, <span className="font-medium">Multi-Agent</span> for detailed research
          </p>
        </div>
      </div>
    </div>
  );
}
