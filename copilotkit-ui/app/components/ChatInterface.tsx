'use client'

import { useState, useRef, useEffect } from "react";
import { Send, Loader2, Trash2, TrendingUp, MessageSquare, Zap, Users, Brain } from "lucide-react";
import { useAuth, useUser } from "@clerk/nextjs";
import { chat as chatApi, AgentMode } from "@/lib/api";
import { formatError } from "@/lib/errors";

const AGENT_MODES: { value: AgentMode; label: string; icon: React.ReactNode; desc: string }[] = [
  { value: "simple", label: "Hızlı", icon: <Zap className="w-4 h-4" />, desc: "Tek agent, hızlı yanıt" },
  { value: "multi", label: "Multi-Agent", icon: <Users className="w-4 h-4" />, desc: "Supervisor + Researcher + Coder + Writer" },
  { value: "deep", label: "Deep Research", icon: <Brain className="w-4 h-4" />, desc: "MCP + Full pipeline" },
];

type Message = {
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
};

// Modern chat hook with mode support
function useCopilotChat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [threadId, setThreadId] = useState<string | null>(null);
  const [mode, setMode] = useState<AgentMode>("simple");
  const { user } = useUser();
  const { getToken } = useAuth();

  useEffect(() => {
    try {
      const stored = localStorage.getItem("threadId_chat");
      if (stored) setThreadId(stored);
      const storedMode = localStorage.getItem("agent_mode") as AgentMode;
      if (storedMode && ["simple", "multi", "deep"].includes(storedMode)) {
        setMode(storedMode);
      }
    } catch {}
  }, []);

  const changeMode = (newMode: AgentMode) => {
    setMode(newMode);
    try {
      localStorage.setItem("agent_mode", newMode);
    } catch {}
  };

  const appendMessage = async (content: string, role: "user" | "assistant" = "user") => {
    const newMessage: Message = {
      role,
      content,
      timestamp: new Date(),
    };
    
    setMessages(prev => [...prev, newMessage]);

    if (role === "user") {
      setIsLoading(true);
      try {
        const token = await getToken();
        const data = await chatApi({
          message: content,
          threadId,
          useCache: true,
          token,
          mode,
        });
        if (data.thread_id) {
          setThreadId(data.thread_id);
          try {
            localStorage.setItem("threadId_chat", data.thread_id);
          } catch {}
        }
        
        setMessages(prev => [...prev, {
          role: "assistant",
          content: data.response || "Sonuç alınamadı.",
          timestamp: new Date(),
        }]);
      } catch (error) {
        setMessages(prev => [...prev, {
          role: "assistant",
          content: `❌ ${formatError(error)}`,
          timestamp: new Date(),
        }]);
      } finally {
        setIsLoading(false);
      }
    }
  };

  const deleteMessage = (index: number) => {
    setMessages(prev => prev.filter((_, i) => i !== index));
  };

  const clearMessages = () => {
    setMessages([]);
  };

  return {
    messages,
    isLoading,
    appendMessage,
    deleteMessage,
    clearMessages,
    mode,
    changeMode,
  };
}

export default function ChatInterface() {
  const { messages, isLoading, appendMessage, deleteMessage, clearMessages, mode, changeMode } = useCopilotChat();
  const [input, setInput] = useState("");
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage = input.trim();
    setInput("");
    await appendMessage(userMessage);
  };

  const exampleQuestions = [
    "Python pandas nedir?",
    "Web scraping nasıl yapılır?",
    "FastAPI authentication örneği"
  ];

  return (
    <div className="flex h-full bg-background">
      <div className="flex-1 flex flex-col max-w-5xl mx-auto w-full">
        {/* Header */}
        <div className="p-5 border-b bg-background flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <MessageSquare className="w-5 h-5 text-muted-foreground" />
            <div>
              <h2 className="text-sm font-semibold">Chat</h2>
              <p className="text-xs text-muted-foreground">Full screen</p>
            </div>
          </div>
          
          {/* Mode Selector */}
          <div className="flex items-center gap-1 bg-muted/50 rounded-lg p-1">
            {AGENT_MODES.map((m) => (
              <button
                key={m.value}
                onClick={() => changeMode(m.value)}
                title={m.desc}
                className={`flex items-center gap-1.5 px-3 py-1.5 text-xs rounded-md transition-all ${
                  mode === m.value
                    ? "bg-background text-foreground shadow-sm font-medium"
                    : "text-muted-foreground hover:text-foreground"
                }`}
              >
                {m.icon}
                <span className="hidden sm:inline">{m.label}</span>
              </button>
            ))}
          </div>

          {messages.length > 0 && (
            <button
              onClick={clearMessages}
              className="flex items-center gap-2 px-3 py-2 text-xs text-muted-foreground hover:text-foreground rounded-md border"
            >
              <Trash2 className="w-4 h-4" />
              Clear
            </button>
          )}
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-6 space-y-4">
          {messages.length === 0 ? (
            <div className="max-w-xl">
              <p className="text-sm text-muted-foreground mb-4">
                Ask a question to start. Examples:
              </p>
              <div className="space-y-2">
                {exampleQuestions.map((q, idx) => (
                  <button
                    key={idx}
                    onClick={() => appendMessage(q)}
                    className="w-full text-left rounded-md border px-3 py-2 text-sm hover:bg-muted"
                  >
                    {q}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            messages.map((msg, idx) => (
              <div
                key={idx}
                className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"} group animate-in fade-in slide-in-from-bottom-2 duration-300`}
              >
                <div className="flex flex-col max-w-3xl">
                  <div
                    className={`rounded-2xl px-6 py-4 shadow-md ${
                      msg.role === "user"
                        ? "bg-slate-900 text-white"
                        : "bg-white border"
                    }`}
                  >
                    <div className="whitespace-pre-wrap text-sm leading-relaxed">
                      {msg.content}
                    </div>
                  </div>
                  <div className="flex items-center space-x-2 mt-2 px-2">
                    <span className="text-xs text-gray-400">
                      {msg.timestamp.toLocaleTimeString('tr-TR')}
                    </span>
                    <button
                      onClick={() => deleteMessage(idx)}
                      className="text-xs text-gray-400 hover:text-red-500 opacity-0 group-hover:opacity-100 transition-opacity"
                    >
                      Sil
                    </button>
                  </div>
                </div>
              </div>
            ))
          )}
          
          {isLoading && (
            <div className="flex justify-start animate-in fade-in slide-in-from-bottom-2">
              <div className="bg-white rounded-xl px-4 py-3 border">
                <div className="flex items-center space-x-3">
                  <Loader2 className="w-4 h-4 text-muted-foreground animate-spin" />
                  <span className="text-sm text-muted-foreground">Thinking…</span>
                </div>
              </div>
            </div>
          )}
          
          <div ref={messagesEndRef} />
        </div>

        {/* Input */}
        <div className="border-t bg-background p-4">
          <form onSubmit={handleSubmit} className="flex space-x-3">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Bir şey sorun..."
              disabled={isLoading}
              className="flex-1 px-4 py-3 border rounded-md focus:outline-none focus:ring-2 focus:ring-slate-300 disabled:bg-muted"
            />
            <button
              type="submit"
              disabled={isLoading || !input.trim()}
              className="px-4 py-3 bg-slate-900 text-white rounded-md hover:bg-slate-800 disabled:opacity-50 disabled:cursor-not-allowed transition-all font-medium flex items-center gap-2"
            >
              {isLoading ? (
                <Loader2 className="w-5 h-5 animate-spin" />
              ) : (
                <Send className="w-5 h-5" />
              )}
              <span>Gönder</span>
            </button>
          </form>
          
          <div className="mt-2 text-xs text-muted-foreground">
            Powered by DeepAgents • {messages.length} messages
          </div>
        </div>
      </div>
    </div>
  );
}
