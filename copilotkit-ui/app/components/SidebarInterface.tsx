'use client'

import { useState, useRef, useEffect } from "react";
import { Send, Loader2, X, MessageCircle, BarChart3, Users, TrendingUp, DollarSign } from "lucide-react";
import { useAuth, useUser } from "@clerk/nextjs";
import { chat as chatApi } from "@/lib/api";
import { formatError } from "@/lib/errors";

type Message = {
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
};

function useCopilotChat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [threadId, setThreadId] = useState<string | null>(null);
  const { user } = useUser();
  const { getToken } = useAuth();

  useEffect(() => {
    try {
      const stored = localStorage.getItem("threadId_sidebar");
      if (stored) setThreadId(stored);
    } catch {}
  }, []);

  const appendMessage = async (content: string, role: "user" | "assistant" = "user") => {
    const newMessage: Message = { role, content, timestamp: new Date() };
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
        });
        if (data.thread_id) {
          setThreadId(data.thread_id);
          try {
            localStorage.setItem("threadId_sidebar", data.thread_id);
          } catch {}
        }
        setMessages(prev => [...prev, {
          role: "assistant",
          content: data.response,
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

  return { messages, isLoading, appendMessage };
}

export default function SidebarInterface() {
  const { messages, isLoading, appendMessage } = useCopilotChat();
  const [input, setInput] = useState("");
  const [isOpen, setIsOpen] = useState(true);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;
    const userMessage = input.trim();
    setInput("");
    await appendMessage(userMessage);
  };

  const stats = [
    { label: "Total Revenue", value: "$45,231.89", change: "+20.1%", icon: DollarSign, color: "blue" as const },
    { label: "Active Users", value: "+2,350", change: "+180.1%", icon: Users, color: "green" as const },
    { label: "Sales", value: "+12,234", change: "+19%", icon: TrendingUp, color: "indigo" as const },
    { label: "Active Now", value: "+573", change: "+201", icon: BarChart3, color: "purple" as const },
  ];

  // Tailwind production build için dinamik class kullanmıyoruz (bg-${color}-100 vs çalışmayabilir)
  const colorClasses = {
    blue: { bg: "bg-blue-100", text: "text-blue-600" },
    green: { bg: "bg-green-100", text: "text-green-600" },
    indigo: { bg: "bg-indigo-100", text: "text-indigo-600" },
    purple: { bg: "bg-purple-100", text: "text-purple-600" },
  } as const;

  return (
    <div className="flex h-full bg-background">
      {/* Main Content - Modern Dashboard */}
      <div className="flex-1 p-6 bg-muted/20 overflow-y-auto">
        <div className="max-w-7xl mx-auto">
          <h1 className="text-lg font-semibold mb-1">Dashboard</h1>
          <p className="text-sm text-muted-foreground mb-6">Example content area</p>
          
          {/* Stats Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
            {stats.map((stat, idx) => {
              const Icon = stat.icon;
              const c = colorClasses[stat.color];
              return (
                <div
                  key={idx}
                  className="bg-white p-6 rounded-2xl shadow-sm border border-gray-200 hover:shadow-lg transition-all hover:-translate-y-1"
                >
                  <div className="flex items-center justify-between mb-4">
                    <p className="text-sm font-medium text-gray-600">{stat.label}</p>
                    <div className={`w-10 h-10 rounded-lg ${c.bg} flex items-center justify-center`}>
                      <Icon className={`w-5 h-5 ${c.text}`} />
                    </div>
                  </div>
                  <p className="text-3xl font-bold text-gray-900 mb-1">{stat.value}</p>
                  <p className={`text-sm ${c.text} font-medium`}>{stat.change} from last month</p>
                </div>
              );
            })}
          </div>

          {/* Chart Area */}
          <div className="bg-white p-6 rounded-2xl shadow-sm border border-gray-200">
            <h3 className="text-lg font-semibold mb-6 text-gray-800">Revenue Overview</h3>
            <div className="h-64 flex items-end space-x-3">
              {[45, 62, 48, 85, 68, 92, 73, 88].map((height, i) => (
                <div key={i} className="flex-1 flex flex-col items-center">
                  <div
                    className="w-full bg-gradient-to-t from-blue-500 to-indigo-500 rounded-t-lg hover:from-blue-600 hover:to-indigo-600 transition-all cursor-pointer"
                    style={{ height: `${height}%` }}
                  />
                  <span className="text-xs text-gray-500 mt-2">Day {i + 1}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Sidebar Chat */}
      <div
        className={`bg-white border-l shadow-2xl transition-all duration-300 flex flex-col ${
          isOpen ? "w-[420px]" : "w-0"
        }`}
      >
        {isOpen && (
          <>
            <div className="p-4 border-b bg-background flex items-center justify-between">
              <div className="flex items-center space-x-3">
                <div className="w-9 h-9 rounded-md bg-slate-900 flex items-center justify-center">
                  <MessageCircle className="w-4 h-4 text-white" />
                </div>
                <div>
                  <h2 className="font-semibold text-sm">Assistant</h2>
                  <p className="text-xs text-muted-foreground">Sidebar mode</p>
                </div>
              </div>
              <button
                onClick={() => setIsOpen(false)}
                className="text-muted-foreground hover:text-foreground transition-colors"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="flex-1 overflow-y-auto p-4 space-y-4 bg-muted/30">
              {messages.length === 0 ? (
                <div className="text-center py-12">
                  <div className="text-5xl mb-4">👋</div>
                  <p className="text-sm text-muted-foreground">Ask a question to start.</p>
                </div>
              ) : (
                messages.map((msg, idx) => (
                  <div
                    key={idx}
                    className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
                  >
                    <div
                      className={`max-w-[85%] rounded-xl px-4 py-2.5 text-sm shadow-sm ${
                        msg.role === "user"
                          ? "bg-slate-900 text-white"
                          : "bg-white border text-gray-800"
                      }`}
                    >
                      {msg.content}
                    </div>
                  </div>
                ))
              )}
              {isLoading && (
                <div className="flex justify-start">
                  <div className="bg-white border rounded-xl px-4 py-2.5 shadow-sm">
                    <div className="flex items-center space-x-2">
                      <Loader2 className="w-3 h-3 text-muted-foreground animate-spin" />
                      <span className="text-xs text-muted-foreground">Thinking...</span>
                    </div>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>

            <div className="p-4 border-t bg-background">
              <form onSubmit={handleSubmit} className="flex space-x-2">
                <input
                  type="text"
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  placeholder="Type a message..."
                  disabled={isLoading}
                  className="flex-1 px-4 py-2.5 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
                <button
                  type="submit"
                  disabled={isLoading || !input.trim()}
                  className="px-4 py-2.5 bg-slate-900 text-white rounded-lg hover:bg-slate-800 disabled:opacity-50 transition-all shadow-sm flex items-center gap-1"
                >
                  <Send className="w-4 h-4" />
                </button>
              </form>
            </div>
          </>
        )}
      </div>

      {/* Toggle Button */}
      {!isOpen && (
        <button
          onClick={() => setIsOpen(true)}
          className="fixed bottom-6 right-6 w-14 h-14 bg-gradient-to-r from-blue-500 to-indigo-600 text-white rounded-full shadow-2xl hover:scale-110 transition-transform flex items-center justify-center"
        >
          <MessageCircle className="w-6 h-6" />
        </button>
      )}
    </div>
  );
}
