'use client'

import { useState, useRef, useEffect } from "react";
import { Send, Loader2, Trash2, TrendingUp, MessageSquare } from "lucide-react";
import { useUser } from "@clerk/nextjs";
import { chat as chatApi } from "@/lib/api";

type Message = {
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
};

// Modern chat hook
function useCopilotChat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [threadId, setThreadId] = useState<string | null>(null);
  const { user } = useUser();

  useEffect(() => {
    try {
      const stored = localStorage.getItem("threadId_chat");
      if (stored) setThreadId(stored);
    } catch {}
  }, []);

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
        const data = await chatApi({
          message: content,
          userId: user?.id,
          threadId,
          useCache: true,
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
          content: `❌ Backend hatası: ${error}. Backend'i çalıştırın: python -m uvicorn src.simple_copilot_backend:app --reload`,
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
  };
}

export default function ChatInterface() {
  const { messages, isLoading, appendMessage, deleteMessage, clearMessages } = useCopilotChat();
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
    <div className="flex h-[calc(100vh-64px)] bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50">
      <div className="flex-1 flex flex-col max-w-5xl mx-auto w-full">
        {/* Header */}
        <div className="p-6 bg-white/80 backdrop-blur-sm border-b flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="relative">
              <MessageSquare className="w-8 h-8 text-blue-600" />
              <div className="absolute -bottom-1 -right-1 w-3 h-3 bg-green-500 border-2 border-white rounded-full"></div>
            </div>
            <div>
              <h2 className="text-xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
                CopilotChat
              </h2>
              <p className="text-sm text-gray-500 flex items-center gap-2">
                <TrendingUp className="w-3 h-3" />
                Full screen interface
              </p>
            </div>
          </div>
          {messages.length > 0 && (
            <button
              onClick={clearMessages}
              className="flex items-center gap-2 px-4 py-2 text-sm text-gray-600 hover:text-red-600 hover:bg-red-50 rounded-lg transition-all border border-gray-200 hover:border-red-300"
            >
              <Trash2 className="w-4 h-4" />
              Clear
            </button>
          )}
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {messages.length === 0 ? (
            <div className="flex items-center justify-center h-full">
              <div className="text-center max-w-2xl">
                <div className="relative inline-block mb-8">
                  <div className="absolute inset-0 bg-gradient-to-r from-blue-400 to-indigo-400 blur-3xl opacity-20 rounded-full"></div>
                  <div className="relative text-7xl">👋</div>
                </div>
                
                <h2 className="text-3xl font-bold mb-4 bg-gradient-to-r from-gray-900 to-gray-600 bg-clip-text text-transparent">
                  Merhaba! Size nasıl yardımcı olabilirim?
                </h2>
                
                <p className="text-lg text-gray-600 mb-8">
                  Web araştırması yapabilir, kod örnekleri oluşturabilir ve detaylı raporlar hazırlayabilirim.
                </p>
                
                <div className="bg-white rounded-2xl shadow-lg border border-gray-200 p-6">
                  <div className="flex items-center gap-2 mb-4">
                    <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse"></div>
                    <p className="text-sm font-semibold text-gray-700">Örnek Sorular</p>
                  </div>
                  <div className="grid gap-3">
                    {exampleQuestions.map((question, idx) => (
                      <button
                        key={idx}
                        onClick={() => appendMessage(question)}
                        className="text-left p-4 rounded-xl bg-gradient-to-r from-blue-50 to-indigo-50 hover:from-blue-100 hover:to-indigo-100 border border-blue-200 hover:border-blue-300 transition-all group"
                      >
                        <p className="text-sm text-gray-700 group-hover:text-gray-900 font-medium">
                          {question}
                        </p>
                      </button>
                    ))}
                  </div>
                </div>
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
                        ? "bg-gradient-to-r from-blue-500 to-indigo-600 text-white"
                        : "bg-white border border-gray-200"
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
              <div className="bg-white shadow-md rounded-2xl px-6 py-4 border border-gray-200">
                <div className="flex items-center space-x-3">
                  <Loader2 className="w-4 h-4 text-blue-500 animate-spin" />
                  <span className="text-sm text-gray-600">AI düşünüyor...</span>
                </div>
              </div>
            </div>
          )}
          
          <div ref={messagesEndRef} />
        </div>

        {/* Input */}
        <div className="border-t bg-white/80 backdrop-blur-sm p-6">
          <form onSubmit={handleSubmit} className="flex space-x-3">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Bir şey sorun..."
              disabled={isLoading}
              className="flex-1 px-6 py-4 border-2 border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-gray-50 disabled:cursor-not-allowed bg-white shadow-sm transition-all"
            />
            <button
              type="submit"
              disabled={isLoading || !input.trim()}
              className="px-8 py-4 bg-gradient-to-r from-blue-500 to-indigo-600 text-white rounded-xl hover:from-blue-600 hover:to-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all font-medium shadow-lg hover:shadow-xl hover:-translate-y-0.5 flex items-center gap-2"
            >
              {isLoading ? (
                <Loader2 className="w-5 h-5 animate-spin" />
              ) : (
                <Send className="w-5 h-5" />
              )}
              <span>Gönder</span>
            </button>
          </form>
          
          <div className="flex items-center justify-between mt-4 px-2">
            <p className="text-xs text-gray-500">
              Powered by <span className="font-semibold text-blue-600">DeepAgents</span> • {messages.length} mesaj
            </p>
            {messages.length > 0 && (
              <p className="text-xs text-gray-400">
                Shift + Enter for new line
              </p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
