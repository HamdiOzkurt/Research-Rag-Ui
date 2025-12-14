'use client'

import { useState, useRef, useEffect } from "react";
import { Send, Loader2, X, MessageCircle, Sparkles } from "lucide-react";
import { useAuth, useUser } from "@clerk/nextjs";
import { chat as chatApi } from "@/lib/api";
import { formatError } from "@/lib/errors";

type Message = {
  role: "user" | "assistant";
  content: string;
};

function useCopilotChat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [threadId, setThreadId] = useState<string | null>(null);
  const { user } = useUser();
  const { getToken } = useAuth();

  useEffect(() => {
    try {
      const stored = localStorage.getItem("threadId_popup");
      if (stored) setThreadId(stored);
    } catch {}
  }, []);

  const appendMessage = async (content: string, role: "user" | "assistant" = "user") => {
    setMessages(prev => [...prev, { role, content }]);
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
            localStorage.setItem("threadId_popup", data.thread_id);
          } catch {}
        }
        setMessages(prev => [...prev, { role: "assistant", content: data.response }]);
      } catch (error) {
        setMessages(prev => [...prev, { role: "assistant", content: `❌ ${formatError(error)}` }]);
      } finally {
        setIsLoading(false);
      }
    }
  };

  return { messages, isLoading, appendMessage };
}

export default function PopupInterface() {
  const { messages, isLoading, appendMessage } = useCopilotChat();
  const [input, setInput] = useState("");
  const [isOpen, setIsOpen] = useState(false);
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

  return (
    <div className="h-full bg-background">
      {/* Main Content */}
      <div className="p-8 max-w-7xl mx-auto">
        <h1 className="text-lg font-semibold mb-1">Your Main Application</h1>
        <p className="text-sm text-muted-foreground mb-6">Floating assistant demo</p>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {[1, 2, 3, 4, 5, 6].map((i) => (
            <div key={i} className="bg-white p-6 rounded-xl shadow-sm border hover:shadow-md transition-all">
              <div className="w-10 h-10 bg-muted rounded-lg flex items-center justify-center mb-4">
                <Sparkles className="w-5 h-5 text-muted-foreground" />
              </div>
              <h3 className="font-semibold text-lg mb-2">Feature {i}</h3>
              <p className="text-sm text-muted-foreground">
                Your app content goes here. The chat popup floats over this content.
              </p>
            </div>
          ))}
        </div>
      </div>

      {/* Floating Popup */}
      {isOpen && (
        <div className="fixed bottom-6 right-6 w-[400px] h-[560px] bg-white rounded-xl shadow-2xl border flex flex-col overflow-hidden animate-in slide-in-from-bottom-8 duration-300">
          {/* Header */}
          <div className="p-4 bg-background border-b flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-9 h-9 bg-slate-900 rounded-md flex items-center justify-center">
                <MessageCircle className="w-4 h-4 text-white" />
              </div>
              <div>
                <h2 className="font-semibold text-sm">Assistant</h2>
                <p className="text-xs text-muted-foreground">Popup</p>
              </div>
            </div>
            <button
              onClick={() => setIsOpen(false)}
              className="text-muted-foreground hover:text-foreground transition-colors"
            >
              <X className="w-5 h-5" />
            </button>
          </div>

          {/* Messages */}
          <div className="flex-1 overflow-y-auto p-4 space-y-3 bg-muted/30">
            {messages.length === 0 ? (
              <div className="text-center py-12">
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

          {/* Input */}
          <div className="p-4 border-t bg-background">
            <form onSubmit={handleSubmit} className="flex space-x-2">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Type a message..."
                disabled={isLoading}
                className="flex-1 px-4 py-2.5 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-purple-500"
              />
              <button
                type="submit"
                disabled={isLoading || !input.trim()}
                className="px-4 py-2.5 bg-slate-900 text-white rounded-lg hover:bg-slate-800 disabled:opacity-50 transition-all shadow-sm"
              >
                <Send className="w-4 h-4" />
              </button>
            </form>
          </div>
        </div>
      )}

      {/* Floating Button */}
      {!isOpen && (
        <button
          onClick={() => setIsOpen(true)}
          className="fixed bottom-6 right-6 w-14 h-14 bg-slate-900 text-white rounded-full shadow-2xl hover:bg-slate-800 transition-transform flex items-center justify-center group"
        >
          <MessageCircle className="w-7 h-7 group-hover:rotate-12 transition-transform" />
          <div className="absolute -top-1 -right-1 w-5 h-5 bg-red-500 rounded-full flex items-center justify-center text-xs font-bold animate-pulse">
            !
          </div>
        </button>
      )}
    </div>
  );
}
