'use client'

import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { MessageSquare, Loader2 } from "lucide-react";
import { listThreads } from "@/lib/api";

type Thread = {
  thread_id: string;
  last_message_at: string;
  preview: string;
};

type Props = {
  onResumeThread: (threadId: string) => void;
};

export default function ThreadsList({ onResumeThread }: Props) {
  const [threads, setThreads] = useState<Thread[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const loadThreads = async () => {
      try {
        setLoading(true);
        const data = await listThreads(20);
        setThreads(Array.isArray(data) ? data : []);
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : "Failed to load threads";
        setError(msg);
      } finally {
        setLoading(false);
      }
    };
    loadThreads();
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="w-6 h-6 animate-spin text-muted-foreground" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-6 text-center text-sm text-destructive">
        {error}
      </div>
    );
  }

  if (!threads.length) {
    return (
      <div className="p-6 text-center text-sm text-muted-foreground">
        No conversation threads yet. Start chatting to create one!
      </div>
    );
  }

  return (
    <div className="p-4 space-y-3">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-semibold">Conversation History</h3>
        <Badge variant="secondary" className="text-xs">{threads.length} threads</Badge>
      </div>
      {threads.map((thread) => (
        <Card key={thread.thread_id} className="hover:shadow-md transition-shadow">
          <CardHeader className="pb-3">
            <div className="flex items-start justify-between gap-2">
              <div className="flex items-center gap-2 min-w-0">
                <MessageSquare className="w-4 h-4 text-muted-foreground shrink-0" />
                <CardDescription className="text-xs truncate">
                  {new Date(thread.last_message_at).toLocaleString()}
                </CardDescription>
              </div>
            </div>
          </CardHeader>
          <CardContent className="pt-0">
            <p className="text-sm text-muted-foreground mb-3 line-clamp-2">
              {thread.preview || "(No preview)"}
            </p>
            <div className="flex items-center gap-2">
              <Button
                size="sm"
                variant="outline"
                onClick={() => onResumeThread(thread.thread_id)}
                className="text-xs"
              >
                Resume
              </Button>
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  );
}
