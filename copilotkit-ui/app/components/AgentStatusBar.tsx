'use client'

import { Badge } from "@/components/ui/badge";
import { Loader2 } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import type { AgentStatus } from "@/lib/api";

type Props = {
  status: AgentStatus | null;
  mode?: "simple" | "multi" | "deep";
};

const statusConfig = {
  initializing: { color: "bg-blue-500/10 text-blue-600 border-blue-500/20" },
  planning: { color: "bg-purple-500/10 text-purple-600 border-purple-500/20" },
  searching: { color: "bg-cyan-500/10 text-cyan-600 border-cyan-500/20" },
  researching: { color: "bg-yellow-500/10 text-yellow-600 border-yellow-500/20" },
  coding: { color: "bg-green-500/10 text-green-600 border-green-500/20" },
  writing: { color: "bg-indigo-500/10 text-indigo-600 border-indigo-500/20" },
  done: { color: "bg-emerald-500/10 text-emerald-600 border-emerald-500/20" },
  error: { color: "bg-red-500/10 text-red-600 border-red-500/20" },
  needs_approval: { color: "bg-orange-500/10 text-orange-600 border-orange-500/20" },
};

const stageLabels: Record<AgentStatus["status"], string> = {
  initializing: "Başlatılıyor",
  planning: "Plan",
  searching: "Arama",
  researching: "Araştırma",
  coding: "Kod",
  writing: "Yazım",
  done: "Tamamlandı",
  error: "Hata",
  needs_approval: "Onay Bekleniyor",
};

type Stage = { key: AgentStatus["status"]; label: string };

function getStages(mode?: Props["mode"]): Stage[] {
  if (mode === "simple") {
    return [
      { key: "planning", label: "Plan" },
      { key: "writing", label: "Yanıt" },
      { key: "done", label: "Bitti" },
    ];
  }
  // multi + deep
  return [
    { key: "planning", label: "Plan" },
    { key: "searching", label: "Arama" },
    { key: "researching", label: "Araştırma" },
    { key: "coding", label: "Kod" },
    { key: "writing", label: "Yazım" },
    { key: "done", label: "Bitti" },
  ];
}

function formatElapsed(ms: number) {
  const totalSeconds = Math.max(0, Math.floor(ms / 1000));
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}:${seconds.toString().padStart(2, "0")}`;
}

const agentNames: Record<string, string> = {
  simple: "Simple Agent",
  multi: "Multi-Agent",
  supervisor: "Supervisor",
  search: "Web Search",
  researcher: "Researcher",
  coder: "Coder",
  writer: "Writer",
  deep: "Deep Research",
};

export default function AgentStatusBar({ status, mode }: Props) {
  if (!status) return null;

  const config = (statusConfig as Record<string, { color: string }>)[status.status] ?? statusConfig.planning;
  const agentName = status.agent
    ? (agentNames[status.agent] ?? status.agent)
    : agentNames[mode || "simple"];

  const stages = useMemo(() => getStages(mode), [mode]);
  const currentStageIndex = useMemo(() => {
    const idx = stages.findIndex((s) => s.key === status.status);
    if (idx >= 0) return idx;
    // For statuses not in the selected pipeline (e.g., initializing/error), keep it at the closest sensible point.
    if (status.status === "initializing") return 0;
    if (status.status === "error") return Math.max(0, stages.length - 1);
    return 0;
  }, [stages, status.status]);

  const runKey = status.thread_id ?? "__no_thread__";
  const runStartRef = useRef<number | null>(null);
  const lastRunKeyRef = useRef<string>(runKey);
  const [now, setNow] = useState(() => Date.now());

  // Reset timer on new thread OR when a new run starts.
  useEffect(() => {
    const isNewThread = lastRunKeyRef.current !== runKey;
    const isNewRunStart = status.status === "planning";
    if (isNewThread || isNewRunStart || runStartRef.current == null) {
      runStartRef.current = Date.now();
      lastRunKeyRef.current = runKey;
    }
  }, [runKey, status.status]);

  useEffect(() => {
    if (status.status === "done" || status.status === "error") return;
    const id = window.setInterval(() => setNow(Date.now()), 500);
    return () => window.clearInterval(id);
  }, [status.status]);

  const elapsed = runStartRef.current ? formatElapsed(now - runStartRef.current) : "0:00";

  const stageTitle = stageLabels[status.status] ?? "Durum";
  const detail = (status.message || "").trim();
  const showDetail = detail.length > 0 && detail.toLowerCase() !== stageTitle.toLowerCase();

  return (
    <div className="flex items-center gap-3 px-4 py-2 bg-muted/50 border-b">
      <Badge variant="outline" className={`${config.color} border font-medium shrink-0`}>
        {agentName}
      </Badge>

      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2">
          <div className="flex items-center gap-1.5 text-xs text-muted-foreground shrink-0">
            <span className="font-medium text-foreground">{stageTitle}</span>
            {status.status !== "done" && status.status !== "error" && (
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
            )}
            <span className="tabular-nums">{elapsed}</span>
          </div>

          <div className="hidden sm:flex items-center gap-2 min-w-0">
            <div className="flex items-center gap-2">
              {stages.map((s, i) => {
                const isDone = i < currentStageIndex;
                const isActive = i === currentStageIndex;

                return (
                  <div key={s.key} className="flex items-center gap-2">
                    <div
                      className={
                        "h-2 w-2 rounded-full " +
                        (isActive
                          ? "bg-foreground"
                          : isDone
                            ? "bg-foreground/70"
                            : "bg-muted-foreground/30")
                      }
                      aria-label={s.label}
                      title={s.label}
                    />
                    {i < stages.length - 1 && (
                      <div
                        className={
                          "h-px w-6 " +
                          (i < currentStageIndex ? "bg-foreground/40" : "bg-muted-foreground/20")
                        }
                      />
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        {showDetail && (
          <div className="text-sm text-muted-foreground truncate">
            {detail}
          </div>
        )}
      </div>

      {status.cached && (
        <Badge variant="secondary" className="shrink-0 text-xs">
          Cached
        </Badge>
      )}
    </div>
  );
}
