'use client'

import { ReactNode, useMemo, useState } from "react";
import { UserButton } from "@clerk/nextjs";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import {
  LayoutDashboard,
  MessageSquareText,
  PanelRightOpen,
  Sparkles,
  ChevronLeft,
  ChevronRight,
  Circle,
} from "lucide-react";

export type UIMode = "dashboard" | "chat" | "sidebar" | "popup";

type Props = {
  mode: UIMode;
  setMode: (m: UIMode) => void;
  backendStatus: "checking" | "online" | "offline";
  stats: any;
  userName?: string | null;
  children: ReactNode;
};

export default function AppShell({
  mode,
  setMode,
  backendStatus,
  stats,
  userName,
  children,
}: Props) {
  const [collapsed, setCollapsed] = useState(false);
  const navBtnClass = (active: boolean) =>
    [
      "w-full justify-start gap-2",
      collapsed ? "px-2" : "",
      // keep sidebar consistently dark (do NOT rely on shadcn `secondary` which is light in :root)
      "text-slate-200 hover:text-white hover:bg-slate-900/60",
      active ? "bg-slate-900/80 text-white shadow-inner" : "",
      "focus-visible:ring-2 focus-visible:ring-slate-700 focus-visible:ring-offset-0",
    ].join(" ");

  const statusText = useMemo(() => {
    if (backendStatus === "online") return "Connected";
    if (backendStatus === "offline") return "Offline";
    return "Checking";
  }, [backendStatus]);

  const statusColor = useMemo(() => {
    if (backendStatus === "online") return "text-emerald-500";
    if (backendStatus === "offline") return "text-rose-500";
    return "text-amber-500";
  }, [backendStatus]);

  return (
    <div className="h-screen w-full bg-muted/30 text-foreground">
      <div className="flex h-full">
        {/* Left sidebar (CopilotKit-docs style) */}
        <aside
          className={[
            "h-full shrink-0 border-r bg-slate-950 text-slate-50",
            collapsed ? "w-[68px]" : "w-[260px]",
            "transition-[width] duration-200",
          ].join(" ")}
        >
          <div className="flex h-14 items-center justify-between px-3">
            <div className="flex items-center gap-2 overflow-hidden">
              <div className="grid h-8 w-8 place-items-center rounded-md bg-slate-800">
                <Sparkles className="h-4 w-4 text-slate-200" />
              </div>
              {!collapsed && (
                <div className="leading-tight">
                  <div className="text-sm font-semibold">AI Research</div>
                  <div className="text-[11px] text-slate-400">Workspace</div>
                </div>
              )}
            </div>
            <Button
              variant="ghost"
              size="icon"
              className="text-slate-200 hover:bg-slate-900"
              onClick={() => setCollapsed((v) => !v)}
            >
              {collapsed ? (
                <ChevronRight className="h-4 w-4" />
              ) : (
                <ChevronLeft className="h-4 w-4" />
              )}
            </Button>
          </div>

          <Separator className="bg-slate-900" />

          <nav className="px-2 py-3">
            <div className={collapsed ? "hidden" : "px-2 pb-2 text-[11px] text-slate-400"}>
              Navigation
            </div>

            <div className="space-y-1">
              <Button
                variant="ghost"
                className={navBtnClass(mode === "dashboard")}
                onClick={() => setMode("dashboard")}
              >
                <LayoutDashboard className="h-4 w-4" />
                {!collapsed && "Dashboard"}
              </Button>
              <Button
                variant="ghost"
                className={navBtnClass(mode === "chat")}
                onClick={() => setMode("chat")}
              >
                <MessageSquareText className="h-4 w-4" />
                {!collapsed && "Chat"}
              </Button>
              <Button
                variant="ghost"
                className={navBtnClass(mode === "sidebar")}
                onClick={() => setMode("sidebar")}
              >
                <PanelRightOpen className="h-4 w-4" />
                {!collapsed && "Sidebar Chat"}
              </Button>
              <Button
                variant="ghost"
                className={navBtnClass(mode === "popup")}
                onClick={() => setMode("popup")}
              >
                <Sparkles className="h-4 w-4" />
                {!collapsed && "Popup"}
              </Button>
            </div>
          </nav>

          <div className="mt-auto p-3">
            <div className="rounded-md bg-slate-900/60 p-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Circle className={`h-3 w-3 ${statusColor}`} fill="currentColor" />
                  {!collapsed && (
                    <div className="text-xs text-slate-200">
                      {statusText}
                      {backendStatus === "online" && stats?.cache ? (
                        <span className="text-slate-400"> • {stats.cache.entries} cached</span>
                      ) : null}
                    </div>
                  )}
                </div>
                {!collapsed && stats?.rate_limit ? (
                  <Badge
                    variant={stats.rate_limit.remaining > 5 ? "secondary" : "destructive"}
                    className={
                      stats.rate_limit.remaining > 5
                        ? "bg-slate-800 text-slate-100 border-slate-700"
                        : "border-transparent"
                    }
                  >
                    {stats.rate_limit.remaining}/{stats.rate_limit.max_requests_per_minute}
                  </Badge>
                ) : null}
              </div>
            </div>
          </div>
        </aside>

        {/* Main */}
        <main className="flex h-full min-w-0 flex-1 flex-col">
          <header className="h-14 border-b bg-background px-4">
            <div className="flex h-full items-center justify-between">
              <div className="min-w-0">
                <div className="truncate text-sm font-medium">
                  {mode === "dashboard"
                    ? "Dashboard"
                    : mode === "chat"
                    ? "Chat"
                    : mode === "sidebar"
                    ? "Sidebar Chat"
                    : "Popup"}
                </div>
                <div className="text-xs text-muted-foreground truncate">
                  {userName ? userName : "Signed in"} • Backend: {statusText}
                </div>
              </div>
              <div className="flex items-center gap-2">
                <UserButton afterSignOutUrl="/sign-in" />
              </div>
            </div>
          </header>

          <div className="flex-1 min-h-0">{children}</div>
        </main>
      </div>
    </div>
  );
}


