'use client'

import { useMemo } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

type Props = {
  stats: any;
};

function pct(used: number, limit: number) {
  if (!limit || limit <= 0) return 0;
  return Math.max(0, Math.min(100, Math.round((used / limit) * 100)));
}

export default function BillingPage({ stats }: Props) {
  const billing = stats?.billing || null;
  const planName: string = billing?.plan || "free";
  const used: number = billing?.monthly_used ?? 0;
  const limit: number = billing?.monthly_limit ?? 0;
  const percent = useMemo(() => pct(used, limit), [used, limit]);

  return (
    <div className="h-full overflow-y-auto bg-gradient-to-br from-slate-50 via-white to-slate-50">
      <div className="max-w-5xl mx-auto p-8 space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold">Usage</h1>
            <p className="text-sm text-muted-foreground">Free plan • aylık kullanım</p>
          </div>
          <Button variant="secondary" disabled>
            Stripe kapalı (Free)
          </Button>
        </div>

        <Card className="border-slate-200/60 shadow-sm">
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle>Mevcut Plan</CardTitle>
              <Badge variant={planName === "free" ? "secondary" : "default"} className={planName === "free" ? "" : "bg-slate-900 text-white"}>
                {planName.toUpperCase()}
              </Badge>
            </div>
            <CardDescription>Aylık kullanım takibi (istek sayısı)</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex items-center justify-between text-sm mb-2">
              <div className="text-muted-foreground">Kullanım</div>
              <div className="font-medium">
                {used}/{limit || "—"}
              </div>
            </div>
            <div className="h-2 w-full rounded-full bg-slate-100 overflow-hidden">
              <div className="h-full bg-slate-900" style={{ width: `${percent}%` }} />
            </div>
            <div className="mt-2 text-xs text-muted-foreground">{percent}%</div>
          </CardContent>
        </Card>

        <div className="rounded-md border border-slate-200 bg-white px-4 py-3 text-sm text-slate-700">
          Şu an ödeme sistemi kapalı. Free limitleri backend’de `FREE_MONTHLY_REQUESTS` ile yönetiyoruz.
        </div>
      </div>
    </div>
  );
}


