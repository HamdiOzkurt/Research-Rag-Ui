from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional, Dict, Any, Tuple

from src.config import settings

try:
    from supabase import create_client, Client
except Exception:  # pragma: no cover
    create_client = None
    Client = object  # type: ignore


def _supabase() -> "Client":
    if not create_client:
        raise RuntimeError("supabase package not installed")
    if not settings.supabase_url or not settings.supabase_key:
        raise RuntimeError("Missing SUPABASE_URL or SUPABASE_KEY")
    return create_client(settings.supabase_url, settings.supabase_key)


def supabase_client() -> "Client":
    """Public accessor (avoid importing private _supabase from other modules)."""
    return _supabase()


@dataclass(frozen=True)
class Plan:
    name: str
    monthly_limit: int
    price_id: Optional[str] = None


def resolve_plan(price_id: Optional[str]) -> Plan:
    # Stripe is optional. If price ids are not configured, everything falls back to free.
    if settings.stripe_price_id_team and price_id == settings.stripe_price_id_team:
        return Plan(name="team", monthly_limit=settings.team_monthly_requests, price_id=price_id)
    if settings.stripe_price_id_pro and price_id == settings.stripe_price_id_pro:
        return Plan(name="pro", monthly_limit=settings.pro_monthly_requests, price_id=price_id)
    return Plan(name="free", monthly_limit=settings.free_monthly_requests, price_id=None)

def get_active_subscription(clerk_user_id: str) -> Optional[Dict[str, Any]]:
    """
    Stripe is optional. If billing tables are not created or Stripe isn't configured,
    this returns None and the system behaves as Free plan.
    """
    try:
        sb = _supabase()
        res = (
            sb.table("billing_subscriptions")
            .select("status,price_id,current_period_end,cancel_at_period_end,stripe_subscription_id")
            .eq("clerk_user_id", clerk_user_id)
            .order("current_period_end", desc=True)
            .limit(1)
            .execute()
        )
        if not res.data:
            return None
        row = res.data[0]
        if row.get("status") in ("active", "trialing"):
            return row
        return None
    except Exception:
        return None


def month_start(d: Optional[date] = None) -> date:
    d = d or date.today()
    return date(d.year, d.month, 1)


def get_usage(clerk_user_id: str, period: Optional[date] = None) -> int:
    p = month_start(period)
    try:
        sb = _supabase()
        res = (
            sb.table("usage_counters")
            .select("request_count")
            .eq("clerk_user_id", clerk_user_id)
            .eq("period_start", str(p))
            .limit(1)
            .execute()
        )
        if not res.data:
            return 0
        return int(res.data[0].get("request_count") or 0)
    except Exception:
        # Fallback: in-memory usage (dev-friendly)
        key = f"{clerk_user_id}:{p}"
        return int(_USAGE_MEMORY.get(key, 0))


def increment_usage(clerk_user_id: str, period: Optional[date] = None, by: int = 1) -> int:
    p = month_start(period)
    current = get_usage(clerk_user_id, p)
    new_val = current + by
    try:
        sb = _supabase()
        sb.table("usage_counters").upsert(
            {"clerk_user_id": clerk_user_id, "period_start": str(p), "request_count": new_val},
            on_conflict="clerk_user_id,period_start",
        ).execute()
    except Exception:
        key = f"{clerk_user_id}:{p}"
        _USAGE_MEMORY[key] = new_val
    return new_val


# Dev fallback store (when Supabase usage table doesn't exist yet)
_USAGE_MEMORY: Dict[str, int] = {}


