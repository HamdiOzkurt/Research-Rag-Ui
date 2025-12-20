-- Billing schema for Clerk + Stripe
-- Note: If you use Clerk (not Supabase Auth), true end-user RLS policies are non-trivial.
-- Recommended: keep these tables private and access them ONLY via backend using Supabase service role key.

-- Customers map (Clerk user -> Stripe customer)
create table if not exists billing_customers (
  clerk_user_id text primary key,
  stripe_customer_id text unique not null,
  email text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

-- Subscriptions (current status)
create table if not exists billing_subscriptions (
  id uuid primary key default gen_random_uuid(),
  clerk_user_id text not null references billing_customers(clerk_user_id) on delete cascade,
  stripe_subscription_id text unique not null,
  status text,
  price_id text,
  cancel_at_period_end boolean not null default false,
  current_period_start timestamptz,
  current_period_end timestamptz,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index if not exists billing_subscriptions_user_idx on billing_subscriptions (clerk_user_id);
create index if not exists billing_subscriptions_status_idx on billing_subscriptions (status);

-- Usage counters (monthly)
create table if not exists usage_counters (
  clerk_user_id text not null references billing_customers(clerk_user_id) on delete cascade,
  period_start date not null,
  request_count integer not null default 0,
  updated_at timestamptz not null default now(),
  primary key (clerk_user_id, period_start)
);

-- (Optional) disable public access via RLS (recommended for Clerk setups)
alter table billing_customers disable row level security;
alter table billing_subscriptions disable row level security;
alter table usage_counters disable row level security;


