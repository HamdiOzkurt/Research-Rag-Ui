-- Supabase Conversation History Schema (minimal)
-- Bu schema, mevcut `src/memory/supabase_memory.py` ile uyumludur.
-- NOT: Production'da RLS policy'leri eklemek istersiniz; backend'de service_role key kullanmanız önerilir.

create table if not exists public.conversations (
  id bigserial primary key,
  session_id text not null,
  message_role text not null,
  message_content text not null,
  metadata jsonb,
  created_at timestamptz not null default now()
);

create index if not exists conversations_session_id_idx on public.conversations (session_id);
create index if not exists conversations_created_at_idx on public.conversations (created_at);

-- =============================================================================
-- RLS (SaaS) - Optional
-- =============================================================================
-- Bu policy'ler iki farklı kullanım için not olarak eklenmiştir:
-- 1) Backend service_role key ile yazıyorsanız: RLS genelde kapalı tutulur (service_role bypass eder).
-- 2) Eğer client-side (anon/authenticated) ile direkt erişim isterseniz: aşağıdaki RLS yaklaşımı gerekir.
--
-- Bizim backend mimarisinde öneri:
-- - SUPABASE_KEY olarak service_role kullan (server-side only)
-- - RLS açık olsa bile service_role bypass ettiği için çalışır
-- - Client tarafına supabase key vermeyiz.
--
-- Yine de ileride "client access" düşünürseniz diye örnek:
--
-- alter table public.conversations enable row level security;
--
-- -- Kullanıcı kendi satırlarını görebilsin (metadata->>user_id = auth.uid() veya clerk id eşlemesi gerekir)
-- -- Not: Clerk user_id UUID değildir; Supabase auth.uid() ile birebir uymaz.
-- -- Bu yüzden client-side erişim için ayrı mapping table gerekir.

