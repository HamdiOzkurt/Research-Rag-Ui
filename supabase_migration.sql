-- Supabase Conversation Memory Table
-- Bu SQL'i Supabase Dashboard > SQL Editor'da çalıştırın

-- Conversations tablosu
CREATE TABLE IF NOT EXISTS conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id TEXT NOT NULL,
    message_role TEXT NOT NULL CHECK (message_role IN ('user', 'assistant', 'system')),
    message_content TEXT NOT NULL,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Index'ler (hızlı sorgular için)
CREATE INDEX IF NOT EXISTS idx_conversations_session ON conversations(session_id);
CREATE INDEX IF NOT EXISTS idx_conversations_created ON conversations(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_conversations_role ON conversations(message_role);

-- Row Level Security (RLS) - Opsiyonel
-- ALTER TABLE conversations ENABLE ROW LEVEL SECURITY;

-- Herkes okuyabilir (public read)
-- CREATE POLICY "Public read access" ON conversations FOR SELECT USING (true);

-- Herkes yazabilir (public write) - Dikkat: Production'da kısıtlayın!
-- CREATE POLICY "Public write access" ON conversations FOR INSERT WITH CHECK (true);

COMMENT ON TABLE conversations IS 'DeepAgents conversation history storage';
COMMENT ON COLUMN conversations.session_id IS 'Unique session identifier for grouping conversations';
COMMENT ON COLUMN conversations.message_role IS 'Message sender: user, assistant, or system';
COMMENT ON COLUMN conversations.message_content IS 'The actual message text';
COMMENT ON COLUMN conversations.metadata IS 'Optional JSON metadata (model, tokens, etc.)';
