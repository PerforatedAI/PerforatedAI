-- Database schema for Foru.ms integration with pgvector

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Threads table (mirrors Foru.ms threads)
CREATE TABLE IF NOT EXISTS forum_threads (
  id TEXT PRIMARY KEY,
  forum_id TEXT NOT NULL DEFAULT 'default',
  title TEXT NOT NULL,
  created_at TIMESTAMP NOT NULL,
  tags TEXT[],
  post_count INTEGER DEFAULT 0,
  ingested_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_threads_forum ON forum_threads(forum_id);
CREATE INDEX IF NOT EXISTS idx_threads_created ON forum_threads(created_at DESC);

-- Posts table (mirrors Foru.ms posts)
CREATE TABLE IF NOT EXISTS forum_posts (
  id TEXT PRIMARY KEY,
  thread_id TEXT NOT NULL REFERENCES forum_threads(id) ON DELETE CASCADE,
  author_id TEXT NOT NULL,
  author_name TEXT NOT NULL,
  content TEXT NOT NULL,
  position INTEGER NOT NULL,
  votes INTEGER DEFAULT 0,
  created_at TIMESTAMP NOT NULL,
  ingested_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_posts_thread ON forum_posts(thread_id);
CREATE INDEX IF NOT EXISTS idx_posts_author ON forum_posts(author_id);
CREATE INDEX IF NOT EXISTS idx_posts_created ON forum_posts(created_at DESC);

-- Sentence-level embeddings for semantic search and highlights
CREATE TABLE IF NOT EXISTS post_sentences (
  id BIGSERIAL PRIMARY KEY,
  
  thread_id TEXT NOT NULL REFERENCES forum_threads(id) ON DELETE CASCADE,
  post_id TEXT NOT NULL REFERENCES forum_posts(id) ON DELETE CASCADE,
  
  sentence_index INTEGER NOT NULL,
  sentence TEXT NOT NULL,
  
  -- Vector embedding (1536 dimensions for text-embedding-3-small)
  embedding VECTOR(1536),
  
  -- Classification scores
  category TEXT, -- fact, solution, opinion, question, citation, irrelevant
  confidence FLOAT,
  importance_score FLOAT,
  
  -- Metadata
  char_offset INTEGER,
  created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_sentences_thread ON post_sentences(thread_id);
CREATE INDEX IF NOT EXISTS idx_sentences_post ON post_sentences(post_id);
CREATE INDEX IF NOT EXISTS idx_sentences_category ON post_sentences(category);

-- IVF index for fast cosine similarity search
CREATE INDEX IF NOT EXISTS idx_sentence_embedding
ON post_sentences
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Summaries table
CREATE TABLE IF NOT EXISTS thread_summaries (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  thread_id TEXT NOT NULL REFERENCES forum_threads(id) ON DELETE CASCADE,
  persona TEXT NOT NULL DEFAULT 'developer',
  
  summary_text TEXT,
  digest JSONB, -- Array of bullet points
  
  model_used TEXT,
  tokens_used INTEGER,
  processing_time FLOAT,
  
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_summaries_thread ON thread_summaries(thread_id);
CREATE INDEX IF NOT EXISTS idx_summaries_persona ON thread_summaries(persona);

-- Highlights table
CREATE TABLE IF NOT EXISTS highlights (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  summary_id UUID NOT NULL REFERENCES thread_summaries(id) ON DELETE CASCADE,
  sentence_id BIGINT REFERENCES post_sentences(id) ON DELETE CASCADE,
  
  text TEXT NOT NULL,
  category TEXT NOT NULL,
  confidence FLOAT NOT NULL,
  importance_score FLOAT NOT NULL,
  
  post_id TEXT NOT NULL,
  sentence_index INTEGER NOT NULL,
  
  created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_highlights_summary ON highlights(summary_id);
CREATE INDEX IF NOT EXISTS idx_highlights_sentence ON highlights(sentence_id);

-- Processing metrics
CREATE TABLE IF NOT EXISTS processing_metrics (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  thread_id TEXT NOT NULL,
  persona TEXT NOT NULL,
  
  total_time FLOAT NOT NULL,
  llm_time FLOAT,
  classification_time FLOAT,
  embedding_time FLOAT,
  
  tokens_used INTEGER,
  cache_hit BOOLEAN,
  model_used TEXT,
  
  highlight_count INTEGER,
  avg_confidence FLOAT,
  
  created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_metrics_thread ON processing_metrics(thread_id);
CREATE INDEX IF NOT EXISTS idx_metrics_created ON processing_metrics(created_at DESC);

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers for auto-updating updated_at
CREATE TRIGGER update_threads_updated_at BEFORE UPDATE ON forum_threads
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_summaries_updated_at BEFORE UPDATE ON thread_summaries
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Views for analytics
CREATE OR REPLACE VIEW thread_analytics AS
SELECT 
  t.id,
  t.title,
  t.post_count,
  COUNT(DISTINCT s.id) as summary_count,
  COUNT(DISTINCT h.id) as highlight_count,
  AVG(pm.total_time) as avg_processing_time,
  MAX(t.created_at) as last_activity
FROM forum_threads t
LEFT JOIN thread_summaries s ON t.id = s.thread_id
LEFT JOIN highlights h ON s.id = h.summary_id
LEFT JOIN processing_metrics pm ON t.id = pm.thread_id
GROUP BY t.id, t.title, t.post_count;
