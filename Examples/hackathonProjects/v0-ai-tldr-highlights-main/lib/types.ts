// Type definitions for TL;DR + Smart Highlights

export enum Persona {
  NOVICE = "novice",
  DEVELOPER = "developer",
  EXECUTIVE = "executive",
}

export enum HighlightCategory {
  FACT = "fact",
  SOLUTION = "solution",
  OPINION = "opinion",
  QUESTION = "question",
  CITATION = "citation",
  IRRELEVANT = "irrelevant",
}

export interface Post {
  id: string
  author_id: string
  author_name?: string
  content: string
  position: number
  upvotes: number
  created_at: string
}

export interface Thread {
  id: string
  forum_id: string
  title: string
  posts: Post[]
}

export interface Highlight {
  id: string
  text: string
  category: HighlightCategory
  confidence: number
  importance_score: number
  post_id: string
  author_id: string
  post_position: number
  sentence_index: number
  start_offset: number
  end_offset: number
  created_at: string
}

export interface SummaryResponse {
  summary_id: string
  thread_id: string
  forum_id: string
  persona: Persona
  summary_text?: string
  digest: string[]
  highlights: Highlight[]
  processing_time: number
  tokens_used?: number
  read_time_saved: number
  cache_hit: boolean
  created_at: string
  updated_at: string
  confidence_score?: number
  is_reliable?: boolean
}

export interface StreamEvent {
  type: "token" | "highlight" | "digest" | "complete" | "error" | "progress"
  data: any
  timestamp: string
}

export interface SentenceWithMeta {
  text: string
  post_id: string
  author_id: string
  post_position: number
  sentence_index: number
  char_offset: number
  upvotes: number
  created_at: string
}

export interface ProcessingMetricsSummary {
  latency: {
    firstToken: string
    digest: string
    highlights: string
    total: string
  }
  quality: {
    highlightCount: number
    avgConfidence: string
  }
  cost: {
    tokensUsed: number
    estimatedCost: string
  }
}
