# Foru.ms Integration Guide

Complete integration with Foru.ms headless forum API for production-ready thread summarization.

## Overview

This application integrates with Foru.ms (https://foru.ms) to provide:
- Real-time thread fetching
- Sentence-level provenance linking
- Vote-aware highlight ranking
- Semantic search with pgvector
- Webhook support for live updates

## Architecture

```
Foru.ms API
   ↓
Forums Client (fetch threads)
   ↓
Normalizer (convert to internal types)
   ↓
Thread Processor (split sentences)
   ↓
Embedding Generation
   ↓
PostgreSQL + pgvector
   ↓
Semantic Search + Highlighting
   ↓
Frontend UI (provenance links)
```

## Setup

### 1. Get Foru.ms API Credentials

1. Sign up at https://foru.ms
2. Create a new instance (or use existing)
3. Navigate to Settings to find your credentials:
   - **Instance ID**: Unique identifier for your forum instance
   - **API Key**: Authentication token for API requests
   - **Instance Handle**: URL-friendly identifier (e.g., "tl-dr")
   - **Instance Name**: Display name for your forum

4. Add to `.env.local`:

```bash
FORUM_API_BASE=https://api.foru.ms/v1
FORUM_API_TOKEN=your_api_key_here
FORUM_INSTANCE_ID=your_instance_id_here
FORUM_INSTANCE_HANDLE=your_instance_handle
FORUM_INSTANCE_NAME="Your Forum Name"
```

**Example Configuration:**
```bash
FORUM_API_BASE=https://api.foru.ms/v1
FORUM_API_TOKEN=06f833a8-b799-4668-8954-e893243c2320
FORUM_INSTANCE_ID=164c1c88-9a97-4335-878c-c0c216280445
FORUM_INSTANCE_HANDLE=tl-dr
FORUM_INSTANCE_NAME="tl dr"
```

### 2. Database Setup (Optional - for persistence)

```bash
# Install PostgreSQL with pgvector
brew install postgresql@15
brew install pgvector

# Create database
createdb tldr_db

# Run schema
psql tldr_db < scripts/create_forums_schema.sql
```

### 3. Configure Environment

```bash
DATABASE_URL=postgresql://localhost:5432/tldr_db
PGVECTOR_ENABLED=true
```

## Usage

### Fetch Thread from Foru.ms

```typescript
import { forumsClient } from "@/lib/forums/client"

// Fetch single thread
const thread = await forumsClient.getThread("thread_123")

// Fetch with pagination
const posts = await forumsClient.getAllThreadPosts("thread_123")

// Get latest threads
const threads = await forumsClient.getLatestThreads(1, 20)
```

### Normalize Data

```typescript
import { forumsNormalizer } from "@/lib/forums/normalizer"

// Convert to internal format
const normalizedThread = forumsNormalizer.normalizeThread(forumsThread)

// Convert to plain text
const text = forumsNormalizer.threadToText(normalizedThread)
```

### Ingest into Database

```typescript
import { forumsIngestion } from "@/lib/forums/ingestion"

// Ingest single thread
const result = await forumsIngestion.ingestThread("thread_123")

// Batch ingest
const results = await forumsIngestion.ingestThreads([
  "thread_1",
  "thread_2",
  "thread_3",
])
```

### API Routes

#### GET /api/forums/thread/[id]

Fetch thread from Foru.ms:

```bash
curl http://localhost:3000/api/forums/thread/demo
```

Response:
```json
{
  "success": true,
  "thread": {
    "id": "demo",
    "forum_id": "tech",
    "title": "Thread title",
    "posts": [...]
  }
}
```

#### POST /api/forums/ingest

Ingest threads into database:

```bash
curl -X POST http://localhost:3000/api/forums/ingest \
  -H "Content-Type: application/json" \
  -d '{"thread_id": "demo"}'
```

Response:
```json
{
  "success": true,
  "results": [{
    "thread_id": "demo",
    "posts_count": 15,
    "sentences_count": 142,
    "processing_time": 1234,
    "success": true
  }],
  "summary": {
    "total": 1,
    "successful": 1,
    "failed": 0
  }
}
```

## Database Schema

### Tables

**forum_threads**
- Mirrors Foru.ms threads
- Tracks metadata and tags

**forum_posts**
- Mirrors Foru.ms posts
- Includes votes and author info

**post_sentences**
- Sentence-level granularity
- Vector embeddings (1536 dims)
- Classification scores
- Provenance tracking

**thread_summaries**
- Generated summaries per persona
- Token usage tracking

**highlights**
- Ranked highlights
- Links to source sentences

### Indexes

- IVF index on embeddings for fast cosine search
- B-tree indexes on thread_id, post_id
- Composite indexes for common queries

## Semantic Search

Query similar sentences across threads:

```sql
-- Find semantically similar sentences
SELECT
  sentence,
  post_id,
  1 - (embedding <=> $1) AS similarity
FROM post_sentences
WHERE thread_id = $2
ORDER BY embedding <=> $1
LIMIT 10;
```

TypeScript wrapper:

```typescript
import { searchSimilarSentences } from "@/lib/forums/search"

const results = await searchSimilarSentences(
  "thread_123",
  "How to fix async issues?",
  10
)
```

## Provenance Linking

Click a highlight to jump to source:

```typescript
function jumpToSource(postId: string, sentenceIdx: number) {
  const element = document.getElementById(
    `post-${postId}-sentence-${sentenceIdx}`
  )
  
  if (element) {
    element.scrollIntoView({ behavior: "smooth", block: "center" })
    element.classList.add("highlight-flash")
  }
}
```

CSS animation:

```css
.highlight-flash {
  animation: flash 1s ease-in-out;
}

@keyframes flash {
  0%, 100% { background-color: transparent; }
  50% { background-color: rgb(var(--highlight) / 0.3); }
}
```

## Vote-Aware Ranking

Boost highlights from highly-voted posts:

```typescript
function calculateImportanceScore(
  confidence: number,
  votes: number,
  postPosition: number
): number {
  let score = confidence * 0.5
  
  // Boost by votes (logarithmic)
  score += Math.log1p(votes) * 0.2
  
  // Boost early posts
  if (postPosition < 3) {
    score += 0.15
  }
  
  return Math.min(score, 1.0)
}
```

## Webhooks (Optional)

Receive real-time updates when new posts are created:

### Setup Webhook Endpoint

```typescript
// app/api/webhooks/forums/route.ts
export async function POST(request: Request) {
  const signature = request.headers.get("X-Forum-Signature")
  const body = await request.json()
  
  // Verify signature
  if (!verifyWebhookSignature(body, signature)) {
    return new Response("Invalid signature", { status: 401 })
  }
  
  // Handle events
  if (body.event === "post.created") {
    const { thread_id, post } = body.data
    
    // Re-ingest thread or incrementally add post
    await forumsIngestion.ingestThread(thread_id)
  }
  
  return new Response("OK", { status: 200 })
}
```

### Register Webhook in Foru.ms

```bash
curl -X POST https://api.foru.ms/v1/webhooks \
  -H "Authorization: Bearer $FORUM_API_TOKEN" \
  -d '{
    "url": "https://your-app.vercel.app/api/webhooks/forums",
    "events": ["post.created", "post.updated"],
    "secret": "your_webhook_secret"
  }'
```

## Performance Optimization

### Caching Strategy

```typescript
// Cache thread data for 5 minutes
const thread = await cache.remember(
  `thread:${threadId}`,
  300,
  () => forumsClient.getThread(threadId)
)
```

### Incremental Updates

Only process new posts since last ingestion:

```typescript
async function incrementalUpdate(threadId: string) {
  const lastIngested = await getLastIngestedTimestamp(threadId)
  const newPosts = await forumsClient.getThreadPosts(threadId)
    .filter(p => new Date(p.created_at) > lastIngested)
  
  // Only process new posts
  for (const post of newPosts) {
    await processSinglePost(post)
  }
}
```

### Batch Embedding Generation

Generate embeddings for multiple sentences in one API call:

```typescript
import { generateEmbeddings } from "@/lib/embeddings"

const sentences = posts.flatMap(p => splitIntoSentences(p.content))
const embeddings = await generateEmbeddings(sentences) // Batch API call
```

## Testing

### Mock Foru.ms Data

```typescript
// tests/mocks/forums.ts
export const mockForumsThread = {
  id: "test_thread",
  title: "Test Thread",
  posts: [
    {
      id: "post1",
      author: { username: "alice" },
      content: "Question about async...",
      votes: 5,
      created_at: "2024-01-01T00:00:00Z"
    }
  ]
}
```

### Integration Tests

```typescript
import { forumsClient } from "@/lib/forums/client"

describe("Forums Integration", () => {
  it("fetches thread successfully", async () => {
    const thread = await forumsClient.getThread("demo")
    expect(thread.id).toBe("demo")
    expect(thread.posts.length).toBeGreaterThan(0)
  })
})
```

## Monitoring

Track Foru.ms API usage:

```typescript
// lib/forums/metrics.ts
export function trackApiCall(
  endpoint: string,
  duration: number,
  success: boolean
) {
  console.log(`[Forums API] ${endpoint} - ${duration}ms - ${success ? "OK" : "ERROR"}`)
  
  // Send to analytics
  analytics.track("forums_api_call", {
    endpoint,
    duration,
    success,
    timestamp: new Date().toISOString()
  })
}
```

## Cost Analysis

### API Calls
- Thread fetch: 1 call
- Batch posts: 1 call per 50 posts
- Estimated: ~2-3 calls per thread

### Embeddings
- Cost: $0.00002 per 1K tokens (text-embedding-3-small)
- Average thread: 500 sentences = ~50K tokens
- Cost per thread: $0.001

### LLM Summarization
- Cost: $0.15 per 1M tokens (gpt-4o-mini)
- Average thread: 5K input tokens + 500 output
- Cost per summary: $0.0008

**Total per thread: ~$0.002**

## Production Checklist

- [ ] Set up Foru.ms API credentials
- [ ] Configure PostgreSQL with pgvector
- [ ] Run database migrations
- [ ] Set up Redis for caching
- [ ] Configure webhook endpoints
- [ ] Set up monitoring and logging
- [ ] Test with real Foru.ms threads
- [ ] Optimize embedding generation
- [ ] Configure rate limiting
- [ ] Set up backup strategy

## Troubleshooting

### "Failed to fetch thread"

Check:
1. API token is valid
2. Thread ID exists in Foru.ms
3. Network connectivity
4. Rate limits not exceeded

### "Database connection failed"

Check:
1. DATABASE_URL is correct
2. PostgreSQL is running
3. pgvector extension is installed
4. Migrations have been run

### "Embedding generation slow"

Solutions:
1. Use batch embedding generation
2. Cache embeddings in Redis
3. Use smaller embedding model
4. Process asynchronously

## Examples

### Complete Workflow

```typescript
// 1. Fetch thread from Foru.ms
const forumsThread = await forumsClient.getThread("demo")

// 2. Normalize
const thread = forumsNormalizer.normalizeThread(forumsThread)

// 3. Process and split sentences
const processor = new ThreadProcessor()
const processed = processor.process(thread)

// 4. Generate embeddings
const embeddings = await generateEmbeddings(
  processed.sentences.map(s => s.text)
)

// 5. Store in database
await storeSentences(processed.sentences, embeddings)

// 6. Generate summary
const summary = await generateSummary(processed.text, "developer")

// 7. Extract highlights with semantic search
const highlights = await extractHighlights(thread.id, summary)

// 8. Return to frontend
return { thread, summary, highlights }
```

## Resources

- Foru.ms API Docs: https://docs.foru.ms
- pgvector GitHub: https://github.com/pgvector/pgvector
- OpenAI Embeddings: https://platform.openai.com/docs/guides/embeddings
- Next.js Docs: https://nextjs.org/docs
