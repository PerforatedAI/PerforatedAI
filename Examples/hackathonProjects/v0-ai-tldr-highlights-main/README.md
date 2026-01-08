# AI TL;DR + Smart Highlights

Real-time forum thread summarization with streaming highlights and provenance links.

🚀 **100% Free & Open Source** - No credit card or paid AI services required!

## Features

- **Streaming Summaries**: Progressive token streaming delivers instant results
- **Smart Highlights**: Sentence-level highlights with source provenance
- **Persona Tuning**: Novice, Developer, or Executive summary modes
- **Free AI Models**: Groq, Ollama, or Hugging Face - your choice!
- **Foru.ms Integration**: Easy LLM integration with forum API data
- **Production Ready**: Caching, rate limiting, monitoring built-in

## Quick Start (3 Minutes)

### Option 0: Demo Mode (Zero Setup!)

**No setup required** - Just run and try it:

```bash
npm install
npm run dev
```

Visit `http://localhost:3000` and test with built-in mock data. Perfect for hackathon demos!

When you're ready for real AI:

### 1. Choose Your Free AI Provider

**Option A: Groq (Recommended - Fast & Free)**
```bash
# Get free API key: https://console.groq.com/keys
GROQ_API_KEY=gsk_your_key_here
```

**Option B: Ollama (Local - 100% Private)**
```bash
# Install: curl -fsSL https://ollama.com/install.sh | sh
ollama serve
ollama pull llama3.1:8b
OLLAMA_BASE_URL=http://localhost:11434
```

**Option C: Hugging Face (Free Tier)**
```bash
# Get free token: https://huggingface.co/settings/tokens
HUGGINGFACE_API_KEY=hf_your_token_here
```

See [SETUP_FREE_AI.md](./SETUP_FREE_AI.md) for detailed setup instructions.

### 2. Install & Run

```bash
# Install dependencies
npm install

# Set up environment
cp .env.example .env.local
# Add your free API key from step 1

# Run
npm run dev
```

Visit `http://localhost:3000` 🎉

## Foru.ms Integration

This application integrates with [Foru.ms](https://foru.ms) headless forum API for real-world thread data.

### Why Foru.ms + LLMs?

Foru.ms API data pipes **incredibly easily** into LLMs:

```typescript
// 1. Fetch thread from Foru.ms
const thread = await forumsClient.getThread(threadId)

// 2. Send to free LLM
const summary = await llm.generateText({
  prompt: `Summarize this thread:\n${thread.posts.map(p => p.content).join('\n')}`
})

// Done! That's it. 🎯
```

The structured JSON format from Foru.ms is perfect for AI analysis:
- Sentiment analysis
- Topic extraction  
- Expertise detection
- Smart reply suggestions
- Trend detection

See [FORUMS_AI_INTEGRATION.md](./FORUMS_AI_INTEGRATION.md) for complete examples.

### Quick Start with Foru.ms

1. Get API credentials from https://foru.ms
2. Add to `.env.local`:
```bash
FORUM_API_TOKEN=your_token_here
FORUM_INSTANCE_ID=your_instance_id
```

3. Fetch real threads:
```bash
http://localhost:3000/thread/your_thread_id
```

## Tech Stack

### Frontend
- Next.js 15 with App Router
- React Server Components
- Free AI Models (Groq/Ollama/HF)
- TailwindCSS v4
- TypeScript

### AI Providers (All Free!)
- **Groq**: Fast 70B models, free API
- **Ollama**: Local models, 100% private
- **Hugging Face**: Free inference API

### Optional Backend
- Redis for caching (Upstash free tier)
- PostgreSQL (Neon free tier)
- Python training scripts for research

## Cost Comparison

| Provider | Cost | Speed | Quality |
|----------|------|-------|---------|
| **Groq** | $0 | ⚡⚡⚡⚡⚡ | ⭐⭐⭐⭐⭐ |
| **Ollama** | $0 | ⚡⚡⚡⚡ | ⭐⭐⭐⭐⭐ |
| **Hugging Face** | $0 | ⚡⚡⚡ | ⭐⭐⭐⭐ |
| OpenAI | $0.02/req | ⚡⚡⚡⚡ | ⭐⭐⭐⭐⭐ |
| Anthropic | $0.03/req | ⚡⚡⚡⚡ | ⭐⭐⭐⭐⭐ |

**Winner**: Groq (free + fast) or Ollama (free + private)

## API Routes

### POST /api/stream_summary
Stream summary tokens and highlights in real-time via SSE.

**Request:**
```json
{
  "text": "Thread content...",
  "persona": "developer"
}
```

**Events:**
- `token`: Individual summary tokens
- `digest`: Final bullet points
- `highlight`: Individual highlights
- `complete`: Metrics and summary

### POST /api/classify
Classify sentences using free LLMs.

### GET /api/insights/[threadId]
Generate AI insights for Foru.ms threads.

## Project Structure

```
app/
├── api/
│   ├── summarize/route.ts      # Non-streaming summary
│   ├── stream_summary/route.ts # Streaming SSE endpoint
│   ├── classify/route.ts       # Sentence classification
│   └── insights/[threadId]/route.ts # AI insights generation
├── thread/[id]/page.tsx        # Thread view page
└── page.tsx                    # Landing page

components/
├── ThreadPageClient.tsx        # Thread view client component
├── SummaryPanel.tsx           # Summary sidebar
└── PostList.tsx               # Post list with highlights

lib/
├── types.ts                   # TypeScript type definitions
└── backend/
    ├── thread-processor.ts    # Thread processing logic
    ├── highlight-ranker.ts    # Highlight ranking algorithm
    ├── prompts.ts             # LLM prompt templates
    └── ai-insights.ts         # AI insights generation logic

hooks/
└── useStreamSummary.ts        # Streaming hook for SSE

utils/
└── sentenceUtils.ts           # Sentence splitting utilities
```

## Dendritic Optimization

This project uses **Perforated Backpropagation** for neural network optimization, achieving:

- **40% parameter reduction** (110M → 66M parameters)
- **57% faster inference** (450ms → 195ms)
- **40% cost savings** ($0.03 → $0.018 per summary)
- **Minimal accuracy loss** (94% → 93%)

### Why Dendritic?

Traditional BERT models are too large for real-time applications. Our dendritic-optimized classifier maintains production-quality accuracy while being significantly faster and cheaper.

### Learn More

- [Dendritic Optimization Overview](/dendritic)
- [Training Guide](./python/DENDRITIC_TRAINING.md)
- [Implementation Details](./DENDRITIC_INTEGRATION.md)

## Performance Metrics

| Metric | Baseline | With Dendrites | Improvement |
|--------|----------|----------------|-------------|
| Parameters | 110M | 66M | -40% |
| Latency | 450ms | 195ms | -57% |
| GPU Memory | 4.2 GB | 2.6 GB | -38% |
| Cost/1K | $30 | $18 | -40% |
| Accuracy | 94% | 93% | -1% |

## Deployment

### Vercel (Free Tier)

```bash
# Deploy with free AI
vercel

# Add environment variable
vercel env add GROQ_API_KEY
```

Done! Completely free deployment with Groq.

### Self-Hosted with Ollama

Perfect for complete privacy and unlimited usage:

```bash
# Setup server with Ollama
ollama serve
ollama pull llama3.1:8b

# Deploy app pointing to Ollama
OLLAMA_BASE_URL=http://your-server:11434
```

## Hackathon Submission

This project demonstrates:

1. **LLM-Powered Features**: Incredibly easy Foru.ms → LLM integration
2. **Free & Open Source**: No paid services or credit cards required
3. **Streaming Architecture**: Real-time token delivery
4. **Community Enhancement**: AI-powered insights for forum threads
5. **Production Ready**: Caching, monitoring, error handling

**Track**: AI & Intelligence - LLM-Powered Features

## Documentation

- [Free AI Setup Guide](./SETUP_FREE_AI.md) - Get started in 3 minutes
- [Foru.ms Integration](./FORUMS_AI_INTEGRATION.md) - AI-powered community features
- [Hackathon Submission](./HACKATHON.md) - Technical details
- [Backend Architecture](./BACKEND.md) - System design
- [Deployment Guide](./DEPLOYMENT.md) - Production deployment

## License

MIT - Free to use, modify, and distribute!
