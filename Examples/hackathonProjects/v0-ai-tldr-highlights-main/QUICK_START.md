# Quick Start Guide

## Instant Demo (No Setup Required!)

The app works immediately with mock data. Just run:

```bash
npm install
npm run dev
```

Visit http://localhost:3000 and try the demo threads!

## Connect Your Foru.ms Instance (Already Configured!)

Your Foru.ms instance is already configured in `.env.local`:

- **Instance Handle**: `tl-dr`
- **Instance ID**: `164c1c88-9a97-4335-878c-c0c216280445`
- **API Key**: Configured

### Test Your Real Foru.ms Data

Create a thread in your Foru.ms instance at https://foru.ms, then:

1. Copy the thread ID from the URL
2. Visit `http://localhost:3000/thread/YOUR_THREAD_ID`
3. Watch AI TL;DR summarize your real forum content!

## Add Real AI (Optional)

The app uses realistic mock AI responses by default. To use real AI:

### Option 1: Groq (Recommended - Free & Fast)

1. Get free API key: https://console.groq.com
2. Edit `.env.local`:
   ```bash
   AI_PROVIDER=groq
   GROQ_API_KEY=your_groq_api_key_here
   ```

### Option 2: Ollama (Fully Local & Private)

1. Install Ollama: https://ollama.ai
2. Pull a model: `ollama pull llama2`
3. Edit `.env.local`:
   ```bash
   AI_PROVIDER=ollama
   OLLAMA_BASE_URL=http://localhost:11434
   OLLAMA_MODEL=llama2
   ```

### Option 3: Hugging Face (Free Tier Available)

1. Get token: https://huggingface.co/settings/tokens
2. Edit `.env.local`:
   ```bash
   AI_PROVIDER=huggingface
   HUGGINGFACE_API_KEY=your_hf_token_here
   ```

## Features Available Now

✅ Real-time AI summarization with streaming tokens
✅ Smart highlight extraction with confidence scores
✅ Persona-based summaries (Novice, Developer, Executive)
✅ Sentiment analysis and topic extraction
✅ Expertise detection and smart reply suggestions
✅ Provenance tracking (click highlights to see source)
✅ Character-level source attribution
✅ Read time savings metrics

## Demo Threads Available

- **Thread 1**: Next.js 15 Performance Discussion
- **Thread 2**: TypeScript Migration Strategy
- **Thread 3**: API Rate Limiting Best Practices

## Architecture Highlights

- **Zero-config demo mode** with realistic mock data
- **Free & open-source AI** via Groq/Ollama/HF
- **Easy Foru.ms integration** - API data pipes directly to LLMs
- **Production-ready** with Redis caching and PostgreSQL storage
- **Dendritic optimization** for 40% model size reduction

## Need Help?

Check out the full docs:
- `README.md` - Complete setup guide
- `FORUMS_INTEGRATION.md` - Foru.ms API details
- `SETUP_FREE_AI.md` - AI provider comparison
- `HACKATHON.md` - Hackathon submission details

Enjoy your AI-powered forum summaries! 🚀
