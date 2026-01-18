# Setup Guide: Free & Open-Source AI Models

This guide shows you how to run AI TL;DR + Smart Highlights **completely free** using open-source models. No credit card required!

## Quick Start (3 Options)

### Option 0: Demo Mode (No Setup Required! 🎯)

**Pros:** Zero setup, instant demo, works offline  
**Cons:** Uses pre-generated mock data (not real AI)

Perfect for hackathon demos and testing the UI!

1. **That's it!** No setup needed.
   ```bash
   npm install
   npm run dev
   ```

2. **Visit:** `http://localhost:3000`

3. **Try the demo threads:**
   - "Best practices for streaming LLM responses?" (Tech discussion)
   - "Dendritic optimization for neural networks?" (AI research)
   - "State management patterns for complex forms?" (React patterns)

All highlights, summaries, and insights are pre-generated using real AI models. When you're ready for live AI:

### Option 1: Groq (Recommended ⚡)

**Pros:** Fastest, 70B models, completely free  
**Cons:** Requires internet, API key signup

1. Go to [console.groq.com](https://console.groq.com/keys)
2. Create a free account (no credit card needed)
3. Generate an API key
4. Add to `.env.local`:
   ```bash
   GROQ_API_KEY=gsk_your_key_here
   ```

**Available Models:**
- `llama-3.1-70b-versatile` (fastest, recommended)
- `llama-3.3-70b-versatile` (newer)
- `mixtral-8x7b-32768` (long context)
- `gemma2-9b-it` (smaller, faster)

### Option 2: Ollama (100% Local 🔒)

**Pros:** Completely private, no API keys, unlimited usage  
**Cons:** Requires local setup, uses your GPU/CPU

1. **Install Ollama:**
   ```bash
   # macOS/Linux
   curl -fsSL https://ollama.com/install.sh | sh
   
   # Windows: Download from ollama.com
   ```

2. **Start Ollama:**
   ```bash
   ollama serve
   ```

3. **Pull a model:**
   ```bash
   # Recommended: Llama 3.1 8B (4.7GB)
   ollama pull llama3.1:8b
   
   # Alternatives:
   ollama pull llama3.2:3b        # Smaller, faster (2GB)
   ollama pull mistral:7b          # Good alternative (4.1GB)
   ollama pull phi3:mini           # Tiny, fast (2.3GB)
   ```

4. **Add to `.env.local`:**
   ```bash
   OLLAMA_BASE_URL=http://localhost:11434
   ```

5. **That's it!** No API keys needed.

### Option 3: Hugging Face (Free Tier 🤗)

**Pros:** Free, many models, easy setup  
**Cons:** Rate limits, slower than Groq

1. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Create a free account
3. Generate a "Read" token
4. Add to `.env.local`:
   ```bash
   HUGGINGFACE_API_KEY=hf_your_token_here
   ```

**Available Models:**
- `meta-llama/Llama-3.2-3B-Instruct` (free tier)
- `mistralai/Mistral-7B-Instruct-v0.2`
- `google/flan-t5-large`

## Provider Priority

The app automatically selects providers in this order:

1. **Groq** (if `GROQ_API_KEY` is set)
2. **Ollama** (if `OLLAMA_BASE_URL` is set or running locally)
3. **Hugging Face** (if `HUGGINGFACE_API_KEY` is set)

You can configure multiple providers and the app will fallback automatically.

## Performance Comparison

| Provider | Speed | Quality | Privacy | Cost |
|----------|-------|---------|---------|------|
| **Groq** | ⚡⚡⚡⚡⚡ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | FREE |
| **Ollama** | ⚡⚡⚡⚡ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | FREE |
| **Hugging Face** | ⚡⚡⚡ | ⭐⭐⭐⭐ | ⭐⭐⭐ | FREE |

## Recommended Setup for Hackathon Demo

**Best Demo Experience:**
```bash
# Primary: Groq for speed
GROQ_API_KEY=gsk_your_key_here

# Backup: Ollama for offline demos
OLLAMA_BASE_URL=http://localhost:11434
```

This gives you:
- Lightning-fast responses with Groq
- Offline capability with Ollama
- Zero cost for both

## Installation Steps

1. **Clone the repo:**
   ```bash
   git clone https://github.com/yourusername/ai-tldr.git
   cd ai-tldr
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

3. **Setup environment:**
   ```bash
   cp .env.example .env.local
   # Edit .env.local with your free API keys
   ```

4. **Run the app:**
   ```bash
   npm run dev
   ```

5. **Open browser:**
   ```
   http://localhost:3000
   ```

## Troubleshooting

### Groq Issues

**"No API key provided"**
- Make sure `GROQ_API_KEY` is in `.env.local`
- Restart your dev server after adding the key

**"Rate limit exceeded"**
- Groq free tier has generous limits (14,400 requests/day for Llama 3.1)
- Wait a minute and try again
- Switch to Ollama for unlimited requests

### Ollama Issues

**"Connection refused"**
- Make sure Ollama is running: `ollama serve`
- Check the URL: `http://localhost:11434`

**"Model not found"**
- Pull the model first: `ollama pull llama3.1:8b`
- List available models: `ollama list`

**Slow performance**
- Try a smaller model: `ollama pull llama3.2:3b`
- Close other GPU-intensive apps
- Check RAM usage (models need 4-16GB)

### Hugging Face Issues

**"Model is loading"**
- Free tier models "cold start" when first used
- Wait 30-60 seconds and retry
- Use Groq or Ollama for instant responses

**"Rate limit"**
- Free tier has 1,000 requests/month
- Upgrade to PRO ($9/month) or use Groq/Ollama

## Cost Comparison

| Service | Free Tier | Paid Tier | Credit Card |
|---------|-----------|-----------|-------------|
| **Groq** | 14,400 req/day | N/A | ❌ No |
| **Ollama** | Unlimited | N/A | ❌ No |
| **Hugging Face** | 1,000 req/month | $9/month | ❌ No |
| OpenAI | $5 credit (expires) | Pay-as-you-go | ✅ Yes |
| Anthropic | N/A | Pay-as-you-go | ✅ Yes |

## Advanced: Running Multiple Models

You can run multiple Ollama models simultaneously:

```bash
# Terminal 1: Fast model for summaries
OLLAMA_HOST=localhost:11434 ollama serve &
ollama pull llama3.1:8b

# Terminal 2: Larger model for insights
OLLAMA_HOST=localhost:11435 ollama serve &
ollama pull llama3.1:70b
```

Then configure different endpoints for different features.

## Production Deployment

### Vercel (Free Tier)

1. Deploy normally with Groq API key
2. Add environment variable in Vercel dashboard:
   ```
   GROQ_API_KEY=gsk_your_key
   ```
3. Done! Completely free with great performance.

### Self-Hosted with Ollama

1. Setup a server with GPU (optional but faster)
2. Install Ollama
3. Pull models
4. Set `OLLAMA_BASE_URL` to your server
5. Completely free, unlimited usage!

## FAQ

**Q: Do I need a GPU for Ollama?**  
A: No! CPU works fine for smaller models (3-8B). GPU is faster but optional.

**Q: Which provider should I use?**  
A: Groq for speed, Ollama for privacy/unlimited usage.

**Q: Can I use multiple providers?**  
A: Yes! The app will automatically try them in priority order.

**Q: Is this really free?**  
A: Yes! All three options are completely free with no credit card required.

**Q: What about OpenAI/Anthropic?**  
A: They require credit cards and charge per token. We use 100% free alternatives!

## Support

Having issues? Check:
1. [Groq Discord](https://discord.gg/groq)
2. [Ollama GitHub Issues](https://github.com/ollama/ollama/issues)
3. [Hugging Face Forums](https://discuss.huggingface.co/)

## Next Steps

- [📖 Read the full documentation](./README.md)
- [🎯 See the hackathon submission](./HACKATHON.md)
- [🔧 Explore advanced features](./BACKEND.md)
