# Deployment Guide

## Quick Deploy to Vercel (Recommended)

### 1. Prerequisites
- GitHub account
- Vercel account (free tier works)
- OpenAI API key

### 2. Deploy Steps

```bash
# Install Vercel CLI
npm i -g vercel

# Login to Vercel
vercel login

# Deploy
vercel

# Follow prompts to connect your GitHub repo
```

### 3. Configure Environment Variables

In your Vercel dashboard:

1. Go to **Project Settings** → **Environment Variables**
2. Add the following:
   - `OPENAI_API_KEY`: Your OpenAI API key
   - `NEXT_PUBLIC_STREAM_URL`: `/api/stream_summary` (default)

### 4. Deploy to Production

```bash
vercel --prod
```

Your app will be live at `https://your-project.vercel.app`

---

## Alternative: Docker Deployment

### 1. Build Docker Image

```bash
# Build
docker build -t tldr-app .

# Run locally
docker run -p 3000:3000 \
  -e OPENAI_API_KEY=your_key \
  tldr-app
```

### 2. Docker Compose (with Redis)

```yaml
# docker-compose.yml
version: '3.8'
services:
  app:
    build: .
    ports:
      - "3000:3000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - REDIS_URL=redis://redis:6379
    depends_on:
      - redis
  
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

volumes:
  redis_data:
```

Run with:
```bash
docker-compose up -d
```

---

## Production Checklist

### Security
- [ ] Rotate API keys regularly
- [ ] Set up rate limiting
- [ ] Enable HTTPS only
- [ ] Configure CORS properly
- [ ] Add authentication if needed

### Performance
- [ ] Enable caching (Redis)
- [ ] Configure CDN (Vercel does this automatically)
- [ ] Set up monitoring (Vercel Analytics)
- [ ] Optimize images with Next.js Image

### Monitoring
- [ ] Set up error tracking (Sentry)
- [ ] Monitor API usage costs
- [ ] Track latency metrics
- [ ] Set up uptime monitoring

### Cost Optimization
- [ ] Use gpt-4o-mini for cost efficiency
- [ ] Cache frequently requested summaries
- [ ] Implement request deduplication
- [ ] Set token limits per request

---

## Environment-Specific Configs

### Development
```bash
OPENAI_API_KEY=sk-...
NEXT_PUBLIC_STREAM_URL=/api/stream_summary
```

### Staging
```bash
OPENAI_API_KEY=sk-...
NEXT_PUBLIC_STREAM_URL=/api/stream_summary
CACHE_ENABLED=true
REDIS_URL=redis://staging-redis:6379
```

### Production
```bash
OPENAI_API_KEY=sk-...
NEXT_PUBLIC_STREAM_URL=/api/stream_summary
CACHE_ENABLED=true
REDIS_URL=redis://prod-redis:6379
DATABASE_URL=postgresql://...
```

---

## Troubleshooting

### Issue: White screen on deploy
**Solution**: Check browser console for errors. Ensure all environment variables are set.

### Issue: Streaming not working
**Solution**: Verify `OPENAI_API_KEY` is set and valid. Check API quotas.

### Issue: High latency
**Solution**: Enable caching with Redis. Use gpt-4o-mini model. Deploy to region closest to users.

### Issue: Cost concerns
**Solution**: Set max token limits. Implement caching. Use cheaper models for drafts.

---

## Scaling

### Horizontal Scaling (Vercel)
Vercel automatically scales based on traffic. No configuration needed.

### Vertical Scaling (Self-hosted)
- Increase memory allocation for LLM inference
- Use faster Redis (Redis Cloud)
- Add database read replicas

### Load Balancing
If self-hosting, use nginx or similar:

```nginx
upstream tldr_app {
    server localhost:3000;
    server localhost:3001;
    server localhost:3002;
}

server {
    listen 80;
    location / {
        proxy_pass http://tldr_app;
    }
}
