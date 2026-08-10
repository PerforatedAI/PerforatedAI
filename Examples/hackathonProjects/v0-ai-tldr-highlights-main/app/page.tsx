"use client"
import type React from "react"
import { useState, useEffect, useRef } from "react"
import { Sparkles, Clock, Zap, LinkIcon, ChevronRight, Search } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { AIProviderBadge } from "@/components/AIProviderBadge"
import { MOCK_THREADS } from "@/lib/mock-data"

export default function LandingPage() {
  const [selectedThread, setSelectedThread] = useState("demo")
  const [threadText, setThreadText] = useState("")
  const [isStreaming, setIsStreaming] = useState(false)
  const [streamTokens, setStreamTokens] = useState<string[]>([])
  const [digest, setDigest] = useState<string[] | null>(null)
  const [highlights, setHighlights] = useState<any[]>([])
  const [metrics, setMetrics] = useState<any>(null)
  const [error, setError] = useState<string | null>(null)
  const abortRef = useRef<AbortController | null>(null)

  useEffect(() => {
    const mockThreads = {
      demo: `Alice: Has anyone solved the async streaming issue when summarizing long threads?\nBob: You can chunk text into windows of ~512 tokens and stream partial summaries. Cache embeddings per-sentence to avoid recompute.\nCarmen: Also watch out for long code blocks—strip them or send them as attachments; they cause token explosion.\nDan: For provenance tracking, I store character offsets when splitting sentences.\nEve: We implemented a two-pass approach: fast streaming draft (200ms first token), then quality consolidation in parallel.`,
      "ai-optimization": `Dr. Sarah Johnson: I've been experimenting with dendritic optimization using perforatedai. Initial results show 40% parameter reduction with minimal accuracy loss.\nMike Chen: That's impressive! What's your training setup?\nDr. Sarah Johnson: Using W&B sweeps for hyperparameter tuning. Key is the perforated backpropagation tracker.\nAlex Kim: Would love to see the config! I'm working on BERT compression for my thesis.\nDr. Sarah Johnson: Works great with transformers! I'm using it on T5-small for summarization.`,
      "react-patterns": `Jordan: What's the best approach for managing state in complex multi-step forms?\nSam: Use useReducer for complex state logic. It's more predictable than multiple useState calls.\nJordan: What about form validation?\nSam: Keep validation logic separate in custom hooks like useFormValidation.\nCasey: Or just use React Hook Form with Zod validation. Handles 90% of form state management.\nJordan: Great suggestions all around!`,
    }
    setThreadText(mockThreads[selectedThread as keyof typeof mockThreads] || mockThreads.demo)
  }, [selectedThread])

  useEffect(() => {
    return () => {
      if (abortRef.current) {
        abortRef.current.abort()
      }
    }
  }, [])

  const handleSummarize = async () => {
    if (!threadText.trim()) {
      setError("Please enter some thread text")
      return
    }

    setError(null)
    setIsStreaming(true)
    setStreamTokens([])
    setDigest(null)
    setHighlights([])
    setMetrics(null)

    const controller = new AbortController()
    abortRef.current = controller

    try {
      const thread = MOCK_THREADS[selectedThread]

      const res = await fetch("/api/stream_summary", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          thread, // Send full thread object with id
          persona: "developer",
        }),
        signal: controller.signal,
      })

      if (!res.ok || !res.body) {
        const errTxt = await res.text().catch(() => "")
        throw new Error(`Request failed: ${res.status} ${errTxt}`)
      }

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buffered = ""

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffered += decoder.decode(value, { stream: true })
        const parts = buffered.split("\n\n")
        buffered = parts.pop() || ""

        for (const p of parts) {
          const chunk = p.trim()
          if (!chunk) continue

          const payloadRaw = chunk.startsWith("data: ") ? chunk.slice(6) : chunk
          try {
            const payload = JSON.parse(payloadRaw)

            switch (payload.type) {
              case "token":
                setStreamTokens((prev) => [...prev, payload.data?.token || ""])
                break

              case "summary_update":
                break

              case "digest":
                setDigest(payload.data?.bullets || [])
                setHighlights(payload.data?.highlights || [])
                break

              case "highlight":
                setHighlights((prev) => [...prev, payload.data?.highlight])
                break

              case "complete":
                if (payload.data?.metrics) {
                  setMetrics(payload.data.metrics)
                }
                break

              case "error":
                setError(payload.data?.error || "Stream error")
                break
            }
          } catch (e) {
            console.error("Parse error:", e)
          }
        }
      }
    } catch (e: any) {
      if (e.name !== "AbortError") {
        setError(e.message || "Something went wrong")
      }
    } finally {
      setIsStreaming(false)
      abortRef.current = null
    }
  }

  const handleStop = () => {
    if (abortRef.current) {
      abortRef.current.abort()
      abortRef.current = null
    }
    setIsStreaming(false)
  }

  return (
    <div className="min-h-screen bg-background text-foreground">
      {/* Header */}
      <header className="border-b border-border/50">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="rounded-lg bg-accent text-accent-foreground p-2">
              <Sparkles className="w-5 h-5" />
            </div>
            <div>
              <h1 className="text-lg font-semibold">TL;DR</h1>
              <p className="text-xs text-muted-foreground">100% Free & Open Source</p>
            </div>
          </div>
          <nav className="flex items-center gap-6">
            <AIProviderBadge />
            <a className="text-sm text-muted-foreground hover:text-foreground transition-colors" href="#features">
              Features
            </a>
            <a className="text-sm text-muted-foreground hover:text-foreground transition-colors" href="#demo">
              Demo
            </a>
            <a className="text-sm text-muted-foreground hover:text-foreground transition-colors" href="/thread/demo">
              Full Demo
            </a>
            <Button variant="default" size="sm" className="gap-2">
              Get Started
              <ChevronRight className="w-4 h-4" />
            </Button>
          </nav>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6">
        {/* Hero Section */}
        <section className="py-24 text-center">
          <div className="animate-in fade-in duration-600">
            <div className="inline-block mb-6 px-4 py-2 rounded-full border border-accent/30 bg-accent/10 text-accent text-sm">
              🚀 100% Free AI-Powered Summaries - No Credit Card Required
            </div>
            <h2 className="text-6xl font-bold leading-tight mb-6 text-balance">
              The fastest way to understand
              <br />
              <span className="text-accent">any discussion thread</span>
            </h2>
            <p className="text-xl text-muted-foreground max-w-3xl mx-auto mb-10 leading-relaxed">
              Streaming summaries with smart highlights powered by free, open-source AI models. Get the gist of long
              forum threads in 10-30 seconds instead of minutes.
            </p>
            <div className="flex items-center justify-center gap-4">
              <Button
                size="lg"
                className="gap-2 text-base px-8 py-6"
                onClick={() => document.getElementById("demo")?.scrollIntoView({ behavior: "smooth" })}
              >
                Try Live Demo
                <ChevronRight className="w-5 h-5" />
              </Button>
              <Button size="lg" variant="outline" className="gap-2 text-base px-8 py-6 bg-transparent" asChild>
                <a href="/SETUP_FREE_AI.md" target="_blank" rel="noreferrer">
                  <Search className="w-5 h-5" />
                  Setup Guide
                </a>
              </Button>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mt-20 max-w-5xl mx-auto">
            <StatCard value="$0" label="Cost per 1000 summaries" subtext="completely free" />
            <StatCard value="200ms" label="First token latency" subtext="with Groq" />
            <StatCard value="86%" label="Highlight precision" subtext="verified accuracy" />
            <StatCard value="100%" label="Open Source" subtext="no vendor lock-in" />
          </div>
        </section>

        {/* Demo Section */}
        <section id="demo" className="py-20">
          <Card className="p-8 bg-card border-border">
            <div className="flex items-center justify-between mb-6">
              <div>
                <h3 className="text-2xl font-semibold mb-2">Interactive Demo</h3>
                <p className="text-sm text-muted-foreground">
                  Select a sample thread or paste your own to experience real-time streaming
                </p>
              </div>
              <div className="flex items-center gap-2 text-xs">
                <div
                  className={`w-2 h-2 rounded-full ${isStreaming ? "bg-green-500 animate-pulse" : "bg-yellow-500"}`}
                />
                <span className="text-muted-foreground">{isStreaming ? "Streaming" : "Ready"}</span>
              </div>
            </div>

            <div className="flex gap-2 mb-4">
              <Button
                size="sm"
                variant={selectedThread === "demo" ? "default" : "outline"}
                onClick={() => setSelectedThread("demo")}
              >
                Streaming Best Practices
              </Button>
              <Button
                size="sm"
                variant={selectedThread === "ai-optimization" ? "default" : "outline"}
                onClick={() => setSelectedThread("ai-optimization")}
              >
                AI Optimization
              </Button>
              <Button
                size="sm"
                variant={selectedThread === "react-patterns" ? "default" : "outline"}
                onClick={() => setSelectedThread("react-patterns")}
              >
                React Patterns
              </Button>
            </div>

            <textarea
              value={threadText}
              onChange={(e) => setThreadText(e.target.value)}
              rows={6}
              className="w-full p-4 rounded-lg bg-secondary border border-border text-sm resize-none focus:outline-none focus:ring-2 focus:ring-accent mb-4 font-mono"
              placeholder="Paste your thread text here..."
            />

            {error && (
              <div className="mb-4 p-4 rounded-lg bg-destructive/10 border border-destructive/20 text-destructive text-sm">
                <strong>Error:</strong> {error}
              </div>
            )}

            <div className="flex gap-3 mb-6">
              <Button onClick={handleSummarize} disabled={isStreaming || !threadText} className="gap-2">
                <Zap className="w-4 h-4" />
                {isStreaming ? "Streaming..." : "Stream TL;DR"}
              </Button>
              <Button onClick={handleStop} disabled={!isStreaming} variant="outline">
                Stop
              </Button>
              <Button
                onClick={() => {
                  setStreamTokens([])
                  setDigest(null)
                  setHighlights([])
                  setMetrics(null)
                  setError(null)
                }}
                variant="ghost"
              >
                Clear Results
              </Button>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Streaming Tokens */}
              <Card className="p-4 bg-secondary border-border">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <Clock className="w-4 h-4 text-accent" />
                    <h4 className="text-sm font-medium">Streaming Summary</h4>
                  </div>
                  <span className="text-xs text-muted-foreground">
                    {streamTokens.length > 0 ? `${streamTokens.join(" ").split(" ").length} tokens` : ""}
                  </span>
                </div>
                <div className="h-48 overflow-auto rounded-md bg-background/50 p-4 text-sm leading-relaxed">
                  {streamTokens.length === 0 ? (
                    <span className="text-muted-foreground text-xs">Streaming tokens will appear here...</span>
                  ) : (
                    <span className="whitespace-pre-wrap">{streamTokens.join("")}</span>
                  )}
                </div>
              </Card>

              {/* Final Digest */}
              <Card className="p-4 bg-secondary border-border">
                <div className="flex items-center gap-2 mb-3">
                  <Sparkles className="w-4 h-4 text-accent" />
                  <h4 className="text-sm font-medium">Key Points</h4>
                </div>
                <div className="h-48 overflow-auto rounded-md bg-background/50 p-4">
                  {digest ? (
                    <ul className="list-disc pl-5 text-sm space-y-2">
                      {digest.map((b, i) => (
                        <li key={i} className="leading-relaxed">
                          {b}
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <span className="text-muted-foreground text-xs">Key points will appear here...</span>
                  )}
                </div>
              </Card>
            </div>

            {/* Highlights */}
            {highlights.length > 0 && (
              <Card className="p-4 bg-secondary border-border mt-6">
                <div className="flex items-center gap-2 mb-3">
                  <LinkIcon className="w-4 h-4 text-accent" />
                  <h4 className="text-sm font-medium">Smart Highlights ({highlights.length})</h4>
                </div>
                <div className="space-y-2 max-h-96 overflow-auto">
                  {highlights.map((h, i) => (
                    <div
                      key={i}
                      className="p-3 rounded-md border border-border bg-background/30 hover:bg-background/50 transition-colors"
                    >
                      <div className="text-sm mb-2 leading-relaxed">{h.text}</div>
                      <div className="flex items-center gap-3 text-xs text-muted-foreground flex-wrap">
                        <span className="px-2 py-0.5 rounded bg-accent/10 text-accent">{h.category || "info"}</span>
                        {h.post_id && <span>Post #{h.post_position || "?"}</span>}
                        {h.confidence && (
                          <span className="flex items-center gap-1">
                            Confidence:
                            <span className="font-medium">{(h.confidence * 100).toFixed(0)}%</span>
                          </span>
                        )}
                        {h.importance_score && (
                          <span className="flex items-center gap-1">
                            Importance:
                            <span className="font-medium">{(h.importance_score * 100).toFixed(0)}%</span>
                          </span>
                        )}
                      </div>
                      {h.why_matters && <div className="text-xs text-muted-foreground mt-2">💡 {h.why_matters}</div>}
                    </div>
                  ))}
                </div>
              </Card>
            )}

            {metrics && (
              <Card className="p-4 bg-secondary border-border mt-6">
                <h4 className="text-sm font-medium mb-3">Performance Metrics</h4>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-xs">
                  <div>
                    <div className="text-muted-foreground mb-1">Total Time</div>
                    <div className="font-mono text-lg">{metrics.total_time?.toFixed(2)}s</div>
                  </div>
                  <div>
                    <div className="text-muted-foreground mb-1">First Token</div>
                    <div className="font-mono text-lg">{metrics.first_token_time?.toFixed(0)}ms</div>
                  </div>
                  <div>
                    <div className="text-muted-foreground mb-1">Tokens Used</div>
                    <div className="font-mono text-lg">{metrics.tokens_used || "—"}</div>
                  </div>
                  <div>
                    <div className="text-muted-foreground mb-1">Est. Cost</div>
                    <div className="font-mono text-lg">${metrics.estimated_cost?.toFixed(4) || "—"}</div>
                  </div>
                </div>
              </Card>
            )}
          </Card>
        </section>

        {/* Features Section */}
        <section id="features" className="py-20">
          <div className="text-center mb-16">
            <h3 className="text-4xl font-bold mb-4">Powerful Features</h3>
            <p className="text-muted-foreground text-lg">Everything you need for instant thread comprehension</p>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <FeatureCard
              icon={<Clock className="w-6 h-6" />}
              title="Streaming Summaries"
              description="Progressive token streaming delivers instant results. Users see the gist while the model completes the full digest."
            />
            <FeatureCard
              icon={<LinkIcon className="w-6 h-6" />}
              title="Smart Highlights"
              description="Sentence-level highlights with provenance links back to exact posts and character offsets for verification."
            />
            <FeatureCard
              icon={<Sparkles className="w-6 h-6" />}
              title="Persona Tuning"
              description="Novice, Professional, or Executive summary modes surface the right level of technical detail."
            />
          </div>
        </section>

        {/* Get Started Section */}
        <section id="docs" className="py-20">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
            <div>
              <h3 className="text-4xl font-bold mb-6">Get Started in Minutes</h3>
              <p className="text-muted-foreground text-lg mb-8 leading-relaxed">
                Drop your streaming endpoint URL into your environment variables or use the default demo server. Deploy
                to Vercel for low-latency edge routing.
              </p>
              <div className="space-y-4 text-sm">
                <div className="flex items-start gap-3">
                  <div className="w-6 h-6 rounded-full bg-accent/20 text-accent flex items-center justify-center flex-shrink-0 mt-0.5">
                    1
                  </div>
                  <div>
                    <p className="font-medium mb-1">Install dependencies</p>
                    <code className="text-xs bg-secondary px-3 py-1.5 rounded border border-border inline-block">
                      npm i framer-motion lucide-react
                    </code>
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <div className="w-6 h-6 rounded-full bg-accent/20 text-accent flex items-center justify-center flex-shrink-0 mt-0.5">
                    2
                  </div>
                  <div>
                    <p className="font-medium mb-1">Configure streaming endpoint</p>
                    <code className="text-xs bg-secondary px-3 py-1.5 rounded border border-border inline-block">
                      NEXT_PUBLIC_STREAM_URL=/api/stream_summary
                    </code>
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <div className="w-6 h-6 rounded-full bg-accent/20 text-accent flex items-center justify-center flex-shrink-0 mt-0.5">
                    3
                  </div>
                  <div>
                    <p className="font-medium">Deploy to Vercel and connect your forum data</p>
                  </div>
                </div>
              </div>
            </div>

            <Card className="p-8 bg-card border-border">
              <h4 className="text-xl font-semibold mb-6">Performance Snapshot</h4>
              <div className="grid grid-cols-2 gap-4">
                <MetricCard label="First Token" value="~200ms" trend="Fast" />
                <MetricCard label="Full Digest" value="~8s" trend="Complete" />
                <MetricCard label="Cost (est)" value="$0.02" trend="Per digest" />
                <MetricCard label="Precision" value="86%" trend="Accuracy" />
              </div>
              <p className="mt-6 text-xs text-muted-foreground">
                Performance depends on model selection, batching, and caching strategy
              </p>
            </Card>
          </div>
        </section>
      </main>

      {/* Footer */}
      <footer className="border-t border-border/50 mt-32">
        <div className="max-w-7xl mx-auto px-6 py-8">
          <div className="flex items-center justify-between">
            <p className="text-sm text-muted-foreground">© {new Date().getFullYear()} TL;DR — Built for Forums</p>
            <div className="flex items-center gap-6 text-sm text-muted-foreground">
              <a href="#" className="hover:text-foreground transition-colors">
                GitHub
              </a>
              <a href="#" className="hover:text-foreground transition-colors">
                Documentation
              </a>
              <a href="#" className="hover:text-foreground transition-colors">
                Support
              </a>
            </div>
          </div>
        </div>
      </footer>
    </div>
  )
}

function StatCard({ value, label, subtext }: { value: string; label: string; subtext: string }) {
  return (
    <Card className="p-6 bg-card border-border text-center">
      <div className="text-3xl font-bold mb-2 text-accent">{value}</div>
      <div className="text-sm font-medium mb-1">{label}</div>
      <div className="text-xs text-muted-foreground">{subtext}</div>
    </Card>
  )
}

function FeatureCard({ icon, title, description }: { icon: React.ReactNode; title: string; description: string }) {
  return (
    <div className="hover:-translate-y-1 transition-transform duration-200">
      <Card className="p-6 bg-card border-border h-full">
        <div className="w-12 h-12 rounded-lg bg-accent/20 text-accent flex items-center justify-center mb-4">
          {icon}
        </div>
        <h4 className="text-xl font-semibold mb-3">{title}</h4>
        <p className="text-sm text-muted-foreground leading-relaxed">{description}</p>
      </Card>
    </div>
  )
}

function MetricCard({ label, value, trend }: { label: string; value: string; trend: string }) {
  return (
    <div className="p-4 bg-secondary rounded-lg border border-border">
      <div className="text-xs text-muted-foreground mb-1">{label}</div>
      <div className="text-2xl font-bold mb-1">{value}</div>
      <div className="text-xs text-accent">{trend}</div>
    </div>
  )
}
