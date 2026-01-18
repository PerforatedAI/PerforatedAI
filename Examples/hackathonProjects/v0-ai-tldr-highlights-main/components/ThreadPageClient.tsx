"use client"
import { useState } from "react"
import SummaryPanel from "./SummaryPanel"
import PostList from "./PostList"
import AIInsightsPanel from "./AIInsightsPanel"
import useStreamSummary from "@/hooks/useStreamSummary"
import type { Thread, ProcessingMetricsSummary } from "@/lib/types"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"

type Sentence = { idx: number; text: string }
type Post = {
  id: string
  author: string
  content: string
  created_at?: string
  votes?: number
  sentences?: Sentence[]
}

export default function ThreadPageClient({
  threadId,
  thread,
  posts,
}: {
  threadId: string
  thread: Thread
  posts: Post[]
}) {
  const [persona, setPersona] = useState<"novice" | "developer" | "executive">("developer")
  const [highlights, setHighlights] = useState<any[]>([])
  const [digest, setDigest] = useState<string[] | null>(null)
  const [streamTokens, setStreamTokens] = useState<string[]>([])
  const [metrics, setMetrics] = useState<ProcessingMetricsSummary>()
  const { startStream, stopStream, isStreaming } = useStreamSummary()

  async function handleStart() {
    setStreamTokens([])
    setHighlights([])
    setDigest(null)
    setMetrics(undefined)

    await startStream({
      text: "", // Not needed when thread is provided
      thread,
      persona,
      onToken: (tok: string) => {
        setStreamTokens((s) => [...s, tok])
      },
      onSummaryUpdate: (finalSummary: string) => {
        // Summary quality pass complete
      },
      onDigest: (bullets: string[], hlts: any[]) => {
        setDigest(bullets)
        // Initial highlights from digest
        if (hlts.length > 0) {
          setHighlights(hlts)
        }
      },
      onHighlight: (h: any) => {
        setHighlights((prev) => {
          // Avoid duplicates
          const exists = prev.some((p) => p.text === h.text)
          return exists ? prev : [...prev, h]
        })
      },
      onMetrics: (m: ProcessingMetricsSummary) => {
        setMetrics(m)
      },
      onError: (err) => {
        console.error("Stream error:", err)
        alert(`Error: ${err.message}`)
      },
    })
  }

  return (
    <div className="grid grid-cols-1 lg:grid-cols-[minmax(0,1fr)_420px] gap-8">
      <main>
        <div className="mb-4">
          <h1 className="text-3xl font-bold mb-2">{thread.title}</h1>
          <div className="text-sm text-muted-foreground">
            Thread {threadId} · {posts.length} posts
          </div>
        </div>
        <PostList posts={posts} highlights={highlights} />
      </main>

      <aside className="lg:sticky lg:top-6 lg:self-start space-y-4">
        <Tabs defaultValue="summary" className="w-full">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="summary">TL;DR</TabsTrigger>
            <TabsTrigger value="insights">AI Insights</TabsTrigger>
          </TabsList>
          <TabsContent value="summary" className="mt-4">
            <SummaryPanel
              persona={persona}
              setPersona={setPersona}
              isStreaming={isStreaming}
              onStart={handleStart}
              onStop={() => stopStream()}
              tokens={streamTokens}
              digest={digest}
              highlights={highlights}
              metrics={metrics}
              onJumpTo={(postId, sentenceIdx) => {
                const ev = new CustomEvent("jump-to-sentence", { detail: { postId, sentenceIdx } })
                window.dispatchEvent(ev)
              }}
            />
          </TabsContent>
          <TabsContent value="insights" className="mt-4">
            <AIInsightsPanel threadId={threadId} />
          </TabsContent>
        </Tabs>
      </aside>
    </div>
  )
}
