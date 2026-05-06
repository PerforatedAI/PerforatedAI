"use client"
import { useEffect, useState } from "react"
import { Card } from "@/components/ui/card"
import { ChevronDown, ChevronUp } from "lucide-react"

type Sentence = { idx: number; text: string }
type Post = {
  id: string
  author: string
  content: string
  created_at?: string
  votes?: number
  sentences?: Sentence[]
}

export default function PostList({ posts, highlights }: { posts: Post[]; highlights: any[] }) {
  const [collapsedPosts, setCollapsedPosts] = useState<Set<string>>(new Set())

  useEffect(() => {
    const handler = (e: any) => {
      const { postId, sentenceIdx } = e.detail || {}
      if (!postId) return

      setCollapsedPosts((prev) => {
        const next = new Set(prev)
        next.delete(postId)
        return next
      })

      setTimeout(() => {
        const el = document.getElementById(`post-${postId}`)
        if (!el) return
        el.scrollIntoView({ behavior: "smooth", block: "center" })
        const sentEl = document.getElementById(`post-${postId}-sent-${sentenceIdx}`)
        if (sentEl) {
          sentEl.classList.add("ring-2", "ring-accent", "bg-accent/10", "rounded")
          setTimeout(() => sentEl.classList.remove("ring-2", "ring-accent", "bg-accent/10", "rounded"), 3500)
        }
      }, 100)
    }
    window.addEventListener("jump-to-sentence", handler)
    return () => window.removeEventListener("jump-to-sentence", handler)
  }, [])

  const highlightMap: Record<string, number[]> = {}
  for (const h of highlights || []) {
    const pid = h.postId || h.post_id
    const si = h.sentenceIdx ?? h.sentence_index
    if (!pid) continue
    highlightMap[pid] = highlightMap[pid] || []
    if (typeof si === "number") highlightMap[pid].push(si)
  }

  const toggleCollapse = (postId: string) => {
    setCollapsedPosts((prev) => {
      const next = new Set(prev)
      if (next.has(postId)) {
        next.delete(postId)
      } else {
        next.add(postId)
      }
      return next
    })
  }

  return (
    <div className="space-y-4">
      {posts.map((p) => {
        const isCollapsed = collapsedPosts.has(p.id)
        return (
          <Card key={p.id} id={`post-${p.id}`} className="bg-card border-border p-4">
            <div className="flex items-start justify-between">
              <div className="flex-1">
                <div className="text-sm font-medium">{p.author}</div>
                <div className="text-xs text-muted-foreground">
                  {p.created_at ? new Date(p.created_at).toLocaleString() : ""}
                </div>
              </div>
              <div className="flex items-center gap-3">
                <div className="text-xs text-muted-foreground">{p.votes} votes</div>
                <button
                  onClick={() => toggleCollapse(p.id)}
                  className="p-1 hover:bg-slate-100 rounded"
                  aria-label={isCollapsed ? "Expand post" : "Collapse post"}
                  aria-expanded={!isCollapsed}
                >
                  {isCollapsed ? <ChevronDown size={16} /> : <ChevronUp size={16} />}
                </button>
              </div>
            </div>

            {!isCollapsed && (
              <div className="mt-3 text-sm space-y-2" role="region" aria-label={`Post content by ${p.author}`}>
                {p.sentences && p.sentences.length > 0 ? (
                  p.sentences.map((s) => (
                    <p
                      key={s.idx}
                      id={`post-${p.id}-sent-${s.idx}`}
                      className={`leading-relaxed px-1 py-0.5 transition-all duration-300 ${
                        highlightMap[p.id] && highlightMap[p.id].includes(s.idx)
                          ? "bg-yellow-500/20 border-l-2 border-yellow-500 pl-2"
                          : ""
                      }`}
                    >
                      {s.text}
                    </p>
                  ))
                ) : (
                  <p className="leading-relaxed">{p.content}</p>
                )}
              </div>
            )}

            {isCollapsed && (
              <div className="mt-2 text-sm text-muted-foreground italic">
                {p.content.slice(0, 100)}
                {p.content.length > 100 ? "..." : ""}
              </div>
            )}
          </Card>
        )
      })}
    </div>
  )
}
