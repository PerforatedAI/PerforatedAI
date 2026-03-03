"use client"
import { ArrowRight } from "lucide-react"

export default function HighlightBadge({
  h,
  onJump,
}: {
  h: {
    text: string
    postId?: string
    post_id?: string
    sentenceIdx?: number
    sentence_index?: number
    confidence?: number
    category?: string
    why_matters?: string
  }
  onJump?: (postId: string, sentenceIdx: number) => void
}) {
  const conf = Math.max(0, Math.min(1, h.confidence ?? 0.6))
  const pct = Math.round(conf * 100)
  const colorClass = conf > 0.8 ? "bg-emerald-400" : conf > 0.6 ? "bg-amber-400" : "bg-red-400"

  return (
    <div className="p-3 rounded-md border bg-white hover:shadow-md transition-shadow">
      <div className="flex items-start justify-between gap-2">
        <div className="text-sm text-slate-800 leading-relaxed flex-1">{h.text}</div>
        <div className="text-xs text-slate-500 px-2 py-0.5 bg-slate-100 rounded">{h.category ?? "info"}</div>
      </div>

      {h.why_matters && <div className="text-xs text-slate-600 mt-2 italic">{h.why_matters}</div>}

      <div className="mt-3 flex items-center justify-between gap-3">
        <div className="flex-1">
          <div className="h-2 w-full bg-slate-100 rounded overflow-hidden">
            <div className={`h-full ${colorClass} transition-all duration-300`} style={{ width: `${pct}%` }} />
          </div>
          <div className="text-xs text-slate-500 mt-1">Confidence: {pct}%</div>
        </div>
        <button
          onClick={() => {
            const postId = h.postId || h.post_id || ""
            const sentenceIdx = h.sentenceIdx ?? h.sentence_index ?? 0
            if (onJump && postId) onJump(postId, sentenceIdx)
          }}
          className="px-3 py-1.5 text-xs text-indigo-600 rounded hover:bg-indigo-50 flex items-center gap-1 transition-colors"
        >
          Jump <ArrowRight size={12} />
        </button>
      </div>
    </div>
  )
}
