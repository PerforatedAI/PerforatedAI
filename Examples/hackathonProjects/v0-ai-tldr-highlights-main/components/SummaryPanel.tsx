"use client"
import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import TokenTicker from "./TokenTicker"
import HighlightBadge from "./HighlightBadge"
import { LoadingSkeleton } from "./LoadingSkeleton"
import useLocalStorage from "@/hooks/useLocalStorage"
import useKeyboardShortcuts from "@/hooks/useKeyboardShortcuts"
import { useToast } from "./Toast"
import { Copy, Share2, Play, Square } from "lucide-react"
import type { ProcessingMetricsSummary } from "@/lib/types"

export default function SummaryPanel({
  persona,
  setPersona,
  isStreaming,
  onStart,
  onStop,
  tokens,
  digest,
  highlights,
  metrics,
  onJumpTo,
}: {
  persona: string
  setPersona: (p: string) => void
  isStreaming: boolean
  onStart: () => void
  onStop: () => void
  tokens: string[]
  digest: string[] | null
  highlights: any[]
  metrics?: ProcessingMetricsSummary
  onJumpTo: (postId: string, sentenceIdx: number) => void
}) {
  const toast = useToast()
  const [streamRate, setStreamRate] = useLocalStorage<number>("tldr_rate", 1.0)

  useKeyboardShortcuts({
    toggleStream: () => (isStreaming ? onStop() : onStart()),
    stop: () => onStop(),
    nextHighlight: () => {
      if (highlights && highlights.length > 0) {
        const first = highlights[0]
        onJumpTo(first.post_id || first.postId || "", first.sentence_index ?? first.sentenceIdx ?? 0)
      }
    },
  })

  const copyDigest = async () => {
    const text = (digest || []).map((d) => `• ${d}`).join("\n")
    try {
      await navigator.clipboard.writeText(text)
      toast.push({ title: "Copied digest to clipboard", tone: "success" })
    } catch {
      toast.push({ title: "Copy failed", tone: "error", body: "Clipboard permission denied" })
    }
  }

  const shareDigest = () => {
    toast.push({ title: "Share", body: "Share functionality coming soon", tone: "info" })
  }

  const categoryStats = highlights.reduce(
    (acc, h) => {
      const cat = h.category || "unknown"
      acc[cat] = (acc[cat] || 0) + 1
      return acc
    },
    {} as Record<string, number>,
  )

  return (
    <Card className="bg-card border-border p-4">
      <div className="flex items-center justify-between">
        <h3 className="font-semibold">Live TL;DR</h3>
        <div className="flex items-center gap-2">
          <select
            aria-label="Reader persona"
            value={persona}
            onChange={(e) => setPersona(e.target.value)}
            className="text-xs border rounded px-2 py-1 bg-background"
            disabled={isStreaming}
          >
            <option value="novice">Novice</option>
            <option value="developer">Developer</option>
            <option value="executive">Executive</option>
          </select>
          <button
            onClick={copyDigest}
            className="p-2 rounded hover:bg-slate-50 disabled:opacity-50"
            title="Copy digest (Ctrl+C)"
            disabled={!digest || digest.length === 0}
            aria-label="Copy digest"
          >
            <Copy size={14} />
          </button>
          <button
            onClick={shareDigest}
            className="p-2 rounded hover:bg-slate-50"
            title="Share digest"
            aria-label="Share digest"
          >
            <Share2 size={14} />
          </button>
        </div>
      </div>

      <div className="mt-3">
        <div className="text-xs text-muted-foreground flex items-center justify-between">
          <span>Streaming status</span>
          {isStreaming ? (
            <span className="flex items-center gap-1 text-green-600">
              <span className="inline-block w-2 h-2 bg-green-500 rounded-full animate-pulse" />
              Live
            </span>
          ) : (
            <span>Ready</span>
          )}
        </div>
        <div className="mt-2 flex items-center gap-2">
          <div className="flex-1 h-2 bg-slate-100 rounded overflow-hidden">
            <div
              className={`h-full bg-indigo-400 transition-all duration-300 ${isStreaming ? "animate-pulse" : ""}`}
              style={{ width: `${Math.min(100, (tokens.length / 60) * 100)}%` }}
            />
          </div>
          <div className="text-xs text-slate-500 w-16 text-right">{tokens.length} tok</div>
        </div>

        <div className="mt-3">
          <TokenTicker tokens={tokens} />
        </div>

        <div className="mt-3">
          <div className="flex items-center gap-2">
            <label className="text-xs text-slate-500">Speed</label>
            <input
              type="range"
              min={0.25}
              max={2.0}
              step={0.25}
              value={streamRate}
              onChange={(e) => setStreamRate(Number(e.target.value))}
              className="w-full"
              aria-label="Streaming speed"
            />
            <div className="text-xs text-slate-500 w-8 text-right">{streamRate}x</div>
          </div>
        </div>
      </div>

      <div className="mt-3 flex gap-2">
        <Button
          onClick={() => (isStreaming ? onStop() : onStart())}
          size="sm"
          className="flex-1 gap-2"
          aria-pressed={isStreaming}
        >
          {isStreaming ? (
            <>
              <Square size={14} /> Stop
            </>
          ) : (
            <>
              <Play size={14} /> Start Stream
            </>
          )}
        </Button>
      </div>

      <div className="mt-4">
        <div className="text-xs text-muted-foreground mb-2">Key Points Digest</div>
        <div className="min-h-[6rem] rounded border border-border p-3 bg-background text-sm">
          {!digest ? (
            <LoadingSkeleton lines={3} />
          ) : digest.length === 0 ? (
            <div className="text-xs text-muted-foreground italic">Digest will appear after streaming completes</div>
          ) : (
            <ul className="space-y-2">
              {digest.map((b, i) => (
                <li key={i} className="flex gap-2">
                  <span className="text-muted-foreground mt-0.5">•</span>
                  <span className={b.startsWith("Verdict:") ? "font-medium text-accent" : ""}>{b}</span>
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>

      <div className="mt-4">
        <div className="text-xs text-muted-foreground mb-2 flex items-center justify-between">
          <span>Smart Highlights</span>
          {highlights.length > 0 && (
            <div className="flex gap-1">
              {Object.entries(categoryStats).map(([cat, count]) => (
                <Badge key={cat} variant="outline" className="text-[10px] px-1 py-0">
                  {cat}: {count}
                </Badge>
              ))}
            </div>
          )}
        </div>
        <div className="space-y-2 max-h-64 overflow-auto">
          {highlights.length === 0 ? (
            <div className="text-xs text-muted-foreground italic">Highlights will be extracted after summary</div>
          ) : (
            highlights.map((h, i) => <HighlightBadge key={i} h={h} onJump={onJumpTo} />)
          )}
        </div>
      </div>

      <div className="mt-4 p-2 bg-slate-50 rounded text-xs">
        <div className="font-medium mb-1">Keyboard Shortcuts</div>
        <div className="space-y-0.5 text-muted-foreground">
          <div>
            <kbd className="px-1 bg-white border rounded">Space</kbd> Start/Stop
          </div>
          <div>
            <kbd className="px-1 bg-white border rounded">S</kbd> Stop
          </div>
          <div>
            <kbd className="px-1 bg-white border rounded">J</kbd> Jump to first highlight
          </div>
        </div>
      </div>

      {metrics && (
        <div className="mt-4 pt-4 border-t border-border">
          <div className="text-xs text-muted-foreground mb-2">Performance Metrics</div>
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div>
              <div className="text-muted-foreground">First Token</div>
              <div className="font-medium">{metrics?.latency?.firstToken || "N/A"}</div>
            </div>
            <div>
              <div className="text-muted-foreground">Total Time</div>
              <div className="font-medium">{metrics?.latency?.total || "N/A"}</div>
            </div>
            <div>
              <div className="text-muted-foreground">Tokens Used</div>
              <div className="font-medium">{metrics?.cost?.tokensUsed || "N/A"}</div>
            </div>
            <div>
              <div className="text-muted-foreground">Est. Cost</div>
              <div className="font-medium">{metrics?.cost?.estimatedCost || "N/A"}</div>
            </div>
          </div>
        </div>
      )}
    </Card>
  )
}
