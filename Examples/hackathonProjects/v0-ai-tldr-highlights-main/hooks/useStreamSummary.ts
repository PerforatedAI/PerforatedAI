"use client"
import { useRef, useState } from "react"
import type { Thread, ProcessingMetricsSummary } from "@/lib/types"

type StartOpts = {
  text: string
  thread?: Thread
  persona?: string
  onToken?: (t: string) => void
  onDigest?: (bullets: string[], highlights: any[]) => void
  onHighlight?: (h: any) => void
  onSummaryUpdate?: (finalSummary: string) => void
  onMetrics?: (metrics: ProcessingMetricsSummary) => void
  onError?: (e: any) => void
}

export default function useStreamSummary() {
  const controllerRef = useRef<AbortController | null>(null)
  const [isStreaming, setIsStreaming] = useState(false)

  async function startStream(opts: StartOpts) {
    const url = process.env.NEXT_PUBLIC_STREAM_URL || "/api/stream_summary"
    if (isStreaming) {
      stopStream()
    }
    const controller = new AbortController()
    controllerRef.current = controller
    setIsStreaming(true)

    console.log("[v0] Starting stream...")

    try {
      const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: opts.text,
          thread: opts.thread,
          persona: opts.persona,
        }),
        signal: controller.signal,
      })

      if (!res.ok || !res.body) {
        const errTxt = await res.text().catch(() => "")
        throw new Error(`Stream request failed: ${res.status} ${errTxt}`)
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
                if (opts.onToken) opts.onToken(payload.data?.token || "")
                break

              case "summary_update":
                if (opts.onSummaryUpdate) opts.onSummaryUpdate(payload.data?.final_summary || "")
                break

              case "digest":
                if (opts.onDigest) {
                  opts.onDigest(payload.data?.bullets || [], payload.data?.highlights || [])
                }
                break

              case "highlight":
                if (opts.onHighlight) opts.onHighlight(payload.data?.highlight)
                break

              case "complete":
                if (opts.onMetrics && payload.data?.metrics) {
                  opts.onMetrics(payload.data.metrics)
                }
                console.log("[v0] Stream complete:", payload.data)
                break

              case "error":
                if (opts.onError) opts.onError(new Error(payload.data?.error || "Stream error"))
                break
            }
          } catch (e) {
            // Fallback for non-JSON chunks
            if (opts.onToken) opts.onToken(chunk)
          }
        }
      }

      console.log("[v0] Stream finished")
    } catch (e: any) {
      console.error("[v0] Stream error:", e)
      if (opts.onError) opts.onError(e)
    } finally {
      setIsStreaming(false)
      controllerRef.current = null
    }
  }

  function stopStream() {
    if (controllerRef.current) {
      controllerRef.current.abort()
      controllerRef.current = null
    }
    setIsStreaming(false)
  }

  return { startStream, stopStream, isStreaming }
}
