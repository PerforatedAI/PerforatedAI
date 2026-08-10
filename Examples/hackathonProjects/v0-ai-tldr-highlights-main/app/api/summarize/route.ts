import { NextResponse } from "next/server"
import { generateText } from "ai"
import type { SummaryResponse, Thread, Persona, HighlightCategory } from "@/lib/types"
import { ThreadProcessor } from "@/lib/backend/thread-processor"
import { HighlightRanker } from "@/lib/backend/highlight-ranker"
import { PromptTemplates } from "@/lib/backend/prompts"
import { ProcessingMetrics } from "@/lib/backend/metrics"

const processor = new ThreadProcessor()
const ranker = new HighlightRanker()
const prompts = new PromptTemplates()

export async function POST(request: Request) {
  const metrics = new ProcessingMetrics()
  metrics.start()

  try {
    const body = await request.json()
    const { thread, persona = "developer", max_highlights = 10 } = body

    if (!thread || !thread.id) {
      return NextResponse.json({ error: "Thread data required" }, { status: 400 })
    }

    console.log("[v0] Starting non-streaming summarization")

    // Step 1: Process thread
    const processed = processor.process(thread as Thread)

    // Step 2: Generate consolidated summary (single pass)
    const summaryPrompt = prompts.buildStreamingSummaryPrompt(processed.text, persona as Persona)

    const { text: summaryText, usage: summaryUsage } = await generateText({
      model: "openai/gpt-4o-mini",
      prompt: summaryPrompt,
      temperature: 0.2,
      maxTokens: 500,
    })

    // Step 3: Generate digest
    const digestPrompt = prompts.buildDigestPrompt(processed.text, persona as Persona)

    const { text: digestText, usage: digestUsage } = await generateText({
      model: "openai/gpt-4o-mini",
      prompt: digestPrompt,
      temperature: 0.2,
      maxTokens: 300,
    })

    const digest = digestText
      .split("\n")
      .filter((line) => line.trim().startsWith("-"))
      .map((line) => line.replace(/^-\s*/, "").trim())
      .slice(0, 5)

    // Extract verdict
    const verdictMatch = digestText.match(/Verdict:\s*(.+)/i)
    if (verdictMatch) {
      digest.push(`Verdict: ${verdictMatch[1].trim()}`)
    }

    // Step 4: Extract highlights with LLM
    const highlightPrompt = prompts.buildHighlightExtractionPrompt(
      processed.sentences.map((s) => s.text),
      max_highlights,
    )

    const { text: highlightResponse, usage: highlightUsage } = await generateText({
      model: "openai/gpt-4o-mini",
      prompt: highlightPrompt,
      temperature: 0.1,
      maxTokens: 1500,
    })

    let highlights: any[] = []

    try {
      const highlightData = JSON.parse(highlightResponse)

      const classifiedSentences = highlightData.map((h: any) => {
        const sentence = processed.sentences[h.sentence_index]
        return {
          ...sentence,
          category: h.category as HighlightCategory,
          confidence: h.confidence,
          why_matters: h.why_matters,
          importance_score: 0,
        }
      })

      // Rank and select
      const scored = ranker.calculateImportanceScores(classifiedSentences, persona as Persona, summaryText)
      const topHighlights = ranker.selectTopHighlights(scored, max_highlights)

      highlights = topHighlights.map((h, i) => ({
        id: `hl_${i}_${Date.now()}`,
        text: h.text,
        category: h.category,
        confidence: h.confidence,
        importance_score: h.importance_score,
        why_matters: h.why_matters,
        post_id: h.post_id,
        author_id: h.author_id,
        post_position: h.post_position,
        sentence_index: h.sentence_index,
        start_offset: h.char_offset,
        end_offset: h.char_offset + h.text.length,
        created_at: new Date().toISOString(),
      }))

      const avgConfidence = highlights.reduce((sum, h) => sum + h.confidence, 0) / (highlights.length || 1)
      metrics.recordHighlights(highlights.length, avgConfidence)
    } catch (e) {
      console.error("[v0] Failed to parse highlights:", e)
    }

    // Step 5: Record metrics
    const totalTokens =
      (summaryUsage?.totalTokens || 0) + (digestUsage?.totalTokens || 0) + (highlightUsage?.totalTokens || 0)
    metrics.recordTokens(totalTokens, "gpt-4o-mini")
    metrics.finish()

    const processingTime = (metrics.getMetrics().totalTime || 0) / 1000
    const readTimeSaved = Math.max(
      0,
      processor.estimateReadTime(processed.text) - processor.estimateReadTime(summaryText),
    )

    // Step 6: Assess confidence
    const confidencePrompt = prompts.buildConfidencePrompt(processed.text, summaryText)

    const { text: confidenceResponse } = await generateText({
      model: "openai/gpt-4o-mini",
      prompt: confidencePrompt,
      temperature: 0.1,
      maxTokens: 200,
    })

    let confidenceData = { confidence_score: 0.85, is_reliable: true }
    try {
      confidenceData = JSON.parse(confidenceResponse)
    } catch {}

    // Step 7: Format response
    const response: SummaryResponse = {
      summary_id: `sum_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`,
      thread_id: processed.thread_id,
      forum_id: processed.forum_id,
      persona: persona as Persona,
      summary_text: summaryText,
      digest,
      highlights,
      processing_time: processingTime,
      tokens_used: totalTokens,
      read_time_saved: readTimeSaved,
      cache_hit: false,
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      confidence_score: confidenceData.confidence_score,
      is_reliable: confidenceData.is_reliable,
    }

    console.log("[v0] Summarization complete:", metrics.getSummary())

    return NextResponse.json(response)
  } catch (error: any) {
    console.error("[v0] Summarize error:", error)
    return NextResponse.json({ error: error.message || "Failed to generate summary" }, { status: 500 })
  }
}
