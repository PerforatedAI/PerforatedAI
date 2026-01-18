import { FREE_LLM_CONFIGS } from "@/lib/ai/free-llm-client"
import { ThreadProcessor } from "@/lib/backend/thread-processor"
import { PromptTemplates } from "@/lib/backend/prompts"
import { HighlightRanker } from "@/lib/backend/highlight-ranker"
import { getMockSummaryData, simulateTokenStream } from "@/lib/mock-ai-responses"

const processor = new ThreadProcessor()
const prompts = new PromptTemplates()
const ranker = new HighlightRanker()

const getLLMConfig = () => {
  // Priority: Groq (fast & free) > Ollama (local) > HuggingFace (free)
  if (process.env.GROQ_API_KEY) {
    return FREE_LLM_CONFIGS.groq
  } else if (process.env.OLLAMA_BASE_URL || process.env.NODE_ENV === "development") {
    return FREE_LLM_CONFIGS.ollama
  } else if (process.env.HUGGINGFACE_API_KEY) {
    return FREE_LLM_CONFIGS.huggingface
  }

  throw new Error("No free LLM provider configured. See .env.example for setup instructions.")
}

export async function POST(request: Request) {
  try {
    const body = await request.json()
    const { thread, persona = "developer", max_highlights = 10 } = body

    if (!thread || !thread.id) {
      return new Response(JSON.stringify({ error: "Thread ID required" }), { status: 400 })
    }

    // Get mock data for this thread
    const mockData = getMockSummaryData(thread.id)
    if (!mockData) {
      return new Response(JSON.stringify({ error: "Mock data not available for this thread" }), { status: 404 })
    }

    const encoder = new TextEncoder()
    let totalTokens = 0

    const stream = new ReadableStream({
      async start(controller) {
        try {
          const startTime = Date.now()

          // Simulate streaming tokens
          for await (const token of simulateTokenStream(mockData.tokens, 30)) {
            totalTokens++

            const event = {
              type: "token",
              data: { token, totalTokens },
              timestamp: new Date().toISOString(),
            }
            controller.enqueue(encoder.encode(`data: ${JSON.stringify(event)}\n\n`))
          }

          // Simulate processing time
          await new Promise((resolve) => setTimeout(resolve, 500))

          // Send summary update
          const summaryUpdateEvent = {
            type: "summary_update",
            data: { final_summary: mockData.final_summary, confidence: "high" },
            timestamp: new Date().toISOString(),
          }
          controller.enqueue(encoder.encode(`data: ${JSON.stringify(summaryUpdateEvent)}\n\n`))

          // Send digest
          const digestEvent = {
            type: "digest",
            data: { bullets: mockData.digest, highlights: mockData.highlights },
            timestamp: new Date().toISOString(),
          }
          controller.enqueue(encoder.encode(`data: ${JSON.stringify(digestEvent)}\n\n`))

          // Stream highlights one by one
          for (const highlight of mockData.highlights) {
            const highlightEvent = {
              type: "highlight",
              data: { highlight },
              timestamp: new Date().toISOString(),
            }
            controller.enqueue(encoder.encode(`data: ${JSON.stringify(highlightEvent)}\n\n`))
            await new Promise((resolve) => setTimeout(resolve, 100))
          }

          // Send completion with metrics
          const endTime = Date.now()
          const totalTime = (endTime - startTime) / 1000

          const completeEvent = {
            type: "complete",
            data: {
              message: "Summary complete",
              metrics: {
                total_time: totalTime,
                first_token_time: 0.15,
                tokens_per_second: totalTokens / totalTime,
                highlight_extraction_time: 0.8,
                total_tokens: totalTokens,
              },
              read_time_saved: 180,
            },
            timestamp: new Date().toISOString(),
          }
          controller.enqueue(encoder.encode(`data: ${JSON.stringify(completeEvent)}\n\n`))

          controller.close()
        } catch (error: any) {
          const errorEvent = {
            type: "error",
            data: { error: error.message, phase: "streaming" },
            timestamp: new Date().toISOString(),
          }
          controller.enqueue(encoder.encode(`data: ${JSON.stringify(errorEvent)}\n\n`))
          controller.close()
        }
      },
    })

    return new Response(stream, {
      headers: {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache, no-transform",
        Connection: "keep-alive",
        "X-Accel-Buffering": "no",
      },
    })
  } catch (error: any) {
    return new Response(JSON.stringify({ error: error.message }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    })
  }
}
