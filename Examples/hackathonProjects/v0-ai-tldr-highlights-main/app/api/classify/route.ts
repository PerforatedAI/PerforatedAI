import { NextResponse } from "next/server"
import { generateFreeStructuredOutput } from "@/lib/ai/free-llm-client"

const getLLMConfig = () => {
  if (process.env.GROQ_API_KEY) {
    return { provider: "groq" as const, model: "llama-3.1-70b-versatile", apiKey: process.env.GROQ_API_KEY }
  } else if (process.env.OLLAMA_BASE_URL || process.env.NODE_ENV === "development") {
    return { provider: "ollama" as const, model: "llama3.1:8b" }
  } else if (process.env.HUGGINGFACE_API_KEY) {
    return {
      provider: "huggingface" as const,
      model: "meta-llama/Llama-3.2-3B-Instruct",
      apiKey: process.env.HUGGINGFACE_API_KEY,
    }
  }
  throw new Error("No free LLM provider configured")
}

export async function POST(request: Request) {
  try {
    const body = await request.json()
    const { sentences } = body

    if (!sentences || !Array.isArray(sentences)) {
      return NextResponse.json({ error: "Sentences array required" }, { status: 400 })
    }

    const prompt = `You are a sentence classifier for forum threads. Classify each sentence into one of these categories:

- fact: Objective information, data, or verifiable statements
- solution: Proposed solutions, fixes, or answers to problems
- opinion: Personal views, subjective statements, or preferences
- question: Questions or requests for information
- citation: References, links, or quoted sources
- irrelevant: Off-topic, spam, or non-informative content

Sentences to classify:
${sentences.map((s: string, i: number) => `${i}. ${s}`).join("\n")}

Provide a confidence score (0-1) for each classification.`

    const schema = `{
  "classifications": [
    {
      "sentence_index": number,
      "category": "fact" | "solution" | "opinion" | "question" | "citation" | "irrelevant",
      "confidence": number,
      "reasoning": string (optional)
    }
  ]
}`

    const config = getLLMConfig()
    const result = await generateFreeStructuredOutput(config, prompt, schema)

    return NextResponse.json({ classifications: result.classifications })
  } catch (error: any) {
    console.error("[API] Classify error:", error)
    return NextResponse.json({ error: error.message || "Failed to classify sentences" }, { status: 500 })
  }
}
