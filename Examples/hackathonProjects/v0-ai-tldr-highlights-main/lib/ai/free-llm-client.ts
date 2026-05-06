/**
 * Free & Open-Source LLM Client
 * Supports: Groq (free), Ollama (local), HuggingFace Inference API (free)
 * NO credit card or paid services required!
 */

export interface LLMConfig {
  provider: "groq" | "ollama" | "huggingface"
  model?: string
  apiKey?: string
  baseURL?: string
}

export interface StreamOptions {
  prompt: string
  temperature?: number
  maxTokens?: number
  onToken?: (token: string) => void
}

export class FreeLLMClient {
  private config: LLMConfig

  constructor(config: LLMConfig) {
    this.config = config
  }

  /**
   * Stream text generation - works with all free providers
   */
  async *streamText(options: StreamOptions): AsyncGenerator<string> {
    switch (this.config.provider) {
      case "groq":
        yield* this.streamGroq(options)
        break
      case "ollama":
        yield* this.streamOllama(options)
        break
      case "huggingface":
        yield* this.streamHuggingFace(options)
        break
    }
  }

  /**
   * Generate complete text (non-streaming)
   */
  async generateText(options: StreamOptions): Promise<string> {
    let fullText = ""
    for await (const token of this.streamText(options)) {
      fullText += token
    }
    return fullText
  }

  /**
   * GROQ - Free, fast LLM API (70B+ models, no credit card!)
   */
  private async *streamGroq(options: StreamOptions): AsyncGenerator<string> {
    const apiKey = this.config.apiKey || process.env.GROQ_API_KEY

    if (!apiKey) {
      throw new Error("Groq API key required. Get free key at: https://console.groq.com")
    }

    const model = this.config.model || "llama-3.1-70b-versatile"

    const response = await fetch("https://api.groq.com/openai/v1/chat/completions", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${apiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model,
        messages: [{ role: "user", content: options.prompt }],
        temperature: options.temperature ?? 0.2,
        max_tokens: options.maxTokens ?? 1000,
        stream: true,
      }),
    })

    if (!response.ok) {
      throw new Error(`Groq API error: ${response.statusText}`)
    }

    const reader = response.body?.getReader()
    const decoder = new TextDecoder()

    if (!reader) throw new Error("No response body")

    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      const chunk = decoder.decode(value)
      const lines = chunk.split("\n").filter((line) => line.trim().startsWith("data:"))

      for (const line of lines) {
        const data = line.replace("data:", "").trim()
        if (data === "[DONE]") continue

        try {
          const json = JSON.parse(data)
          const token = json.choices?.[0]?.delta?.content
          if (token) {
            options.onToken?.(token)
            yield token
          }
        } catch (e) {
          // Skip malformed JSON
        }
      }
    }
  }

  /**
   * OLLAMA - Run models locally (100% free, private)
   */
  private async *streamOllama(options: StreamOptions): AsyncGenerator<string> {
    const baseURL = this.config.baseURL || "http://localhost:11434"
    const model = this.config.model || "llama3.1:8b"

    const response = await fetch(`${baseURL}/api/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model,
        prompt: options.prompt,
        stream: true,
        options: {
          temperature: options.temperature ?? 0.2,
          num_predict: options.maxTokens ?? 1000,
        },
      }),
    })

    if (!response.ok) {
      throw new Error(`Ollama error: ${response.statusText}. Make sure Ollama is running: ollama serve`)
    }

    const reader = response.body?.getReader()
    const decoder = new TextDecoder()

    if (!reader) throw new Error("No response body")

    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      const chunk = decoder.decode(value)
      const lines = chunk.split("\n").filter((line) => line.trim())

      for (const line of lines) {
        try {
          const json = JSON.parse(line)
          const token = json.response
          if (token) {
            options.onToken?.(token)
            yield token
          }
        } catch (e) {
          // Skip malformed JSON
        }
      }
    }
  }

  /**
   * HUGGING FACE - Free Inference API
   */
  private async *streamHuggingFace(options: StreamOptions): AsyncGenerator<string> {
    const apiKey = this.config.apiKey || process.env.HUGGINGFACE_API_KEY

    if (!apiKey) {
      throw new Error("HuggingFace API key required. Get free key at: https://huggingface.co/settings/tokens")
    }

    const model = this.config.model || "meta-llama/Llama-3.2-3B-Instruct"

    const response = await fetch(`https://api-inference.huggingface.co/models/${model}`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${apiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        inputs: options.prompt,
        parameters: {
          temperature: options.temperature ?? 0.2,
          max_new_tokens: options.maxTokens ?? 1000,
          return_full_text: false,
        },
      }),
    })

    if (!response.ok) {
      const error = await response.text()
      throw new Error(`HuggingFace API error: ${error}`)
    }

    const result = await response.json()
    const text = result[0]?.generated_text || result.generated_text || ""

    // Simulate streaming for HF (doesn't support native streaming)
    const words = text.split(" ")
    for (const word of words) {
      const token = word + " "
      options.onToken?.(token)
      yield token
      await new Promise((resolve) => setTimeout(resolve, 50))
    }
  }
}

/**
 * Generate structured output (JSON) using free LLMs
 */
export async function generateFreeStructuredOutput<T>(config: LLMConfig, prompt: string, schema: string): Promise<T> {
  const client = new FreeLLMClient(config)

  const fullPrompt = `${prompt}

You MUST respond with valid JSON matching this schema:
${schema}

Respond ONLY with the JSON object, no other text.`

  const text = await client.generateText({ prompt: fullPrompt, temperature: 0.1 })

  // Extract JSON from response
  const jsonMatch = text.match(/\{[\s\S]*\}/)
  if (!jsonMatch) {
    throw new Error("No valid JSON found in response")
  }

  return JSON.parse(jsonMatch[0])
}

/**
 * Default configurations for each provider
 */
export const FREE_LLM_CONFIGS = {
  groq: {
    provider: "groq" as const,
    model: "llama-3.1-70b-versatile", // Fast, powerful, FREE!
    apiKey: process.env.GROQ_API_KEY,
  },
  ollama: {
    provider: "ollama" as const,
    model: "llama3.1:8b", // Run locally, 100% free & private
    baseURL: process.env.OLLAMA_BASE_URL || "http://localhost:11434",
  },
  huggingface: {
    provider: "huggingface" as const,
    model: "meta-llama/Llama-3.2-3B-Instruct", // Free tier
    apiKey: process.env.HUGGINGFACE_API_KEY,
  },
}
