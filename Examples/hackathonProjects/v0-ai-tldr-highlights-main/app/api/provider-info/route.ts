import { NextResponse } from "next/server"

export async function GET() {
  // Determine which provider is configured
  let provider: "groq" | "ollama" | "huggingface" | "unknown" = "unknown"

  if (process.env.GROQ_API_KEY) {
    provider = "groq"
  } else if (process.env.OLLAMA_BASE_URL || process.env.NODE_ENV === "development") {
    provider = "ollama"
  } else if (process.env.HUGGINGFACE_API_KEY) {
    provider = "huggingface"
  }

  return NextResponse.json({
    provider,
    configured: provider !== "unknown",
    message:
      provider === "unknown"
        ? "No free AI provider configured. See SETUP_FREE_AI.md for instructions."
        : `Using ${provider} for AI inference`,
  })
}
