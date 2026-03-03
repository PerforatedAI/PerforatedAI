"use client"

import { useEffect, useState } from "react"
import { Zap, Server, Globe } from "lucide-react"

export function AIProviderBadge() {
  const [provider, setProvider] = useState<"groq" | "ollama" | "huggingface" | "unknown">("unknown")
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    // Detect which provider is configured
    const detectProvider = async () => {
      try {
        const response = await fetch("/api/provider-info")
        const data = await response.json()
        setProvider(data.provider)
      } catch (error) {
        setProvider("unknown")
      } finally {
        setIsLoading(false)
      }
    }

    detectProvider()
  }, [])

  if (isLoading) return null

  const providerConfig = {
    groq: {
      name: "Groq",
      icon: Zap,
      color: "text-orange-500",
      bg: "bg-orange-500/10",
      description: "Lightning fast, free 70B models",
    },
    ollama: {
      name: "Ollama",
      icon: Server,
      color: "text-blue-500",
      bg: "bg-blue-500/10",
      description: "Running locally, 100% private",
    },
    huggingface: {
      name: "Hugging Face",
      icon: Globe,
      color: "text-yellow-500",
      bg: "bg-yellow-500/10",
      description: "Free inference API",
    },
    unknown: {
      name: "Not Configured",
      icon: Globe,
      color: "text-gray-500",
      bg: "bg-gray-500/10",
      description: "Setup required",
    },
  }

  const config = providerConfig[provider]
  const Icon = config.icon

  return (
    <div className={`inline-flex items-center gap-2 rounded-full px-3 py-1.5 text-sm ${config.bg}`}>
      <Icon className={`h-4 w-4 ${config.color}`} />
      <div className="flex flex-col">
        <span className="font-medium">{config.name}</span>
        <span className="text-xs text-muted-foreground">{config.description}</span>
      </div>
    </div>
  )
}
