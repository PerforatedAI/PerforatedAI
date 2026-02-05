"use client"

import { useState } from "react"
import { Card } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Sparkles, TrendingUp, Users, MessageSquarePlus, Tag, Loader2 } from "lucide-react"
import type { CommunityInsight } from "@/lib/ai/community-insights"

interface AIInsightsPanelProps {
  threadId: string
  onLoadInsights?: () => void
}

export default function AIInsightsPanel({ threadId, onLoadInsights }: AIInsightsPanelProps) {
  const [insights, setInsights] = useState<CommunityInsight[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function loadInsights() {
    setLoading(true)
    setError(null)
    onLoadInsights?.()

    try {
      const response = await fetch(`/api/insights/${threadId}`)
      if (!response.ok) throw new Error("Failed to load insights")

      const data = await response.json()
      setInsights(data.insights)
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error")
    } finally {
      setLoading(false)
    }
  }

  const getInsightIcon = (type: string) => {
    switch (type) {
      case "sentiment":
        return <Sparkles className="w-4 h-4" />
      case "topic":
        return <Tag className="w-4 h-4" />
      case "expertise":
        return <Users className="w-4 h-4" />
      case "suggestion":
        return <MessageSquarePlus className="w-4 h-4" />
      case "trend":
        return <TrendingUp className="w-4 h-4" />
      default:
        return <Sparkles className="w-4 h-4" />
    }
  }

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return "bg-green-500/10 text-green-700 border-green-200"
    if (confidence >= 0.6) return "bg-yellow-500/10 text-yellow-700 border-yellow-200"
    return "bg-orange-500/10 text-orange-700 border-orange-200"
  }

  return (
    <Card className="p-6">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Sparkles className="w-5 h-5 text-primary" />
          <h3 className="text-lg font-semibold">AI Community Insights</h3>
        </div>
        <Button onClick={loadInsights} disabled={loading} size="sm">
          {loading ? (
            <>
              <Loader2 className="w-4 h-4 mr-2 animate-spin" />
              Analyzing...
            </>
          ) : (
            "Generate Insights"
          )}
        </Button>
      </div>

      {error && (
        <div className="p-4 bg-destructive/10 border border-destructive/20 rounded-lg text-sm text-destructive">
          {error}
        </div>
      )}

      {insights.length > 0 && (
        <div className="space-y-4">
          {insights.map((insight, idx) => (
            <Card key={idx} className="p-4 bg-muted/50">
              <div className="flex items-start gap-3">
                <div className="p-2 bg-background rounded-lg">{getInsightIcon(insight.type)}</div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-2">
                    <h4 className="font-medium text-sm">{insight.title}</h4>
                    <Badge variant="outline" className={`text-xs ${getConfidenceColor(insight.confidence)}`}>
                      {Math.round(insight.confidence * 100)}% confident
                    </Badge>
                  </div>
                  <p className="text-sm text-muted-foreground mb-3">{insight.description}</p>

                  {insight.type === "sentiment" && insight.metadata && (
                    <div className="flex flex-wrap gap-1">
                      {insight.metadata.emotions?.map((emotion: string, i: number) => (
                        <Badge key={i} variant="secondary" className="text-xs">
                          {emotion}
                        </Badge>
                      ))}
                    </div>
                  )}

                  {insight.type === "topic" && insight.metadata && (
                    <div className="space-y-2">
                      <div className="flex flex-wrap gap-1">
                        {insight.metadata.technologies?.map((tech: string, i: number) => (
                          <Badge key={i} variant="outline" className="text-xs">
                            {tech}
                          </Badge>
                        ))}
                      </div>
                      {insight.metadata.tags && (
                        <div className="text-xs text-muted-foreground">
                          Suggested tags: {insight.metadata.tags.join(", ")}
                        </div>
                      )}
                    </div>
                  )}

                  {insight.type === "expertise" && insight.metadata?.experts && (
                    <div className="space-y-2">
                      {insight.metadata.experts.map((expert: any, i: number) => (
                        <div key={i} className="flex items-center justify-between text-xs bg-background/50 p-2 rounded">
                          <span className="font-medium">{expert.username}</span>
                          <div className="flex items-center gap-2">
                            <Badge variant="secondary" className="text-xs">
                              {expert.expertise_level}
                            </Badge>
                            <span className="text-muted-foreground">{expert.specialty}</span>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}

                  {insight.type === "suggestion" && insight.metadata?.suggestions && (
                    <div className="space-y-2">
                      {insight.metadata.suggestions.map((suggestion: any, i: number) => (
                        <div key={i} className="bg-background/50 p-3 rounded text-xs">
                          <div className="flex items-center gap-2 mb-1">
                            <Badge variant="outline" className="text-xs">
                              {suggestion.type}
                            </Badge>
                          </div>
                          <p className="text-foreground mb-1">{suggestion.text}</p>
                          <p className="text-muted-foreground italic">{suggestion.rationale}</p>
                        </div>
                      ))}
                    </div>
                  )}

                  {insight.type === "trend" && insight.metadata && (
                    <div className="space-y-2">
                      {insight.metadata.trending_topics?.map((topic: any, i: number) => (
                        <div key={i} className="flex items-center justify-between text-xs bg-background/50 p-2 rounded">
                          <span className="font-medium">{topic.topic}</span>
                          <Badge variant={topic.growth === "rising" ? "default" : "secondary"} className="text-xs">
                            {topic.growth}
                          </Badge>
                        </div>
                      ))}
                    </div>
                  )}
                  {/* </CHANGE> */}
                </div>
              </div>
            </Card>
          ))}
        </div>
      )}

      {!loading && insights.length === 0 && (
        <div className="text-center py-8 text-muted-foreground text-sm">
          <Sparkles className="w-12 h-12 mx-auto mb-3 opacity-30" />
          <p>Click "Generate Insights" to analyze this thread with AI</p>
          <p className="text-xs mt-2">Powered by Foru.ms + LLM integration</p>
        </div>
      )}
    </Card>
  )
}
