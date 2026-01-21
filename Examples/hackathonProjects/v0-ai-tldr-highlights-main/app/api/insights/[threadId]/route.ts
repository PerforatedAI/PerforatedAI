import { type NextRequest, NextResponse } from "next/server"
import { getMockInsights } from "@/lib/mock-ai-responses"
import { getMockThread } from "@/lib/mock-data"
import type { CommunityInsight } from "@/lib/ai/community-insights"

export async function GET(request: NextRequest, { params }: { params: { threadId: string } }) {
  const { threadId } = params

  try {
    // Check if thread exists
    const thread = getMockThread(threadId)
    if (!thread) {
      return NextResponse.json({ error: "Thread not found" }, { status: 404 })
    }

    // Get mock insights
    const rawInsights = getMockInsights(threadId)
    if (!rawInsights) {
      return NextResponse.json({ error: "Insights not available for this thread" }, { status: 404 })
    }

    const insights: CommunityInsight[] = []

    // Sentiment insight
    if (rawInsights.sentiment) {
      insights.push({
        type: "sentiment",
        title: `Overall Sentiment: ${rawInsights.sentiment.overall}`,
        description: `Community sentiment is ${rawInsights.sentiment.overall} with ${rawInsights.sentiment.distribution.positive}% positive, ${rawInsights.sentiment.distribution.neutral}% neutral, and ${rawInsights.sentiment.distribution.negative}% negative responses. Trend is ${rawInsights.sentiment.trend}.`,
        confidence: 0.92,
        metadata: {
          emotions: Object.keys(rawInsights.sentiment.distribution),
          trend: rawInsights.sentiment.trend,
          distribution: rawInsights.sentiment.distribution,
        },
      })
    }

    // Topic insights
    if (rawInsights.topics && rawInsights.topics.length > 0) {
      insights.push({
        type: "topic",
        title: "Main Topics Discussed",
        description: `${rawInsights.topics.length} key topics identified in this thread with high relevance.`,
        confidence: 0.88,
        metadata: {
          technologies: rawInsights.topics.map((t: any) => t.name),
          tags: rawInsights.topics.slice(0, 3).map((t: any) => t.name.toLowerCase().replace(/\s+/g, "-")),
        },
      })
    }

    // Expertise insights
    if (rawInsights.expertise && rawInsights.expertise.length > 0) {
      insights.push({
        type: "expertise",
        title: "Community Experts Identified",
        description: `${rawInsights.expertise.length} users demonstrated strong expertise in this discussion.`,
        confidence: 0.85,
        metadata: {
          experts: rawInsights.expertise.map((e: any) => ({
            username: e.username,
            expertise_level: e.expertise_score >= 0.9 ? "Expert" : "Advanced",
            specialty: e.recommended_for,
          })),
        },
      })
    }

    // Smart reply suggestions
    if (rawInsights.smart_replies && rawInsights.smart_replies.length > 0) {
      insights.push({
        type: "suggestion",
        title: "Smart Reply Suggestions",
        description: `${rawInsights.smart_replies.length} context-aware reply suggestions generated.`,
        confidence: 0.78,
        metadata: {
          suggestions: rawInsights.smart_replies.map((s: any) => ({
            type: "reply",
            text: s.suggested_reply,
            rationale: `For: ${s.context}`,
          })),
        },
      })
    }

    // Trend insights
    if (rawInsights.trends && rawInsights.trends.length > 0) {
      insights.push({
        type: "trend",
        title: "Community Trends",
        description: `${rawInsights.trends.length} emerging patterns detected in the discussion.`,
        confidence: 0.91,
        metadata: {
          trending_topics: rawInsights.trends.map((t: any) => ({
            topic: t.pattern,
            growth: t.confidence > 0.9 ? "rising" : "stable",
            implications: t.implications,
          })),
        },
      })
    }
    // </CHANGE>

    // Simulate API delay
    await new Promise((resolve) => setTimeout(resolve, 1500))

    return NextResponse.json({
      success: true,
      threadId,
      insights,
      generatedAt: new Date().toISOString(),
    })
  } catch (error) {
    return NextResponse.json(
      {
        error: "Failed to generate insights",
        message: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 },
    )
  }
}
