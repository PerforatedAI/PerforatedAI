import { generateFreeStructuredOutput } from "@/lib/ai/free-llm-client"
import type { ForumsThread, ForumsPost } from "@/lib/forums/client"

export interface CommunityInsight {
  type: "sentiment" | "topic" | "expertise" | "trend" | "suggestion"
  title: string
  description: string
  confidence: number
  metadata?: Record<string, any>
}

const getLLMConfig = () => {
  if (process.env.GROQ_API_KEY) {
    return { provider: "groq" as const, model: "llama-3.1-70b-versatile", apiKey: process.env.GROQ_API_KEY }
  } else if (process.env.OLLAMA_BASE_URL) {
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

export async function analyzeSentiment(posts: ForumsPost[]): Promise<CommunityInsight> {
  const content = posts.map((p) => `${p.author.username}: ${p.content}`).join("\n\n")

  const prompt = `Analyze the sentiment and emotional tone of this forum thread.

Thread posts:
${content.slice(0, 4000)}

Identify:
1. Overall sentiment (positive/neutral/negative/mixed)
2. Key emotions expressed (e.g., frustration, excitement, confusion)
3. Any significant tone shifts during the conversation
4. Confidence in your assessment

Be specific and cite examples from the posts.`

  const schema = `{
  "overall_sentiment": "positive" | "neutral" | "negative" | "mixed",
  "confidence": number,
  "key_emotions": string[],
  "tone_shift": string (optional),
  "reasoning": string
}`

  const config = getLLMConfig()
  const result = await generateFreeStructuredOutput(config, prompt, schema)

  return {
    type: "sentiment",
    title: `${result.overall_sentiment.charAt(0).toUpperCase() + result.overall_sentiment.slice(1)} Discussion`,
    description: result.reasoning,
    confidence: result.confidence,
    metadata: {
      sentiment: result.overall_sentiment,
      emotions: result.key_emotions,
      tone_shift: result.tone_shift,
    },
  }
}

export async function extractTopics(thread: ForumsThread): Promise<CommunityInsight> {
  const content = `Title: ${thread.title}\n\n${thread.posts.map((p) => p.content).join("\n\n")}`

  const prompt = `Extract topics, themes, and relevant tags from this forum thread.

Thread content:
${content.slice(0, 5000)}

Identify:
1. Main topic or problem being discussed
2. Key subtopics or related issues
3. Technologies, tools, or frameworks mentioned
4. Suggested tags for categorization

Be concise and specific.`

  const schema = `{
  "main_topic": string,
  "subtopics": string[],
  "related_technologies": string[],
  "suggested_tags": string[],
  "confidence": number
}`

  const config = getLLMConfig()
  const result = await generateFreeStructuredOutput(config, prompt, schema)

  return {
    type: "topic",
    title: result.main_topic,
    description: `Subtopics: ${result.subtopics.join(", ")}`,
    confidence: result.confidence,
    metadata: {
      main_topic: result.main_topic,
      subtopics: result.subtopics,
      technologies: result.related_technologies,
      tags: result.suggested_tags,
    },
  }
}

export async function identifyExpertise(posts: ForumsPost[]): Promise<CommunityInsight> {
  const userPosts = posts.reduce(
    (acc, p) => {
      const user = p.author.username || p.author.name || "Anonymous"
      if (!acc[user]) acc[user] = []
      acc[user].push(p.content)
      return acc
    },
    {} as Record<string, string[]>,
  )

  const topContributors = Object.entries(userPosts)
    .map(([user, posts]) => ({
      user,
      post_count: posts.length,
      content: posts.join(" ").slice(0, 1000),
    }))
    .sort((a, b) => b.post_count - a.post_count)
    .slice(0, 3)

  const prompt = `Analyze user expertise based on their contributions to this thread.

Top contributors:
${topContributors.map((c) => `${c.user} (${c.post_count} posts): ${c.content}`).join("\n\n")}

For each user, determine:
1. Expertise level (beginner/intermediate/expert/authority)
2. Their specialty or area of knowledge
3. How helpful their contributions are (0-1 score)

Be fair and evidence-based.`

  const schema = `{
  "experts": [
    {
      "username": string,
      "expertise_level": "beginner" | "intermediate" | "expert" | "authority",
      "specialty": string,
      "helpful_score": number
    }
  ],
  "confidence": number,
  "summary": string
}`

  const config = getLLMConfig()
  const result = await generateFreeStructuredOutput(config, prompt, schema)

  return {
    type: "expertise",
    title: "Community Expertise",
    description: result.summary,
    confidence: result.confidence,
    metadata: {
      experts: result.experts,
    },
  }
}

export async function suggestReplies(thread: ForumsThread): Promise<CommunityInsight> {
  const recentPosts = thread.posts.slice(-5)
  const content = recentPosts.map((p) => `${p.author.username}: ${p.content}`).join("\n\n")

  const prompt = `Generate 3-5 helpful reply suggestions for this forum thread.

Thread title: ${thread.title}

Recent posts:
${content}

For each suggestion:
1. Choose a type (question, clarification, solution, resource, encouragement)
2. Write a natural, helpful reply
3. Explain why this reply would be valuable

Be constructive and community-focused.`

  const schema = `{
  "suggestions": [
    {
      "type": "question" | "clarification" | "solution" | "resource" | "encouragement",
      "text": string,
      "rationale": string
    }
  ],
  "confidence": number
}`

  const config = getLLMConfig()
  const result = await generateFreeStructuredOutput(config, prompt, schema)

  return {
    type: "suggestion",
    title: "Smart Reply Suggestions",
    description: `${result.suggestions.length} AI-generated replies to help the discussion`,
    confidence: result.confidence,
    metadata: {
      suggestions: result.suggestions,
    },
  }
}

export async function detectTrends(threads: ForumsThread[]): Promise<CommunityInsight> {
  const threadSummaries = threads.slice(0, 10).map((t) => ({
    title: t.title,
    post_count: t.posts.length,
    tags: t.tags || [],
    preview: t.posts[0]?.content.slice(0, 200),
  }))

  const prompt = `Analyze trends across these forum threads.

Threads:
${JSON.stringify(threadSummaries, null, 2)}

Identify:
1. Trending topics (with estimated frequency and growth trajectory)
2. Emerging issues or new problems
3. Common questions being asked
4. Overall community trends

Provide actionable insights for community managers.`

  const schema = `{
  "trending_topics": [
    {
      "topic": string,
      "frequency": number,
      "growth": "rising" | "stable" | "declining"
    }
  ],
  "emerging_issues": string[],
  "common_questions": string[],
  "confidence": number,
  "summary": string
}`

  const config = getLLMConfig()
  const result = await generateFreeStructuredOutput(config, prompt, schema)

  return {
    type: "trend",
    title: "Community Trends",
    description: result.summary,
    confidence: result.confidence,
    metadata: {
      trending_topics: result.trending_topics,
      emerging_issues: result.emerging_issues,
      common_questions: result.common_questions,
    },
  }
}

export async function getComprehensiveInsights(thread: ForumsThread): Promise<CommunityInsight[]> {
  const [sentiment, topics, expertise, replies] = await Promise.all([
    analyzeSentiment(thread.posts),
    extractTopics(thread),
    identifyExpertise(thread.posts),
    suggestReplies(thread),
  ])

  return [sentiment, topics, expertise, replies]
}
