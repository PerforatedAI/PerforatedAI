// Thread processing utilities

import type { Thread, SentenceWithMeta } from "@/lib/types"

export class ThreadProcessor {
  /**
   * Process thread into structured format with sentences
   */
  process(thread: Thread): {
    thread_id: string
    forum_id: string
    title: string
    text: string
    sentences: SentenceWithMeta[]
    post_count: number
    total_chars: number
  } {
    const sentences: SentenceWithMeta[] = []
    let fullText = ""

    for (const post of thread.posts) {
      const cleanContent = this.cleanContent(post.content)
      const postSentences = this.splitIntoSentences(cleanContent)

      for (let i = 0; i < postSentences.length; i++) {
        const sentence = postSentences[i]
        if (sentence.trim().length < 10) continue

        const charOffset = cleanContent.indexOf(sentence)

        sentences.push({
          text: sentence,
          post_id: post.id,
          author_id: post.author_id,
          post_position: post.position,
          sentence_index: i,
          char_offset: charOffset,
          upvotes: post.upvotes,
          created_at: post.created_at,
        })
      }

      fullText += cleanContent + "\n\n"
    }

    return {
      thread_id: thread.id,
      forum_id: thread.forum_id,
      title: thread.title,
      text: fullText,
      sentences,
      post_count: thread.posts.length,
      total_chars: fullText.length,
    }
  }

  private cleanContent(content: string): string {
    // Remove signatures
    const lines = content.split("\n")
    const cleanedLines = []
    for (const line of lines) {
      if (line.trim() === "--" || line.trim() === "~~") break
      cleanedLines.push(line)
    }

    let cleaned = cleanedLines.join("\n")

    // Remove URLs
    cleaned = cleaned.replace(/https?:\/\/\S+/gi, "")

    // Remove excessive whitespace
    cleaned = cleaned.replace(/\s+/g, " ")

    return cleaned.trim()
  }

  private splitIntoSentences(text: string): string[] {
    // Simple sentence splitting
    const sentences = text.split(/(?<=[.!?])\s+/)
    return sentences.map((s) => s.trim()).filter(Boolean)
  }

  /**
   * Estimate reading time in seconds
   */
  estimateReadTime(text: string, wordsPerMinute = 200): number {
    const words = text.split(/\s+/).length
    const minutes = words / wordsPerMinute
    return Math.round(minutes * 60)
  }
}
