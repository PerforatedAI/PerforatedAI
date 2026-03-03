// Normalize Foru.ms data to internal types

import type { Thread, Post } from "@/lib/types"
import type { ForumsThread, ForumsPost } from "./client"

export class ForumsNormalizer {
  /**
   * Normalize Foru.ms thread to internal Thread type
   */
  normalizeThread(forumsThread: ForumsThread, forumId = "default"): Thread {
    return {
      id: forumsThread.id,
      forum_id: forumId,
      title: forumsThread.title,
      posts: forumsThread.posts.map((post, index) => this.normalizePost(post, index)),
    }
  }

  /**
   * Normalize Foru.ms post to internal Post type
   */
  normalizePost(forumsPost: ForumsPost, position: number): Post {
    const author = forumsPost.author || {}
    const authorId = author.id || author.username || author.name || "unknown"
    const authorName = author.username || author.name || "Anonymous"

    return {
      id: forumsPost.id,
      author_id: authorId,
      author_name: authorName,
      content: forumsPost.content || "",
      position,
      created_at: forumsPost.created_at || new Date().toISOString(),
      upvotes: forumsPost.votes || 0,
    }
  }

  /**
   * Convert thread to plain text for summarization
   */
  threadToText(thread: Thread): string {
    let text = `${thread.title}\n\n`

    for (const post of thread.posts) {
      text += `${post.author_name}: ${post.content}\n\n`
    }

    return text.trim()
  }
}

export const forumsNormalizer = new ForumsNormalizer()
