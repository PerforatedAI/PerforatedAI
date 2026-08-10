// Foru.ms API client for fetching forum data

const FORUM_API_BASE = process.env.FORUM_API_BASE || "https://api.foru.ms/v1"
const FORUM_API_TOKEN = process.env.FORUM_API_TOKEN
const FORUM_INSTANCE_ID = process.env.FORUM_INSTANCE_ID
const FORUM_INSTANCE_HANDLE = process.env.FORUM_INSTANCE_HANDLE

export interface ForumsThread {
  id: string
  title: string
  created_at: string
  tags?: string[]
  post_count?: number
  posts: ForumsPost[]
}

export interface ForumsPost {
  id: string
  author: {
    id?: string
    username?: string
    name?: string
  }
  content: string
  created_at: string
  votes?: number
}

export class ForumsClient {
  private baseUrl: string
  private token?: string
  private instanceId?: string
  private instanceHandle?: string

  constructor(baseUrl?: string, token?: string, instanceId?: string, instanceHandle?: string) {
    this.baseUrl = baseUrl || FORUM_API_BASE
    this.token = token || FORUM_API_TOKEN
    this.instanceId = instanceId || FORUM_INSTANCE_ID
    this.instanceHandle = instanceHandle || FORUM_INSTANCE_HANDLE
  }

  /**
   * Fetch data from Foru.ms API
   */
  private async fetch(url: string): Promise<any> {
    const headers: HeadersInit = {
      "Content-Type": "application/json",
    }

    if (this.token) {
      headers["X-API-Key"] = this.token
    }

    if (this.instanceId) {
      headers["X-Instance-ID"] = this.instanceId
    }

    const fullUrl = `${this.baseUrl}${url}`

    try {
      const response = await fetch(fullUrl, {
        headers,
        next: { revalidate: 60 }, // Cache for 1 minute
      })

      if (!response.ok) {
        // Try to get error body, but don't fail if it's not JSON
        let errorBody = ""
        try {
          const text = await response.text()
          errorBody = text.substring(0, 200) // Only show first 200 chars
        } catch (e) {
          // Ignore body parsing errors
        }

        throw new Error(`fetch to ${fullUrl} failed with status ${response.status} and body: ${errorBody}`)
      }

      return await response.json()
    } catch (error: any) {
      // Re-throw with context
      throw new Error(error.message || `Failed to fetch from ${fullUrl}`)
    }
  }

  /**
   * Get a thread with all posts
   */
  async getThread(threadId: string): Promise<ForumsThread> {
    const path = this.instanceHandle ? `/instances/${this.instanceHandle}/threads/${threadId}` : `/threads/${threadId}`
    return await this.fetch(path)
  }

  /**
   * Get a single post
   */
  async getPost(postId: string): Promise<ForumsPost> {
    const path = this.instanceHandle ? `/instances/${this.instanceHandle}/posts/${postId}` : `/posts/${postId}`
    return await this.fetch(path)
  }

  /**
   * Get latest threads with pagination
   */
  async getLatestThreads(page = 1, limit = 20): Promise<ForumsThread[]> {
    const path = this.instanceHandle
      ? `/instances/${this.instanceHandle}/threads?page=${page}&limit=${limit}`
      : `/threads?page=${page}&limit=${limit}`
    const data = await this.fetch(path)
    return data.threads || data
  }

  /**
   * Get posts for a thread with pagination
   */
  async getThreadPosts(threadId: string, page = 1, limit = 50): Promise<ForumsPost[]> {
    const path = this.instanceHandle
      ? `/instances/${this.instanceHandle}/threads/${threadId}/posts?page=${page}&limit=${limit}`
      : `/threads/${threadId}/posts?page=${page}&limit=${limit}`
    const data = await this.fetch(path)
    return data.posts || data
  }

  /**
   * Fetch all posts for a thread (handles pagination)
   */
  async getAllThreadPosts(threadId: string): Promise<ForumsPost[]> {
    const allPosts: ForumsPost[] = []
    let page = 1
    let hasMore = true

    while (hasMore) {
      const posts = await this.getThreadPosts(threadId, page, 50)
      if (!posts || posts.length === 0) {
        hasMore = false
      } else {
        allPosts.push(...posts)
        page++
      }
    }

    return allPosts
  }

  /**
   * Get thread statistics
   */
  async getThreadStats(threadId: string): Promise<any> {
    const path = this.instanceHandle
      ? `/instances/${this.instanceHandle}/threads/${threadId}/stats`
      : `/threads/${threadId}/stats`
    return await this.fetch(path)
  }
}

// Export singleton instance
export const forumsClient = new ForumsClient()
