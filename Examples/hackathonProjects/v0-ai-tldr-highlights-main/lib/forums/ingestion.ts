// Data ingestion pipeline for Foru.ms threads

import { forumsClient } from "./client"
import { forumsNormalizer } from "./normalizer"
import { ThreadProcessor } from "@/lib/backend/thread-processor"

export interface IngestionResult {
  thread_id: string
  posts_count: number
  sentences_count: number
  processing_time: number
  success: boolean
  error?: string
}

export class ForumsIngestion {
  private processor: ThreadProcessor

  constructor() {
    this.processor = new ThreadProcessor()
  }

  /**
   * Ingest a thread from Foru.ms into the database
   */
  async ingestThread(threadId: string, forumId = "default"): Promise<IngestionResult> {
    const startTime = Date.now()

    try {
      // Step 1: Fetch thread from Foru.ms
      const forumsThread = await forumsClient.getThread(threadId)

      // Step 2: Normalize to internal format
      const thread = forumsNormalizer.normalizeThread(forumsThread, forumId)

      // Step 3: Process thread (split into sentences)
      const processed = this.processor.process(thread)

      // In production, you would:
      // - Store thread in database
      // - Generate embeddings for sentences
      // - Store embeddings in pgvector
      // - Run classification on sentences
      // - Store classified sentences

      // For now, return success metrics
      return {
        thread_id: threadId,
        posts_count: thread.posts.length,
        sentences_count: processed.sentences.length,
        processing_time: Date.now() - startTime,
        success: true,
      }
    } catch (error: any) {
      return {
        thread_id: threadId,
        posts_count: 0,
        sentences_count: 0,
        processing_time: Date.now() - startTime,
        success: false,
        error: error.message,
      }
    }
  }

  /**
   * Ingest multiple threads in batch
   */
  async ingestThreads(threadIds: string[], forumId = "default"): Promise<IngestionResult[]> {
    const results: IngestionResult[] = []

    for (const threadId of threadIds) {
      const result = await this.ingestThread(threadId, forumId)
      results.push(result)
    }

    return results
  }
}

export const forumsIngestion = new ForumsIngestion()
