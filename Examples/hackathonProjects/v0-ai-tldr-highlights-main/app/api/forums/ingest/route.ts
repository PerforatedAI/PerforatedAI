// API route to ingest Foru.ms threads into database

import type { NextRequest } from "next/server"
import { forumsIngestion } from "@/lib/forums/ingestion"

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { thread_id, thread_ids, forum_id = "default" } = body

    if (!thread_id && !thread_ids) {
      return Response.json(
        {
          success: false,
          error: "Either thread_id or thread_ids must be provided",
        },
        { status: 400 },
      )
    }

    let results

    if (thread_id) {
      // Ingest single thread
      const result = await forumsIngestion.ingestThread(thread_id, forum_id)
      results = [result]
    } else {
      // Ingest multiple threads
      results = await forumsIngestion.ingestThreads(thread_ids, forum_id)
    }

    const successful = results.filter((r) => r.success).length
    const failed = results.filter((r) => !r.success).length

    return Response.json({
      success: true,
      results,
      summary: {
        total: results.length,
        successful,
        failed,
      },
    })
  } catch (error: any) {
    console.error("[Ingest API] Error:", error)

    return Response.json(
      {
        success: false,
        error: error.message,
      },
      { status: 500 },
    )
  }
}
