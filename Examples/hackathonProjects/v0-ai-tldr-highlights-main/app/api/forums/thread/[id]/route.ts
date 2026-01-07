// API route to fetch thread from Foru.ms

import type { NextRequest } from "next/server"
import { forumsClient } from "@/lib/forums/client"
import { forumsNormalizer } from "@/lib/forums/normalizer"

export async function GET(request: NextRequest, { params }: { params: Promise<{ id: string }> }) {
  try {
    const { id } = await params

    // Fetch thread from Foru.ms
    const forumsThread = await forumsClient.getThread(id)

    // Normalize to internal format
    const thread = forumsNormalizer.normalizeThread(forumsThread)

    return Response.json({
      success: true,
      thread,
    })
  } catch (error: any) {
    console.error("[Forums API] Error fetching thread:", error)

    return Response.json(
      {
        success: false,
        error: error.message,
      },
      { status: error.message.includes("404") ? 404 : 500 },
    )
  }
}
