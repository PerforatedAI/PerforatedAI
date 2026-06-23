// Webhook handler for Foru.ms real-time updates

import type { NextRequest } from "next/server"
import { forumsIngestion } from "@/lib/forums/ingestion"
import crypto from "crypto"

const WEBHOOK_SECRET = process.env.FORUM_WEBHOOK_SECRET

/**
 * Verify webhook signature from Foru.ms
 */
function verifySignature(payload: string, signature: string | null): boolean {
  if (!WEBHOOK_SECRET || !signature) {
    return false
  }

  const expectedSignature = crypto.createHmac("sha256", WEBHOOK_SECRET).update(payload).digest("hex")

  return crypto.timingSafeEqual(Buffer.from(signature), Buffer.from(expectedSignature))
}

export async function POST(request: NextRequest) {
  try {
    // Get raw body for signature verification
    const rawBody = await request.text()
    const signature = request.headers.get("X-Forum-Signature")

    // Verify signature
    if (!verifySignature(rawBody, signature)) {
      console.warn("[Webhook] Invalid signature")
      return new Response("Unauthorized", { status: 401 })
    }

    const body = JSON.parse(rawBody)
    const { event, data } = body

    console.log(`[Webhook] Received event: ${event}`)

    // Handle different events
    switch (event) {
      case "post.created": {
        const { thread_id, post } = data

        console.log(`[Webhook] New post in thread ${thread_id}: ${post.id}`)

        // Re-ingest thread to update summaries
        await forumsIngestion.ingestThread(thread_id)

        break
      }

      case "post.updated": {
        const { thread_id, post } = data

        console.log(`[Webhook] Post updated in thread ${thread_id}: ${post.id}`)

        // Re-process thread
        await forumsIngestion.ingestThread(thread_id)

        break
      }

      case "post.deleted": {
        const { thread_id, post_id } = data

        console.log(`[Webhook] Post deleted from thread ${thread_id}: ${post_id}`)

        // Clean up database records
        // In production: delete post_sentences, highlights, etc.

        break
      }

      case "thread.created": {
        const { thread_id } = data

        console.log(`[Webhook] New thread created: ${thread_id}`)

        // Optionally auto-ingest new threads
        // await forumsIngestion.ingestThread(thread_id)

        break
      }

      default:
        console.log(`[Webhook] Unknown event type: ${event}`)
    }

    return new Response("OK", { status: 200 })
  } catch (error: any) {
    console.error("[Webhook] Error processing webhook:", error)
    return new Response(error.message, { status: 500 })
  }
}

// Verify webhook configuration endpoint
export async function GET() {
  return Response.json({
    status: "active",
    webhook_configured: !!WEBHOOK_SECRET,
    events: ["post.created", "post.updated", "post.deleted", "thread.created"],
  })
}
