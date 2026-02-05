import ThreadPageClient from "@/components/ThreadPageClient"
import { splitIntoSentences } from "@/utils/sentenceUtils"
import type { Thread } from "@/lib/types"
import { forumsClient } from "@/lib/forums/client"
import { forumsNormalizer } from "@/lib/forums/normalizer"
import { getMockThread } from "@/lib/mock-data"

type Props = {
  params: Promise<{ id: string }>
}

export default async function ThreadPage({ params }: Props) {
  const { id: threadId } = await params
  let thread: Thread

  const mockThread = getMockThread(threadId)

  if (mockThread) {
    console.log(`[Thread Page] Using mock thread data for ${threadId}`)
    thread = mockThread
  } else {
    try {
      console.log(`[Thread Page] Attempting to fetch thread ${threadId} from Foru.ms...`)
      const forumsThread = await forumsClient.getThread(threadId)
      thread = forumsNormalizer.normalizeThread(forumsThread)
      console.log(`[Thread Page] Successfully fetched thread ${threadId} from Foru.ms`)
    } catch (error: any) {
      console.warn(`[Thread Page] Foru.ms fetch failed for ${threadId}:`, error.message)

      // Final fallback to basic demo
      console.log(`[Thread Page] No mock data found, using basic demo for ${threadId}`)
      thread = {
        id: threadId,
        forum_id: "demo-forum",
        title: `Demo Thread: ${threadId}`,
        posts: [
          {
            id: "p1",
            author_id: "demo_user",
            author_name: "Demo User",
            content: `This is a demo thread (${threadId}). Try using thread IDs: demo, ai-optimization, or react-patterns for full mock data.`,
            position: 0,
            created_at: new Date().toISOString(),
            upvotes: 0,
          },
        ],
      }
    }
  }

  // Pre-split each post into sentences server-side
  const postsWithSentences = thread.posts.map((p) => ({
    ...p,
    sentences: splitIntoSentences(p.content).map((s, idx) => ({ idx, text: s })),
  }))

  return (
    <div className="min-h-screen bg-background">
      <div className="max-w-6xl mx-auto px-6 py-10">
        <ThreadPageClient threadId={threadId} thread={thread} posts={postsWithSentences} />
      </div>
    </div>
  )
}
