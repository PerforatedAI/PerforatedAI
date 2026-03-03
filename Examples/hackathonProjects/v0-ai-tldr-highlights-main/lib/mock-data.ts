import type { Thread } from "./types"

export const MOCK_THREADS: Record<string, Thread> = {
  demo: {
    id: "demo",
    forum_id: "tech-forum",
    title: "Best practices for streaming LLM responses?",
    posts: [
      {
        id: "p1",
        author_id: "alice",
        author_name: "Alice Chen",
        content:
          "Has anyone solved the async streaming issue when summarizing long threads? I tried an SSE server but it chokes on large docs. Looking for best practices.",
        position: 0,
        created_at: "2025-12-01T10:00:00Z",
        upvotes: 12,
      },
      {
        id: "p2",
        author_id: "bob",
        author_name: "Bob Smith",
        content:
          "You can chunk text into windows of ~512 tokens and stream partial summaries. Cache embeddings per-sentence to avoid recompute. Use a faster model like GPT-4o-mini for streaming, then consolidate with a quality pass.",
        position: 1,
        created_at: "2025-12-01T10:04:00Z",
        upvotes: 6,
      },
      {
        id: "p3",
        author_id: "carmen",
        author_name: "Carmen Rodriguez",
        content:
          "Also watch out for long code blocks—strip them or send them as attachments; they cause token explosion. Using a shorter draft model for streaming helped reduce latency to under 200ms for first token.",
        position: 2,
        created_at: "2025-12-01T10:05:00Z",
        upvotes: 3,
      },
      {
        id: "p4",
        author_id: "dan",
        author_name: "Dan Lee",
        content:
          "For provenance tracking, I store character offsets when splitting sentences. This lets you link highlights back to exact locations in posts. Essential for verification!",
        position: 3,
        created_at: "2025-12-01T10:08:00Z",
        upvotes: 8,
      },
      {
        id: "p5",
        author_id: "eve",
        author_name: "Eve Thompson",
        content:
          "We implemented a two-pass approach: fast streaming draft (200ms first token), then quality consolidation in parallel. Cut total latency by 40% while maintaining accuracy.",
        position: 4,
        created_at: "2025-12-01T10:12:00Z",
        upvotes: 15,
      },
    ],
  },
  "ai-optimization": {
    id: "ai-optimization",
    forum_id: "ai-research",
    title: "Dendritic optimization for neural networks?",
    posts: [
      {
        id: "p1",
        author_id: "researcher1",
        author_name: "Dr. Sarah Johnson",
        content:
          "I've been experimenting with dendritic optimization using perforatedai. Initial results show 40% parameter reduction with minimal accuracy loss. Has anyone else tried this approach?",
        position: 0,
        created_at: "2025-11-28T14:30:00Z",
        upvotes: 24,
      },
      {
        id: "p2",
        author_id: "mleng",
        author_name: "Mike Chen",
        content:
          "That's impressive! What's your training setup? I've seen similar results with sparse attention patterns, but dendritic optimization seems more principled.",
        position: 1,
        created_at: "2025-11-28T14:45:00Z",
        upvotes: 8,
      },
      {
        id: "p3",
        author_id: "researcher1",
        author_name: "Dr. Sarah Johnson",
        content:
          "Using W&B sweeps for hyperparameter tuning. Key is the perforated backpropagation tracker—it automatically prunes low-impact connections during training. I can share my config if interested.",
        position: 2,
        created_at: "2025-11-28T15:00:00Z",
        upvotes: 16,
      },
      {
        id: "p4",
        author_id: "student",
        author_name: "Alex Kim",
        content:
          "Would love to see the config! I'm working on BERT compression for my thesis. Does this work with transformers or just CNNs?",
        position: 3,
        created_at: "2025-11-28T15:20:00Z",
        upvotes: 4,
      },
      {
        id: "p5",
        author_id: "researcher1",
        author_name: "Dr. Sarah Johnson",
        content:
          "Works great with transformers! I'm using it on T5-small for summarization. The key is wrapping attention layers properly. Check out the PAI module wrapper docs.",
        position: 4,
        created_at: "2025-11-28T15:45:00Z",
        upvotes: 12,
      },
    ],
  },
  "react-patterns": {
    id: "react-patterns",
    forum_id: "webdev",
    title: "State management patterns for complex forms?",
    posts: [
      {
        id: "p1",
        author_id: "frontend_dev",
        author_name: "Jordan Taylor",
        content:
          "What's the best approach for managing state in complex multi-step forms? Currently using useState for everything and it's getting messy.",
        position: 0,
        created_at: "2025-12-02T09:00:00Z",
        upvotes: 18,
      },
      {
        id: "p2",
        author_id: "react_expert",
        author_name: "Sam Wilson",
        content:
          "Use useReducer for complex state logic. It's more predictable than multiple useState calls and easier to test. Combine with useContext for sharing state across components.",
        position: 1,
        created_at: "2025-12-02T09:15:00Z",
        upvotes: 22,
      },
      {
        id: "p3",
        author_id: "frontend_dev",
        author_name: "Jordan Taylor",
        content: "Thanks! What about form validation? Should that be in the reducer or separate?",
        position: 2,
        created_at: "2025-12-02T09:30:00Z",
        upvotes: 5,
      },
      {
        id: "p4",
        author_id: "react_expert",
        author_name: "Sam Wilson",
        content:
          "Keep validation logic separate in custom hooks like useFormValidation. The reducer handles state transitions, validation hooks handle business logic. Cleaner separation of concerns.",
        position: 3,
        created_at: "2025-12-02T09:45:00Z",
        upvotes: 14,
      },
      {
        id: "p5",
        author_id: "another_dev",
        author_name: "Casey Brown",
        content:
          "Or just use React Hook Form with Zod validation. Handles 90% of form state management out of the box and has great TypeScript support.",
        position: 4,
        created_at: "2025-12-02T10:00:00Z",
        upvotes: 28,
      },
      {
        id: "p6",
        author_id: "frontend_dev",
        author_name: "Jordan Taylor",
        content:
          "Great suggestions all around! I'll try React Hook Form first for the quick win, and if I need more custom logic I'll go with the reducer pattern.",
        position: 5,
        created_at: "2025-12-02T10:20:00Z",
        upvotes: 7,
      },
    ],
  },
}

// Helper to get mock thread
export function getMockThread(threadId: string): Thread | null {
  return MOCK_THREADS[threadId] || null
}

// Helper to list available mock threads
export function listMockThreads(): Array<{ id: string; title: string; postCount: number }> {
  return Object.values(MOCK_THREADS).map((thread) => ({
    id: thread.id,
    title: thread.title,
    postCount: thread.posts.length,
  }))
}
