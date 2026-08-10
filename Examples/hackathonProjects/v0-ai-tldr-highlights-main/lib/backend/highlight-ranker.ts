import type { SentenceWithMeta, HighlightCategory } from "@/lib/types"
import { Persona, HighlightCategory as HC } from "@/lib/types"

export interface ClassifiedSentence extends SentenceWithMeta {
  category: HighlightCategory
  confidence: number
  importance_score: number
  why_matters?: string
}

export class HighlightRanker {
  /**
   * Calculate importance scores with enhanced criteria
   */
  calculateImportanceScores(
    sentences: Array<SentenceWithMeta & { category: HighlightCategory; confidence: number; why_matters?: string }>,
    persona: Persona,
    summaryText?: string,
  ): ClassifiedSentence[] {
    return sentences.map((sentence) => {
      let score = 0.0

      // Base score from confidence (40% weight)
      score += sentence.confidence * 0.4

      // Category importance (30% weight)
      const categoryWeights = {
        [HC.SOLUTION]: 0.3,
        [HC.FACT]: 0.25,
        [HC.CITATION]: 0.15,
        [HC.QUESTION]: 0.1,
        [HC.OPINION]: 0.05,
        [HC.IRRELEVANT]: 0.0,
      }
      score += categoryWeights[sentence.category] || 0.1

      // Position in thread (15% weight)
      // Early posts (questions) and late posts (solutions) are valuable
      if (sentence.post_position <= 2) {
        score += 0.1 // Question or problem statement
      } else if (sentence.post_position >= 5) {
        score += 0.05 // Later solutions
      }

      // Voting/engagement signals (10% weight)
      const normalizedVotes = Math.min(sentence.upvotes / 50, 1.0)
      score += normalizedVotes * 0.1

      // Sentence length penalty (too short or too long)
      const wordCount = sentence.text.split(/\s+/).length
      if (wordCount < 5 || wordCount > 100) {
        score *= 0.7 // Penalize
      }

      // Persona-specific boosts (5% weight)
      score += this.getPersonaBoost(sentence, persona)

      // Semantic overlap with summary (if provided)
      if (summaryText) {
        const overlap = this.calculateSemanticOverlap(sentence.text, summaryText)
        score += overlap * 0.1
      }

      return {
        ...sentence,
        importance_score: Math.min(score, 1.0),
      }
    })
  }

  private getPersonaBoost(sentence: SentenceWithMeta & { category: HighlightCategory }, persona: Persona): number {
    const textLower = sentence.text.toLowerCase()

    if (persona === Persona.NOVICE) {
      // Boost explanatory content
      if (sentence.category === HC.SOLUTION || sentence.category === HC.FACT) return 0.05
      if (textLower.includes("how to") || textLower.includes("step")) return 0.03
    } else if (persona === Persona.DEVELOPER) {
      // Boost technical content
      if (textLower.includes("code") || textLower.includes("implement") || textLower.includes("api")) return 0.05
      if (sentence.category === HC.CITATION) return 0.03
    } else if (persona === Persona.EXECUTIVE) {
      // Boost business/decision content
      const keywords = ["recommend", "suggest", "should", "conclusion", "summary", "decision", "impact"]
      if (keywords.some((kw) => textLower.includes(kw))) return 0.05
    }

    return 0.0
  }

  private calculateSemanticOverlap(sentence: string, summary: string): number {
    // Simple lexical overlap (in production, use embeddings)
    const sentenceWords = new Set(sentence.toLowerCase().split(/\s+/))
    const summaryWords = summary.toLowerCase().split(/\s+/)

    const overlaps = summaryWords.filter((w) => sentenceWords.has(w) && w.length > 3).length
    return Math.min(overlaps / 10, 1.0)
  }

  /**
   * Select top highlights with Maximal Marginal Relevance (MMR)
   * Balances relevance and diversity
   */
  selectTopHighlights(sentences: ClassifiedSentence[], maxHighlights = 10, lambda = 0.7): ClassifiedSentence[] {
    // Filter out irrelevant and low-quality
    const filtered = sentences.filter(
      (s) =>
        s.category !== HC.IRRELEVANT &&
        s.confidence >= 0.6 && // Higher threshold
        s.importance_score >= 0.3 && // Higher threshold
        s.text.split(/\s+/).length >= 5, // Minimum length
    )

    if (filtered.length === 0) return []

    // Sort by importance
    const sorted = filtered.sort((a, b) => b.importance_score - a.importance_score)

    // MMR selection
    const selected: ClassifiedSentence[] = []
    const remaining = [...sorted]

    // Select first (highest scoring)
    selected.push(remaining.shift()!)

    // Iteratively select based on MMR
    while (selected.length < maxHighlights && remaining.length > 0) {
      let bestScore = -1
      let bestIndex = -1

      for (let i = 0; i < remaining.length; i++) {
        const candidate = remaining[i]

        // Relevance score
        const relevance = candidate.importance_score

        // Maximum similarity to already selected
        const similarities = selected.map((s) => this.calculateSimilarity(candidate, s))
        const maxSimilarity = Math.max(...similarities, 0)

        // MMR score
        const mmrScore = lambda * relevance - (1 - lambda) * maxSimilarity

        if (mmrScore > bestScore) {
          bestScore = mmrScore
          bestIndex = i
        }
      }

      if (bestIndex >= 0) {
        selected.push(remaining.splice(bestIndex, 1)[0])
      } else {
        break
      }
    }

    return selected
  }

  private calculateSimilarity(a: ClassifiedSentence, b: ClassifiedSentence): number {
    // Post-level similarity
    if (a.post_id === b.post_id) {
      // Same post
      const sentenceDiff = Math.abs(a.sentence_index - b.sentence_index)
      if (sentenceDiff <= 1) return 0.9 // Adjacent sentences
      if (sentenceDiff <= 3) return 0.6 // Nearby sentences
      return 0.3
    }

    // Category similarity
    if (a.category === b.category) return 0.4

    // Lexical similarity
    const wordsA = new Set(
      a.text
        .toLowerCase()
        .split(/\s+/)
        .filter((w) => w.length > 3),
    )
    const wordsB = new Set(
      b.text
        .toLowerCase()
        .split(/\s+/)
        .filter((w) => w.length > 3),
    )

    const intersection = [...wordsA].filter((w) => wordsB.has(w)).length
    const union = wordsA.size + wordsB.size - intersection

    return union > 0 ? intersection / union : 0
  }

  /**
   * Re-rank highlights based on user interaction
   */
  reRankByEngagement(highlights: ClassifiedSentence[], clickedIds: string[]): ClassifiedSentence[] {
    return highlights
      .map((h) => {
        const wasClicked = clickedIds.includes(`${h.post_id}_${h.sentence_index}`)
        return {
          ...h,
          importance_score: wasClicked ? h.importance_score * 1.1 : h.importance_score,
        }
      })
      .sort((a, b) => b.importance_score - a.importance_score)
  }
}
