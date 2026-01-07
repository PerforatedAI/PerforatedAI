export interface MetricsCollector {
  // Timing metrics
  startTime: number
  firstTokenTime?: number
  digestTime?: number
  highlightsTime?: number
  totalTime?: number

  // Quality metrics
  tokensUsed: number
  highlightCount: number
  avgConfidence: number

  // Cost metrics
  estimatedCost: number

  // Engagement metrics
  clickThroughCount: number
  sourceVerifications: number
}

export class ProcessingMetrics {
  private metrics: Partial<MetricsCollector> = {}

  start() {
    this.metrics.startTime = Date.now()
  }

  recordFirstToken() {
    if (this.metrics.startTime) {
      this.metrics.firstTokenTime = Date.now() - this.metrics.startTime
    }
  }

  recordDigest() {
    if (this.metrics.startTime) {
      this.metrics.digestTime = Date.now() - this.metrics.startTime
    }
  }

  recordHighlights(count: number, avgConfidence: number) {
    if (this.metrics.startTime) {
      this.metrics.highlightsTime = Date.now() - this.metrics.startTime
    }
    this.metrics.highlightCount = count
    this.metrics.avgConfidence = avgConfidence
  }

  recordTokens(count: number, model: string) {
    this.metrics.tokensUsed = count

    // Estimate cost (rough approximation)
    const costPer1kTokens = {
      "gpt-4o": 0.005,
      "gpt-4o-mini": 0.0002,
      "claude-3-haiku": 0.00025,
    }

    const modelKey = model.includes("mini") ? "gpt-4o-mini" : model.includes("haiku") ? "claude-3-haiku" : "gpt-4o"

    this.metrics.estimatedCost = (count / 1000) * costPer1kTokens[modelKey]
  }

  finish() {
    if (this.metrics.startTime) {
      this.metrics.totalTime = Date.now() - this.metrics.startTime
    }
  }

  getMetrics(): MetricsCollector {
    return this.metrics as MetricsCollector
  }

  getSummary() {
    const m = this.metrics
    return {
      latency: {
        firstToken: m.firstTokenTime ? `${m.firstTokenTime}ms` : "N/A",
        digest: m.digestTime ? `${m.digestTime}ms` : "N/A",
        highlights: m.highlightsTime ? `${m.highlightsTime}ms` : "N/A",
        total: m.totalTime ? `${m.totalTime}ms` : "N/A",
      },
      quality: {
        highlightCount: m.highlightCount || 0,
        avgConfidence: m.avgConfidence ? m.avgConfidence.toFixed(2) : "N/A",
      },
      cost: {
        tokensUsed: m.tokensUsed || 0,
        estimatedCost: m.estimatedCost ? `$${m.estimatedCost.toFixed(4)}` : "N/A",
      },
    }
  }
}
