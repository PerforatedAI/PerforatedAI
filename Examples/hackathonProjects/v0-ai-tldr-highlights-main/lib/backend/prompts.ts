import { Persona } from "@/lib/types"

export class PromptTemplates {
  /**
   * Build streaming summary prompt (Pass A - fast draft)
   * Optimized for low latency and progressive revelation
   */
  buildStreamingSummaryPrompt(threadContent: string, persona: Persona): string {
    const personaInstructions = {
      [Persona.NOVICE]:
        "Use plain language, explain technical terms in simple words, focus on what the user needs to know.",
      [Persona.DEVELOPER]:
        "Focus on technical details, implementation specifics, code patterns, and developer considerations. Be concise and precise.",
      [Persona.EXECUTIVE]:
        "Focus on business impact, key decisions, outcomes, and high-level recommendations. Skip technical implementation details.",
    }

    const instruction = personaInstructions[persona] || personaInstructions[Persona.DEVELOPER]

    return `You are a concise assistant creating a progressive TL;DR of a forum thread. ${instruction}

Thread content (most recent posts last):
${threadContent.slice(0, 8000)}

Provide a streaming summary in 2-4 sentences that captures:
1. The main question or topic
2. The consensus answer or key insight
3. Any important caveats or follow-ups

Be direct and actionable. Start immediately with the content.`
  }

  /**
   * Build consolidation summary prompt (Pass B - quality)
   * More comprehensive analysis after streaming
   */
  buildConsolidationPrompt(threadContent: string, persona: Persona, streamedSummary: string): string {
    return `You are refining a forum thread summary for ${persona} readers.

Initial streaming summary:
${streamedSummary}

Full thread content:
${threadContent.slice(0, 12000)}

Provide an improved, fact-checked summary that:
- Corrects any errors from the streaming summary
- Adds important details that were missed
- Maintains the same concise style (2-4 sentences)
- Indicates uncertainty if confidence is low

Return only the refined summary text.`
  }

  /**
   * Build digest generation prompt
   * Creates bullet point digest with verdict
   */
  buildDigestPrompt(threadContent: string, persona: Persona): string {
    const personaContext = {
      [Persona.NOVICE]: "a beginner who needs clear, simple explanations",
      [Persona.DEVELOPER]: "a technical practitioner who wants actionable details",
      [Persona.EXECUTIVE]: "a decision-maker who needs business context and recommendations",
    }

    return `Generate a concise digest of this forum thread for ${personaContext[persona]}.

Thread content:
${threadContent.slice(0, 10000)}

Format:
- [3-5 bullet points starting with "-"]
- Each bullet: one key insight, solution, or finding
- End with: "Verdict: [one sentence conclusion]"

Focus on actionable information and consensus answers.`
  }

  /**
   * Build highlight extraction prompt with provenance
   * Identifies key sentences with exact source tracking
   */
  buildHighlightExtractionPrompt(sentences: string[], maxHighlights = 10): string {
    return `From the following sentences, identify the top ${maxHighlights} that best answer the thread's main question or provide key solutions.

For each highlight, classify it as:
- fact: Verifiable information, data, or established knowledge
- solution: Direct answer, fix, or implementation approach
- opinion: Subjective viewpoint or recommendation
- question: Important clarifying question
- citation: Reference to external resource or documentation

Sentences (with indices):
${sentences
  .slice(0, 100)
  .map((s, i) => `[${i}] ${s}`)
  .join("\n")}

Return a JSON array with EXACTLY this structure:
[
  {
    "sentence_index": 0,
    "category": "solution",
    "confidence": 0.95,
    "why_matters": "Provides the accepted solution to the main problem"
  }
]

Select diverse, non-redundant highlights. Prioritize solutions and facts over opinions.`
  }

  /**
   * Build classification prompt for individual sentences
   */
  buildClassificationPrompt(sentence: string): string {
    return `Classify this forum post sentence into ONE category:

Categories:
- fact: Verifiable information or data
- solution: Proposed solution, answer, or fix
- opinion: Personal viewpoint or subjective statement
- question: Question being asked
- citation: Reference to external source or documentation
- irrelevant: Off-topic, greeting, signature, or noise

Sentence: "${sentence}"

Return JSON: {"category": "solution", "confidence": 0.92, "reasoning": "brief explanation"}`
  }

  /**
   * Build persona adaptation prompt
   * Re-frames content for specific audience
   */
  buildPersonaAdaptationPrompt(summary: string, fromPersona: Persona, toPersona: Persona): string {
    return `Adapt this forum thread summary from a ${fromPersona} perspective to a ${toPersona} perspective.

Original summary (${fromPersona}):
${summary}

Rewrite for ${toPersona} readers:
${toPersona === Persona.NOVICE ? "- Use simple language\n- Explain technical terms\n- Add context" : ""}
${toPersona === Persona.DEVELOPER ? "- Add technical details\n- Include code/implementation hints\n- Be precise" : ""}
${toPersona === Persona.EXECUTIVE ? "- Focus on business impact\n- Remove technical jargon\n- Emphasize decisions and outcomes" : ""}

Return only the adapted summary (2-4 sentences).`
  }

  /**
   * Build confidence assessment prompt
   * Determines if summary is reliable
   */
  buildConfidencePrompt(threadContent: string, summary: string): string {
    return `Assess the reliability of this summary for the given thread.

Thread (first 5000 chars):
${threadContent.slice(0, 5000)}

Summary:
${summary}

Evaluate:
1. Accuracy: Does the summary correctly represent the thread?
2. Completeness: Are key points covered?
3. Potential hallucinations: Any information not in the source?

Return JSON: {
  "confidence_score": 0.85,
  "is_reliable": true,
  "concerns": ["specific concern if any"],
  "recommendation": "use as-is" | "verify claims" | "regenerate"
}`
  }
}
