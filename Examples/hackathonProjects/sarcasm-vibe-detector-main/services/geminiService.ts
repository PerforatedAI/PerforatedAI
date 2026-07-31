
import { GoogleGenAI } from "@google/genai";

export class GeminiService {
  private getClient() {
    return new GoogleGenAI({ apiKey: process.env.API_KEY });
  }

  async analyzeSarcasm(text: string): Promise<string> {
    try {
      const ai = this.getClient();
      const response = await ai.models.generateContent({
        model: 'gemini-3-flash-preview',
        contents: `Act as a NeurIPS-level ML Research Scientist specializing in Pragmatics and Hinglish NLP.
        Analyze the following text for Sarcasm and Emotion (Multitask context). 
        The input is likely Hinglish (Hindi + English code-mixed).
        
        Provide a concise research-style technical breakdown (max 3 sentences):
        1. Linguistic cue identified (normalized).
        2. Emotional anchor mapping.
        3. Structural reasoning for sarcasm classification.
        
        Text: "${text}"`,
        config: {
          temperature: 0.7,
          maxOutputTokens: 200,
        }
      });

      return response.text || "Structural analysis inconclusive.";
    } catch (error) {
      console.error("Gemini Analysis Error:", error);
      return "Unable to perform deep analysis at this time.";
    }
  }
}

export const geminiService = new GeminiService();
