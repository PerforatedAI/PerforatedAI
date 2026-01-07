
import { GoogleGenAI, Modality } from "@google/genai";

export class TTSService {
  private getClient() {
    return new GoogleGenAI({ apiKey: process.env.API_KEY });
  }

  private decodeBase64(base64: string): Uint8Array {
    const binaryString = atob(base64);
    const bytes = new Uint8Array(binaryString.length);
    for (let i = 0; i < binaryString.length; i++) {
      bytes[i] = binaryString.charCodeAt(i);
    }
    return bytes;
  }

  private async decodeAudioData(
    data: Uint8Array,
    ctx: AudioContext,
    sampleRate: number,
    numChannels: number
  ): Promise<AudioBuffer> {
    const dataInt16 = new Int16Array(data.buffer);
    const frameCount = dataInt16.length / numChannels;
    const buffer = ctx.createBuffer(numChannels, frameCount, sampleRate);

    for (let channel = 0; channel < numChannels; channel++) {
      const channelData = buffer.getChannelData(channel);
      for (let i = 0; i < frameCount; i++) {
        channelData[i] = dataInt16[i * numChannels + channel] / 32768.0;
      }
    }
    return buffer;
  }

  async speak(text: string, vibe: 'Sarcasm' | 'Positive' | 'Neutral' | 'Negative'): Promise<void> {
    try {
      const ai = this.getClient();
      
      // Select voice based on vibe
      // Puck is cheeky/sarcastic, Zephyr is clear/neutral
      const voiceName = vibe === 'Sarcasm' ? 'Puck' : (vibe === 'Positive' ? 'Kore' : 'Zephyr');
      const instruction = vibe === 'Sarcasm' 
        ? `Say this with heavy sarcasm and irony: "${text}"` 
        : `Say this naturally: "${text}"`;

      const response = await ai.models.generateContent({
        model: "gemini-2.5-flash-preview-tts",
        contents: [{ parts: [{ text: instruction }] }],
        config: {
          responseModalities: [Modality.AUDIO],
          speechConfig: {
            voiceConfig: {
              prebuiltVoiceConfig: { voiceName },
            },
          },
        },
      });

      const base64Audio = response.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
      if (!base64Audio) throw new Error("No audio data received");

      const audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 24000 });
      const audioData = this.decodeBase64(base64Audio);
      const audioBuffer = await this.decodeAudioData(audioData, audioCtx, 24000, 1);

      const source = audioCtx.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(audioCtx.destination);
      source.start();
    } catch (error) {
      console.error("TTS Error:", error);
    }
  }
}

export const ttsService = new TTSService();
