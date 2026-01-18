
import { PredictionResult, VibeLabel, BatchPredictionResult, EmotionLabel } from '../types';

export const SAMPLES = [
  "Wah! 2 ghante late ho aur bol rahe ho 'bas 5 min'. Genius! 🙄",
  "Bhai kya hi logic lagaya hai tune, Nobel prize milna chahiye! /s",
  "Love this product, sach mein badiya kaam kiya hai team ne! 🙌",
  "Bohot badhiya service hai aapki, 10 din se call wait pe hai. Amazing. 👏",
  "Kya mast room saaf kiya hai, lag raha hai yahan tsunami aayi thi. 😂",
  "Great timing! My internet died right during the meeting. Lovely.",
  "Bhai, tum toh genius ho (bilkul nahi)",
  "Sach mein bohot help hui, thanks a ton! 😍",
  "Haan haan, tumhare paas toh bohot paisa hai na, hum hi gareeb hain.",
  "Wow, another meeting that could have been an email. My favorite.",
  "Zabardast yaar, keep it up! Proud of you.",
  "Sure, let's wait another hour. I have nothing better to do. 😑",
  "Bohot accha kiya jo bata diya, varna main toh bhul hi jata.",
  "Kya baat hai, itna fast delivery? Amazon fail hai tumhare samne. /s",
  "Badiya badiya, aise hi kaam karte raho (don't).",
  "Tumhari baatein sun ke lagta hai main hi galat tha. Obviously.",
  "Hahaha, sahi hai bhai, tum toh antaryami ho.",
  "Mast weather hai yaar, chalo chai peete hain! ☕",
  "Oh perfect, it's raining and I forgot my umbrella. Just my luck."
];

export function predictVibe(text: string): PredictionResult {
  const t = text.toLowerCase();
  let scoreSarcasm = 0.0;
  let scorePos = 0.0;
  let scoreNeg = 0.0;

  // Sarcasm Logic (Heuristics)
  const sarcasmMarkers = ["/s", "not.", "yeah right", "sure", "as if", "totally", "wow, great", "great timing", "genius", "bilkul nahi", "nobel prize", "obviously", "perfect", "just my luck"];
  const positiveWords = ["love", "awesome", "great", "mast", "zabardast", "badiya", "proud", "helpful", "thanks", "accha"];
  const negativeWords = ["hate", "terrible", "sucks", "ghatiya", "bekaar", "dead", "waste", "late", "died", "gareeb"];
  
  sarcasmMarkers.forEach(m => { if (t.includes(m)) scoreSarcasm += 0.4; });
  positiveWords.forEach(m => { if (t.includes(m)) scorePos += 0.2; });
  negativeWords.forEach(m => { if (t.includes(m)) scoreNeg += 0.25; });
  
  // Detection of Praise + Negative context (Classic Sarcasm)
  if ((t.includes("genius") || t.includes("wah") || t.includes("perfect") || t.includes("great")) && 
      (t.includes("late") || t.includes("died") || t.includes("tsunami") || t.includes("wait") || t.includes("dead"))) {
    scoreSarcasm += 0.6;
  }

  const exclam = (text.match(/!/g) || []).length;
  if (exclam >= 2) scoreSarcasm += 0.15;

  const sarcasmEmojis = ["🙃", "🙄", "😒", "😑", "😂", "😉", "👏"];
  sarcasmEmojis.forEach(e => { if (text.includes(e)) scoreSarcasm += 0.25; });

  let label: VibeLabel = 'Neutral';
  let maxScore = 0.0;

  if (scoreSarcasm > scorePos && scoreSarcasm > 0.45) {
    label = 'Sarcasm';
    maxScore = scoreSarcasm;
  } else if (scorePos > scoreNeg && scorePos > 0.3) {
    label = 'Positive';
    maxScore = scorePos;
  } else if (scoreNeg > 0.3) {
    label = 'Negative';
    maxScore = scoreNeg;
  }

  const confidence = Math.min(0.99, 0.6 + (maxScore / 2.5));
  
  // Simple Emotion mapping for mock
  let emotion: EmotionLabel = 'Neutral';
  if (label === 'Sarcasm') emotion = 'Anger';
  else if (label === 'Positive') emotion = 'Joy';
  else if (label === 'Negative') emotion = 'Sadness';

  return { 
    text, 
    label, 
    confidence, 
    emotion,
    normalizedText: text.toLowerCase().trim()
  };
}

export function predictBatch(texts: string[]): BatchPredictionResult {
  const results = texts.map(t => predictVibe(t));
  const avgConf = results.reduce((acc, r) => acc + r.confidence, 0) / results.length;
  return { results, averageConfidence: avgConf };
}
