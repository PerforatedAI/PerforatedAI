
import { PredictionResult, VibeLabel, EmotionLabel } from '../types';

/**
 * Phonetic Correction Dictionary (Research-Curated)
 */
const PHONETIC_CORRECTIONS: Record<string, string> = {
  "steshan": "station",
  "pulis": "police",
  "jaldi": "jaldi",
  "jldi": "jaldi",
  "kal": "kal",
  "kl": "kal",
  "bohot": "bahut",
  "accha": "achha",
  "acha": "achha",
  "masttt": "mast",
  "yaarrr": "yaar",
  "apake": "apke",
  "karna": "krna",
  "juggad": "jugaad"
};

/**
 * Advanced Hinglish Normalizer
 * Handles character elongation, phonetic misspellings, and transliteration noise.
 */
export function normalizeHinglish(text: string): string {
  let processed = text.toLowerCase();
  
  // Rule 1: Remove excessive punctuation (!!! -> !)
  processed = processed.replace(/[\!\"#\$%\&\'\(\)\*\+,\.\/:;<=>\?@\[\\\]\^_`{\|}~]{2,}/g, ' ');

  // Rule 2: Remove character elongation (broooo -> broo)
  processed = processed.replace(/(.)\1{2,}/g, '$1$1');

  // Rule 3: Word-level phonetic correction
  const words = processed.split(/\s+/);
  const normalized = words.map(w => PHONETIC_CORRECTIONS[w] || w);
  
  return normalized.join(' ').trim();
}

/**
 * Entropy-based Confidence Calibration
 */
export function estimateUncertainty(text: string, confidence: number): number {
  const complexity = text.split(' ').length / 25;
  const baseUncertainty = 1 - confidence;
  return Math.min(0.99, baseUncertainty + (complexity * 0.05));
}

/**
 * Hybrid Saliency Extraction (Mock for Token + Char CNN)
 */
export function getTokenImportance(text: string, label: VibeLabel): Array<{ token: string; weight: number }> {
  const words = text.split(/\s+/);
  const indicators = ["/s", "not.", "yeah", "right", "sure", "waah", "genius", "bilkul"];
  
  return words.map(word => {
    let weight = 0.1 + Math.random() * 0.15;
    const lower = word.toLowerCase().replace(/[^\w]/g, '');
    if (indicators.some(i => lower.includes(i)) && label === 'Sarcasm') {
      weight += 0.6;
    }
    return { token: word, weight: Math.min(1, weight) };
  });
}

/**
 * Multitask Emotion Head
 */
export function detectEmotion(text: string, label: VibeLabel): EmotionLabel {
  const t = text.toLowerCase();
  if (label === 'Sarcasm') return 'Anger';
  if (t.includes('love') || t.includes('mast') || t.includes('best') || t.includes('badiya')) return 'Joy';
  if (t.includes('nahi') || t.includes('bekaar') || t.includes('sucks')) return 'Anger';
  if (t.includes('shock') || t.includes('kya') || t.includes('waah')) return 'Surprise';
  return 'Neutral';
}
