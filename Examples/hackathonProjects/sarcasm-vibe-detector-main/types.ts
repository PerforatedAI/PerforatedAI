
export type VibeLabel = 'Sarcasm' | 'Positive' | 'Neutral' | 'Negative';
export type EmotionLabel = 'Joy' | 'Anger' | 'Sadness' | 'Fear' | 'Surprise' | 'Neutral';

export interface PredictionResult {
  text: string;
  normalizedText?: string;
  label: VibeLabel;
  emotion?: EmotionLabel;
  confidence: number;
  uncertainty?: number;
  explanation?: string;
  tokenImportance?: Array<{ token: string; weight: number }>;
}

export interface BatchPredictionResult {
  results: PredictionResult[];
  averageConfidence: number;
}

export interface ModelInfo {
  name: string;
  version: string;
  parameters: string;
  latency: string;
  size: string;
  accuracy: string;
}

export interface DendriteStat {
  layer: string;
  active: number;
  total: number;
  activation: number;
}
