
import { PredictionResult, BatchPredictionResult, VibeLabel } from '../types';

const API_BASE_URL = 'http://localhost:8000';

export interface BackendModelInfo {
  backend: string;
  onnx_path: string | null;
  pytorch_path: string | null;
  max_length: number;
  labels: string[];
}

export const apiService = {
  async predict(text: string): Promise<PredictionResult> {
    try {
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      });
      if (!response.ok) throw new Error('Backend prediction failed');
      const data = await response.json();
      return {
        text: data.result.text,
        label: data.result.label as VibeLabel,
        confidence: data.result.probs[data.result.pred_index],
      };
    } catch (error) {
      console.warn('Backend not available, using mock fallback');
      // Import mock as fallback if backend is down
      const { predictVibe } = await import('./mockPredictor');
      return predictVibe(text);
    }
  },

  async predictBatch(texts: string[]): Promise<BatchPredictionResult> {
    try {
      const response = await fetch(`${API_BASE_URL}/predict_batch`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ texts }),
      });
      if (!response.ok) throw new Error('Batch prediction failed');
      const data = await response.json();
      const results = data.results.map((r: any) => ({
        text: r.text,
        label: r.label as VibeLabel,
        confidence: r.probs[r.pred_index],
      }));
      const avgConf = results.reduce((acc: number, r: any) => acc + r.confidence, 0) / results.length;
      return { results, averageConfidence: avgConf };
    } catch (error) {
      console.warn('Backend batch not available, using mock fallback');
      const { predictBatch } = await import('./mockPredictor');
      return predictBatch(texts);
    }
  },

  async startTraining(params: any) {
    const formData = new FormData();
    Object.entries(params).forEach(([key, value]) => {
      formData.append(key, String(value));
    });

    const response = await fetch(`${API_BASE_URL}/start_training`, {
      method: 'POST',
      body: formData,
    });
    return response.json();
  },

  async getJobStatus(jobId: string) {
    const response = await fetch(`${API_BASE_URL}/job_status/${jobId}`);
    return response.json();
  },

  async getModelInfo(): Promise<BackendModelInfo> {
    const response = await fetch(`${API_BASE_URL}/model_info`);
    return response.json();
  }
};
