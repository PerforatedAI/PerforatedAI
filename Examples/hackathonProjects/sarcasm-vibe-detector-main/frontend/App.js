import React, { useState } from 'react';

const SarcasmDetector = () => {
  const [text, setText] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const analyzeVibe = async () => {
    setLoading(true);
    try {
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      });
      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error("Error analyzing vibe:", error);
    }
    setLoading(false);
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white p-8 font-sans">
      <div className="max-w-2xl mx-auto">
        <h1 className="text-4xl font-bold mb-2 bg-gradient-to-r from-cyan-400 to-purple-500 bg-clip-text text-transparent">
          Sarcasm Vibe Detector
        </h1>
        <p className="text-gray-400 mb-8">TinyBERT + Dendrites | Hinglish Optimized</p>

        <div className="bg-gray-800 rounded-2xl p-6 shadow-xl border border-gray-700">
          <textarea
            className="w-full bg-gray-900 border border-gray-700 rounded-xl p-4 text-lg focus:ring-2 focus:ring-cyan-500 outline-none transition-all"
            rows="4"
            placeholder="Type your Hinglish message here... (e.g., 'Wah bete, mauj kardi!')"
            value={text}
            onChange={(e) => setText(e.target.value)}
          />
          
          <button
            onClick={analyzeVibe}
            disabled={loading || !text}
            className="w-full mt-4 py-3 bg-gradient-to-r from-cyan-600 to-purple-600 rounded-xl font-semibold hover:opacity-90 disabled:opacity-50 transition-all"
          >
            {loading ? 'Analyzing Vibe...' : 'Detect Sarcasm'}
          </button>
        </div>

        {result && (
          <div className="mt-8 animate-fade-in">
            <div className="grid grid-cols-2 gap-4">
              <div className="bg-gray-800 p-4 rounded-xl border border-gray-700">
                <p className="text-gray-400 text-sm">Label</p>
                <p className="text-2xl font-bold text-cyan-400">{result.label}</p>
              </div>
              <div className="bg-gray-800 p-4 rounded-xl border border-gray-700">
                <p className="text-gray-400 text-sm">Vibe Score</p>
                <p className="text-2xl font-bold text-purple-400">{(result.vibe_score * 100).toFixed(1)}%</p>
              </div>
            </div>
            <div className="mt-4 bg-gray-800 p-4 rounded-xl border border-gray-700">
              <p className="text-gray-400 text-sm">Latency</p>
              <p className="text-lg">{result.latency_ms.toFixed(2)} ms</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default SarcasmDetector;
