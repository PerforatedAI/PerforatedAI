
import React, { useState, useCallback, useEffect } from 'react';
import { 
  ShieldCheck, 
  Cpu, 
  TrendingUp, 
  RefreshCcw,
  Sparkles,
  ChevronDown,
  Info,
  Terminal,
  Zap,
  Volume2
} from 'lucide-react';
import { VibeLabel, PredictionResult } from './types';
import { apiService } from './services/apiService';
import { geminiService } from './services/geminiService';
import { ttsService } from './services/ttsService';
import { normalizeHinglish, detectEmotion, getTokenImportance, estimateUncertainty } from './services/aiEngine';
import Navigation from './components/Navigation';
import BatchTest from './components/BatchTest';
import ModelStats from './components/ModelStats';
import TrainingDashboard from './components/TrainingDashboard';
import AIInsights from './components/AIInsights';
import LandingPage from './components/LandingPage';
import { SAMPLES } from './services/mockPredictor';

const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState('home');
  const [inputText, setInputText] = useState("Wah, kya timing hai — bilkul perfect... /s");
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [deepAnalysis, setDeepAnalysis] = useState<string | null>(null);
  const [showInsights, setShowInsights] = useState(false);

  const handlePredict = useCallback(async () => {
    if (!inputText.trim()) return;
    
    const normalized = normalizeHinglish(inputText);
    const baseResult = await apiService.predict(normalized);
    
    const result: PredictionResult = {
      ...baseResult,
      text: inputText,
      normalizedText: normalized,
      emotion: detectEmotion(normalized, baseResult.label),
      uncertainty: estimateUncertainty(normalized, baseResult.confidence),
      tokenImportance: getTokenImportance(inputText, baseResult.label)
    };
    
    setPrediction(result);
    setDeepAnalysis(null);
  }, [inputText]);

  const handleDeepAnalysis = async () => {
    if (!inputText.trim()) return;
    setIsAnalyzing(true);
    try {
      const explanation = await geminiService.analyzeSarcasm(inputText);
      setDeepAnalysis(explanation);
      setShowInsights(true); 
    } catch (err) {
      setDeepAnalysis("Deep analysis failed.");
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleSpeak = async () => {
    if (!prediction) return;
    setIsSpeaking(true);
    await ttsService.speak(prediction.text, prediction.label);
    setIsSpeaking(false);
  };

  const fillRandomSample = () => {
    const sample = SAMPLES[Math.floor(Math.random() * SAMPLES.length)];
    setInputText(sample);
  };

  useEffect(() => {
    if (activeTab === 'demo') {
      handlePredict();
    }
  }, [activeTab, handlePredict]);

  const getVibeStyles = (label: VibeLabel) => {
    switch (label) {
      case 'Sarcasm': return 'border-pink-500/30 bg-pink-500/5 text-pink-400';
      case 'Positive': return 'border-emerald-500/30 bg-emerald-500/5 text-emerald-400';
      case 'Negative': return 'border-orange-500/30 bg-orange-500/5 text-orange-400';
      default: return 'border-slate-500/30 bg-slate-500/5 text-slate-400';
    }
  };

  const getVibeIcon = (label: VibeLabel) => {
    switch (label) {
      case 'Sarcasm': return '🙃';
      case 'Positive': return '😊';
      case 'Negative': return '😠';
      default: return '🤔';
    }
  };

  return (
    <div className="min-h-screen flex flex-col items-center py-6 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto selection:bg-emerald-500/30">
      <Navigation activeTab={activeTab} setActiveTab={setActiveTab} />

      <main className="w-full min-h-[700px]">
        {activeTab === 'home' && <LandingPage onStart={() => setActiveTab('demo')} />}
        
        {activeTab === 'demo' && (
          <div className="flex flex-col lg:grid lg:grid-cols-12 gap-8 animate-in fade-in slide-in-from-bottom-6 duration-700">
            
            {/* Left Content: Hero & Specs */}
            <div className="lg:col-span-7 space-y-8">
              <header className="space-y-4">
                <div className="flex flex-wrap gap-2">
                  <span className="px-3 py-1 rounded-lg bg-emerald-500/10 text-emerald-400 text-xs font-bold uppercase tracking-widest border border-emerald-500/20 flex items-center gap-1.5 glow-emerald">
                    <Zap size={14} className="fill-current" /> SOTA Multimodal Engine
                  </span>
                  <span className="px-3 py-1 rounded-lg bg-slate-800 text-slate-400 text-xs font-bold uppercase tracking-widest border border-white/5">
                    Hinglish-Ready
                  </span>
                </div>
                <h2 className="text-5xl font-extrabold leading-tight tracking-tight">
                  Researcher <span className="gradient-text">Studio v3.1</span>
                </h2>
                <p className="text-slate-400 text-lg max-w-xl leading-relaxed">
                  Analyze specific code-mixed inputs to observe dendritic activation and multitask emotion anchors in real-time.
                </p>
              </header>

              <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                {[
                  { label: 'Recall', val: '89.4%', color: 'text-emerald-400' },
                  { label: 'Latency', val: '38ms', color: 'text-blue-400' },
                  { label: 'Params', val: '13.2M', color: 'text-purple-400' },
                  { label: 'ECE', val: '0.042', color: 'text-orange-400' },
                ].map((stat, i) => (
                  <div key={i} className="glass p-4 rounded-2xl text-center space-y-1">
                    <p className="text-[10px] text-slate-500 font-bold uppercase tracking-tighter">{stat.label}</p>
                    <p className={`text-xl font-bold ${stat.color}`}>{stat.val}</p>
                  </div>
                ))}
              </div>

              {prediction && showInsights ? (
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <h3 className="text-sm font-bold uppercase tracking-widest text-slate-400 flex items-center gap-2">
                      <Terminal size={16} /> Neural Interpretation
                    </h3>
                  </div>
                  <AIInsights result={prediction} />
                </div>
              ) : (
                <div className="glass p-8 rounded-3xl border-dashed border-white/10 flex flex-col items-center justify-center text-center space-y-4 min-h-[300px]">
                  <div className="w-16 h-16 rounded-full bg-slate-800 flex items-center justify-center text-slate-500">
                    <Info size={32} />
                  </div>
                  <div className="space-y-1">
                    <h3 className="text-slate-300 font-bold">Awaiting Prediction</h3>
                    <p className="text-slate-500 text-sm">Analyze text on the right to see neural weights & dendritic activation.</p>
                  </div>
                </div>
              )}
            </div>

            {/* Right Sidebar: Interaction & Results */}
            <div className="lg:col-span-5 space-y-6">
              <div className="glass p-8 rounded-[2.5rem] space-y-8 sticky top-6 border-white/10 shadow-2xl">
                <div className="flex items-center justify-between">
                  <h3 className="text-xl font-black tracking-tight text-white flex items-center gap-2">
                    <TrendingUp size={22} className="text-emerald-400" /> Interaction
                  </h3>
                  <button 
                    onClick={fillRandomSample}
                    className="p-2 rounded-xl bg-white/5 text-slate-400 hover:text-white transition-all hover:bg-white/10"
                    title="Load Random Sample"
                  >
                    <RefreshCcw size={18} />
                  </button>
                </div>

                <div className="space-y-4">
                  <div className="relative">
                    <textarea 
                      className="w-full h-40 bg-black/30 rounded-3xl p-6 text-lg border border-white/5 focus:ring-2 focus:ring-emerald-500/30 focus:border-emerald-500/50 outline-none transition-all placeholder:text-slate-700 resize-none leading-relaxed"
                      placeholder="Type code-mixed text... (e.g. 'Wah kya genius idea hai /s')"
                      value={inputText}
                      onChange={(e) => setInputText(e.target.value)}
                    />
                    <div className="absolute bottom-4 right-6 text-[10px] text-slate-600 font-bold uppercase pointer-events-none">
                      Hinglish v3.1
                    </div>
                  </div>

                  <div className="flex gap-3">
                    <button 
                      onClick={handlePredict}
                      className="flex-[2] py-4 bg-emerald-500 text-slate-950 font-black rounded-2xl hover:bg-emerald-400 transition-all shadow-xl shadow-emerald-500/20 active:scale-95 text-lg"
                    >
                      Run Inference
                    </button>
                    <button 
                      onClick={handleDeepAnalysis}
                      disabled={isAnalyzing}
                      className="flex-1 py-4 glass rounded-2xl font-bold flex items-center justify-center gap-2 hover:bg-white/10 transition-all disabled:opacity-50"
                    >
                      {isAnalyzing ? (
                        <RefreshCcw size={20} className="animate-spin" />
                      ) : (
                        <Sparkles size={20} className="text-blue-400" />
                      )}
                    </button>
                  </div>
                </div>

                {prediction && (
                  <div className={`p-8 rounded-[2rem] border transition-all duration-500 animate-in zoom-in-95 ${getVibeStyles(prediction.label)}`}>
                    <div className="flex items-center justify-between mb-6">
                      <div className="flex items-center gap-4">
                        <span className="text-6xl filter drop-shadow-lg">{getVibeIcon(prediction.label)}</span>
                        <div>
                          <h4 className="text-4xl font-black tracking-tighter uppercase">{prediction.label}</h4>
                          <p className="text-xs font-bold uppercase tracking-widest opacity-60">Vibe Signature</p>
                        </div>
                      </div>
                      <div className="text-right flex flex-col items-end gap-2">
                        <p className="text-2xl font-black">{(prediction.confidence * 100).toFixed(0)}%</p>
                        <button 
                          onClick={handleSpeak}
                          disabled={isSpeaking}
                          className="p-2 rounded-lg bg-white/10 hover:bg-white/20 text-current transition-all"
                        >
                          {isSpeaking ? <RefreshCcw className="animate-spin" size={16} /> : <Volume2 size={16} />}
                        </button>
                      </div>
                    </div>
                    
                    <button 
                      onClick={() => setShowInsights(!showInsights)}
                      className="w-full py-3 rounded-xl bg-white/5 border border-current/10 text-xs font-black uppercase tracking-widest hover:bg-white/10 transition-all flex items-center justify-center gap-2"
                    >
                      {showInsights ? 'Hide Details' : 'View AI Insights'} <ChevronDown size={14} className={showInsights ? 'rotate-180 transition-transform' : 'transition-transform'} />
                    </button>

                    {deepAnalysis && (
                      <div className="mt-6 pt-6 border-t border-current/10 space-y-3">
                        <div className="flex items-center gap-2 text-[10px] font-black uppercase opacity-50">
                          <Cpu size={12} /> Researcher Probe Output
                        </div>
                        <p className="text-sm font-medium leading-relaxed italic opacity-90">{deepAnalysis}</p>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {activeTab === 'batch' && <BatchTest />}
        {activeTab === 'models' && <ModelStats />}
        {activeTab === 'training' && <TrainingDashboard />}
      </main>

      <footer className="w-full mt-24 py-12 border-t border-white/5">
        <div className="flex flex-col md:flex-row items-center justify-between gap-6 px-4 opacity-50">
          <div className="flex items-center gap-8 text-sm font-medium">
            <a href="#" className="hover:text-emerald-400 transition-colors">Paper</a>
            <a href="#" className="hover:text-emerald-400 transition-colors">Architecture</a>
            <a href="#" className="hover:text-emerald-400 transition-colors">Safety Docs</a>
          </div>
          <p className="text-xs font-bold tracking-widest uppercase">
            © 2024 VibeDetector — Powered by PerforatedAI & Gemini
          </p>
        </div>
      </footer>
    </div>
  );
};

export default App;
