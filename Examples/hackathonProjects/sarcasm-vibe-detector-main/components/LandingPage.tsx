
import React, { useEffect, useState } from 'react';
import { 
  Zap, 
  ShieldCheck, 
  Cpu, 
  ArrowRight, 
  MessageSquare, 
  Users, 
  Globe,
  CheckCircle2,
  Volume2
} from 'lucide-react';
import { SAMPLES, predictVibe } from '../services/mockPredictor';
import { ttsService } from '../services/ttsService';
import { PredictionResult } from '../types';

interface LandingPageProps {
  onStart: () => void;
}

const LandingPage: React.FC<LandingPageProps> = ({ onStart }) => {
  const [liveResults, setLiveResults] = useState<PredictionResult[]>([]);
  const [playingId, setPlayingId] = useState<string | null>(null);

  useEffect(() => {
    const initial = SAMPLES.slice(0, 3).map(s => predictVibe(s));
    setLiveResults(initial);

    const interval = setInterval(() => {
      const randomSample = SAMPLES[Math.floor(Math.random() * SAMPLES.length)];
      const result = predictVibe(randomSample);
      setLiveResults(prev => [result, ...prev.slice(0, 2)]);
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  const handleSpeak = async (res: PredictionResult) => {
    setPlayingId(res.text);
    await ttsService.speak(res.text, res.label);
    setPlayingId(null);
  };

  return (
    <div className="space-y-24 pb-20 animate-in fade-in duration-1000">
      {/* Hero Section */}
      <section className="relative pt-12 text-center space-y-8">
        <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-sm font-bold tracking-widest uppercase shadow-[0_0_20px_rgba(16,185,129,0.1)]">
          <Zap size={16} className="fill-current" /> v3.1 Multimodal Engine Online
        </div>
        
        <h1 className="text-6xl md:text-8xl font-black tracking-tighter leading-[0.9] max-w-5xl mx-auto">
          Hear the <span className="gradient-text">Subtext</span> in Hinglish Sarcasm
        </h1>
        
        <p className="text-slate-400 text-xl md:text-2xl max-w-2xl mx-auto leading-relaxed">
          The first edge-optimized sarcasm detector that doesn't just read the text—it understands the <span className="text-white font-semibold">tone</span>. 
          Powered by Gemini 2.5 Flash TTS for multimodal irony analysis.
        </p>

        <div className="flex flex-col sm:flex-row items-center justify-center gap-4 pt-4">
          <button 
            onClick={onStart}
            className="w-full sm:w-auto px-10 py-5 bg-emerald-500 text-slate-950 font-black rounded-2xl hover:bg-emerald-400 transition-all shadow-2xl shadow-emerald-500/20 text-xl flex items-center justify-center gap-3 group"
          >
            Launch Demo <ArrowRight className="group-hover:translate-x-2 transition-transform" />
          </button>
          <button 
            className="w-full sm:w-auto px-10 py-5 glass rounded-2xl font-black hover:bg-white/10 transition-all border border-white/10 text-xl"
          >
            Read Whitepaper
          </button>
        </div>
      </section>

      {/* Live Feed Simulation with TTS */}
      <section className="max-w-4xl mx-auto space-y-6">
        <div className="flex items-center justify-between px-2">
          <h3 className="text-xs font-black uppercase tracking-[0.3em] text-slate-500 flex items-center gap-2">
            <Globe size={14} className="animate-pulse" /> Multimodal Inference Stream
          </h3>
          <span className="text-[10px] font-black uppercase text-emerald-500 flex items-center gap-1">
            <CheckCircle2 size={10} /> Audio prosody matched
          </span>
        </div>
        
        <div className="space-y-4">
          {liveResults.map((res, i) => (
            <div 
              key={`${res.text}-${i}`} 
              className="glass p-6 rounded-3xl border border-white/5 flex items-center justify-between group hover:border-white/20 transition-all animate-in slide-in-from-bottom-4"
            >
              <div className="flex items-center gap-6">
                <button 
                  onClick={() => handleSpeak(res)}
                  disabled={playingId === res.text}
                  className={`w-12 h-12 rounded-2xl flex items-center justify-center text-2xl transition-all ${
                    playingId === res.text ? 'bg-emerald-500/20 text-emerald-400' : 'bg-white/5 hover:bg-white/10 text-slate-400 hover:text-white'
                  }`}
                >
                  {playingId === res.text ? <div className="animate-ping w-2 h-2 bg-emerald-400 rounded-full" /> : <Volume2 size={20} />}
                </button>
                <div>
                  <p className="text-slate-300 font-bold leading-tight line-clamp-1">"{res.text}"</p>
                  <p className="text-[10px] font-black uppercase tracking-widest text-slate-500 mt-1">Sarcasm Score: {(res.confidence * 100).toFixed(0)}%</p>
                </div>
              </div>
              <div className="text-right">
                <p className={`text-lg font-black ${res.label === 'Sarcasm' ? 'text-pink-400' : 'text-emerald-400'}`}>
                  {res.label}
                </p>
                <div className="flex items-center gap-2 justify-end">
                  <div className="w-16 h-1 bg-white/5 rounded-full overflow-hidden">
                    <div className="h-full bg-current opacity-50" style={{ width: `${res.confidence * 100}%` }}></div>
                  </div>
                  <p className="text-[10px] font-bold opacity-40">{res.emotion}</p>
                </div>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Features Grid */}
      <section className="grid grid-cols-1 md:grid-cols-3 gap-8">
        {[
          {
            icon: ShieldCheck,
            title: "Prosody-Aware",
            desc: "The model doesn't just detect sarcasm; it speaks with the exact cynical tone used in urban India.",
            color: "text-blue-400"
          },
          {
            icon: Cpu,
            title: "Dendritic Scaling",
            desc: "Our Tiny BERT backbone recovers its full capacity for complex code-mixed Roman-Hindi semantics.",
            color: "text-emerald-400"
          },
          {
            icon: Volume2,
            title: "Multimodal Studio",
            desc: "Integrated TTS allows for real-time validation of ironic prosody across 5 pre-built voices.",
            color: "text-purple-400"
          }
        ].map((feat, i) => (
          <div key={i} className="glass p-10 rounded-[2.5rem] border-white/5 space-y-4 hover:translate-y-[-8px] transition-all">
            <div className={`p-4 rounded-2xl bg-white/5 inline-block ${feat.color}`}>
              <feat.icon size={32} />
            </div>
            <h4 className="text-2xl font-black">{feat.title}</h4>
            <p className="text-slate-400 leading-relaxed">{feat.desc}</p>
          </div>
        ))}
      </section>
    </div>
  );
};

export default LandingPage;
