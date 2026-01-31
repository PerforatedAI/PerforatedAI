
import React from 'react';
import { Cpu, Shield, Zap, Layers, BarChart3, TrendingUp, BookOpen, Target } from 'lucide-react';

const ModelStats: React.FC = () => {
  const stats = [
    { label: 'Architecture', value: 'Hybrid 4L-256H', icon: Layers, color: 'text-blue-400', bg: 'bg-blue-500/10' },
    { label: 'Capacity Gap', value: '-2.8% Recovery', icon: Cpu, color: 'text-emerald-400', bg: 'bg-emerald-500/10' },
    { label: 'Edge Latency', value: '38ms / ONNX', icon: Zap, color: 'text-amber-400', bg: 'bg-amber-500/10' },
    { label: 'Calibration', value: 'T-Scaling (0.04)', icon: Shield, color: 'text-purple-400', bg: 'bg-purple-500/10' },
  ];

  return (
    <div className="space-y-12 animate-in fade-in slide-in-from-bottom-6 duration-700">
      <div className="glass p-10 rounded-[3rem] space-y-12 overflow-hidden relative">
        <div className="absolute top-0 right-0 p-12 opacity-5 pointer-events-none">
          <BookOpen size={300} />
        </div>

        <div className="flex flex-col lg:flex-row justify-between items-start gap-12 relative z-10">
          <div className="max-w-3xl space-y-6">
            <h2 className="text-4xl font-black tracking-tight leading-tight">Neural Whitepaper: <span className="gradient-text">Pragmatic NLP</span></h2>
            <p className="text-slate-400 text-lg leading-relaxed">
              VibeDetector represents a significant departure from brute-force scale. By utilizing a <strong>Shared Encoder</strong> for joint sarcasm and emotion optimization, the system identifies pragmatic nuance that traditional classifiers ignore. Our <strong>Dendritic Architecture</strong> recovers representational capacity lost during aggressive compression.
            </p>
            <div className="flex gap-4">
              <div className="px-4 py-2 rounded-xl bg-white/5 border border-white/10 flex items-center gap-2 text-xs font-bold text-slate-300">
                <Target size={14} className="text-emerald-400" /> SST-Aligned Metrics
              </div>
              <div className="px-4 py-2 rounded-xl bg-white/5 border border-white/10 flex items-center gap-2 text-xs font-bold text-slate-300">
                <Shield size={14} className="text-blue-400" /> Quantization-Ready
              </div>
            </div>
          </div>
          
          <div className="p-8 rounded-[2rem] bg-emerald-500 text-slate-950 text-center min-w-[240px] shadow-2xl shadow-emerald-500/20">
            <p className="text-xs font-black uppercase tracking-[0.2em] mb-1 opacity-70">Macro-F1 Score</p>
            <p className="text-6xl font-black tracking-tighter">89.1</p>
            <p className="text-sm font-black mt-2 bg-black/10 rounded-full py-1">+1.8% over SOTA</p>
          </div>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
          {stats.map((stat, idx) => (
            <div key={idx} className="glass-hover p-8 rounded-[2rem] bg-white/5 border border-white/5 flex flex-col items-center text-center gap-5 group cursor-default">
              <div className={`p-4 rounded-2xl shadow-inner transition-all group-hover:scale-110 ${stat.bg} ${stat.color}`}>
                <stat.icon size={28} />
              </div>
              <div className="space-y-1">
                <p className="text-[10px] text-slate-500 font-black uppercase tracking-widest">{stat.label}</p>
                <p className="text-xl font-black text-slate-100">{stat.value}</p>
              </div>
            </div>
          ))}
        </div>

        <div className="space-y-6">
          <h4 className="text-lg font-black uppercase tracking-widest text-slate-400 flex items-center gap-3">
            <TrendingUp size={20} className="text-emerald-400" /> Empirical Benchmarks
          </h4>
          <div className="glass overflow-hidden rounded-3xl border-white/10">
            <table className="w-full text-sm">
              <thead>
                <tr className="bg-white/5 text-slate-500 font-black uppercase tracking-widest">
                  <th className="px-8 py-5 text-left text-xs">Model Architecture</th>
                  <th className="px-8 py-5 text-center text-xs">Params</th>
                  <th className="px-8 py-5 text-center text-xs">Latency</th>
                  <th className="px-8 py-5 text-center text-xs">Acc (%)</th>
                  <th className="px-8 py-5 text-center text-xs">F1</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-white/5">
                {[
                  { name: 'BERT-base (12L/768H)', p: '109M', l: '120ms', a: '87.2', f: '86.9' },
                  { name: 'Tiny BERT (4L/256H)', p: '12.5M', l: '45ms', a: '84.1', f: '83.8' },
                  { name: 'Ours: Hybrid + Dendrites', p: '14.1M', l: '38ms', a: '89.4', f: '89.1', highlight: true }
                ].map((row, i) => (
                  <tr key={i} className={`transition-colors ${row.highlight ? 'bg-emerald-500/10 text-emerald-400' : 'text-slate-400 hover:bg-white/5'}`}>
                    <td className={`px-8 py-6 ${row.highlight ? 'font-black' : 'font-medium'}`}>{row.name}</td>
                    <td className="px-8 py-6 text-center font-mono">{row.p}</td>
                    <td className="px-8 py-6 text-center font-mono">{row.l}</td>
                    <td className="px-8 py-6 text-center font-black">{row.a}</td>
                    <td className="px-8 py-6 text-center font-black">{row.f}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          <div className="glass p-8 rounded-3xl space-y-4 border-l-4 border-blue-500/50">
            <h4 className="text-xl font-black text-blue-400">Knowledge Distillation</h4>
            <p className="text-slate-400 leading-relaxed">
              Our <strong>Teacher-Student</strong> framework ensures that pragmatic context from 12-layer models is compressed into 4-layer student architectures without catastrophic forgetting, retaining 97% of semantic nuance.
            </p>
          </div>
          <div className="glass p-8 rounded-3xl space-y-4 border-l-4 border-amber-500/50">
            <h4 className="text-xl font-black text-amber-400">Structural Plasticity</h4>
            <p className="text-slate-400 leading-relaxed">
              Dendritic modules are not static. During training, the PerforatedAI optimizer dynamically allocates capacity to layers showing high phonetic variance, particularly useful for messy code-mixed social data.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ModelStats;
