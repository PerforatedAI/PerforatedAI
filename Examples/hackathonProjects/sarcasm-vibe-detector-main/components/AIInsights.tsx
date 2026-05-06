
import React from 'react';
import { PredictionResult, DendriteStat } from '../types';
import { Activity, Brain, Fingerprint, Layers, ShieldAlert, Cpu } from 'lucide-react';

interface AIInsightsProps {
  result: PredictionResult;
}

const AIInsights: React.FC<AIInsightsProps> = ({ result }) => {
  const dendriteStats: DendriteStat[] = [
    { layer: 'Dendrite 1: Punctuation/Tone', active: 3, total: 4, activation: 0.88 },
    { layer: 'Dendrite 2: Cultural Slang', active: 4, total: 4, activation: 0.94 },
    { layer: 'Dendrite 3: Contrastive Logic', active: 2, total: 4, activation: 0.58 },
    { layer: 'Dendrite 4: Emoji Sentiment', active: 4, total: 4, activation: 0.91 },
  ];

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        
        {/* Normalization Panel */}
        <div className="glass p-6 rounded-3xl space-y-4 border-l-4 border-emerald-500/50">
          <div className="flex items-center gap-3 text-emerald-400">
            <div className="p-2 rounded-lg bg-emerald-500/10">
              <Fingerprint size={20} />
            </div>
            <h4 className="text-xs font-black uppercase tracking-widest">Normalization Engine</h4>
          </div>
          <div className="space-y-4">
            <div className="space-y-1">
              <p className="text-[10px] text-slate-500 font-black uppercase">Input Stream</p>
              <p className="text-sm italic text-slate-400 font-medium truncate">"{result.text}"</p>
            </div>
            <div className="space-y-1">
              <p className="text-[10px] text-slate-500 font-black uppercase">Cleaned Tensor</p>
              <p className="text-sm font-bold text-slate-200">"{result.normalizedText || result.text}"</p>
            </div>
          </div>
        </div>

        {/* Calibration Panel */}
        <div className="glass p-6 rounded-3xl space-y-4 border-l-4 border-orange-500/50">
          <div className="flex items-center gap-3 text-orange-400">
            <div className="p-2 rounded-lg bg-orange-500/10">
              <ShieldAlert size={20} />
            </div>
            <h4 className="text-xs font-black uppercase tracking-widest">Trust Metrics</h4>
          </div>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <p className="text-[10px] text-slate-500 font-black uppercase">Risk Factor</p>
              <p className="text-xl font-black text-slate-200">{(result.uncertainty || 0).toFixed(3)}</p>
            </div>
            <div className="w-full h-1.5 bg-white/5 rounded-full overflow-hidden">
              <div 
                className="h-full bg-orange-500 transition-all duration-1000" 
                style={{ width: `${(result.uncertainty || 0) * 100}%` }}
              ></div>
            </div>
            <p className="text-[10px] text-slate-500 leading-tight">ECE calibrated post-processing ensures confidence scores reflect actual model error margins.</p>
          </div>
        </div>
      </div>

      {/* Multitask Section */}
      <div className="glass p-6 rounded-3xl space-y-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3 text-purple-400">
            <div className="p-2 rounded-lg bg-purple-500/10">
              <Activity size={20} />
            </div>
            <h4 className="text-xs font-black uppercase tracking-widest">Multitask Heads</h4>
          </div>
          <div className="flex items-center gap-3">
            <span className="text-xs font-bold text-slate-500 uppercase tracking-widest">Active Task: Emotion Mapping</span>
            <span className="text-2xl font-black text-white">{result.emotion || 'Neutral'}</span>
          </div>
        </div>

        {/* Attention Map */}
        <div className="space-y-4">
          <div className="flex items-center gap-3 text-blue-400">
            <div className="p-2 rounded-lg bg-blue-500/10">
              <Brain size={20} />
            </div>
            <h4 className="text-xs font-black uppercase tracking-widest">Attention Saliency Map</h4>
          </div>
          <div className="flex flex-wrap gap-2">
            {result.tokenImportance?.map((item, idx) => (
              <div key={idx} className="relative group">
                <span 
                  className="px-3 py-1.5 rounded-xl text-sm font-bold border border-white/5 transition-all cursor-default"
                  style={{ 
                    backgroundColor: `rgba(59, 130, 246, ${item.weight * 0.4})`,
                    color: item.weight > 0.45 ? '#fff' : '#94a3b8',
                  }}
                >
                  {item.token}
                </span>
                <div className="absolute -top-10 left-1/2 -translate-x-1/2 opacity-0 group-hover:opacity-100 transition-opacity bg-slate-900 text-[10px] font-black p-2 rounded-lg border border-white/10 z-10 whitespace-nowrap shadow-2xl">
                  {Math.round(item.weight * 100)}% Influence
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Structural Stats Section */}
      <div className="glass p-8 rounded-3xl space-y-6">
        <div className="flex items-center gap-3 text-amber-400">
          <div className="p-2 rounded-lg bg-amber-500/10">
            <Layers size={20} />
          </div>
          <h4 className="text-xs font-black uppercase tracking-widest">Structural Capacity Recovery</h4>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-x-12 gap-y-6">
          {dendriteStats.map((stat, idx) => (
            <div key={idx} className="space-y-2 group">
              <div className="flex justify-between items-end">
                <div className="space-y-1">
                  <p className="text-[10px] font-black uppercase text-slate-500 tracking-tighter">{stat.layer}</p>
                  <p className="text-xs font-bold text-slate-300">{(stat.activation * 100).toFixed(0)}% Optimization</p>
                </div>
                <div className="text-[10px] font-black px-2 py-1 rounded-md bg-white/5 text-amber-500">
                  {stat.active} BRANCHES
                </div>
              </div>
              <div className="w-full h-2 bg-black/40 rounded-full overflow-hidden p-0.5 border border-white/5">
                <div 
                  className="h-full bg-gradient-to-r from-amber-600 to-amber-400 rounded-full transition-all duration-1000 group-hover:brightness-125 shadow-[0_0_10px_rgba(245,158,11,0.2)]" 
                  style={{ width: `${stat.activation * 100}%` }}
                ></div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default AIInsights;
