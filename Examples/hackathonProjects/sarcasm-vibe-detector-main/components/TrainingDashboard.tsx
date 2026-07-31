
import React, { useState } from 'react';
import { Rocket, Activity, PlayCircle, StopCircle, Clock, Database, CheckCircle2, FlaskConical, Layers } from 'lucide-react';

const TrainingDashboard: React.FC = () => {
  const [isTraining, setIsTraining] = useState(false);
  const [isAugmenting, setIsAugmenting] = useState(false);
  const [progress, setProgress] = useState(0);

  const startTraining = () => {
    setIsTraining(true);
    setProgress(0);
    const interval = setInterval(() => {
      setProgress((p) => {
        if (p >= 100) {
          clearInterval(interval);
          setIsTraining(false);
          return 100;
        }
        return p + Math.random() * 8;
      });
    }, 800);
  };

  const startAugmentation = () => {
    setIsAugmenting(true);
    setTimeout(() => setIsAugmenting(false), 2000);
  };

  return (
    <div className="space-y-6 animate-in fade-in duration-500">
      {/* Configuration Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="glass p-6 rounded-3xl space-y-4">
          <h3 className="text-lg font-bold flex items-center gap-2">
            <FlaskConical className="text-emerald-400" size={20} />
            Data Preparation
          </h3>
          <div className="space-y-3">
            <div className="p-3 rounded-xl bg-white/5 border border-white/5 flex justify-between items-center">
              <span className="text-sm text-slate-400">Back-translation Augmentation</span>
              <button 
                onClick={startAugmentation}
                disabled={isAugmenting}
                className="text-xs font-bold text-emerald-400 hover:text-emerald-300 disabled:opacity-50"
              >
                {isAugmenting ? 'Processing...' : 'Run Now'}
              </button>
            </div>
            <div className="p-3 rounded-xl bg-white/5 border border-white/5 flex justify-between items-center">
              <span className="text-sm text-slate-400">Teacher Pseudo-labeling (1M set)</span>
              <button className="text-xs font-bold text-blue-400 hover:text-blue-300">Queue Task</button>
            </div>
          </div>
        </div>

        <div className="glass p-6 rounded-3xl space-y-4">
          <h3 className="text-lg font-bold flex items-center gap-2">
            <Layers className="text-purple-400" size={20} />
            Structural Constraints
          </h3>
          <div className="grid grid-cols-2 gap-3">
            <div className="p-2 bg-black/20 rounded-lg text-center">
              <p className="text-[10px] text-slate-500 font-bold uppercase">Max Dendrites</p>
              <p className="text-lg font-bold">4 / layer</p>
            </div>
            <div className="p-2 bg-black/20 rounded-lg text-center">
              <p className="text-[10px] text-slate-500 font-bold uppercase">Distill Temp</p>
              <p className="text-lg font-bold">2.0</p>
            </div>
          </div>
        </div>
      </div>

      {/* Main training panel */}
      <div className="glass p-8 rounded-3xl space-y-6">
        <div className="flex items-center justify-between">
          <h3 className="text-xl font-bold flex items-center gap-2">
            <Rocket className="text-purple-400" size={22} />
            Dendritic Fine-tuning Studio
          </h3>
          <span className={`px-3 py-1 rounded-full text-[10px] font-black uppercase tracking-widest ${isTraining ? 'bg-amber-500/10 text-amber-400 animate-pulse' : 'bg-emerald-500/10 text-emerald-400'}`}>
            {isTraining ? 'Training in progress' : 'System Ready'}
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="p-4 rounded-2xl bg-white/5 border border-white/5 space-y-2">
            <div className="flex items-center gap-2 text-slate-500">
              <Database size={14} />
              <span className="text-[10px] font-bold uppercase tracking-widest">Enhanced Corpus</span>
            </div>
            <p className="text-xl font-bold">58,400 <span className="text-xs text-slate-500">samples</span></p>
          </div>
          <div className="p-4 rounded-2xl bg-white/5 border border-white/5 space-y-2">
            <div className="flex items-center gap-2 text-slate-500">
              <Clock size={14} />
              <span className="text-[10px] font-bold uppercase tracking-widest">Est. Duration</span>
            </div>
            <p className="text-xl font-bold">14m 20s</p>
          </div>
          <div className="p-4 rounded-2xl bg-white/5 border border-white/5 space-y-2">
            <div className="flex items-center gap-2 text-slate-500">
              <Activity size={14} />
              <span className="text-[10px] font-bold uppercase tracking-widest">Target F1</span>
            </div>
            <p className="text-xl font-bold">&gt; 0.86</p>
          </div>
        </div>

        <div className="space-y-4 pt-4">
          <div className="flex justify-between text-sm font-bold">
            <span className="text-slate-400">Optimization Progress</span>
            <span className="text-emerald-400">{Math.round(progress)}%</span>
          </div>
          <div className="w-full h-4 bg-black/40 rounded-full overflow-hidden p-1 border border-white/5">
            <div 
              className="h-full bg-gradient-to-r from-emerald-500 to-blue-500 rounded-full transition-all duration-500 shadow-[0_0_15px_rgba(16,185,129,0.3)]"
              style={{ width: `${progress}%` }}
            ></div>
          </div>
        </div>

        <div className="flex gap-4">
          {!isTraining ? (
            <button 
              onClick={startTraining}
              className="flex-1 py-4 bg-purple-600 text-white font-bold rounded-xl hover:bg-purple-500 transition-all shadow-lg shadow-purple-600/20 flex items-center justify-center gap-2"
            >
              <PlayCircle size={20} />
              Begin Fine-Tuning
            </button>
          ) : (
            <button 
              className="flex-1 py-4 bg-rose-600/20 text-rose-400 font-bold rounded-xl border border-rose-500/30 hover:bg-rose-600/30 transition-all flex items-center justify-center gap-2"
              onClick={() => setIsTraining(false)}
            >
              <StopCircle size={20} />
              Abort Job
            </button>
          )}
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="glass p-6 rounded-2xl space-y-4">
          <h4 className="font-bold">Recent Job Logs</h4>
          <div className="space-y-3">
            {[1, 2, 3].map((i) => (
              <div key={i} className="flex items-center justify-between text-xs p-3 rounded-lg bg-black/20">
                <div className="flex items-center gap-3">
                  <CheckCircle2 size={14} className="text-emerald-500" />
                  <span className="text-slate-300">Model Patch v2.1.{i} exported (ONNX)</span>
                </div>
                <span className="text-slate-500">{i*3}h ago</span>
              </div>
            ))}
          </div>
        </div>
        <div className="glass p-6 rounded-2xl space-y-4">
          <h4 className="font-bold">Optimization Log</h4>
          <div className="space-y-2">
            <p className="text-[10px] text-slate-500 font-bold uppercase tracking-widest">Dendrite Restructuring</p>
            <p className="text-xs text-slate-400 leading-relaxed italic">
              "Epoch 4: Model restructured with new dendrites in Encoder L2/L3 to capture phonetic variations detected in 1M conversational set."
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TrainingDashboard;
