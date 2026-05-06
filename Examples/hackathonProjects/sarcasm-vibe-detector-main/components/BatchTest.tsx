
import React, { useState } from 'react';
// Added RefreshCcw to imports
import { LayoutGrid, Play, Trash2, Plus, Download, FileText, CheckCircle2, RefreshCcw } from 'lucide-react';
import { apiService } from '../services/apiService';
import { BatchPredictionResult } from '../types';

const BatchTest: React.FC = () => {
  const [texts, setTexts] = useState<string[]>(["Wah kya timing hai!", "Nice job... not.", "Zabardast kaam kiya yaar!"]);
  const [newText, setNewText] = useState("");
  const [results, setResults] = useState<BatchPredictionResult | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);

  const addText = () => {
    if (newText.trim()) {
      setTexts([...texts, newText.trim()]);
      setNewText("");
    }
  };

  const removeText = (index: number) => {
    setTexts(texts.filter((_, i) => i !== index));
    setResults(null);
  };

  const runBatch = async () => {
    if (texts.length === 0) return;
    setIsProcessing(true);
    try {
      const res = await apiService.predictBatch(texts);
      setResults(res);
    } catch (err) {
      console.error(err);
    } finally {
      setIsProcessing(false);
    }
  };

  const downloadCSV = () => {
    if (!results) return;
    const csvRows = [
      ["Text", "Prediction", "Confidence"],
      ...results.results.map(r => [r.text, r.label, (r.confidence * 100).toFixed(1) + "%"])
    ];
    const csvContent = "data:text/csv;charset=utf-8," + csvRows.map(e => e.join(",")).join("\n");
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", "vibe_predictions.csv");
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="space-y-8 animate-in fade-in slide-in-from-bottom-6 duration-700">
      <div className="glass p-10 rounded-[3rem] space-y-10 border-white/10 shadow-2xl">
        <div className="flex flex-col md:flex-row items-center justify-between gap-6">
          <div className="space-y-1">
            <h3 className="text-3xl font-black tracking-tight text-white flex items-center gap-3">
              <LayoutGrid className="text-blue-400" size={32} />
              Batch Processor
            </h3>
            <p className="text-slate-500 font-bold text-xs uppercase tracking-widest">Efficiency Benchmark Tool</p>
          </div>
          <div className="flex items-center gap-2 px-6 py-3 rounded-2xl bg-black/40 border border-white/5">
            <span className="text-xs font-black text-slate-500 uppercase">Queue Depth</span>
            <span className="text-lg font-black text-emerald-400">{texts.length}</span>
          </div>
        </div>

        <div className="flex gap-3">
          <div className="relative flex-1 group">
            <input
              type="text"
              value={newText}
              onChange={(e) => setNewText(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && addText()}
              placeholder="Add new Hinglish text to batch pipeline..."
              className="w-full bg-black/30 rounded-[1.5rem] px-6 py-5 border border-white/5 focus:ring-2 focus:ring-emerald-500/30 focus:border-emerald-500/50 outline-none transition-all placeholder:text-slate-700 font-medium"
            />
            <div className="absolute right-6 top-1/2 -translate-y-1/2 flex items-center gap-2 opacity-30 group-focus-within:opacity-100 transition-opacity">
              <span className="text-[10px] font-black uppercase text-slate-500">Press Enter</span>
            </div>
          </div>
          <button 
            onClick={addText} 
            className="px-6 rounded-2xl bg-white/5 border border-white/5 hover:border-emerald-500/50 hover:bg-emerald-500/10 text-emerald-400 transition-all active:scale-95"
          >
            <Plus size={24} />
          </button>
        </div>

        <div className="space-y-3 max-h-[400px] overflow-y-auto pr-3 custom-scrollbar">
          {texts.map((text, idx) => (
            <div key={idx} className="group flex items-center justify-between p-5 rounded-2xl bg-white/5 border border-white/5 hover:border-white/20 transition-all">
              <div className="flex items-center gap-4">
                <div className="w-1.5 h-1.5 rounded-full bg-emerald-500/50 group-hover:bg-emerald-500"></div>
                <span className="text-sm font-bold text-slate-300">{text}</span>
              </div>
              <button onClick={() => removeText(idx)} className="opacity-0 group-hover:opacity-100 p-2 text-slate-500 hover:text-rose-400 transition-all hover:bg-rose-500/10 rounded-lg">
                <Trash2 size={18} />
              </button>
            </div>
          ))}
          {texts.length === 0 && (
            <div className="py-20 text-center space-y-4">
              <div className="inline-flex p-5 rounded-3xl bg-white/5 border border-dashed border-white/10 text-slate-600">
                <FileText size={40} />
              </div>
              <p className="text-slate-500 font-bold">Your batch queue is empty.</p>
            </div>
          )}
        </div>

        <div className="flex flex-col sm:flex-row gap-4 pt-4 border-t border-white/5">
          <button
            onClick={runBatch}
            disabled={texts.length === 0 || isProcessing}
            className="flex-[2] py-4 bg-emerald-500 text-slate-950 font-black rounded-2xl hover:bg-emerald-400 transition-all shadow-xl shadow-emerald-500/10 flex items-center justify-center gap-3 disabled:opacity-50 text-lg"
          >
            {isProcessing ? <RefreshCcw size={22} className="animate-spin" /> : <Play size={22} className="fill-current" />}
            Analyze Batch
          </button>
          {results && (
            <button onClick={downloadCSV} className="flex-1 py-4 glass rounded-2xl font-black text-xs uppercase tracking-widest flex items-center justify-center gap-3 hover:bg-white/10 transition-all border border-white/10">
              <Download size={20} />
              Export Results (CSV)
            </button>
          )}
        </div>
      </div>

      {results && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 animate-in fade-in slide-in-from-bottom-8 duration-1000">
          {results.results.map((r, i) => (
            <div key={i} className={`p-8 rounded-[2rem] border transition-all hover:scale-105 duration-300 relative overflow-hidden group ${
              r.label === 'Sarcasm' ? 'bg-pink-500/5 border-pink-500/20' : 
              r.label === 'Positive' ? 'bg-emerald-500/5 border-emerald-500/20' : 
              'bg-slate-500/5 border-slate-500/20'
            }`}>
              <div className="flex justify-between items-start mb-4">
                <div>
                  <p className="text-[10px] font-black uppercase tracking-widest opacity-50 mb-1">Inference Label</p>
                  <span className={`text-sm font-black uppercase tracking-tighter ${
                    r.label === 'Sarcasm' ? 'text-pink-400' : 
                    r.label === 'Positive' ? 'text-emerald-400' : 
                    'text-slate-400'
                  }`}>{r.label}</span>
                </div>
                <div className="text-right">
                  <p className="text-lg font-black leading-none">{(r.confidence * 100).toFixed(0)}%</p>
                  <p className="text-[10px] font-bold opacity-50">Match</p>
                </div>
              </div>
              <p className="text-md font-bold text-slate-200 leading-relaxed mb-6">"{r.text}"</p>
              <div className="absolute bottom-0 right-0 p-4 opacity-5 group-hover:opacity-10 transition-opacity">
                <CheckCircle2 size={64} />
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default BatchTest;
