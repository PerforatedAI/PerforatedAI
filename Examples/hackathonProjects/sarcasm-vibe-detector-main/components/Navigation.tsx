
import React, { useState } from 'react';
import { Zap, LayoutGrid, Cpu, Github, Menu, X, Rocket, Home } from 'lucide-react';

interface NavigationProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
}

const Navigation: React.FC<NavigationProps> = ({ activeTab, setActiveTab }) => {
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

  const tabs = [
    { id: 'home', label: 'Home', icon: Home },
    { id: 'demo', label: 'Demo', icon: Zap },
    { id: 'batch', label: 'Batch', icon: LayoutGrid },
    { id: 'models', label: 'Whitepaper', icon: Cpu },
    { id: 'training', label: 'Studio', icon: Rocket },
  ];

  const handleTabClick = (id: string) => {
    setActiveTab(id);
    setIsMobileMenuOpen(false);
  };

  return (
    <nav className="w-full relative z-50 mb-16">
      <div className="flex items-center justify-between p-3 glass rounded-[1.5rem] border border-white/10 shadow-xl">
        {/* Logo Section */}
        <div className="flex items-center gap-4 px-3">
          <div className="w-11 h-11 rounded-2xl bg-gradient-to-br from-emerald-400 to-blue-600 flex items-center justify-center shadow-lg shadow-emerald-500/20 group">
            <Zap size={22} className="text-slate-900 fill-slate-900 group-hover:scale-110 transition-transform" />
          </div>
          <div className="hidden sm:block">
            <h1 className="text-lg font-black tracking-tighter text-white leading-none">VIBE-NET</h1>
            <p className="text-emerald-500 text-[10px] font-black uppercase tracking-widest mt-1">Dendritic AI</p>
          </div>
        </div>

        {/* Desktop Tabs */}
        <div className="hidden md:flex items-center gap-1 bg-black/40 p-1.5 rounded-2xl border border-white/5">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => handleTabClick(tab.id)}
              className={`flex items-center gap-2 px-5 py-2.5 rounded-xl text-sm font-bold transition-all duration-300 ${
                activeTab === tab.id
                  ? 'bg-emerald-500 text-slate-950 shadow-lg glow-emerald'
                  : 'text-slate-400 hover:text-white hover:bg-white/5'
              }`}
            >
              <tab.icon size={16} />
              {tab.label}
            </button>
          ))}
        </div>

        {/* Desktop Right Actions */}
        <div className="hidden md:flex items-center px-3">
          <a 
            href="#" 
            target="_blank" 
            rel="noopener noreferrer" 
            className="p-2.5 text-slate-400 hover:text-white transition-all bg-white/5 rounded-2xl border border-white/5 hover:border-white/20"
            aria-label="GitHub Repository"
          >
            <Github size={20} />
          </a>
        </div>

        {/* Mobile Menu Toggle */}
        <button 
          className="md:hidden p-3 text-slate-300 hover:text-white transition-colors"
          onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
          aria-label="Toggle Menu"
        >
          {isMobileMenuOpen ? <X size={24} /> : <Menu size={24} />}
        </button>
      </div>

      {/* Mobile Menu Overlay */}
      {isMobileMenuOpen && (
        <div className="absolute top-24 left-0 w-full md:hidden animate-in fade-in slide-in-from-top-6 duration-300">
          <div className="glass border border-white/10 rounded-3xl shadow-2xl p-6 space-y-3">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => handleTabClick(tab.id)}
                className={`w-full flex items-center gap-4 px-6 py-5 rounded-2xl text-xl font-black transition-all ${
                  activeTab === tab.id
                    ? 'bg-emerald-500 text-slate-950 shadow-lg'
                    : 'text-slate-300 hover:bg-white/5'
                }`}
              >
                <tab.icon size={22} />
                {tab.label}
              </button>
            ))}
          </div>
        </div>
      )}
    </nav>
  );
};

export default Navigation;
