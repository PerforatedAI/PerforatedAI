'use client';

import { Button } from '@/components/ui/button';
import { BarChart3, Brain, TrendingUp, Cpu } from 'lucide-react';

interface SidebarProps {
  activeTab: 'overview' | 'training' | 'dendritic';
  setActiveTab: (tab: 'overview' | 'training' | 'dendritic') => void;
}

export function Sidebar({ activeTab, setActiveTab }: SidebarProps) {
  const menuItems = [
    { id: 'overview', label: 'Overview', icon: BarChart3 },
    { id: 'training', label: 'Training', icon: TrendingUp },
    { id: 'dendritic', label: 'Dendritic', icon: Brain },
  ];

  return (
    <aside className="w-64 border-r border-border bg-sidebar flex flex-col">
      <div className="p-6 border-b border-border">
        <div className="flex items-center gap-2 mb-2">
          <Cpu className="w-6 h-6 text-primary" />
          <span className="text-lg font-bold text-sidebar-foreground">DVT Labs</span>
        </div>
        <p className="text-xs text-sidebar-foreground/60">Model Analysis Suite</p>
      </div>

      <nav className="flex-1 p-4 space-y-2">
        {menuItems.map((item) => {
          const Icon = item.icon;
          const isActive = activeTab === (item.id as typeof activeTab);
          
          return (
            <button
              key={item.id}
              onClick={() => setActiveTab(item.id as typeof activeTab)}
              className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg transition-colors ${
                isActive
                  ? 'bg-sidebar-primary text-sidebar-primary-foreground'
                  : 'text-sidebar-foreground hover:bg-sidebar-accent/50'
              }`}
            >
              <Icon className="w-5 h-5" />
              <span className="font-medium">{item.label}</span>
            </button>
          );
        })}
      </nav>

      <div className="p-4 border-t border-border space-y-2">
        <div className="px-4 py-2">
          <p className="text-xs font-semibold text-sidebar-foreground/60 uppercase mb-2">Status</p>
          <div className="space-y-1 text-xs text-sidebar-foreground/80">
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-green-500" />
              <span>Models: 4</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-blue-500" />
              <span>Training Active</span>
            </div>
          </div>
        </div>
      </div>
    </aside>
  );
}
