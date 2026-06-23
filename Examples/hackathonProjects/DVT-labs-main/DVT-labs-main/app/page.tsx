'use client';

import { useState } from 'react';
import { Header } from '@/components/header';
import { Sidebar } from '@/components/sidebar';
import { ModelComparison } from '@/components/model-comparison';
import { MetricsGrid } from '@/components/metrics-grid';
import { TrainingCurves } from '@/components/training-curves';
import { DendriticVisualization } from '@/components/dendritic-visualization';

export default function Home() {
  const [selectedModels, setSelectedModels] = useState<string[]>(['standard-vit', 'dendritic-vit-v1']);
  const [activeTab, setActiveTab] = useState<'overview' | 'training' | 'dendritic'>('overview');

  return (
    <div className="flex h-screen bg-background text-foreground">
      <Sidebar activeTab={activeTab} setActiveTab={setActiveTab} />
      <div className="flex-1 flex flex-col overflow-hidden">
        <Header />
        <main className="flex-1 overflow-auto">
          <div className="p-8 space-y-6 max-w-7xl mx-auto">
            {activeTab === 'overview' && (
              <>
                <div>
                  <h1 className="text-3xl font-bold mb-2">Model Comparison Dashboard</h1>
                  <p className="text-muted-foreground">Analyze and compare Vision Transformer architectures</p>
                </div>
                <ModelComparison selectedModels={selectedModels} onModelsChange={setSelectedModels} />
                <MetricsGrid selectedModels={selectedModels} />
              </>
            )}
            
            {activeTab === 'training' && (
              <>
                <div>
                  <h1 className="text-3xl font-bold mb-2">Training Analysis</h1>
                  <p className="text-muted-foreground">Monitor training metrics and convergence patterns</p>
                </div>
                <TrainingCurves selectedModels={selectedModels} />
              </>
            )}
            
            {activeTab === 'dendritic' && (
              <>
                <div>
                  <h1 className="text-3xl font-bold mb-2">Dendritic Analysis</h1>
                  <p className="text-muted-foreground">Visualize branch activations and sparsity patterns</p>
                </div>
                <DendriticVisualization />
              </>
            )}
          </div>
        </main>
      </div>
    </div>
  );
}
