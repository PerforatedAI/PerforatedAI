'use client';

import { Card } from '@/components/ui/card';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, Legend } from 'recharts';
import { useState } from 'react';

const BRANCH_ACTIVATION_DATA = Array.from({ length: 50 }, (_, i) => ({
  id: i + 1,
  activation: Math.random() * 100,
  sparsity: Math.random() * 100,
  importance: Math.random() * 10,
}));

const LAYER_ANALYSIS = [
  { layer: 'Layer 1', active: 85, inactive: 15, avgActivation: 0.72 },
  { layer: 'Layer 2', active: 78, inactive: 22, avgActivation: 0.65 },
  { layer: 'Layer 3', active: 72, inactive: 28, avgActivation: 0.58 },
  { layer: 'Layer 4', active: 68, inactive: 32, avgActivation: 0.54 },
  { layer: 'Layer 5', active: 82, inactive: 18, avgActivation: 0.68 },
  { layer: 'Layer 6', active: 75, inactive: 25, avgActivation: 0.61 },
];

export function DendriticVisualization() {
  const [selectedLayer, setSelectedLayer] = useState<string>('Layer 1');

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        {[
          { label: 'Active Branches', value: '2,847', change: '+12%' },
          { label: 'Avg Activation', value: '0.64', change: '+3.2%' },
          { label: 'Sparsity Score', value: '73.2%', change: '+8.1%' },
          { label: 'Efficiency Gain', value: '2.1x', change: '+0.3x' },
        ].map((stat, i) => (
          <Card key={i} className="p-4">
            <p className="text-sm text-muted-foreground mb-1">{stat.label}</p>
            <p className="text-2xl font-bold text-foreground mb-1">{stat.value}</p>
            <p className="text-xs text-accent">{stat.change} vs Standard ViT</p>
          </Card>
        ))}
      </div>

      <Card className="p-6">
        <h3 className="text-lg font-semibold mb-4">Branch Activation Scatter</h3>
        <ResponsiveContainer width="100%" height={300}>
          <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
            <XAxis dataKey="sparsity" stroke="var(--color-muted-foreground)" name="Sparsity %" />
            <YAxis dataKey="activation" stroke="var(--color-muted-foreground)" name="Activation %" />
            <Tooltip
              contentStyle={{
                backgroundColor: 'var(--color-card)',
                border: '1px solid var(--color-border)',
              }}
              cursor={{ fill: 'transparent' }}
            />
            <Scatter
              name="Branches"
              data={BRANCH_ACTIVATION_DATA}
              fill="var(--color-primary)"
              fillOpacity={0.6}
            />
          </ScatterChart>
        </ResponsiveContainer>
      </Card>

      <Card className="p-6">
        <h3 className="text-lg font-semibold mb-4">Layer-wise Analysis</h3>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={LAYER_ANALYSIS}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
            <XAxis dataKey="layer" stroke="var(--color-muted-foreground)" />
            <YAxis stroke="var(--color-muted-foreground)" />
            <Tooltip
              contentStyle={{
                backgroundColor: 'var(--color-card)',
                border: '1px solid var(--color-border)',
              }}
            />
            <Legend />
            <Bar dataKey="active" stackId="a" fill="var(--color-primary)" name="Active %" />
            <Bar dataKey="inactive" stackId="a" fill="var(--color-border)" name="Inactive %" />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card className="p-6">
          <h3 className="text-lg font-semibold mb-4">Average Activation by Layer</h3>
          <div className="space-y-3">
            {LAYER_ANALYSIS.map((layer) => (
              <div key={layer.layer} className="space-y-1">
                <div className="flex justify-between items-center text-sm">
                  <span className="text-foreground">{layer.layer}</span>
                  <span className="font-semibold text-accent">{(layer.avgActivation * 100).toFixed(1)}%</span>
                </div>
                <div className="w-full bg-card rounded-full h-2 overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-primary to-accent"
                    style={{ width: `${layer.avgActivation * 100}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </Card>

        <Card className="p-6">
          <h3 className="text-lg font-semibold mb-4">Dendritic Insights</h3>
          <div className="space-y-4 text-sm">
            <div className="p-3 bg-primary/10 rounded-lg border border-primary/20">
              <p className="font-semibold text-primary mb-1">Peak Efficiency</p>
              <p className="text-foreground">Layer 1 shows the highest branch density with 85% active connections, enabling efficient early feature extraction.</p>
            </div>
            <div className="p-3 bg-accent/10 rounded-lg border border-accent/20">
              <p className="font-semibold text-accent mb-1">Sparsity Pattern</p>
              <p className="text-foreground">Middle layers (3-4) exhibit optimal sparsity trade-offs, reducing 28-32% of connections while maintaining performance.</p>
            </div>
            <div className="p-3 bg-chart-4/10 rounded-lg border border-chart-4/20">
              <p className="font-semibold text-chart-4 mb-1">Convergence Speed</p>
              <p className="text-foreground">Dendritic architecture converges 2.3x faster than standard ViT during initial training phases.</p>
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
}
