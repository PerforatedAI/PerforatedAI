'use client';

import { Card } from '@/components/ui/card';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, LineChart, Line } from 'recharts';

interface MetricsGridProps {
  selectedModels: string[];
}

const METRICS_DATA = {
  accuracy: [
    { name: 'Standard ViT', value: 87.3 },
    { name: 'Dendritic v1', value: 88.9 },
    { name: 'Dendritic v2', value: 89.7 },
    { name: 'Dendritic Opt', value: 90.2 },
  ],
  efficiency: [
    { model: 'Standard ViT', params: 86, flops: 17.6, latency: 45 },
    { model: 'Dendritic v1', params: 45, flops: 8.2, latency: 28 },
    { model: 'Dendritic v2', params: 58, flops: 11.5, latency: 32 },
    { model: 'Dendritic Opt', params: 52, flops: 9.8, latency: 30 },
  ],
  sparsity: [
    { epoch: 1, sparsity: 12 },
    { epoch: 5, sparsity: 28 },
    { epoch: 10, sparsity: 45 },
    { epoch: 15, sparsity: 58 },
    { epoch: 20, sparsity: 71 },
    { epoch: 25, sparsity: 78 },
  ],
};

export function MetricsGrid({ selectedModels }: MetricsGridProps) {
  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <Card className="p-6">
        <h3 className="text-lg font-semibold mb-4">Accuracy Comparison</h3>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={METRICS_DATA.accuracy}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
            <XAxis dataKey="name" stroke="var(--color-muted-foreground)" />
            <YAxis stroke="var(--color-muted-foreground)" />
            <Tooltip
              contentStyle={{
                backgroundColor: 'var(--color-card)',
                border: '1px solid var(--color-border)',
              }}
            />
            <Bar dataKey="value" fill="var(--color-primary)" radius={[8, 8, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card className="p-6">
        <h3 className="text-lg font-semibold mb-4">Computational Efficiency</h3>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={METRICS_DATA.efficiency}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
            <XAxis dataKey="model" stroke="var(--color-muted-foreground)" width={80} />
            <YAxis stroke="var(--color-muted-foreground)" yAxisId="left" />
            <YAxis yAxisId="right" orientation="right" stroke="var(--color-muted-foreground)" />
            <Tooltip
              contentStyle={{
                backgroundColor: 'var(--color-card)',
                border: '1px solid var(--color-border)',
              }}
            />
            <Legend />
            <Bar yAxisId="left" dataKey="params" fill="var(--color-primary)" name="Params (M)" />
            <Bar yAxisId="right" dataKey="latency" fill="var(--color-accent)" name="Latency (ms)" />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card className="p-6 lg:col-span-2">
        <h3 className="text-lg font-semibold mb-4">Dendritic Sparsity Evolution</h3>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={METRICS_DATA.sparsity}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
            <XAxis dataKey="epoch" stroke="var(--color-muted-foreground)" label={{ value: 'Epoch', position: 'insideBottom', offset: -5 }} />
            <YAxis stroke="var(--color-muted-foreground)" label={{ value: 'Sparsity (%)', angle: -90, position: 'insideLeft' }} />
            <Tooltip
              contentStyle={{
                backgroundColor: 'var(--color-card)',
                border: '1px solid var(--color-border)',
              }}
            />
            <Line
              type="monotone"
              dataKey="sparsity"
              stroke="var(--color-primary)"
              dot={{ fill: 'var(--color-primary)', r: 4 }}
              strokeWidth={2}
              name="Sparsity %"
            />
          </LineChart>
        </ResponsiveContainer>
      </Card>
    </div>
  );
}
