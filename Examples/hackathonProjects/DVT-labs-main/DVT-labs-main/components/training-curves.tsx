'use client';

import { Card } from '@/components/ui/card';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ComposedChart, Bar } from 'recharts';

interface TrainingCurvesProps {
  selectedModels: string[];
}

const generateTrainingData = () => {
  const data = [];
  for (let i = 0; i <= 25; i++) {
    data.push({
      epoch: i,
      standardVit: 60 + i * 1.2 + Math.random() * 2,
      dendriticV1: 62 + i * 1.35 + Math.random() * 1.8,
      dendriticV2: 63 + i * 1.45 + Math.random() * 1.5,
      loss: 2.5 - i * 0.08 + Math.random() * 0.1,
    });
  }
  return data;
};

const generateValidationData = () => {
  const data = [];
  for (let i = 0; i <= 25; i += 5) {
    data.push({
      epoch: i,
      standardVit: 85 + (i / 25) * 2.3,
      dendriticV1: 86.5 + (i / 25) * 2.4,
      dendriticV2: 87 + (i / 25) * 2.7,
    });
  }
  return data;
};

export function TrainingCurves({ selectedModels }: TrainingCurvesProps) {
  const trainingData = generateTrainingData();
  const validationData = generateValidationData();

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <Card className="p-6">
        <h3 className="text-lg font-semibold mb-4">Training Accuracy</h3>
        <ResponsiveContainer width="100%" height={350}>
          <LineChart data={trainingData}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
            <XAxis dataKey="epoch" stroke="var(--color-muted-foreground)" />
            <YAxis stroke="var(--color-muted-foreground)" domain={[50, 90]} />
            <Tooltip
              contentStyle={{
                backgroundColor: 'var(--color-card)',
                border: '1px solid var(--color-border)',
              }}
            />
            <Legend />
            <Line
              type="monotone"
              dataKey="standardVit"
              stroke="var(--color-chart-1)"
              strokeWidth={2}
              dot={false}
              name="Standard ViT"
            />
            <Line
              type="monotone"
              dataKey="dendriticV1"
              stroke="var(--color-primary)"
              strokeWidth={2}
              dot={false}
              name="Dendritic v1"
            />
            <Line
              type="monotone"
              dataKey="dendriticV2"
              stroke="var(--color-accent)"
              strokeWidth={2}
              dot={false}
              name="Dendritic v2"
            />
          </LineChart>
        </ResponsiveContainer>
      </Card>

      <Card className="p-6">
        <h3 className="text-lg font-semibold mb-4">Training Loss</h3>
        <ResponsiveContainer width="100%" height={350}>
          <LineChart data={trainingData}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
            <XAxis dataKey="epoch" stroke="var(--color-muted-foreground)" />
            <YAxis stroke="var(--color-muted-foreground)" />
            <Tooltip
              contentStyle={{
                backgroundColor: 'var(--color-card)',
                border: '1px solid var(--color-border)',
              }}
            />
            <Legend />
            <Line
              type="monotone"
              dataKey="loss"
              stroke="var(--color-chart-4)"
              strokeWidth={2}
              dot={false}
              name="Cross-Entropy Loss"
            />
          </LineChart>
        </ResponsiveContainer>
      </Card>

      <Card className="p-6 lg:col-span-2">
        <h3 className="text-lg font-semibold mb-4">Validation Accuracy Over Time</h3>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={validationData}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
            <XAxis dataKey="epoch" stroke="var(--color-muted-foreground)" />
            <YAxis stroke="var(--color-muted-foreground)" domain={[82, 92]} />
            <Tooltip
              contentStyle={{
                backgroundColor: 'var(--color-card)',
                border: '1px solid var(--color-border)',
              }}
            />
            <Legend />
            <Line
              type="monotone"
              dataKey="standardVit"
              stroke="var(--color-chart-1)"
              strokeWidth={3}
              dot={{ fill: 'var(--color-chart-1)', r: 5 }}
              name="Standard ViT"
            />
            <Line
              type="monotone"
              dataKey="dendriticV1"
              stroke="var(--color-primary)"
              strokeWidth={3}
              dot={{ fill: 'var(--color-primary)', r: 5 }}
              name="Dendritic v1"
            />
            <Line
              type="monotone"
              dataKey="dendriticV2"
              stroke="var(--color-accent)"
              strokeWidth={3}
              dot={{ fill: 'var(--color-accent)', r: 5 }}
              name="Dendritic v2"
            />
          </LineChart>
        </ResponsiveContainer>
      </Card>
    </div>
  );
}
