'use client';

import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { CheckCircle2, Circle } from 'lucide-react';

interface ModelComparisonProps {
  selectedModels: string[];
  onModelsChange: (models: string[]) => void;
}

const MODELS = [
  {
    id: 'standard-vit',
    name: 'Standard ViT',
    accuracy: 87.3,
    params: '86M',
    flops: '17.6G',
    speed: '45ms',
    status: 'Trained',
  },
  {
    id: 'dendritic-vit-v1',
    name: 'Dendritic ViT v1',
    accuracy: 88.9,
    params: '45M',
    flops: '8.2G',
    speed: '28ms',
    status: 'Trained',
  },
  {
    id: 'dendritic-vit-v2',
    name: 'Dendritic ViT v2',
    accuracy: 89.7,
    params: '58M',
    flops: '11.5G',
    speed: '32ms',
    status: 'Training',
  },
  {
    id: 'dendritic-vit-optimized',
    name: 'Dendritic ViT (Optimized)',
    accuracy: 90.2,
    params: '52M',
    flops: '9.8G',
    speed: '30ms',
    status: 'Validating',
  },
];

export function ModelComparison({ selectedModels, onModelsChange }: ModelComparisonProps) {
  const toggleModel = (modelId: string) => {
    if (selectedModels.includes(modelId)) {
      onModelsChange(selectedModels.filter((id) => id !== modelId));
    } else {
      onModelsChange([...selectedModels, modelId]);
    }
  };

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {MODELS.map((model) => {
          const isSelected = selectedModels.includes(model.id);
          return (
            <Card
              key={model.id}
              className={`p-4 cursor-pointer transition-all border-2 ${
                isSelected
                  ? 'border-primary bg-primary/5'
                  : 'border-border hover:border-primary/50'
              }`}
              onClick={() => toggleModel(model.id)}
            >
              <div className="space-y-3">
                <div className="flex items-start justify-between">
                  <h3 className="font-semibold text-foreground">{model.name}</h3>
                  {isSelected ? (
                    <CheckCircle2 className="w-5 h-5 text-primary flex-shrink-0" />
                  ) : (
                    <Circle className="w-5 h-5 text-muted-foreground flex-shrink-0" />
                  )}
                </div>
                <Badge
                  variant={
                    model.status === 'Trained'
                      ? 'default'
                      : model.status === 'Training'
                        ? 'secondary'
                        : 'outline'
                  }
                  className="w-fit"
                >
                  {model.status}
                </Badge>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Accuracy</span>
                    <span className="font-semibold text-accent">{model.accuracy}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Parameters</span>
                    <span className="font-semibold">{model.params}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">FLOPs</span>
                    <span className="font-semibold">{model.flops}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Latency</span>
                    <span className="font-semibold">{model.speed}</span>
                  </div>
                </div>
              </div>
            </Card>
          );
        })}
      </div>
    </div>
  );
}
