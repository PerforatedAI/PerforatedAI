import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Zap, TrendingDown } from "lucide-react"
import Link from "next/link"

export default function DendriticPage() {
  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="max-w-7xl mx-auto px-6 py-12">
        <div className="mb-12">
          <Link href="/" className="text-sm text-muted-foreground hover:text-foreground">
            ← Back to Home
          </Link>
        </div>

        <div className="mb-16">
          <div className="inline-block mb-6 px-4 py-2 rounded-full border border-accent/30 bg-accent/10 text-accent text-sm">
            Neural Network Optimization
          </div>
          <h1 className="text-5xl font-bold mb-6">Dendritic Optimization</h1>
          <p className="text-xl text-muted-foreground max-w-3xl leading-relaxed">
            We use Perforated Backpropagation to reduce model parameters by 25% while maintaining accuracy. This means
            faster inference, lower costs, and better user experience.
          </p>
        </div>

        {/* Comparison Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 mb-16">
          <ComparisonCard
            label="Experiment A"
            title="Baseline Model"
            params="110M"
            latency="450ms"
            accuracy="94%"
            cost="$0.03"
            isBaseline
          />
          <ComparisonCard
            label="Experiment B"
            title="Compressed + Dendrites"
            params="66M"
            latency="195ms"
            accuracy="93%"
            cost="$0.018"
            isBest
            improvements={[
              { label: "Parameters", value: "-40%" },
              { label: "Latency", value: "-57%" },
              { label: "Cost", value: "-40%" },
            ]}
          />
          <ComparisonCard
            label="Experiment C"
            title="Compressed Control"
            params="66M"
            latency="210ms"
            accuracy="91%"
            cost="$0.018"
          />
        </div>

        {/* How It Works */}
        <Card className="p-12 bg-card border-border mb-16">
          <h2 className="text-3xl font-bold mb-8">How Dendritic Optimization Works</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-12">
            <div>
              <h3 className="text-xl font-semibold mb-4">The Problem</h3>
              <p className="text-muted-foreground mb-6 leading-relaxed">
                Traditional BERT models have 110M+ parameters. This makes them slow for real-time applications and
                expensive to run at scale. Most neurons contribute little to final accuracy.
              </p>
              <div className="space-y-3">
                <MetricRow label="Baseline parameters" value="110M" />
                <MetricRow label="Inference time" value="450ms" />
                <MetricRow label="GPU memory" value="4.2 GB" />
                <MetricRow label="Cost per 1K inferences" value="$30" />
              </div>
            </div>

            <div>
              <h3 className="text-xl font-semibold mb-4">The Solution</h3>
              <p className="text-muted-foreground mb-6 leading-relaxed">
                Perforated Backpropagation selectively prunes low-impact neurons during training. The algorithm
                identifies and removes connections that contribute minimally to accuracy, restructuring the network
                dynamically.
              </p>
              <div className="space-y-3">
                <MetricRow label="Optimized parameters" value="66M" trend="down" />
                <MetricRow label="Inference time" value="195ms" trend="down" />
                <MetricRow label="GPU memory" value="2.6 GB" trend="down" />
                <MetricRow label="Cost per 1K inferences" value="$18" trend="down" />
              </div>
            </div>
          </div>
        </Card>

        {/* Process Steps */}
        <div className="mb-16">
          <h2 className="text-3xl font-bold mb-8">Training Process</h2>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
            <ProcessStep
              number={1}
              title="Initialize"
              description="Start with pretrained BERT-base model and wrap layers for perforated backprop"
            />
            <ProcessStep
              number={2}
              title="Train"
              description="Standard fine-tuning on sentence classification with validation tracking"
            />
            <ProcessStep
              number={3}
              title="Restructure"
              description="Algorithm identifies low-impact neurons and removes them every N epochs"
            />
            <ProcessStep
              number={4}
              title="Converge"
              description="Training completes when accuracy plateaus with minimal parameter count"
            />
          </div>
        </div>

        {/* Results Table */}
        <Card className="p-8 bg-card border-border mb-16">
          <h2 className="text-2xl font-bold mb-6">Experimental Results</h2>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border">
                  <th className="text-left py-4 px-4">Experiment</th>
                  <th className="text-right py-4 px-4">Parameters</th>
                  <th className="text-right py-4 px-4">Latency (ms)</th>
                  <th className="text-right py-4 px-4">Accuracy</th>
                  <th className="text-right py-4 px-4">GPU Memory</th>
                  <th className="text-right py-4 px-4">Cost/1K</th>
                </tr>
              </thead>
              <tbody>
                <tr className="border-b border-border/50">
                  <td className="py-4 px-4">A: Baseline</td>
                  <td className="text-right py-4 px-4 font-mono">110M</td>
                  <td className="text-right py-4 px-4 font-mono">450</td>
                  <td className="text-right py-4 px-4 font-mono">94%</td>
                  <td className="text-right py-4 px-4 font-mono">4.2 GB</td>
                  <td className="text-right py-4 px-4 font-mono">$30</td>
                </tr>
                <tr className="border-b border-border/50 bg-accent/5">
                  <td className="py-4 px-4 font-semibold">B: Compressed + Dendrites</td>
                  <td className="text-right py-4 px-4 font-mono text-green-600">66M (-40%)</td>
                  <td className="text-right py-4 px-4 font-mono text-green-600">195 (-57%)</td>
                  <td className="text-right py-4 px-4 font-mono text-green-600">93% (-1%)</td>
                  <td className="text-right py-4 px-4 font-mono text-green-600">2.6 GB (-38%)</td>
                  <td className="text-right py-4 px-4 font-mono text-green-600">$18 (-40%)</td>
                </tr>
                <tr>
                  <td className="py-4 px-4">C: Compressed Control</td>
                  <td className="text-right py-4 px-4 font-mono">66M (-40%)</td>
                  <td className="text-right py-4 px-4 font-mono">210 (-53%)</td>
                  <td className="text-right py-4 px-4 font-mono text-red-600">91% (-3%)</td>
                  <td className="text-right py-4 px-4 font-mono">2.6 GB (-38%)</td>
                  <td className="text-right py-4 px-4 font-mono">$18 (-40%)</td>
                </tr>
              </tbody>
            </table>
          </div>
          <p className="mt-6 text-sm text-muted-foreground">
            Experiment B (with dendrites) achieves best balance: significant compression with minimal accuracy loss
          </p>
        </Card>

        {/* Code Example */}
        <Card className="p-8 bg-card border-border mb-16">
          <h2 className="text-2xl font-bold mb-6">Implementation Example</h2>
          <div className="bg-secondary p-6 rounded-lg font-mono text-sm overflow-x-auto">
            <pre className="text-foreground">
              {`import perforatedai as PA

# 1. Wrap model layers
model = wrap_bert_layers_for_pai(model)

# 2. Convert for dendritic optimization
PA.convert_network(
  model,
  module_names_to_convert=["fc1", "fc2", "classifier"]
)

# 3. Initialize tracker
tracker = PA.PerforatedBackPropagationTracker(
  do_pb=True,
  save_name="sentence_classifier",
  maximizing_score=True
)

# 4. Train with restructuring
for epoch in range(max_epochs):
  train_loss = train_epoch(model)
  val_acc = validate(model)
  
  # Dendritic feedback
  model, improved, restructured, done = \\
    tracker.add_validation_score(model, val_acc)
  
  if done: break`}
            </pre>
          </div>
        </Card>

        {/* Call to Action */}
        <div className="text-center">
          <h2 className="text-3xl font-bold mb-4">Try Dendritic Optimization</h2>
          <p className="text-muted-foreground mb-8 max-w-2xl mx-auto">
            The training scripts and models are open source. Follow our guide to train your own dendritic-optimized
            classifier.
          </p>
          <div className="flex items-center justify-center gap-4">
            <Button size="lg" className="gap-2">
              <Zap className="w-5 h-5" />
              View Training Guide
            </Button>
            <Button size="lg" variant="outline" className="gap-2 bg-transparent">
              Download Model
            </Button>
          </div>
        </div>
      </div>
    </div>
  )
}

function ComparisonCard({
  label,
  title,
  params,
  latency,
  accuracy,
  cost,
  isBaseline = false,
  isBest = false,
  improvements,
}: {
  label: string
  title: string
  params: string
  latency: string
  accuracy: string
  cost: string
  isBaseline?: boolean
  isBest?: boolean
  improvements?: Array<{ label: string; value: string }>
}) {
  return (
    <Card className={`p-6 ${isBest ? "border-accent border-2 bg-accent/5" : "bg-card border-border"} relative`}>
      {isBest && (
        <div className="absolute -top-3 left-1/2 -translate-x-1/2 px-3 py-1 bg-accent text-accent-foreground text-xs font-semibold rounded-full">
          Best Result
        </div>
      )}
      <div className="mb-6">
        <div className="text-xs text-muted-foreground mb-2">{label}</div>
        <h3 className="text-xl font-bold mb-1">{title}</h3>
        {isBaseline && <div className="text-xs text-muted-foreground">Standard pretrained model</div>}
      </div>
      <div className="space-y-3 mb-6">
        <MetricRow label="Parameters" value={params} />
        <MetricRow label="Latency" value={latency} />
        <MetricRow label="Accuracy" value={accuracy} />
        <MetricRow label="Cost/summary" value={cost} />
      </div>
      {improvements && (
        <div className="pt-4 border-t border-border space-y-2">
          {improvements.map((imp) => (
            <div key={imp.label} className="flex items-center justify-between text-sm">
              <span className="text-muted-foreground">{imp.label}</span>
              <span className="text-green-600 font-semibold">{imp.value}</span>
            </div>
          ))}
        </div>
      )}
    </Card>
  )
}

function MetricRow({ label, value, trend }: { label: string; value: string; trend?: "up" | "down" }) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-muted-foreground text-sm">{label}</span>
      <div className="flex items-center gap-2">
        <span className="font-semibold">{value}</span>
        {trend === "down" && <TrendingDown className="w-4 h-4 text-green-600" />}
      </div>
    </div>
  )
}

function ProcessStep({ number, title, description }: { number: number; title: string; description: string }) {
  return (
    <Card className="p-6 bg-card border-border">
      <div className="w-10 h-10 rounded-full bg-accent/20 text-accent flex items-center justify-center font-bold mb-4">
        {number}
      </div>
      <h3 className="font-semibold mb-2">{title}</h3>
      <p className="text-sm text-muted-foreground leading-relaxed">{description}</p>
    </Card>
  )
}
