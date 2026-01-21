import { Button } from '@/components/ui/button';

export function Header() {
  return (
    <header className="border-b border-border bg-card/50 backdrop-blur-sm sticky top-0 z-10">
      <div className="px-8 py-4 flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold">Dendritic Vision Transformer</h2>
          <p className="text-sm text-muted-foreground">Research & Development</p>
        </div>
        <div className="flex items-center gap-4">
          <Button variant="outline" size="sm">
            Export Data
          </Button>
          <Button variant="outline" size="sm">
            Settings
          </Button>
        </div>
      </div>
    </header>
  );
}
