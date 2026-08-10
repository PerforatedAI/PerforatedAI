export function LoadingSkeleton({ lines = 4 }: { lines?: number }) {
  return (
    <div className="animate-pulse space-y-2" role="status" aria-label="Loading">
      {Array.from({ length: lines }).map((_, i) => (
        <div key={i} className="h-3 rounded bg-slate-200 w-full" style={{ width: `${100 - i * 5}%` }} />
      ))}
    </div>
  )
}
