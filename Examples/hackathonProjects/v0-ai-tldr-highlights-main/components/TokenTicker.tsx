"use client"

export default function TokenTicker({ tokens, maxBubbles = 12 }: { tokens: string[]; maxBubbles?: number }) {
  const latest = tokens.slice(-maxBubbles)
  return (
    <div className="flex items-center gap-2 flex-wrap min-h-[2rem]">
      {latest.map((t, i) => (
        <div
          key={`${t}-${i}`}
          className="px-3 py-1 bg-gradient-to-br from-indigo-50 to-indigo-100 rounded-full text-xs text-indigo-700 shadow-sm animate-in fade-in slide-in-from-bottom-2 duration-200"
          title={t}
        >
          {t.length > 18 ? t.slice(0, 15) + "…" : t}
        </div>
      ))}
    </div>
  )
}
