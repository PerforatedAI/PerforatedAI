"use client"
import { createContext, useContext, useState, type ReactNode } from "react"
import { X } from "lucide-react"

type Toast = { id: string; title: string; body?: string; tone?: "info" | "success" | "error" }
type ToastContextType = {
  push: (t: Omit<Toast, "id">) => void
  dismiss: (id: string) => void
}

const ToastContext = createContext<ToastContextType | null>(null)

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([])

  function push(t: Omit<Toast, "id">) {
    const id = Math.random().toString(36).slice(2, 9)
    setToasts((s) => [...s, { ...t, id }])
    setTimeout(() => dismiss(id), 6000)
  }

  function dismiss(id: string) {
    setToasts((s) => s.filter((t) => t.id !== id))
  }

  return (
    <ToastContext.Provider value={{ push, dismiss }}>
      {children}
      <div className="fixed right-4 bottom-6 z-50 flex flex-col gap-3" aria-live="polite" aria-atomic="true">
        {toasts.map((t) => (
          <div
            key={t.id}
            role="status"
            className={`max-w-sm p-3 rounded-md shadow-lg flex items-start gap-3 animate-in slide-in-from-right ${
              t.tone === "error"
                ? "bg-red-50 border border-red-200"
                : t.tone === "success"
                  ? "bg-green-50 border border-green-200"
                  : "bg-white border"
            }`}
          >
            <div className="flex-1">
              <div className="text-sm font-medium">{t.title}</div>
              {t.body && <div className="text-xs text-slate-600 mt-1">{t.body}</div>}
            </div>
            <button
              onClick={() => dismiss(t.id)}
              aria-label="Dismiss notification"
              className="p-1 text-slate-500 hover:text-slate-700 rounded hover:bg-slate-100"
            >
              <X size={16} />
            </button>
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  )
}

export function useToast() {
  const ctx = useContext(ToastContext)
  if (!ctx) throw new Error("useToast must be used inside ToastProvider")
  return ctx
}
