"use client"

import { useEffect } from "react"

type ShortcutActions = {
  toggleStream?: () => void
  stop?: () => void
  nextHighlight?: () => void
}

export default function useKeyboardShortcuts(actions: ShortcutActions) {
  useEffect(() => {
    function handler(e: KeyboardEvent) {
      const tag = (e.target as HTMLElement).tagName
      if (tag === "INPUT" || tag === "TEXTAREA" || (e.target as HTMLElement).isContentEditable) return

      if (e.key === " " || e.code === "Space") {
        e.preventDefault()
        if (actions.toggleStream) actions.toggleStream()
      } else if (e.key === "s" || e.key === "S") {
        e.preventDefault()
        if (actions.stop) actions.stop()
      } else if (e.key === "j" || e.key === "J") {
        e.preventDefault()
        if (actions.nextHighlight) actions.nextHighlight()
      }
    }
    window.addEventListener("keydown", handler)
    return () => window.removeEventListener("keydown", handler)
  }, [actions])
}
