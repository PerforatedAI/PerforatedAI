export function splitIntoSentences(text: string): string[] {
  if (!text || typeof text !== "string") return []
  // basic rule: split on .?! followed by space and a capital letter OR end of string
  const regex = /(?<=\w[.!?])\s+(?=[A-Z0-9""'`(])/g
  const raw = text
    .split(regex)
    .map((s) => s.trim())
    .filter(Boolean)
  // if regex did nothing, fallback to splitting by newline
  if (raw.length <= 1 && text.includes("\n")) {
    return text
      .split("\n")
      .map((s) => s.trim())
      .filter(Boolean)
  }
  return raw
}
