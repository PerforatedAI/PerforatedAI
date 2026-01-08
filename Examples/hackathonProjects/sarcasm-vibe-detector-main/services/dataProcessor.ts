
/**
 * Hinglish Data Augmentation Suite
 * Implements research-grade strategies for low-resource sarcasm data.
 */

const HINGLISH_SYNONYMS: Record<string, string[]> = {
  "accha": ["badhiya", "mast", "sahi", "top"],
  "phone": ["mobile", "fone", "cell"],
  "kaam": ["work", "job", "task", "assignment"],
  "yaar": ["bhai", "bro", "dost"],
  "jaldi": ["fast", "quick", "jldi"],
  "paisa": ["money", "cash", "rokda"]
};

/**
 * Synonym Replacement Augmentation
 */
export function augmentWithSynonyms(text: string): string {
  const words = text.split(' ');
  const augmented = words.map(word => {
    const lower = word.toLowerCase();
    if (HINGLISH_SYNONYMS[lower] && Math.random() > 0.7) {
      const options = HINGLISH_SYNONYMS[lower];
      return options[Math.floor(Math.random() * options.length)];
    }
    return word;
  });
  return augmented.join(' ');
}

/**
 * Synthetic Sarcasm Injection
 * Adds sarcastic cues to neutral/positive statements about negative situations.
 */
export function injectSyntheticSarcasm(text: string): string {
  const cues = ["Waah!", "Bilkul sahi!", "Mast idea hai!", "Genius logic.", "/s"];
  const randomCue = cues[Math.floor(Math.random() * cues.length)];
  return `${randomCue} ${text}`;
}

/**
 * Simulates a Teacher-Student distillation labeling run
 */
export async function simulateTeacherLabeling(texts: string[]): Promise<Array<{ text: string; pseudoLabel: number }>> {
  // Simulating 500ms API delay for teacher model (e.g. BERT-base)
  await new Promise(r => setTimeout(r, 500));
  return texts.map(t => ({
    text: t,
    pseudoLabel: Math.random() > 0.5 ? 1 : 0
  }));
}
