/**
 * What to offer after a deal, so the second question costs a tap.
 *
 * WHY THIS EXISTS. arXiv 2410.10644 — survey N=121, then N=10 — found that
 * users of prompt-driven interfaces regress to short, command-like prompts, and
 * that the fix is a toolbar of those commands *beside* the box rather than a
 * better box. This page had the box and a rail of starter chips that were the
 * same three strings whatever came back, so a second turn meant typing a second
 * sentence.
 *
 * A REFINEMENT COMPOSES; IT DOES NOT REPLACE. This is the whole design and it
 * was measured before anything was built. Asked "a name that means light" and
 * then handed the bare string "two syllables", readAsk parses themes=[] and the
 * pool comes back with 4 of the original 40 — the visitor asked to shorten
 * their results and got a different subject. Sent as "a name that means light,
 * two syllables" it parses themes=["light"] syllables=[2] and keeps 14 of 40.
 * So every prompt here is the visitor's own sentence with a clause added.
 *
 * EVERY AXIS MOVES A DIFFERENT FIELD OF `Prefs`, which is what guarantees a
 * chip changes the answer without having to run retrieval to find out: length
 * sets `syllables`, letter sets `letters`, feel adds a `theme`, source sets
 * `sources`. Two chips can never be the same question wearing two labels.
 *
 * THE WORDS ARE CHOSEN BY WHAT THE PARSER CAN HEAR, not by what reads best.
 * Measured against readAsk: gentle -> compassion, calm -> peace, strong ->
 * strength, wise -> wisdom, pure -> purity, joyful -> joy, royal -> royal.
 * "soft" and "protective" reach no theme at all and are therefore not offered,
 * however natural they sound — a chip that changes nothing is worse than no
 * chip, because it teaches the visitor the controls do not work.
 */
import type { NaamRow } from '@/types/naam'
import { NAAM_COPY } from './copy'

const C = NAAM_COPY.app.refine

export interface Refinement {
  id: string
  /** What the visitor reads. Short and command-like, on purpose. */
  label: string
  /** The visitor's own ask with one clause added. */
  prompt: string
}

/** Words the parser demonstrably resolves to a theme, with the label to show. */
const FEELS = [
  { theme: 'peace', word: 'calm', label: C.calmer },
  { theme: 'strength', word: 'strong', label: C.stronger },
  { theme: 'compassion', word: 'gentle', label: C.gentler },
  { theme: 'wisdom', word: 'wise', label: C.wiser },
] as const

const LETTER_LABEL: Record<string, string> = { B: C.moreB, S: C.moreS, V: C.moreV }

/** The middle of the deal, which is what "shorter" and "longer" are relative to. */
function medianSyllables(rows: readonly NaamRow[]): number {
  const counts = rows.map((r) => r.syllableSplit?.length ?? 2).sort((a, b) => a - b)
  return counts.length === 0 ? 2 : counts[Math.floor(counts.length / 2)]
}

/**
 * Up to four, and they are ordered by how much of the deal they would change:
 * length first because it is the axis people reach for out loud, then the
 * letter — which on this page is never a neutral choice, since B is Bishal and
 * S is Sneha — then feel, then source.
 *
 * `ask` is the sentence the visitor actually sent. If it is empty there is
 * nothing to compose onto and nothing is offered; a chip that silently starts a
 * new subject is the failure this module exists to prevent.
 */
export function refinements(
  dealt: readonly NaamRow[],
  prefs: { syllables: number[]; letters: string[]; themes: string[]; sources: string[] },
  ask: string,
): Refinement[] {
  const base = ask.trim()
  if (!base || dealt.length === 0) return []

  const out: Refinement[] = []
  const add = (id: string, label: string, clause: string) =>
    out.push({ id, label, prompt: `${base}, ${clause}` })

  // ── length ──────────────────────────────────────────────────────────────
  if (prefs.syllables.length === 0) {
    const mid = medianSyllables(dealt)
    if (mid >= 3) add('shorter', C.shorter, 'two syllables')
    else add('longer', C.longer, 'three syllables')
  }

  // ── letter ──────────────────────────────────────────────────────────────
  // The one the deal has least of, so the chip shows something new. Skipped
  // when the visitor already pinned a letter — they asked for that one.
  if (prefs.letters.length === 0) {
    const seen: Record<string, number> = { B: 0, S: 0, V: 0 }
    for (const row of dealt) seen[row.letter] = (seen[row.letter] ?? 0) + 1
    const scarcest = (['B', 'S', 'V'] as const).reduce((a, b) => (seen[a] <= seen[b] ? a : b))
    add(`letter-${scarcest}`, LETTER_LABEL[scarcest], `beginning with ${scarcest}`)
  }

  // ── feel ────────────────────────────────────────────────────────────────
  // A direction they have not already asked for.
  const feel = FEELS.find((f) => !prefs.themes.includes(f.theme))
  if (feel) add(`feel-${feel.theme}`, feel.label, `${feel.word} ones`)

  // ── source ──────────────────────────────────────────────────────────────
  if (prefs.sources.length === 0) add('vedas', C.vedas, 'from the Vedas')

  return out.slice(0, 4)
}
