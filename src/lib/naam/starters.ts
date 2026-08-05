/**
 * Conversation starters — three ways in, for a visitor facing a blank box.
 *
 * ─── WHAT WAS CUT, AND WHY ─────────────────────────────────────────────────
 *
 * The list this came from covered twelve dimensions of choosing a name. Nine of
 * its entries are not answerable HERE, and a starter that cannot be answered is
 * worse than no starter: it invites somebody to ask a question the page will
 * then fumble.
 *
 *   · siblings — this is a first child. There are no sibling names.
 *   · each parent's likes and dislikes — the page holds one shared thread, not
 *     two preference profiles to find the overlap between.
 *   · surname, middle name, full-name flow — never collected, and inventing a
 *     surname to test against would be a worse answer than declining.
 *   · popularity, and timeless-versus-trending — the corpus is Monier-Williams
 *     glosses out of the Vedas and the Sutras. It carries meaning, grammar and
 *     source. It carries no usage frequency at all, so any answer about how
 *     common a name is would be the model guessing.
 *   · nicknames, and hidden-problem checks — no diminutive data, and a
 *     teasing-risk review would need cross-language association data the
 *     document does not carry either.
 *
 * What is left is the eleven the document can actually speak to: sound, shape,
 * register, and meaning.
 *
 * ─── THE PROMPT IS THE VISITOR'S OWN SENTENCE ──────────────────────────────
 *
 * The obvious build gives each starter a hidden retrieval string alongside the
 * visible label. That would put a line in the transcript that is not what was
 * searched for, which is a small lie in a record two people are keeping.
 *
 * So there is no hidden field. Each prompt is one sentence, written the way a
 * visitor would actually say it, and carrying the words the search needs — the
 * meaning starters name their themes outright, because BM25 over the gloss text
 * is what builds the pool the model must choose from. The starters about SOUND
 * and SHAPE deliberately carry few content words: a near-signal-free query
 * lands on the relevance floor and returns a broad spread of the corpus, which
 * is exactly the right pool for a question about how names sound rather than
 * what they mean.
 */

export interface NaamStarter {
  id: string
  /** Chip text. Short enough to sit three-up on a phone. */
  label: string
  /** What is sent — and shown in the thread as the visitor's turn. */
  prompt: string
  /** Used only to spread the three picks across different kinds of question. */
  dimension: 'shape' | 'sound' | 'meaning' | 'use'
}

export const NAAM_STARTERS: readonly NaamStarter[] = [
  {
    id: 'discover-style',
    label: 'Show me the range',
    dimension: 'shape',
    prompt: 'Show me a few names that are nothing like each other, so I can see the range.',
  },
  {
    id: 'short-or-lyrical',
    label: 'Short or lyrical?',
    dimension: 'shape',
    prompt: 'Put a few short, plain names next to some longer, more lyrical ones.',
  },
  {
    id: 'simple-or-elaborate',
    label: 'Simple or elaborate?',
    dimension: 'shape',
    prompt: 'I want to see simple, understated names beside more elaborate ones.',
  },
  {
    id: 'familiar-or-unexpected',
    label: 'Familiar or unexpected?',
    dimension: 'shape',
    prompt: 'Something familiar, and something unexpected that would still work every day.',
  },
  {
    id: 'soft-or-bold',
    label: 'Soft or bold?',
    dimension: 'sound',
    prompt: 'Soft, flowing sounds on one side and crisp, strong ones on the other.',
  },
  {
    id: 'favorite-sounds',
    label: 'Find my sounds',
    dimension: 'sound',
    prompt: 'Give me a spread of very different sounds and endings, and I will tell you which ones I like.',
  },
  {
    id: 'name-rhythm',
    label: 'Names worth saying',
    dimension: 'sound',
    prompt: 'Names with different rhythms — I want to hear which ones are satisfying to say out loud.',
  },
  {
    id: 'meaning-themes',
    label: 'Wisdom, courage, light',
    dimension: 'meaning',
    prompt: 'Names about wisdom, courage, compassion or light — show me what the range looks like.',
  },
  {
    id: 'nature-themes',
    label: 'Something from nature',
    dimension: 'meaning',
    prompt: 'Something out of nature — river, mountain, dawn, rain, the sea, a tree.',
  },
  {
    id: 'cultural-roots',
    label: 'Deep in the source',
    dimension: 'meaning',
    prompt: 'The old names — the ones that sit deepest in the Vedas and the Sutras.',
  },
  {
    id: 'say-and-spell',
    label: 'Easy anywhere',
    dimension: 'use',
    prompt: 'Names that keep their sound and stay easy to spell wherever he ends up.',
  },
]

/**
 * Three starters, each a different kind of question.
 *
 * `used` is the set of ids already asked this session. They stay on screen for
 * the whole conversation now, so without this the row would keep offering a
 * question that has already been answered directly above it — which reads as
 * the page not listening. Once every starter has been used the set is ignored
 * and the pool opens up again, because three chips is better than none.
 *
 * Called after mount rather than during render: the island is `client:load`, so
 * a random choice made while rendering would differ from the server's HTML and
 * hydration would tear.
 */
export function pickStarters(count = 3, used: ReadonlySet<string> = new Set()): NaamStarter[] {
  const fresh = NAAM_STARTERS.filter((starter) => !used.has(starter.id))
  const pool = fresh.length >= count ? fresh : NAAM_STARTERS
  /**
   * Shuffle the FLAT list and take greedily, skipping a starter whose kind is
   * already represented.
   *
   * The first cut shuffled the DIMENSIONS instead and drew one from each, which
   * looked more principled and was measurably worse: with four kinds and three
   * picks, every kind appears three times in four — so `use`, which has exactly
   * one starter in it, put "Easy anywhere" on screen in four loads out of five
   * while the four shape starters shared a single slot between them.
   *
   * Shuffling the starters weights each kind by how many it actually has, and
   * the skip still guarantees three different kinds of question.
   */
  const shuffled = [...pool].sort(() => Math.random() - 0.5)
  const kinds = new Set<string>()
  const out: NaamStarter[] = []
  for (const starter of shuffled) {
    if (out.length >= count || kinds.has(starter.dimension)) continue
    kinds.add(starter.dimension)
    out.push(starter)
  }

  // Fewer kinds than requested starters would leave a short row; top up from
  // whatever is left rather than showing two chips where three fit.
  for (const starter of shuffled) {
    if (out.length >= count) break
    if (!out.includes(starter)) out.push(starter)
  }
  return out
}
