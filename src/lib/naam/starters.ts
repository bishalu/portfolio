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
  /**
   * EVERY CHIP CARRIES A REAL NAME, and that is the whole change. These read
   * as a survey before — "Short or lyrical?", "Soft or bold?" — which asks a
   * visitor to name a taste they have not formed yet. NN/g's fix for the
   * articulation barrier is to show the CONSEQUENCE instead: a chip carrying a
   * name shows what that direction produces before anyone has to choose it.
   *
   * Every specimen below was checked against public/naam/names-core.json. Two
   * drafts failed that check and were replaced: Soham is a family seed and is
   * NOT in the document, and Stambha's gloss is a citation fragment rather
   * than a meaning. A chip promising a name the page cannot deal is worse than
   * a vague chip.
   */
  {
    id: 'short',
    label: 'Short, like Bisa',
    dimension: 'shape',
    prompt: 'Short names — two syllables, easy to call across a room. Something like Bisa.',
  },
  {
    id: 'soft',
    label: 'Soft, like Samaya',
    dimension: 'sound',
    prompt: 'Soft names, gentle to say. Something like Samaya.',
  },
  {
    id: 'strong',
    label: 'Strong, like Virya',
    dimension: 'sound',
    prompt: 'Names with some strength in them. Something like Virya.',
  },
  {
    id: 'water',
    label: 'Water, like Samudda',
    dimension: 'meaning',
    prompt: 'Names that mean water, or the sea. Something like Samudda.',
  },
  {
    id: 'easy-anywhere',
    label: 'Easy anywhere, like Bodha',
    dimension: 'use',
    prompt: 'Names that are easy to say abroad as well as at home. Something like Bodha.',
  },
  {
    id: 'from-the-sutras',
    label: 'From the Sutras, like Shamatha',
    dimension: 'meaning',
    prompt: 'Names out of the Sutras. Something like Shamatha.',
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
