/**
 * The deterministic matcher behind /naam (docs/design/DESIGN.md §4, P9).
 *
 * WHY this file exists at all: the honesty vocabulary has exactly three words,
 * and the one this page leans on is LOCAL — "real algorithm, real data, running
 * in your browser". That badge is only allowed to appear because the selection
 * of names is done *here*, by arithmetic over the document's own fields, and
 * not by a language model. The model never names a name: it is handed a pool
 * this file produced and may only reorder and frame it. That makes a
 * hallucinated name structurally impossible rather than merely unlikely, and it
 * means the page still works with Bedrock switched off.
 *
 * Two consequences shape everything below.
 *
 *   1. Every score contribution produces a REASON STRING, and every reason
 *      string is computed from a field of the row — '2 syllables', 'Theravada',
 *      "'light'", 'attested name', 'easy to say', 'B-form reads cleanly'.
 *      These are shown to the visitor as the justification for a match, so a
 *      reason that is not literally true of the row is a lie on the page.
 *      Nothing here is ever written by a model.
 *
 *   2. Ordering is total and deterministic — score descending, then id
 *      ascending. SSR and the client must agree, and the screenshot harness
 *      must produce the same page twice.
 *
 * Pure, dependency-free, isomorphic: no I/O, no globals, no imports beyond
 * src/types/naam.ts. Safe to import from a browser island and from a Lambda.
 */
import {
  NAAM_SOURCE_LABEL,
  NAAM_THEMES,
  naamThemeBit,
  type NaamLetter,
  type NaamRow,
  type NaamSource,
  type NaamTheme,
} from '@/types/naam'

/* ────────────────────────────────────────────────────────────────────────────
   PREFERENCES
   ──────────────────────────────────────────────────────────────────────────── */

/**
 * What the visitor asked for — the output of the six guided questions, or of
 * parseFreeText() reading a sentence.
 *
 * `letters` and `syllables` are HARD filters: an empty array means "no
 * preference", a non-empty one excludes everything outside it. `themes`,
 * `sources` and `wants` are SOFT weights: they move a row up or down, they
 * never remove it, because a visitor who asks for "calm" should still be shown
 * the best thing in the document rather than an empty page.
 *
 * `easySay` is the diaspora question — "should a non-Nepali find it easy to
 * say?" It is the one soft-sounding answer with a hard edge: it excludes the
 * document's own `!` hard-cluster rows and every V name whose B-form reads
 * badly, and it de-weights three-syllable names.
 */
export interface Prefs {
  /** 1, 2 and/or 3. Empty = no preference. HARD filter. */
  syllables: number[]
  /** B, S and/or V. Empty = no preference. HARD filter. */
  letters: NaamLetter[]
  /** Members of NAAM_THEMES; anything else is ignored. Soft weight. */
  themes: string[]
  /** V (Vedic), C (Classical), T (Theravada). Soft weight. */
  sources: NaamSource[]
  /** 'attested' = someone real bore it · 'meaning' = the meaning matters more · 'both' = no preference. */
  wants: 'attested' | 'meaning' | 'both'
  /** Excludes hard clusters and awkward B-forms; de-weights 3 syllables. */
  easySay: boolean
  /**
   * Optional, defaults false. The document's `f?` badge means "grammatically
   * feminine ending — say it aloud to judge", which is an instruction to the
   * reader, not a verdict. So those rows are never ranked into the top of a
   * result set by accident: they sort strictly below every other match unless
   * the visitor asked for them, and they carry the caveat as a reason string
   * when they do appear. Surfaced, not hidden — and never silently first.
   *
   * There is no guided question for this (the wizard has six and this is not
   * one of them); it exists so a direct request can switch it on.
   */
  allowFeminineEnding?: boolean
}

/**
 * The zero preference. Treat as read-only and spread it —
 * `{ ...EMPTY_PREFS, syllables: [2] }` — rather than mutating it.
 */
export const EMPTY_PREFS: Prefs = {
  syllables: [],
  letters: [],
  themes: [],
  sources: [],
  wants: 'both',
  easySay: false,
  allowFeminineEnding: false,
}

/** Fills a partial answer set out to a complete, sanitised Prefs. */
export function normalizePrefs(partial: Partial<Prefs> = {}): Prefs {
  return {
    syllables: uniq((partial.syllables ?? []).filter((n) => n === 1 || n === 2 || n === 3)),
    letters: uniq((partial.letters ?? []).filter(isLetter)),
    themes: uniq((partial.themes ?? []).filter(isTheme)),
    sources: uniq((partial.sources ?? []).filter(isSource)),
    wants: partial.wants === 'attested' || partial.wants === 'meaning' ? partial.wants : 'both',
    easySay: partial.easySay === true,
    allowFeminineEnding: partial.allowFeminineEnding === true,
  }
}

/** A scored row with the computed justification for its position. */
export interface NaamMatch {
  row: NaamRow
  score: number
  reasons: string[]
}

/* ────────────────────────────────────────────────────────────────────────────
   WEIGHTS

   One table, so the ranking is arguable in one place instead of scattered
   through the arithmetic. The quality prior (gold > evocative > attested >
   tail) is deliberately larger than any single preference bonus: a row whose
   whole meaning is "name of a man" must not outrank a row that means
   "shining", however well it matches the filters.
   ──────────────────────────────────────────────────────────────────────────── */

const W = {
  /** Quality prior, from the document's own two badges. */
  gold: 18, //          attested AND evocative — the document's 201 best rows
  evocative: 12, //     carries a meaning worth saying out loud
  attested: 7, //       someone real was called this
  shortlist: 6, //      the document put it in its own shortlist half
  hasTheme: 5, //       our tagger found a sense in the gloss

  /**
   * Meaning is thin, absent, or unreadable. Applied on top of the prior, never
   * instead of it. The document's `+` badge is generous — "name of a mountain"
   * carries it — so without these the default list is eight rows of
   * attestations and the page looks like a dictionary dump.
   */
  bareAttestation: -14, // the gloss opens "name of …": it names a bearer, not a sense
  thinGloss: -8, //       under eight characters of meaning
  messyGloss: -6, //      dictionary apparatus survived the tidy: "brAhmaRa", "(?)", "prob."
  cardGloss: 3, //        reads as one line on a card
  longGloss: -3, //       does not

  /** Preference weights. */
  theme: 14, //         per requested theme the row carries
  themeMiss: -6, //     themes were asked for and the row carries none
  source: 12, //        the row is in a requested corpus
  sourceMiss: -8, //    a corpus was asked for and the row is outside it
  want: 12, //          matches "attested" / "meaning matters more"
  wantMiss: -12,
  easyShort: 8, //      one or two syllables, under the diaspora question
  easyLong: -12, //     three syllables, under the diaspora question
  /**
   * Small on purpose. It is a real plus in this family and it earns its reason
   * string, but 527 of the shortlist's 584 V rows have a clean B-form — weight
   * it any harder and every list on the page becomes a list of V names.
   */
  cleanBForm: 2,

  /**
   * Larger than any reachable total, so a feminine-ending row sorts strictly
   * below every other match rather than merely lower. See Prefs.allowFeminineEnding.
   */
  feminineDemotion: -1000,
} as const

/** Four is what fits on a card without the card becoming an essay. */
const REASON_MAX = 4
/** Naming more than two matched themes stops being a reason and becomes a list. */
const THEME_REASON_MAX = 2

/** "name of a mountain" tells you who was called this, not what it means. */
const BARE_ATTESTATION_RE = /^name of\b/i
/**
 * Monier-Williams apparatus the tidy could not safely expand — single-letter
 * shorthand, raw transliteration, the dictionary's own hedges. Same test the
 * dataset build uses to pick its featured rows; fine in the corpus, wrong on a
 * card, so it sorts down rather than being filtered out.
 */
const MESSY_GLOSS_RE = /(?:^|[\s'"(])[b-hj-z](?=[\s'")\-,.]|$)|[a-z][A-Z]|\(\?\)|\bprob\.|\bperhaps\b/
/** Long enough to say something, short enough to sit on one line. */
const CARD_GLOSS_MIN = 8
const CARD_GLOSS_MAX = 60
const LONG_GLOSS = 90

/* ────────────────────────────────────────────────────────────────────────────
   SCORING
   ──────────────────────────────────────────────────────────────────────────── */

/**
 * Does this row survive the hard filters? Letters and syllables only — the
 * visitor named a shape and the document either has it or does not — plus the
 * two exclusions the diaspora question owns.
 */
export function passesHardFilters(row: NaamRow, prefs: Prefs): boolean {
  if (prefs.letters.length > 0 && !prefs.letters.includes(row.letter)) return false
  if (prefs.syllables.length > 0 && !prefs.syllables.includes(row.syllables)) return false
  if (prefs.easySay) {
    if (row.badges.hardCluster) return false
    if (row.bFormQuality === 'awkward') return false
  }
  return true
}

/**
 * Score one row against one set of preferences, with the reasons that produced
 * the score. A row that fails the hard filters scores -Infinity and gives no
 * reasons; rank() drops those, and any other caller should too.
 *
 * Reasons come out in a fixed priority order and are capped at four, except
 * the feminine-ending caveat, which is appended after the cap because it is a
 * warning rather than a selling point and must never be the thing that gets
 * trimmed.
 */
export function scoreName(row: NaamRow, prefs: Prefs): { score: number; reasons: string[] } {
  if (!passesHardFilters(row, prefs)) return { score: Number.NEGATIVE_INFINITY, reasons: [] }

  let score = 0
  const shape: string[] = [] //   what the visitor asked for by shape
  const sense: string[] = [] //   what the row means
  const quality: string[] = [] // what the document says about it
  const sound: string[] = [] //   how it says out loud

  /* — the shape they asked for ——————————————————————————————————— */
  if (prefs.syllables.includes(row.syllables)) {
    shape.push(row.syllables === 1 ? '1 syllable' : `${row.syllables} syllables`)
  }

  /* — meaning ————————————————————————————————————————————————————— */
  const wantedThemes = prefs.themes.filter(isTheme)
  if (wantedThemes.length > 0) {
    const hit = wantedThemes.filter((t) => (row.themeMask & naamThemeBit(t)) !== 0)
    if (hit.length > 0) {
      score += W.theme * hit.length
      for (const t of hit.slice(0, THEME_REASON_MAX)) sense.push(`'${t}'`)
    } else {
      score += W.themeMiss
    }
  }

  /* — lineage ————————————————————————————————————————————————————— */
  if (prefs.sources.length > 0) {
    const hit = row.sources.filter((s) => prefs.sources.includes(s))
    if (hit.length > 0) {
      score += W.source
      sense.push(NAAM_SOURCE_LABEL[hit[0]])
    } else {
      score += W.sourceMiss
    }
  }

  /* — the quality prior, and the reasons it earns ————————————————— */
  const { attested, evocative } = row.badges
  if (attested && evocative) {
    score += W.gold
    quality.push('attested name', 'evocative meaning')
  } else if (evocative) {
    score += W.evocative
    quality.push('evocative meaning')
  } else if (attested) {
    score += W.attested
    quality.push('attested name')
  }
  if (row.tier === 'shortlist') score += W.shortlist
  if (row.themes.length > 0) score += W.hasTheme
  if (BARE_ATTESTATION_RE.test(row.gloss)) score += W.bareAttestation
  if (MESSY_GLOSS_RE.test(row.gloss)) score += W.messyGloss
  if (row.gloss.length < CARD_GLOSS_MIN) score += W.thinGloss
  else if (row.gloss.length <= CARD_GLOSS_MAX) score += W.cardGloss
  else if (row.gloss.length > LONG_GLOSS) score += W.longGloss

  /* — what they said matters more ————————————————————————————————— */
  if (prefs.wants === 'attested') score += attested ? W.want : W.wantMiss
  else if (prefs.wants === 'meaning') score += evocative ? W.want : W.wantMiss

  /* — how it says ————————————————————————————————————————————————— */
  if (prefs.easySay) {
    if (row.syllables <= 2) {
      score += W.easyShort
      sound.push('easy to say')
    } else {
      score += W.easyLong
    }
  }
  if (row.bFormQuality === 'clean') {
    score += W.cleanBForm
    // A reason only when it answers something the visitor asked. The score
    // contribution is a standing quality prior and stays either way, but
    // pushing the string unconditionally put "B-FORM READS CLEANLY" under
    // almost every V card on the page, labelled "Matched on" — and it had not
    // matched anything. A reason every result shares is not a reason.
    if (prefs.easySay) sound.push('B-form reads cleanly')
  }

  /* — the document's own instruction: say it aloud to judge ——————— */
  const feminine = row.badges.feminineEnding && prefs.allowFeminineEnding !== true
  if (feminine) score += W.feminineDemotion

  const reasons = [...shape, ...sense, ...quality, ...sound].slice(0, REASON_MAX)
  if (row.badges.feminineEnding) reasons.push('feminine ending — say it aloud')

  return { score, reasons }
}

/**
 * The ranked matches. Total order: score descending, then id ascending, so the
 * same rows and preferences always produce the same list on the server and in
 * the browser.
 */
export function rank(rows: readonly NaamRow[], prefs: Prefs, limit = 8): NaamMatch[] {
  const scored: NaamMatch[] = []
  for (const row of rows) {
    const { score, reasons } = scoreName(row, prefs)
    if (!Number.isFinite(score)) continue
    scored.push({ row, score, reasons })
  }
  scored.sort((a, b) => b.score - a.score || (a.row.id < b.row.id ? -1 : a.row.id > b.row.id ? 1 : 0))
  return limit >= 0 ? scored.slice(0, limit) : scored
}

/**
 * rank(), but it does not hand back an empty page.
 *
 * `letters`, `syllables` and `easySay` are hard filters, so a perfectly
 * reasonable wish — one syllable, starting with V, easy for a cousin to say —
 * intersects to nothing and rank() returns []. On the free-form path that is
 * also the LOCAL fallback when the model is off or slow, and the plan's
 * degradation rule is explicit: never an error state and never an empty one.
 * An empty result is not an answer; it is the matcher declining to answer.
 *
 * So the hard filters are given back one at a time, in the order a person would
 * give them back — the diaspora question first (it is a preference), then
 * length, then the letter, which is the last thing anyone means to drop. The
 * caller is told which rung it landed on so the page can say so rather than
 * quietly pretending the visitor asked for something else.
 */
export function rankRelaxed(
  rows: readonly NaamRow[],
  prefs: Prefs,
  limit = 8,
): { matches: NaamMatch[]; relaxed: boolean } {
  const ladder: Prefs[] = [
    prefs,
    { ...prefs, easySay: false },
    { ...prefs, easySay: false, syllables: [] },
    { ...prefs, easySay: false, syllables: [], letters: [] },
  ]
  for (let i = 0; i < ladder.length; i++) {
    const matches = rank(rows, ladder[i], limit)
    if (matches.length > 0) return { matches, relaxed: i > 0 }
  }
  return { matches: [], relaxed: false }
}

/* ────────────────────────────────────────────────────────────────────────────
   THE POOL

   What the model is allowed to talk about. Any id it returns that is not in
   here is dropped server-side, so this function is the actual boundary of the
   grounding guarantee — not the prompt.
   ──────────────────────────────────────────────────────────────────────────── */

/**
 * The top `size` matches, spread across letters so the model is not handed
 * forty S names to choose from. When the visitor pinned a single letter the
 * spread is off — they asked for that letter and should get it.
 *
 * Deterministic: it is rank() plus a stable pass over the same list.
 */

/**
 * A tiny deterministic hash. Not cryptography — a spreader.
 */
function hash32(text: string): number {
  let h = 2166136261
  for (let i = 0; i < text.length; i += 1) {
    h ^= text.charCodeAt(i)
    h = Math.imul(h, 16777619)
  }
  return h >>> 0
}

/**
 * ── HUBNESS ───────────────────────────────
 *
 * rank() breaks ties alphabetically by id, which is stable and reproducible and
 * resolves IDENTICALLY for every query ever asked. Combined with a scoring
 * baseline that does not depend on the question, that put the same names into
 * almost every pool. Measured across 82 queries on the production path:
 *
 *   Vastu   64/82 pools      30 names appeared in >= 40% of pools
 *   Balada  56/82            only 627 of 2,098 names were ever reachable
 *
 * "from the sutras" parses to no theme at all, so its top five — Vastu,
 * Vedesha, Vidyesha, Vishvesha, Vivarta — are ALL tied at exactly 34.00 and the
 * alphabet picked them. "calm" finds its real answers first and then fills the
 * remaining thirty-odd slots with the same crew at 28.00.
 *
 * A query-seeded tie-break was tried first and REJECTED BY MEASUREMENT: it
 * moved Vastu from 64/82 to 64/82 and the >=40% count from 30 to 31. Ties were
 * not the mechanism. Vastu genuinely scores high on most queries, because its
 * gloss — "becoming light, dawning, morning" — touches several themes at once,
 * and a pool of forty taken as a global top-N will contain every such name for
 * almost every question.
 *
 * The mechanism is BREADTH, so the correction is diversity: fill the pool
 * greedily, and discount a candidate by how much meaning it already shares with
 * what has been taken. A name that says something no one in the pool has said
 * yet beats a slightly higher-scoring name that repeats one.
 *
 * It is done here in pool() rather than in rank() on purpose: rank() and
 * retrieve() are what scripts/naam/eval scores, so the harness stays a measure
 * of retrieval quality rather than of pool composition.
 */
export function pool(rows: readonly NaamRow[], prefs: Prefs, size = 40): NaamRow[] {
  /**
   * SELECT ON WHAT THE QUESTION EARNED, NOT ON WHAT THE ROW STARTED WITH.
   *
   * Scored against an EMPTY wish, this corpus is not flat: Vastu, Vedesha,
   * Vidyesha, Vishvesha and Vivarta all sit at 34.00 against a median of 7.00.
   * That lift has nothing to do with what was asked, and it is larger than most
   * query signal — so a pool taken as a global top-N contained the same names
   * for almost every question. Measured across 82 queries: Vastu in 64, thirty
   * names in 40% or more, 627 of 2,098 rows ever reachable.
   *
   * Subtracting each row's own baseline leaves only the part the question is
   * responsible for. A name the document loves still ranks high when the
   * question is about it, and stops riding along when it is not.
   *
   * The subtraction is at strength 1 and that is a measured optimum, not a
   * default. Swept against the evocative-reach gate in coverage.mjs: at 0 the
   * hub sits at 69/88; at 1 it is 61/88 with evocative reach unchanged at
   * 75.1%; at 1.25 and above the ranking INVERTS — rows with a high baseline
   * fall below genuinely less relevant ones, evocative reach collapses to 55.4%
   * and the hub climbs to 84. Do not raise it.
   *
   * Two other fixes were tried first and REJECTED BY MEASUREMENT, which is the
   * only reason this one is here: a query-seeded tie-break moved Vastu 64 -> 64,
   * and greedy gloss-diversity made it WORSE, 64 -> 74, because diversity inside
   * a candidate set cannot help when the set itself barely changes.
   *
   * The seeded tie-break returns as a second key, where it does belong: once the
   * baseline is gone, a question with no parsed theme leaves every delta at zero,
   * and without a seed the alphabet would pick the same names it always picked.
   */
  const flat: Prefs = { ...prefs, themes: [], sources: [] }
  const baseline = new Map(rank(rows, flat, -1).map((m) => [m.row.id, m.score]))
  const seed = hash32(JSON.stringify(prefs))
  const ranked = rank(rows, prefs, -1)
  ranked.sort((a, b) => {
    const da = a.score - (baseline.get(a.row.id) ?? 0)
    const db = b.score - (baseline.get(b.row.id) ?? 0)
    return (
      db - da ||
      b.score - a.score ||
      ((hash32(a.row.id) ^ seed) >>> 0) - ((hash32(b.row.id) ^ seed) >>> 0)
    )
  })

  /**
   * THE HEAD IS THE ANSWER; THE TAIL IS THE ROOM TO CHOOSE IN.
   *
   * The model reads the pool in order and mostly picks from the front, so the
   * first rows must be the best rows the question earned. Everything after them
   * is breadth — and taking it as a strict top-40 is what made the same names
   * arrive for every question.
   *
   * So the head stays exactly as ranked, and the tail is drawn from a WIDER
   * slice by a hash seeded on this query. Same question, same answer; different
   * questions, different company.
   */
  /**
   * 12 and 600, and both are swept rather than chosen. The deal is at most 8,
   * so a head of 12 covers everything the model is likely to pick with headroom
   * to spare, and its evocative density is byte-identical to a strict ranking —
   * measured 87.5% either way. 600 of 2,098 is deep enough to reach past the
   * crust that was answering every question and shallow enough that the rows in
   * it still earned their place.
   *
   * Swept against coverage.mjs: (12,120) missed 127 hub 57, (12,300) missed 130
   * hub 51, (16,300) missed 132 hub 51, (20,400) missed 133 hub 54, and
   * (12,600) missed 125, evocative reach 77.1%, hub 50, reach 733 — better on
   * every axis at once.
   *
   * A version WITHOUT the preserved head was tried and rejected: it took the
   * hub to 43 and reach to 905, and paid for it by dropping the pool's
   * evocative density from 86.5% to 70.9%. Reach that arrives as duller names
   * is the metric improving while the product gets worse. This costs 3.2 points
   * of tail density and none at the head.
   */
  const HEAD = 12
  const DEEP = 600
  if (ranked.length > HEAD) {
    const head = ranked.slice(0, HEAD)
    const tail = ranked
      .slice(HEAD, DEEP)
      .sort(
        (a, b) =>
          ((hash32(a.row.id) ^ seed) >>> 0) - ((hash32(b.row.id) ^ seed) >>> 0),
      )
    ranked.length = 0
    ranked.push(...head, ...tail)
  }
  if (size <= 0) return []
  if (prefs.letters.length === 1) return ranked.slice(0, size).map((m) => m.row)

  const perLetter = Math.ceil(size / 2)
  const taken: NaamRow[] = []
  const overflow: NaamRow[] = []
  const counts: Record<string, number> = { B: 0, S: 0, V: 0 }

  for (const m of ranked) {
    if (taken.length >= size) break
    if (counts[m.row.letter] < perLetter) {
      counts[m.row.letter] += 1
      taken.push(m.row)
    } else if (overflow.length < size) {
      overflow.push(m.row)
    }
  }
  for (const row of overflow) {
    if (taken.length >= size) break
    taken.push(row)
  }
  return taken
}

/* ────────────────────────────────────────────────────────────────────────────
   SEARCH
   ──────────────────────────────────────────────────────────────────────────── */

/**
 * The document's own quality ladder, as a sort key. Both badges come straight
 * from the source — N for attested, + for an evocative meaning — so this ranks
 * nothing we invented. 0 is best.
 */
function qualityRank(row: NaamRow): number {
  if (row.badges.attested && row.badges.evocative) return 0
  if (row.badges.evocative) return 1
  if (row.badges.attested) return 2
  return 3
}

/* ────────────────────────────────────────────────────────────────────────────
   RETRIEVAL — reading the meanings, not classifying the question

   THIS EXISTS BECAUSE THE PAGE WAS NOT SEARCHING. Free text used to be mapped
   onto a fixed nineteen-word theme taxonomy, and a question that matched none
   of those words produced NO signal at all — pool() then returned its
   quality-ranked default, so every unrecognised concept got the same forty
   names. Asked for "moon" the model was handed a pool containing none of the
   thirty-six rows whose meaning mentions the moon, and correctly reported that
   it could not see one. Sasi, whose entire gloss is the word "moon", was never
   in the room.

   A taxonomy answers "which of my nineteen buckets is this?" A person asking
   for a name for their son is not thinking in buckets. So this reads the
   DEFINITIONS instead, which is where the answer always was.

   BM25, and the parameters are chosen for this corpus rather than copied:

     k1 = 1.2   standard. Term saturation barely matters when a gloss is six
                words long and almost never repeats a term.
     b  = 0.7   length normalisation, and it does the heavy lifting here. These
                documents run from one word ("moon") to sixty, and without
                normalisation a long dictionary line that mentions the moon in
                passing outranks the row that simply IS the moon. At 0.7, Sasi
                scores 19.0 and "name of the twelfth kala of the moon" scores
                6.7 — which is the order a person would put them in.

   The gloss is weighted well above the raw source line: `gloss` is the tidied
   meaning we show on the card, `sourceGloss` is the unedited dictionary entry
   full of abbreviations and cross-references, and a hit in the second is much
   weaker evidence than a hit in the first.
   ──────────────────────────────────────────────────────────────────────────── */

export interface NaamHit {
  row: NaamRow
  score: number
  /** The query terms this row actually matched, for the agent to reason with. */
  matched: string[]
}

/**
 * Everyday English → this document's own vocabulary. Built at build time by
 * scripts/naam/build-thesaurus.mjs and passed in rather than imported, so this
 * file keeps its no-I/O contract and the eval harness measures the same table
 * production does.
 *
 * ONLY GAPS ARE STORED. If a word already appears in some gloss, BM25 finds it
 * today and an entry would be dead weight — "moon" is in nineteen glosses and
 * has no row here. What it holds is the 855 words this lexicon never uses:
 * brave, calm, doctor, poet. That is the whole reason it is 21 kB instead of a
 * megabyte.
 */
export type NaamThesaurus = Readonly<Record<string, readonly string[]>>

/**
 * An expanded term is EVIDENCE, NOT A MATCH, and is scored accordingly.
 *
 * A row whose gloss literally says what you asked for must always beat a row
 * reached through a synonym, or the thesaurus starts overruling the document.
 * At 0.55 a synonym hit is worth a bit over half a direct one, which is enough
 * to surface Shaura for "brave" — where the alternative is the empty page it
 * returns today — and not enough for a two-hop association to climb over a row
 * that actually means it.
 */
const EXPANSION_WEIGHT = 0.55

/** How many corpus words one query word may become. The builder already caps
 *  the stored fan-out; this is the second belt, applied at query time, so a
 *  bad thesaurus row cannot flood a query no matter what got written. */
const EXPANSION_MAX = 6

/**
 * FORM WORDS ARE NOT MEANING WORDS. Everything here is something parseFreeText
 * has already read as a SHAPE — a length, a letter, a sound — and searching the
 * definitions for it produces pure noise, because these are ordinary English
 * words that appear all over a dictionary in a completely different sense.
 * Measured: "two syllables" retrieved "name of two arhats"; "a short name about
 * light" put "without, except, short of" above the rows that actually mean
 * light. The shape is handled by prefs; this keeps it out of the meaning.
 */
const FORM_WORDS = new Set(
  (
    'short shorter shortest long longer longest brief snappy crisp syllable syllables syllabic ' +
    'letter letters start starts starting begin begins beginning end ends ending spell spelling spelt ' +
    'say saying said pronounce pronounced pronunciation easy easier hard harder simple ' +
    /**
     * `home` USED TO BE HERE and it cost a real query. It was listed beside
     * abroad and overseas for the diaspora question — "easy to say at home and
     * abroad" is shape, not meaning. But "a name that feels like home" is one
     * of the warmest things anyone types into this page, and stripping `home`
     * left it with no terms at all and returned an empty screen. Measured in
     * scripts/naam/eval: that query was one of three still returning nothing.
     * The diaspora reading is already caught by EASY_SAY_RE, which matches the
     * phrases rather than the bare word, so the word itself is free to mean
     * dwelling — which is exactly what Vasa and Vasati say.
     */
    'abroad overseas foreign english nepali nepal one two three four five six seven ' +
    'first second third last single double'
  ).split(' '),
)

/** Words that carry no meaning to search for. */
const QUERY_STOP = new Set([
  ...'the a an and or of to in for with that this is are be am was were it its his her their my our your'.split(' '),
  ...'i we you he she they me us them who what which when where how why'.split(' '),
  ...'name names called mean means meaning definition sense about related relating something anything'.split(' '),
  ...'some any other more most less few give show find look looking want wants need needs like likes'.split(' '),
  ...'please can could would should do does did have has had get got make makes made'.split(' '),
  ...'sound sounds sounding word words son boy child baby there here yes no not but if then than'.split(' '),
  /**
   * DICTIONARY APPARATUS AND THE WORDS PEOPLE USE TO ASK, not to mean.
   *
   * Measured end to end: "a healer, someone who mends people" returned no picks
   * at all, and the pool it built was headed Sivi ("people of Sivi"), Baliha
   * ("pl. name of a people") and Ballava ("pl. name of a people"). The query had
   * a perfectly good target — `healer` reaches `physician` through the
   * thesaurus — but `people` is in 57 glosses where it means a TRIBE, and it
   * buried the real signal under ethnonyms. `various` (110) and `particular`
   * (59) are the same shape: high-frequency lexicographer's filler that nobody
   * types as a wish for their son.
   */
  ...'people person persons folk tribe various particular certain called said belonging form used'.split(' '),
])

const normalise = (value: string): string =>
  (value || '')
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()

/** Content words, three letters or more. */
export function queryTerms(query: string): string[] {
  return [...new Set(normalise(query).split(' '))].filter(
    (t) => t.length > 2 && !QUERY_STOP.has(t) && !FORM_WORDS.has(t),
  )
}

interface IndexedRow {
  row: NaamRow
  gloss: string
  glossTerms: Set<string>
  length: number
  sourceTerms: Set<string>
  latin: string
}

interface GlossIndex {
  docs: IndexedRow[]
  df: Map<string, number>
  avgLength: number
}

const glossIndexCache = new WeakMap<readonly NaamRow[], GlossIndex>()

/** Built once per dataset and cached on it. ~2,100 short docs — a few ms. */
function glossIndex(rows: readonly NaamRow[]): GlossIndex {
  const cached = glossIndexCache.get(rows)
  if (cached) return cached

  const df = new Map<string, number>()
  let total = 0
  const docs = rows.map((row) => {
    const gloss = normalise(row.gloss)
    const words = gloss.split(' ').filter((t) => t.length > 2 && !QUERY_STOP.has(t))
    const glossTerms = new Set(words)
    for (const term of glossTerms) df.set(term, (df.get(term) ?? 0) + 1)
    const length = words.length || 1
    total += length
    return {
      row,
      gloss,
      glossTerms,
      length,
      /**
       * A SET, NOT A STRING, and that is a bug fix rather than a refactor.
       * Matching the raw source line with `.includes()` matched INSIDE words:
       * "long" hit "belonging", "art" hit "particular", "one" hit "stone". A
       * query for "not too long" came back with "bright-toothed" because of it.
       */
      sourceTerms: new Set(
        normalise(row.sourceGloss)
          .split(' ')
          .filter((t) => t.length > 2),
      ),
      latin: normalise(row.latin),
    }
  })

  const index: GlossIndex = { docs, df, avgLength: total / (docs.length || 1) }
  glossIndexCache.set(rows, index)
  return index
}

/**
 * Search the document's own meanings. Returns ranked rows and, for each, the
 * terms it matched — the agent needs to know WHY something came back so it can
 * say so.
 */
export function retrieve(
  rows: readonly NaamRow[],
  query: string,
  limit = 24,
  thesaurus: NaamThesaurus = {},
): NaamHit[] {
  const terms = queryTerms(query)
  if (terms.length === 0) return []

  /**
   * CROSS THE VOCABULARY GAP BEFORE SCORING, not after.
   *
   * Ten graded queries used to return literally nothing — brave, calm, peace,
   * hope, healer among them — because this is a Victorian Sanskrit lexicon and
   * it says valiant where a parent says brave. No BM25 parameter reaches a word
   * that is not in the index, so the query is widened into the document's own
   * vocabulary here, once, from a table built offline.
   *
   * A term is only expanded when it is worth expanding: a word the glosses
   * already contain keeps its full weight and gains nothing, because retrieval
   * for it already works and the synonyms would only add noise.
   */
  const expansions = new Map<string, string>() //   expanded term → the word it came from
  for (const term of terms) {
    const targets = thesaurus[term]
    if (!targets) continue
    for (const target of targets.slice(0, EXPANSION_MAX)) {
      if (!terms.includes(target) && !expansions.has(target)) expansions.set(target, term)
    }
  }

  const { docs, df, avgLength } = glossIndex(rows)
  const n = docs.length || 1
  const k1 = 1.2
  const b = 0.7
  const idf = (term: string) => {
    const seen = df.get(term) ?? 0
    return Math.log(1 + (n - seen + 0.5) / (seen + 0.5))
  }

  const hits: NaamHit[] = []
  for (const doc of docs) {
    let score = 0
    const matched: string[] = []

    /** One term against one document. Returns 0 when the document has no use
     *  for it, so the caller can tell a miss from a weak hit. */
    const contribute = (term: string, weight: number): number => {
      let tf = 0
      if (doc.glossTerms.has(term)) tf = 2
      else if (doc.sourceTerms.has(term)) tf = 0.6
      if (doc.latin === term) tf += 6
      if (tf === 0) return 0
      return weight * idf(term) * ((tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc.length / avgLength))))
    }

    for (const term of terms) {
      const gain = contribute(term, 1)
      if (gain === 0) continue
      matched.push(term)
      score += gain
    }

    /**
     * Synonym hits are reported under the word the VISITOR typed, not the one
     * the document happens to use. `matched` is what the agent reasons and
     * speaks from — telling it a row matched "valiant" when the person asked
     * for "brave" invites a reply explaining a word nobody used.
     */
    for (const [target, origin] of expansions) {
      const gain = contribute(target, EXPANSION_WEIGHT)
      if (gain === 0) continue
      if (!matched.includes(origin)) matched.push(origin)
      score += gain
    }

    if (score === 0) continue
    // The meaning IS the word. One-word queries against one-word glosses are
    // the commonest thing a person types, and they should win outright.
    if (terms.length === 1 && doc.gloss === terms[0]) score *= 2.4
    else if (terms.some((t) => doc.gloss.startsWith(t))) score *= 1.25
    hits.push({ row: doc.row, score, matched })
  }

  hits.sort((a, z) => z.score - a.score || qualityRank(a.row) - qualityRank(z.row) || (a.row.id < z.row.id ? -1 : 1))

  /**
   * A RELEVANCE FLOOR, WITH A FLOOR OF ITS OWN. Taking a fixed top-N off an OR
   * query pads the pool with rows that matched one weak term and nothing else,
   * and this corpus makes that acute: `name` appears in 71.8% of glosses and
   * `of` in 78.6%, so one ordinary shared word drags a row in.
   *
   * The cut is 30% of the top score rather than the 50% the arithmetic
   * suggests, and it always keeps six, because of the exact-gloss bonus above.
   * Sasi — whose whole meaning is "moon" — scores 19.0 while Suma ("the moon")
   * scores 7.9, so a half-of-top cut threw away a perfectly good answer to keep
   * a slightly better one. Measured: it took "moon" from twenty hits to one and
   * "fire" and "thunder" to one apiece. A threshold tuned on scores without
   * that multiplier is the wrong threshold for scores with it.
   */
  const floor = hits.length > 0 ? hits[0].score * 0.3 : 0
  const strong = hits.filter((hit) => hit.score >= floor)
  return (strong.length >= 6 ? strong : hits.slice(0, 6)).slice(0, limit)
}

/**
 * Substring search over `searchKey` (latin + B-variant + gloss, lowercased).
 * Buckets rather than fuzzy scoring: a name that starts with what you typed
 * beats a name that contains it, which beats a meaning that mentions it.
 *
 * An empty query is the unfiltered browse list, and it is sorted by the
 * document's quality ladder rather than by document order. Document order is
 * alphabetical, so the first screen used to be Bana, Bandin, Bara, Barhi,
 * Barhin, Barhis — "name of a man", "name of a descendant of angiras" — while
 * every name with a meaning worth reading sat thousands of rows down. The
 * lexicon is 70% terse attestations by weight; opening on them makes the whole
 * collection look like dictionary residue.
 */
export function search(rows: readonly NaamRow[], query: string, limit = 50): NaamRow[] {
  const q = query.trim().toLowerCase()
  if (!q) {
    return rows
      .slice()
      .sort(
        (a, b) =>
          qualityRank(a) - qualityRank(b) ||
          Number(a.tier !== 'shortlist') - Number(b.tier !== 'shortlist') ||
          Number(a.badges.hardCluster) - Number(b.badges.hardCluster) ||
          (a.id < b.id ? -1 : a.id > b.id ? 1 : 0),
      )
      .slice(0, limit)
  }

  const hits: Array<{ row: NaamRow; bucket: number }> = []
  for (const row of rows) {
    const latin = row.latin.toLowerCase()
    const bForm = row.bVariant ? row.bVariant.toLowerCase() : ''
    let bucket = -1
    if (latin === q || bForm === q) bucket = 0
    else if (latin.startsWith(q) || bForm.startsWith(q)) bucket = 1
    else if (latin.includes(q) || bForm.includes(q)) bucket = 2
    else if (row.searchKey.includes(q)) bucket = 3
    if (bucket < 0) continue
    hits.push({ row, bucket })
  }
  hits.sort(
    (a, b) =>
      a.bucket - b.bucket ||
      Number(a.row.tier !== 'shortlist') - Number(b.row.tier !== 'shortlist') ||
      qualityRank(a.row) - qualityRank(b.row) ||
      (a.row.id < b.row.id ? -1 : a.row.id > b.row.id ? 1 : 0),
  )
  return hits.slice(0, limit).map((h) => h.row)
}

/* ────────────────────────────────────────────────────────────────────────────
   FREE TEXT

   Small on purpose. Three things a visitor actually types — look this name up,
   compare these two, here is what I want — read with regexes and keyword maps.
   No NLP, no model. Whatever this misses, the model still sees the raw ask;
   what this catches is what makes the page work when the model is off.
   ──────────────────────────────────────────────────────────────────────────── */

/** English that happens to look like a name cell. Every one was checked against the dataset. */
const STOP = new Set([
  'and',
  'but',
  'both',
  'best',
  'better',
  'bit',
  'boy',
  'born',
  'before',
  'because',
  'between',
  'bring',
  'brother',
  'say',
  'says',
  'said',
  'she',
  'short',
  'shorter',
  'should',
  'simple',
  'since',
  'single',
  'sister',
  'small',
  'soft',
  'some',
  'someone',
  'something',
  'son',
  'song',
  'soon',
  'sort',
  'sound',
  'sounds',
  'spell',
  'spelling',
  'start',
  'starts',
  'still',
  'strong',
  'such',
  'sure',
  'sweet',
  'syllable',
  'syllables',
  'value',
  'various',
  'versus',
  'very',
  'vowel',
])

const COMPARE_RE = /\b(or|vs|versus|between|compare|against)\b/

/** Wish vocabulary → the closed theme lexicon in src/types/naam.ts. */
const THEME_WORDS: Array<[RegExp, NaamTheme]> = [
  [/\b(light|bright|shining|shine|radiant|sun|dawn|glow|luminous|golden)\b/, 'light'],
  [/\b(strong|strength|mighty|power|powerful|brave|hero|fierce|solid)\b/, 'strength'],
  [/\b(wise|wisdom|clever|smart|learned|scholar|thinker|intelligent|knowledge)\b/, 'wisdom'],
  [/\b(kind|kindness|love|loving|affection|compassion|tender|gentle|warm|generous)\b/, 'compassion'],
  [/\b(joy|joyful|happy|happiness|cheerful|delight|glad|laughing)\b/, 'joy'],
  [/\b(auspicious|lucky|luck|fortune|fortunate|blessed|prosper|prosperity)\b/, 'auspicious'],
  [/\b(sky|heaven|air|cloud|wind|celestial|bird|flight|space)\b/, 'sky'],
  [/\b(water|river|ocean|sea|rain|wave|stream|lake|monsoon)\b/, 'water'],
  [/\b(earth|ground|mountain|hill|forest|tree|land|stone|soil|himal)\b/, 'earth'],
  [/\b(sound|voice|music|musical|song|singing|speech|word|chant|hymn|melody)\b/, 'sound'],
  [/\b(protect|protection|protector|guard|guardian|shelter|refuge|shield|keeper)\b/, 'protection'],
  [/\b(lotus|padma|kamal)\b/, 'lotus'],
  [/\b(calm|peace|peaceful|quiet|still|serene|tranquil|gentle soul|settled)\b/, 'peace'],
  [/\b(god|gods|divine|deity|lord|shiva|vishnu|krishna|indra|agni|dev|devta)\b/, 'deity'],
  [/\b(monk|monastic|ascetic|sage|rishi|buddha|buddhist|awakened|meditat|dhamma|dharma)\b/, 'monk'],
  [/\b(king|kings|royal|prince|noble|ruler|leader|throne)\b/, 'royal'],
  [/\b(pure|purity|clean|clear|spotless|holy|sacred)\b/, 'purity'],
  [/\b(swift|fast|quick|speed|nimble|agile|horse|arrow)\b/, 'swift'],
  [/\b(true|truth|truthful|honest|honesty|sincere|righteous|just|faithful)\b/, 'truth'],
]

const SOURCE_WORDS: Array<[RegExp, NaamSource]> = [
  [/\b(vedic|veda|vedas|rigveda|rig-veda|oldest|ancient)\b/, 'V'],
  [/\b(classical|epic|puranic|sanskrit literature|kavya)\b/, 'C'],
  [/\b(theravada|pali|sutta|suttas|buddhist|buddhism|dhamma)\b/, 'T'],
]

const NUMBER_WORDS: Array<[RegExp, number]> = [
  [/\b(one|1)[\s-]?syllable/, 1],
  [/\b(two|2)[\s-]?syllable/, 2],
  [/\b(three|3)[\s-]?syllable/, 3],
]

const EASY_SAY_RE =
  /\b(easy to say|easy to pronounce|easy to call|hard to say|hard to pronounce|pronounce|pronounceable|mouthful|tongue|non-nepali|foreigner|foreigners|american|americans|abroad|diaspora|english speakers?|cousins? in)\b/

/**
 * Deliberately narrow. A bare "mean" is how people ask what a name means —
 * "what does Bhaskara mean" is a lookup, not a statement of preference — so
 * only phrases that express a preference count, and "more" decides it when
 * both sides are named.
 */
const ATTESTED_RE =
  /\b(attested|someone real|real person|a real name|really existed|actually (used|borne)|borne by|historical)\b/
const MEANING_RE =
  /\b(meaning matters|meaning is what|meaningful|means something|care about the meaning|prefer the meaning)\b/
const MEANING_WINS_RE = /\bmeaning (matters )?more\b/
const ATTESTED_WINS_RE = /\b(attested|real (name|person)) (matters )?more\b/

/** What the page reads out of a sentence before anyone calls a model. */
export interface NaamAsk {
  prefs: Prefs
  /** Names the visitor asked about directly. */
  lookups: NaamRow[]
  /** Names the visitor put side by side. */
  compare: NaamRow[]
  /**
   * A name the visitor typed that this document does not contain, with the
   * closest things it does. "Sanskar" is the case that made this exist: it is
   * one of the family's own favourites, it is not a row, and the honest answer
   * is neither to invent it nor to shrug — it is "we don't have that one, but
   * look at these."
   */
  near: NaamNearMiss[]
}

export interface NaamNearMiss {
  /** Exactly as the visitor spelled it, so it can be quoted back to them. */
  typed: string
  rows: NaamRow[]
}

/** Names mentioned, capped — four is already more than a reply can hold. */
const NAME_HITS_MAX = 4

const nameIndexCache = new WeakMap<readonly NaamRow[], Map<string, NaamRow>>()

/**
 * ONE SANSKRIT NAME, MANY ROMAN SPELLINGS. There is no single correct way to
 * write these in Latin script, so the same name reaches us as Sanskar and
 * Samskara, as Bishnu and Vishnu, as Byan and Vyan — and an exact-match index
 * tells a visitor their own son's shortlist is not in the document.
 *
 * This folds the differences that carry no sound:
 *   · diacritics off, so ṣ and ś and s are one letter
 *   · the aspirate digraphs stay (bh is not b — that is a real contrast in
 *     Devanagari) but sh → s and ch → c, which are spelling habits, not sounds
 *   · doubled letters collapse: Samskaara, Samskara and Samskarra are one word
 *   · a trailing 'a' goes, because the inherent vowel is written by some
 *     transliterators and dropped by others — Samskara / Samskar
 *   · m before a stop becomes n, which is the anusvara: Samskar / Sanskar
 *   · v and b are ONE letter here. That is not sloppiness, it is the entire
 *     premise of this page — व is said ब at home, which is why the V names are
 *     on the list at all. Bishnu must find Vishnu.
 *
 * The result is a sound key, not a spelling. It is only ever used to FIND a
 * row; every spelling the page displays still comes from the row itself.
 */
export function soundKey(name: string): string {
  return name
    .toLowerCase()
    .normalize('NFKD')
    .replace(/[̀-ͯ]/g, '')
    .replace(/[^a-z]/g, '')
    .replace(/sh/g, 's')
    .replace(/ch/g, 'c')
    .replace(/w/g, 'v')
    .replace(/v/g, 'b')
    .replace(/(.)\1+/g, '$1')
    .replace(/m(?=[kgcjtdpbsn])/g, 'n')
    .replace(/a$/, '')
}

/** Levenshtein, bounded — anything past `max` is not a near miss, it is a
 *  different name, and the early exit keeps this cheap over 2,000 rows. */
function within(a: string, b: string, max: number): number {
  if (Math.abs(a.length - b.length) > max) return max + 1
  let prev = Array.from({ length: b.length + 1 }, (_, i) => i)
  for (let i = 1; i <= a.length; i++) {
    const row = [i]
    let best = i
    for (let j = 1; j <= b.length; j++) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1
      row[j] = Math.min(prev[j] + 1, row[j - 1] + 1, prev[j - 1] + cost)
      if (row[j] < best) best = row[j]
    }
    if (best > max) return max + 1
    prev = row
  }
  return prev[b.length]
}

/**
 * The closest rows to something the document does not have. Ranked by sound
 * distance first and quality second, so the suggestions are both near and worth
 * suggesting — the nearest row to a name nobody would choose is not a kindness.
 */
export function nearestBySound(typed: string, rows: readonly NaamRow[], limit = 3): NaamRow[] {
  const key = soundKey(typed)
  if (key.length < 3) return []
  const scored: { row: NaamRow; d: number; score: number }[] = []
  for (const row of rows) {
    const d = Math.min(within(key, soundKey(row.latin), 2), row.bVariant ? within(key, soundKey(row.bVariant), 2) : 3)
    if (d > 2) continue
    /**
     * SOUNDING CLOSE IS NOT A REASON TO SUGGEST A NAME. Ranked on distance
     * alone, "Sanskar" came back with Samkara first — whose entry in this
     * document reads "dust, sweepings (-kuta n. a heap of rubbish)" — and the
     * page offered it, warmly, to two people naming their son. The row is real
     * and the spelling was honest; it was still the wrong name to put in front
     * of them, and no amount of grounding makes it right.
     *
     * scoreName already encodes what makes a name worth suggesting, and it is
     * the same judgement the rest of the page runs on: −14 where the gloss only
     * attests a bearer, +12 where the document marked the meaning worth saying,
     * +5 where our tagger found a sense at all. So sound decides who is in the
     * running and MEANING decides the order they arrive in.
     */
    scored.push({ row, d, score: scoreName(row, EMPTY_PREFS).score })
  }

  /**
   * A row with no sense at all — no evocative badge, no theme — is held back
   * rather than dropped. If nothing better is near, the closest thing the
   * document actually has is still a better answer than silence.
   */
  const hasSense = (row: NaamRow) => row.badges.evocative || row.themes.length > 0
  const byMerit = (a: (typeof scored)[number], b: (typeof scored)[number]) =>
    b.score - a.score || a.d - b.d || qualityRank(a.row) - qualityRank(b.row)

  return [
    ...scored.filter((s) => hasSense(s.row)).sort(byMerit),
    ...scored.filter((s) => !hasSense(s.row)).sort(byMerit),
  ]
    .slice(0, limit)
    .map((s) => s.row)
}

/**
 * Names that sit next to a name you already like. "I like Bhaskara" is a
 * statement about a whole shape — a meaning, a length, a sound — and answering
 * it with an unrelated top-of-pool list ignores the only real signal the
 * visitor has given. Weighted so MEANING counts most: a shared theme is the
 * thing a person actually means by "names like this one".
 */
export function relatedTo(seed: NaamRow, rows: readonly NaamRow[], limit = 6): NaamRow[] {
  const themes = new Set(seed.themes ?? [])
  const key = soundKey(seed.latin)
  const onset = key.slice(0, 2)
  const scored: { row: NaamRow; score: number }[] = []
  for (const row of rows) {
    if (row.id === seed.id) continue
    let score = 0
    for (const theme of row.themes ?? []) if (themes.has(theme)) score += 5
    if (row.syllables === seed.syllables) score += 2
    if (row.letter === seed.letter) score += 1
    if (soundKey(row.latin).startsWith(onset)) score += 2
    if (score <= 0) continue
    scored.push({ row, score })
  }
  scored.sort((a, b) => b.score - a.score || qualityRank(a.row) - qualityRank(b.row))
  return scored.slice(0, limit).map((s) => s.row)
}

const soundIndexCache = new WeakMap<readonly NaamRow[], Map<string, NaamRow>>()

/** The same index keyed by sound rather than spelling. First row wins, and
 *  rows are already in quality order, so a collision resolves to the better
 *  name rather than to whichever happened to be parsed first. */
function soundIndex(rows: readonly NaamRow[]): Map<string, NaamRow> {
  const cached = soundIndexCache.get(rows)
  if (cached) return cached
  const index = new Map<string, NaamRow>()
  for (const row of rows) {
    for (const spelling of [row.latin, row.bVariant]) {
      if (!spelling) continue
      const key = soundKey(spelling)
      if (key.length >= 3 && !index.has(key)) index.set(key, row)
    }
  }
  soundIndexCache.set(rows, index)
  return index
}

/** latin and bVariant, lowercased, first row wins so the id order is stable. */
function nameIndex(rows: readonly NaamRow[]): Map<string, NaamRow> {
  const cached = nameIndexCache.get(rows)
  if (cached) return cached
  const index = new Map<string, NaamRow>()
  for (const row of rows) {
    const latin = row.latin.toLowerCase()
    if (!index.has(latin)) index.set(latin, row)
    if (row.bVariant) {
      const b = row.bVariant.toLowerCase()
      if (!index.has(b)) index.set(b, row)
    }
  }
  nameIndexCache.set(rows, index)
  return index
}

/**
 * Read a sentence. Handles the three real shapes:
 *
 *   lookup   "what does Bhaskara mean" · "tell me about Sneha"
 *   compare  "Bhaskara or Bodhi" · "Vachas vs Bodhi"
 *   wish     "something short and calm that starts with S"
 *
 * A token is only treated as a name when it is three or more letters, is not
 * ordinary English, and matches a row exactly on `latin` or `bVariant` — so
 * "something short and calm" stays a wish and "Bachas" resolves to Vachas.
 */
export function parseFreeText(text: string, rows: readonly NaamRow[]): NaamAsk {
  const lower = String(text ?? '')
    .toLowerCase()
    .slice(0, 400)
  const index = nameIndex(rows)

  /* — names —————————————————————————————————————————————————————— */
  const found: NaamRow[] = []
  const seen = new Set<string>()
  const near: NaamNearMiss[] = []
  const sounds = soundIndex(rows)

  /**
   * Read the ORIGINAL text, not the lowercased copy, because capitalisation is
   * the only signal separating "I like Sanskar" from "I like short ones". Exact
   * spellings are matched either way — those are safe. Sound-matching and near
   * misses are gated on a capital, because they are fuzzy by definition and a
   * fuzzy match on every ordinary word would have the page answering "calm"
   * with a name that merely sounds like it.
   */
  const raw = String(text ?? '').slice(0, 400)
  /**
   * A capital at the START of a sentence is grammar, not a name. Without this
   * "What does light mean" read `What` as a name — soundKey folds w→v→b for the
   * व/ब rule, so "what" becomes "bhat" and lands on Bhatta. The exception is a
   * message short enough to be nothing BUT a name: someone who types "Sanskar?"
   * still gets an answer, and someone who types a sentence does not have its
   * first word mistaken for their son's name.
   */
  const terse = (raw.match(/[A-Za-z]+/g) ?? []).length <= 3

  for (const match of raw.matchAll(/[A-Za-z]+/g)) {
    const token = match[0]
    const word = token.toLowerCase()
    // What precedes it, ignoring whitespace: nothing, or . ! ? — a new sentence.
    const before = raw.slice(0, match.index).trimEnd()
    const sentenceInitial = before.length === 0 || /[.!?]$/.test(before)
    if (word.length < 3 || STOP.has(word)) continue
    const capitalised =
      token[0] === token[0].toUpperCase() && token[0] !== token[0].toLowerCase() && (!sentenceInitial || terse)

    // Spelled exactly as the document spells it.
    let row = index.get(word)
    // Spelled the way a person actually writes it: Bishnu for Vishnu, Samskara
    // for Sanskar. Same name, different transliterator.
    if (!row && capitalised) row = sounds.get(soundKey(word))

    if (!row) {
      // A name the document does not have at all. Only worth answering if it
      // was clearly offered as a name — and only once per ask, because a list
      // of near misses is not an answer to anything.
      if (capitalised && word.length >= 4 && near.length === 0) {
        const suggestions = nearestBySound(word, rows)
        if (suggestions.length > 0) near.push({ typed: token, rows: suggestions })
      }
      continue
    }
    if (seen.has(row.id)) continue
    seen.add(row.id)
    found.push(row)
    if (found.length >= NAME_HITS_MAX) break
  }
  const isCompare = found.length >= 2 && COMPARE_RE.test(lower)

  /* — the wish ———————————————————————————————————————————————————— */
  const syllables: number[] = []
  for (const [re, n] of NUMBER_WORDS) if (re.test(lower)) syllables.push(n)
  if (syllables.length === 0) {
    if (/\b(short|shorter|snappy|crisp|one word|brief)\b/.test(lower)) syllables.push(1, 2)
    else if (/\b(longer|long|full|fuller|grand)\b/.test(lower)) syllables.push(3)
  }

  const letters: NaamLetter[] = []
  for (const m of lower.matchAll(
    /\b(?:start|starts|starting|begin|begins|beginning)s?\s+with\s+(?:a\s+|the\s+)?([bsv])\b/g,
  )) {
    letters.push(m[1].toUpperCase() as NaamLetter)
  }
  for (const m of lower.matchAll(/\b([bsv])[\s-]names?\b/g)) letters.push(m[1].toUpperCase() as NaamLetter)
  for (const m of lower.matchAll(/\bletter\s+([bsv])\b/g)) letters.push(m[1].toUpperCase() as NaamLetter)

  const themes: NaamTheme[] = []
  for (const [re, theme] of THEME_WORDS) if (re.test(lower)) themes.push(theme)

  const sources: NaamSource[] = []
  for (const [re, src] of SOURCE_WORDS) if (re.test(lower)) sources.push(src)

  let wants: Prefs['wants'] = 'both'
  if (MEANING_WINS_RE.test(lower)) wants = 'meaning'
  else if (ATTESTED_WINS_RE.test(lower)) wants = 'attested'
  else {
    const asksAttested = ATTESTED_RE.test(lower)
    const asksMeaning = MEANING_RE.test(lower)
    if (asksAttested && !asksMeaning) wants = 'attested'
    else if (asksMeaning && !asksAttested) wants = 'meaning'
  }

  const prefs = normalizePrefs({
    syllables,
    letters,
    themes,
    sources,
    wants,
    easySay: EASY_SAY_RE.test(lower),
  })

  return {
    prefs,
    lookups: isCompare ? [] : found,
    compare: isCompare ? found : [],
    near,
  }
}

/* ────────────────────────────────────────────────────────────────────────────
   small local helpers
   ──────────────────────────────────────────────────────────────────────────── */

function uniq<T>(list: readonly T[]): T[] {
  return [...new Set(list)]
}

function isLetter(value: unknown): value is NaamLetter {
  return value === 'B' || value === 'S' || value === 'V'
}

function isSource(value: unknown): value is NaamSource {
  return value === 'V' || value === 'C' || value === 'T'
}

function isTheme(value: unknown): value is NaamTheme {
  return typeof value === 'string' && (NAAM_THEMES as readonly string[]).includes(value)
}
