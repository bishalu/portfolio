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
export function pool(rows: readonly NaamRow[], prefs: Prefs, size = 40): NaamRow[] {
  const ranked = rank(rows, prefs, -1)
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
}

/** Names mentioned, capped — four is already more than a reply can hold. */
const NAME_HITS_MAX = 4

const nameIndexCache = new WeakMap<readonly NaamRow[], Map<string, NaamRow>>()

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
  for (const token of lower.match(/[a-z]+/g) ?? []) {
    if (token.length < 3 || STOP.has(token)) continue
    const row = index.get(token)
    if (!row || seen.has(row.id)) continue
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
