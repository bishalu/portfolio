/**
 * The six names the wall opens with (docs/design/DESIGN.md §4 rule 4, P9).
 *
 * WHY they exist: the wall is the page's whole point — one person leaving one
 * name with a reason — and a moderated wall is empty on day one. DESIGN.md §4
 * rule 4 says failure is honest, never blank; an empty wall is not a failure
 * but it reads like one, so the family's own shortlist starts it and approved
 * suggestions append below. No attribution line: they read as a starting
 * shortlist, not as anyone's argument.
 *
 * WHY seven of them are hand-written rows: every one of these was checked
 * against Boy_Name_Candidates.pdf before it was written down.
 *
 *   Byan     IS in the document — `Vyan / Byan`, Classical, one syllable,
 *            "to respire, breathe, inhale and exhale", p.120. It renders as a
 *            full card, with its source block, its page cite and a working
 *            व/ब control, because it is itself a V→B name.
 *   Satwik   `Sattvika` is in the document (p.56); this spelling is not.
 *   Sanskar  `Samskara` is in the document (p.83); this spelling is not.
 *   Bishnu   not a row. Vishnu appears only *inside* meanings.
 *   Brihat   not as spelled — the document has Brihanta, Brihata, Brihaka.
 *            (Corrected from `Brihut` on 2026-08-05: the family's spelling is
 *            Brihat. Devanagari updated to बृहत् — bṛhat, "great" — which is
 *            the standard form and the one closest to the document's Brihata.)
 *   Soham    not present at all.
 *   Brihan   not present. The document has Brihanta and Brihata; this is the
 *            bare stem, and it is the family's spelling rather than a citation.
 *   Stambha  not present as a name. `stambha` appears only inside meanings.
 *
 * So the five carry no meaning, no etymology and no source badge — there is
 * nothing to cite, and inventing a citation for a name someone loves would be
 * exactly the failure this page is built to avoid (P9). They get one mono
 * line, NAAM_COPY.card.notInDocument, and that is the honest handling.
 *
 * Their Devanagari is hand-written and marked `checked` rather than `derived`:
 * these are the most-seen Devanagari forms on the page, so they are not
 * transliterator output. The syllable splits follow the dataset's own
 * onset-nucleus convention (consonants attach to the following vowel, as in
 * the document's `sa·mska·ra`) so the sound rail reads the same everywhere.
 */
import type { NaamLetter, NaamRow, NaamSyllables } from '@/types/naam'

/** The one seed that is a real row; the page resolves it from the dataset. */
export const NAAM_SEED_DOCUMENTED = { id: 'vyan' } as const

interface SeedSpec {
  latin: string
  letter: NaamLetter
  devanagari: string
  syllableSplit: string[]
  /** How many people have said this one. Drives how near it floats. */
  votes: number
}

const SEEDS: readonly SeedSpec[] = [
  { latin: 'Satwik', letter: 'S', devanagari: 'सात्विक', syllableSplit: ['sa', 'twik'], votes: 2 },
  { latin: 'Sanskar', letter: 'S', devanagari: 'संस्कार', syllableSplit: ['sa', 'nskar'], votes: 2 },
  { latin: 'Bishnu', letter: 'B', devanagari: 'बिष्णु', syllableSplit: ['bi', 'shnu'], votes: 2 },
  { latin: 'Brihat', letter: 'B', devanagari: 'बृहत्', syllableSplit: ['bri', 'hat'], votes: 2 },
  { latin: 'Soham', letter: 'S', devanagari: 'सोहम्', syllableSplit: ['so', 'ham'], votes: 2 },
  /* Newer, and one voice each so far — which is what puts them furthest back
     in the sky. Devanagari hand-written like the rest of this list:
       बृहन्  bṛhan,   the stem of bṛhat — "great, vast"
       स्तम्भ  stambha, "pillar, post" — the thing that holds a roof up */
  { latin: 'Brihan', letter: 'B', devanagari: 'बृहन्', syllableSplit: ['bri', 'han'], votes: 1 },
  { latin: 'Stambha', letter: 'S', devanagari: 'स्तम्भ', syllableSplit: ['sta', 'mbha'], votes: 1 },
]

/**
 * Votes per seed, by the row id the page builds below.
 *
 * Support lives here rather than being inferred from how many times a name
 * appears in the family list, because these five — now seven — are one row
 * each: the list says WHICH names the family holds, and this says how many of
 * them have said so. The lantern field reads it straight (see lanterns.ts:
 * distinct counts set the band width, so two tiers put the twos in front and
 * the ones behind them).
 */
export const NAAM_SEED_VOTES: Readonly<Record<string, number>> = {
  ...Object.fromEntries(SEEDS.map((seed) => [`seed-${seed.latin.toLowerCase()}`, seed.votes])),
  /* Byan is the one documented row, resolved from the dataset rather than
     built here, so it carries its own id and its own count. */
  vyan: 2,
}

/**
 * NaamRow-shaped so the one card component renders them, with every
 * document-owned field left empty — because it is. Rendered with
 * `undocumented`, which drops the meaning line and the source disclosure.
 */
export const NAAM_SEED_ROWS: readonly NaamRow[] = SEEDS.map((seed) => ({
  id: `seed-${seed.latin.toLowerCase()}`,
  latin: seed.latin,
  bVariant: null,
  letter: seed.letter,
  syllables: seed.syllableSplit.length as NaamSyllables,
  tier: 'more',
  sources: [],
  badges: { attested: false, evocative: false, feminineEnding: false, hardCluster: false },
  sourceGloss: '',
  page: 0,
  gloss: '',
  glossIsVerbatim: true,
  devanagari: seed.devanagari,
  /** No B-form to show: these five are not V rows, so there is nothing to swap. */
  devanagariB: null,
  devanagariConfidence: 'checked',
  syllableSplit: seed.syllableSplit,
  themes: [],
  themeMask: 0,
  searchKey: seed.latin.toLowerCase(),
  bFormQuality: null,
}))
