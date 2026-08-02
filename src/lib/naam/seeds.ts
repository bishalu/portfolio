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
 * WHY five of them are hand-written rows: every one of these was checked
 * against Boy_Name_Candidates.pdf before it was written down.
 *
 *   Byan     IS in the document — `Vyan / Byan`, Classical, one syllable,
 *            "to respire, breathe, inhale and exhale", p.120. It renders as a
 *            full card, with its source block, its page cite and a working
 *            व/ब control, because it is itself a V→B name.
 *   Satwik   `Sattvika` is in the document (p.56); this spelling is not.
 *   Sanskar  `Samskara` is in the document (p.83); this spelling is not.
 *   Bishnu   not a row. Vishnu appears only *inside* meanings.
 *   Brihut   not as spelled — the document has Brihanta, Brihata, Brihaka.
 *   Soham    not present at all.
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
}

const SEEDS: readonly SeedSpec[] = [
  { latin: 'Satwik', letter: 'S', devanagari: 'सात्विक', syllableSplit: ['sa', 'twik'] },
  { latin: 'Sanskar', letter: 'S', devanagari: 'संस्कार', syllableSplit: ['sa', 'nskar'] },
  { latin: 'Bishnu', letter: 'B', devanagari: 'बिष्णु', syllableSplit: ['bi', 'shnu'] },
  { latin: 'Brihut', letter: 'B', devanagari: 'बृहुत', syllableSplit: ['bri', 'hut'] },
  { latin: 'Soham', letter: 'S', devanagari: 'सोहम्', syllableSplit: ['so', 'ham'] },
]

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
