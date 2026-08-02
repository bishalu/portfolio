/**
 * Every word that appears on /naam (docs/design/DESIGN.md §3, P9, P11).
 *
 * WHY one file: the page, the island, the form, the wall and the approve
 * screen are built by different hands at different times, and micro-copy is
 * exactly where a voice comes apart. Import from here; do not write prose in a
 * component.
 *
 * THE VOICE, so it survives edits:
 *   · Warm, specific, unfussy. Active voice. Sentence case in prose; UI labels
 *     go through .label-mono, which uppercases them in CSS — so they are
 *     written here in the case a screen reader should say them.
 *   · A label labels, an example demonstrates, a helper explains. Nothing does
 *     two of those jobs at once, and nothing repeats a line that already
 *     appears somewhere else on the page.
 *   · Nepali the way an urban Nepali family drops it into English: a word, not
 *     a sentence. "Kina?", "nwaran", "bolaune naam", "dhanyabad". Never
 *     transliterated Hindi, never a glossary.
 *   · Errors say what happened and what to do next. No apologies, no blame.
 *   · Never ceremonial, never touristy. No "journey", no "magical", no
 *     "discover the perfect name", no exclamation marks, no emoji.
 *   · Numbers come from src/generated/naam-facts.ts and are never retyped —
 *     the standing rule in src/content.config.ts.
 *
 * The honesty vocabulary is three words and this page uses two of them: LOCAL
 * when the matcher answered, LIVE when the model did (DESIGN.md §4). There is
 * no REPLAY here — a canned reply was never produced by the real system, so
 * every degradation falls back to the matcher, which is real.
 */
import { NAAM_COUNTS } from '@/generated/naam-facts'
import { NAAM_SOURCE_LABEL, NAAM_THEMES, type NaamLetter, type NaamSource, type NaamTheme } from '@/types/naam'
import type { Prefs } from './match'

/* ────────────────────────────────────────────────────────────────────────────
   The guided flow — six questions, each answerable in one tap.
   Option values are typed against Prefs, so a question that stops matching the
   matcher stops compiling.
   ──────────────────────────────────────────────────────────────────────────── */

export interface NaamOption<V> {
  value: V
  /** What the visitor reads on the control. */
  label: string
  /** One clause under it. Optional, and often better absent. */
  note?: string
}

export interface NaamQuestion<K extends keyof Prefs, V> {
  /** The Prefs field this question fills. */
  key: K
  label: string
  helper: string
  /** true = chips, any number of them. false = one answer. */
  multiple: boolean
  /** Only set where a cap is real. */
  max?: number
  options: readonly NaamOption<V>[]
}

export type NaamAnyQuestion =
  | NaamQuestion<'syllables', number>
  | NaamQuestion<'letters', NaamLetter>
  | NaamQuestion<'themes', NaamTheme>
  | NaamQuestion<'sources', NaamSource>
  | NaamQuestion<'wants', Prefs['wants']>
  | NaamQuestion<'easySay', boolean>

/** Labels for the closed theme lexicon, in NAAM_THEMES order. */
const THEME_LABEL: Record<NaamTheme, string> = {
  light: 'Light',
  strength: 'Strength',
  wisdom: 'Wisdom',
  compassion: 'Kindness',
  joy: 'Joy',
  auspicious: 'Good fortune',
  sky: 'Sky',
  water: 'Water',
  earth: 'Earth',
  sound: 'Sound',
  protection: 'Protection',
  lotus: 'Lotus',
  peace: 'Calm',
  deity: 'The gods',
  monk: 'The monastic',
  royal: 'Kings',
  purity: 'Purity',
  swift: 'Speed',
  truth: 'Truth',
}

const QUESTIONS: readonly NaamAnyQuestion[] = [
  {
    key: 'syllables',
    label: 'How long should it be?',
    helper: 'Say it out loud.',
    multiple: true,
    options: [
      { value: 1, label: 'One syllable', note: 'A short list.' },
      { value: 2, label: 'Two', note: 'Easy to call.' },
      { value: 3, label: 'Three', note: 'Room for meaning.' },
    ],
  },
  {
    key: 'letters',
    label: 'Which letter?',
    helper: 'One, two, or all three.',
    multiple: true,
    options: [
      { value: 'B', label: 'B', note: 'Bishal' },
      { value: 'S', label: 'S', note: 'Sneha' },
      { value: 'V', label: 'V', note: 'Which we say as B' },
    ],
  },
  {
    key: 'themes',
    label: 'What should it mean?',
    helper: 'Up to three. Our reading of the document.',
    multiple: true,
    max: 3,
    options: NAAM_THEMES.map((theme) => ({ value: theme, label: THEME_LABEL[theme] })),
  },
  {
    key: 'sources',
    label: 'Where should it come from?',
    helper: 'Which text it comes from.',
    multiple: true,
    options: [
      { value: 'V', label: NAAM_SOURCE_LABEL.V, note: 'The oldest layer.' },
      { value: 'C', label: NAAM_SOURCE_LABEL.C, note: 'Epic and later Sanskrit.' },
      { value: 'T', label: NAAM_SOURCE_LABEL.T, note: 'The Pali suttas.' },
    ],
  },
  {
    key: 'wants',
    label: 'Which matters more?',
    helper: `The document marks both, and ${n(NAAM_COUNTS.gold)} names carry both.`,
    multiple: false,
    options: [
      { value: 'attested', label: 'Someone real bore it', note: 'A person or a god in the text.' },
      { value: 'meaning', label: 'The meaning', note: 'What it says.' },
      { value: 'both', label: 'No preference' },
    ],
  },
  {
    key: 'easySay',
    label: 'Easy for anyone to say?',
    helper: 'Drops the harder clusters.',
    multiple: false,
    options: [
      { value: true, label: 'Yes', note: 'Kathmandu and Ohio both.' },
      { value: false, label: 'Does not matter' },
    ],
  },
]

/* ────────────────────────────────────────────────────────────────────────────
   The relation list. Warmest thing on the page, and genuinely worth knowing.
   ──────────────────────────────────────────────────────────────────────────── */

export const NAAM_RELATIONS: readonly string[] = [
  'Kaka',
  'Kaki',
  'Mama',
  'Maiju',
  'Phupu',
  'Fupaju',
  'Didi',
  'Dai',
  'Bhai',
  'Friend',
  'Colleague',
  'Just passing through',
]

/* ────────────────────────────────────────────────────────────────────────────
   THE DECK
   ──────────────────────────────────────────────────────────────────────────── */

export const NAAM_COPY = {
  meta: {
    title: 'Naam — help us name our son',
    description:
      'Sneha and Bishal are naming their son. Answer six questions or just say what you are after, and send back a shortlist.',
    /** The nav and footer entry. */
    navLabel: 'Naam',
  },

  hero: {
    eyebrow: 'A family ask',
    headline: 'Help us name our son.',
    /**
     * One sentence. The two doors below say what to do, so the standfirst does
     * not need to; "tell us why / we read every one" moved to the send step,
     * where it is the thing actually being asked for. Nine text blocks used to
     * sit above the first control — on a phone that is 700px of scrolling
     * before you can act.
     */
    standfirst: `Sneha and I have ${n(NAAM_COUNTS.total)} candidates and no decision.`,
    /** Quiet, one line, not the headline. */
    letters: 'B is Bishal, S is Sneha. V is here because Nepali says व as ब.',
    /** Stated once, as the reason two names exist. */
    nwaran:
      'On the eleventh day the priest gives a rashi name from the stars; the bolaune naam, the one everyone actually calls him, is ours to pick — and that is the one this page is about.',
  },

  /** The invitation on the home page, inside CloseSection. */
  invite: {
    line: 'Sneha and I are naming our son, and the shortlist is public.',
    cta: 'Help us name him',
  },

  /**
   * The wizard and the ask both need JavaScript; the names below them do not.
   * Without this line a no-JS visitor meets a panel of dead controls and no
   * explanation of where the real content is.
   */
  noscript:
    'The six questions and the ask need JavaScript. Everything below — the names, their meanings and the form — does not.',

  /**
   * Two doors, no helpers. A label that needs a sentence under it explaining
   * what it does is a label that has not been written yet — and two helpers
   * here pushed the first control below the fold on every viewport.
   */
  doors: {
    guided: { label: 'Answer six questions' },
    freeform: { label: 'Say what you are after' },
  },

  guide: {
    questions: QUESTIONS,
    /** Every question is skippable; none of them is required. */
    skip: 'No preference',
    back: 'Back',
    next: 'Next',
    finish: 'Show me names',
    restart: 'Start over',
    change: 'Change an answer',
    stepLabel: (step: number, total: number) => `Question ${step} of ${total}`,
    /** Shown while the six answers are still being taken. */
    answersLabel: 'So far',
  },

  ask: {
    label: 'Tell us what you are after',
    placeholder: 'something short and calm that starts with S',
    hint: 'A line or two is plenty.',
    send: 'Ask',
    examples: [
      'what does Bhaskara mean',
      'Bhaskara or Bodhi',
      'a strong two-syllable Vedic name my cousins can pronounce',
    ],
  },

  results: {
    heading: 'What the document offers',
    /** Above the computed match reasons. They are never written by a model. */
    reasonsLabel: 'Matched on',
    lookupHeading: 'That one, in the document',
    compareHeading: 'Side by side',
    more: 'Show 50 more',
    /** .pulse-line captions. Lowercase — they are mono captions, not labels. */
    loading: 'reading the document…',
    loadingAsk: 'asking…',
    empty: 'Nothing answers all of that. Loosen one answer — syllables is usually the one to give back.',
    emptyAsk: 'Nothing matched that. Try a meaning rather than a spelling.',
    /** Shown when the hard filters intersected to nothing and were given back. */
    relaxed: 'Nothing answered all of that, so one answer has been given back. These are the closest the document has.',
    /** For a name the visitor asked about that the document does not carry. */
    notFound: (name: string) => `${name} is not in this document. The closest things that are:`,
  },

  /** DESIGN.md §4. Two of the three words; there is no REPLAY on this page. */
  badge: {
    live: 'LIVE',
    local: 'LOCAL',
    liveCaption: 'A real call, just now.',
    localCaption: 'Matched in your browser, from the document.',
  },

  /**
   * Honest failure. Every one of these degrades to the matcher's own list,
   * which is real, so the badge stays LOCAL and never becomes an error state.
   */
  failure: {
    modelDown: 'The model did not answer, so these are the document’s own best matches.',
    modelSlow: 'That was taking too long, so these are the document’s own best matches.',
    modelOff: 'No model on this build. These are the document’s own best matches.',
    dataDown: 'The full list did not load. Reload and it will try again.',
  },

  card: {
    /** The disclosure holds the document verbatim; the summary above it is ours. */
    sourceSummary: 'From the document',
    page: (page: number) => `p.${page}`,
    sourceLabel: (source: NaamSource) => NAAM_SOURCE_LABEL[source],
    attested: 'Attested',
    evocative: 'Evocative',
    feminineEnding: 'Feminine ending — say it aloud',
    hardCluster: 'Harder cluster',
    glossVerbatim: 'The document’s line, unchanged',
    glossTidied: 'The document’s line, tidied by us',
    devanagariNote: 'Devanagari is our rendering, not the document’s.',
    /**
     * The swap control on the first syllable of a V name. The visible glyph is
     * two letters and a separator, not "व to ब": .label-mono uppercases, so an
     * English preposition between two Devanagari letters rendered as
     * "व TO ब" — a system message where the page's one code-switch should be.
     */
    swapGlyph: 'व / ब',
    swapAria: 'Switch between the V and the B spelling',
    swapNote: 'Display only. The document’s spelling stays as printed.',
    pick: 'Pick',
    picked: 'Picked',
    unpick: 'Remove',
    /** The seeded family names that are not rows in the document. */
    notInDocument: 'Not in the document — a family name',
  },

  /**
   * The prerendered names, and the reason they have their own heading: this
   * section used to run C.results.heading over C.browse.standfirst — a title
   * that also labels the wizard's result list, over a sentence describing the
   * browse island two sections down. Two different things wearing one phrase,
   * and a standfirst about a list that is not on screen.
   */
  shortlist: {
    heading: 'Where we would start',
    standfirst: (perLetter: number) =>
      `${n(perLetter)} under each letter, out of the ${n(NAAM_COUNTS.gold)} the document marks as both a real name and a meaning worth having.`,
    /** Shown, out of everything the document holds under that letter. */
    letterCount: (shown: number, total: number) => `${n(shown)} of ${n(total)}`,
  },

  browse: {
    heading: 'The whole list',
    standfirst: `${n(NAAM_COUNTS.core)} names load first; the other ${n(NAAM_COUNTS.rest)} on request.`,
    searchLabel: 'Search names and meanings',
    searchPlaceholder: 'bhas, lotus, king…',
    filterLetter: 'Letter',
    filterSyllables: 'Syllables',
    filterSource: 'Source',
    /**
     * Its own axis. "Attested" and "Evocative" sat under the Source legend
     * beside Vedic/Classical/Theravada, which made a chip row that mixed where
     * a name came from with what kind of entry it is — two different questions
     * under one label.
     */
    filterKind: 'Kind',
    filterTheme: 'Meaning',
    clear: 'Clear filters',
    loadRest: 'Load the rest',
    loading: 'loading the rest…',
    empty: 'Nothing matches those filters.',
    count: (shown: number, total: number) => `${n(shown)} of ${n(total)}`,
  },

  tray: {
    heading: 'Your picks',
    empty: 'No picks yet. Pick a name and it waits here.',
    count: (picks: number) => (picks === 1 ? '1 name' : `${n(picks)} names`),
    full: (max: number) => `That is ${max}, which is plenty. Send these first.`,
    clear: 'Clear',
    send: 'Send these to us',
    /** The tray survives the two paths and a reload. */
    persisted: 'Kept on this device until you send them.',
  },

  form: {
    heading: 'Send us your picks',
    standfirst: 'It comes to us as an email. Nothing appears on the wall until we read it.',
    name: {
      label: 'Your name',
      error: 'We need a name to put next to it.',
      tooLong: 'Keep it under 40 characters.',
    },
    relation: {
      label: 'How do we know you?',
      placeholder: 'Pick one',
      options: NAAM_RELATIONS,
      error: 'Pick one — any one.',
    },
    picks: {
      label: 'Your picks',
      empty: 'No picks yet — tap Pick on any name, or type one below.',
    },
    /**
     * The always-usable half of the form. With JavaScript off nothing can fill
     * the hidden picks array, so without this a no-JS visitor could submit the
     * form and send no name at all — the one thing the page is for.
     */
    names: {
      label: 'Or type the names',
      helper: 'Any spelling, any name. It does not have to be in the document.',
      placeholder: 'Bishrut, Saurya',
    },
    reason: {
      /** The one place the code-switch earns its keep. */
      label: 'Kina? — one line is plenty',
      helper: 'Optional. It is the part we will remember.',
      counter: (used: number, max: number) => `${used}/${max}`,
      tooLong: 'A bit long. 240 characters is the limit.',
    },
    submit: 'Send',
    sending: 'sending…',
    confirmation: {
      heading: 'Got it.',
      body: 'It is with us. Nothing goes on the wall until we have read it. Dhanyabad.',
      again: 'Send another',
      /**
       * The email went and the moderation queue did not take it. Both halves
       * are true and the visitor is told which, rather than being shown "sent"
       * for a write that failed — /api/naam-submit returns `stored` precisely
       * so this can be honest.
       */
      emailOnly: 'It is with us by email. Putting it on the wall may take a second try from our side.',
    },
    error: {
      network: 'That did not send. Your picks are still here — try once more.',
      server: 'Something broke on our side. Try again in a minute, or write to bishal@vibeset.ai.',
      rateLimited: 'That is a few too many in a row. Give it a minute.',
    },
  },

  /** Caps, so the tray, the form and the endpoint agree on them. Read, never retyped. */
  limits: {
    name: 40,
    reason: 240,
    picks: 6,
    names: 120,
  },

  wall: {
    heading: 'What people have sent',
    standfirst: 'Suggestions we have read and put up.',
    /** The six family favorites the wall ships with. */
    seedLabel: 'Ours, to start',
    /** Sits directly under the six seed cards, so it says what is missing. */
    empty: 'No suggestions yet. Yours would be the first.',
    loading: 'loading the wall…',
    failure: 'The wall did not load. Reload and it will come back.',
    entry: (name: string, relation: string) => `${name} · ${relation}`,
  },

  approve: {
    eyebrow: 'Moderation',
    heading: 'Approve this suggestion',
    body: 'Approving puts it on the public wall, with the name and the reason as written.',
    submit: 'Approve',
    working: 'approving…',
    success: 'Approved. It is on the wall.',
    already: 'Already approved. Nothing to do.',
    missing: 'That suggestion is not here any more.',
    invalid: 'This link is not valid. It may have been used already.',
    error: 'That did not go through. Reload and press it once more.',
  },
} as const

export type NaamCopy = typeof NAAM_COPY

/** 6715 → "6,715". Deterministic, so SSR and the client render the same string. */
function n(value: number): string {
  return value.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ',')
}
