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
import { NAAM_SOURCE_LABEL, type NaamSource } from '@/types/naam'

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
      'Sneha and Bishal are naming their son. Say what you are after, keep three names from what the document offers, and send them back.',
    /** The nav and footer entry. */
    navLabel: 'Naam',
  },

  /**
   * THE NO-JS FALLBACK'S WORDS, and nothing else now (src/pages/naam.astro).
   * With JavaScript on there is no hero — the greeting in `app` is the opening
   * line and the app owns the viewport. These four strings are the ordinary
   * document a visitor without JavaScript gets instead, so they are still live
   * and must not be retired with the rest of the editorial frame.
   */
  hero: {
    headline: 'Help us name our son.',
    /** One sentence. The form below is the ask, so the standfirst is not. */
    standfirst: `Sneha and I have ${n(NAAM_COUNTS.total)} candidates and no decision.`,
    /** Quiet, one line, not the headline. */
    letters: 'B is Bishal, S is Sneha. V is here because Nepali says व as ब.',
    /** Stated once, as the reason two names exist. */
    nwaran:
      'On the eleventh day the priest gives a rashi name from the stars; the bolaune naam, the one everyone actually calls him, is ours to pick — and that is the one this page is about.',
  },

  /**
   * THE APP (docs/design/DESIGN.md §4, P8, P11).
   *
   * /naam is one screen now — a rail, a stream, three slots, a composer — so
   * the words that used to be headings and standfirsts are turns in a
   * conversation. They live here for the same reason as everything else in
   * this file: the app, the no-JS fallback and the form must not drift apart.
   *
   * The Devanagari in `starters` is deliberate: it is the string that proves
   * Mukta is first in the chip's font stack. If that chip ever renders ▯▯▯▯,
   * the stack is wrong somewhere and this is where it shows first.
   */
  app: {
    /**
     * The one visible heading with JavaScript on, and it is the rail's brand
     * too — the <h1> in the topbar is the wordmark. Lowercase: it is the page's
     * own name, not a title.
     */
    heading: 'naam',
    streamLabel: 'The conversation, and the names it has dealt',
    /**
     * Screen-reader speaker labels. There are no bubbles and no avatars
     * (P11) — turns are typographic blocks, so the speaker is stated in
     * .sr-only text rather than implied by which side of the column it is on.
     */
    speakerAgent: 'Naam',
    speakerYou: 'You',
    greeting: 'नमस्ते. Sneha and I are naming our son, and we would like your help.',
    invitation: 'Tell me how he should sound. Or start with one of these.',
    familyLead: 'The family likes these already',
    starters: [
      'something calm, about love rather than war',
      'two beats, easy to call',
      'names about light',
      'what does भास्कर mean',
    ],
    dismissStarters: 'Hide these',
    /** Over a fresh deal of cards. */
    dealt: 'Keep the ones you would want on the wall.',
    /** .pulse-line captions, lowercase: they are captions, not labels. */
    reading: 'reading the document…',
    asking: 'asking the model…',
    jump: 'Jump to the latest',
    composerLabel: 'Say what you are after',
    composerPlaceholder: 'a feeling, a sound, a syllable…',
    composerSend: 'Send',
    /** The site footer is hidden on this page, so its index comes with us. */
    indexLabel: 'The rest of the site',
    tray: {
      label: 'Your three',
      /** Devanagari numerals. Free, and unmistakably Nepali. */
      ordinals: ['१', '२', '३'],
      empty: (slot: number) => `Slot ${slot}, empty`,
      /** The whole slot is the control; there is no grey ✕. */
      taken: (slot: number, name: string) => `Slot ${slot}, ${name}. Take it back.`,
    },
    /**
     * The last turn: the send fields arrive in the stream when three are kept.
     * `submit` lives here rather than in a `tray` block of its own — the old
     * one described a six-pick sidebar that could be cleared, counted and
     * declared full, and every string in it but this one described something
     * the page no longer has. Two blocks called tray, one of them dead, is how
     * the next reader picks the wrong one.
     */
    send: {
      lead: 'That is your three. Who should we thank for them?',
      picksLabel: 'Sending',
      why: 'Why these? — one line is plenty',
      submit: 'Send these to us',
    },
  },

  /** The invitation on the home page, inside CloseSection. */
  invite: {
    line: 'Sneha and I are naming our son, and the shortlist is public.',
    cta: 'Help us name him',
  },

  /**
   * The app needs JavaScript; the names and the form below it do not. Without
   * this line a no-JS visitor meets a page with no explanation of why the thing
   * they were sent a link to is not on it.
   */
  noscript: 'The naming app needs JavaScript. Everything below — the names, their meanings and the form — does not.',

  /**
   * What the app says ABOUT a set of names, as opposed to the names themselves.
   * Whittled to three: the wizard's headings, its "show 50 more", its
   * side-by-side lookup and its two loading captions all went with the
   * editorial page, and the app's own captions live under `app` instead.
   */
  results: {
    /** Above the computed match reasons. They are never written by a model. */
    reasonsLabel: 'Matched on',
    emptyAsk: 'Nothing matched that. Try a meaning rather than a spelling.',
    /** Shown when the hard filters intersected to nothing and were given back. */
    relaxed: 'Nothing answered all of that, so one answer has been given back. These are the closest the document has.',
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
    /**
     * Keep, not Pick. The tray is three slots a name is *kept* in, and the
     * word on the button is the word the motion illustrates — the card arcs
     * down and is kept. One verb for one gesture, everywhere on the page.
     *
     * There is no `unpick` here any more. Taking a name back is done on the
     * seated slot, not on the card, so the words for it belong to the slot —
     * `app.tray.taken` carries them.
     */
    pick: 'Keep',
    picked: 'Kept',
    /** The seeded family names that are not rows in the document. */
    notInDocument: 'Not in the document — a family name',
  },

  /**
   * ONE HEADING, over the twelve names the no-JS fallback prerenders. Its
   * standfirst and per-letter counter went with the browse island: the fallback
   * shows four under each letter and says so by showing them, and a count of a
   * list nobody can filter is a number doing no work.
   */
  shortlist: {
    heading: 'Where we would start',
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
    /**
     * Reachable, and only one way: the send turn arrives when the third slot
     * fills, and taking names back afterwards leaves the form on screen with
     * nothing to send. It names the gesture the tray actually has — Keep, and
     * three slots — not the old grid's "tap Pick on any name".
     */
    picks: {
      empty: 'All three slots are empty again. Keep a name and it lands here.',
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
    /**
     * Three, and the cap is the point. The tray is three slots that visibly
     * fill, so the limit is what makes a choice cost something — and three
     * considered names are better to receive than thirty. tray.ts's PICK_MAX
     * and /api/naam-submit both read this, so they cannot disagree.
     */
    picks: 3,
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
