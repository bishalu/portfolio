/**
 * Every word that appears on /naam (docs/design/DESIGN.md §3, P9, P11).
 *
 * WHY one file: the page, the island, the form, the wall and the approve
 * screen are built by different hands at different times, and micro-copy is
 * exactly where a voice comes apart. Import from here; do not write prose in a
 * component.
 *
 * THE VOICE, so it survives edits. The page is an invitation to help name a
 * son, and it should read like one — the warmth of a wedding invitation
 * without any of the costume:
 *   · SIMPLE WORDS, SHORT SENTENCES. A cousin in Ohio who does not read
 *     Devanagari and an aunt in Kathmandu should both get it on the first
 *     pass. If a line needs reading twice, it is the line that is wrong.
 *   · WARM, NEVER CEREMONIAL. No "request the honour", no flourishes, no
 *     exclamation marks, no emoji, no "journey", no "magical", no "discover
 *     the perfect name". The warmth is carried by spacing and scale in
 *     naam.astro, so the words themselves can stay plain.
 *   · NEPALI THE WAY A FAMILY ACTUALLY DROPS IT IN: a word, not a sentence,
 *     and gloss it in English right next to it. Three across a surface is
 *     warm; five is costume. The app gets नमस्ते, Kina? and dhanyabad; the
 *     no-JS fallback gets nwaran and Kina?. Never transliterated Hindi.
 *   · THE AUDIENCE IS THE DIASPORA — someone who left, or whose parents did,
 *     and who wants the name to work in two places at once. The copy may
 *     acknowledge that ("easy to say abroad"); it never explains it.
 *   · A label labels, an example demonstrates, a helper explains. Nothing does
 *     two of those jobs at once, and nothing repeats a line that already
 *     appears somewhere else on the same surface.
 *   · Errors say what happened and what to do next. No apologies, no blame.
 *   · Numbers come from src/generated/naam-facts.ts and are never retyped —
 *     the standing rule in src/content.config.ts. The caps in `limits` are
 *     this file's own, so they are declared once in CAPS below and read from
 *     there by the messages that mention them.
 *
 * The honesty vocabulary is three words and this page uses two of them: LIVE
 * when the model answered, LOCAL when the matcher did (DESIGN.md §4). There is
 * no REPLAY here — a canned reply was never produced by the real system. LIVE
 * is the ordinary case now, because the agent leads and nothing is rendered
 * before it answers; LOCAL appears only when a visitor presses `failure.escape`
 * and asks for the document's own list, so the word means what it says instead
 * of labelling a fallback nobody chose.
 */
import { NAAM_COUNTS } from '@/generated/naam-facts'
import { NAAM_SOURCE_LABEL, type NaamSource } from '@/types/naam'

/**
 * The caps, declared before the deck so the messages that quote a number can
 * read it instead of retyping it. Exposed as `NAAM_COPY.limits`; tray.ts and
 * /api/naam-submit both import that, so nothing can disagree about them.
 */
const CAPS = {
  name: 40,
  reason: 240,
  /**
   * Three, and the cap is the point. The tray is three slots that visibly
   * fill, so the limit is what makes a choice cost something — and three
   * considered names are better to receive than thirty.
   */
  picks: 3,
  names: 120,
} as const

/* ────────────────────────────────────────────────────────────────────────────
   How we know you. Four buckets, and every one of them is deliberately wide.

   It used to be twelve Nepali kinship terms — Kaka, Kaki, Mama, Maiju, Phupu,
   Fupaju, Didi, Dai, Bhai — which was charming and wrong in three ways. It had
   no row for the child's own PARENTS or grandparents, which is an odd thing for
   a naming page to omit. It made a friend from work read nine words of someone
   else's family tree to find themselves. And it drew a line down the middle of
   the audience: a Nepali aunt is offered her exact title, an American colleague
   is offered "Colleague", and the form quietly tells the second one they are a
   guest here.

   Four broad ones ask nothing and exclude nobody, and the real answer arrives
   in `send.why` anyway — Kina?, in their own words, where an aunt can say she
   is his Phupu far better than a <select> ever asked her to. The page's Nepali
   lives there and in नमस्ते and Dhanyabad, in the voice, rather than in a
   dropdown that a non-Nepali friend has to opt out of.
   ──────────────────────────────────────────────────────────────────────────── */

export const NAAM_RELATIONS: readonly string[] = ['Family', 'Friend', 'Work', 'Just passing through']

/* ────────────────────────────────────────────────────────────────────────────
   THE DECK
   ──────────────────────────────────────────────────────────────────────────── */

export const NAAM_COPY = {
  meta: {
    title: 'Naam — help us name our son',
    description:
      'Sneha and Bishal are expecting a son. Say what you have in mind, keep three names from the document, and send them to us.',
    /** The nav and footer entry. */
    navLabel: 'Naam',
  },

  /**
   * THE NO-JS FALLBACK'S WORDS, and nothing else now (src/pages/naam.astro).
   * With JavaScript on there is no hero — the greeting in `app` is the opening
   * line and the app owns the viewport. These four strings are the ordinary
   * document a visitor without JavaScript gets instead, so they carry the same
   * invitation in the same register, and they must not be retired with the
   * rest of the editorial frame.
   */
  hero: {
    headline: 'Help us name our son.',
    /** The invitation, in one breath, with the one number that matters. */
    standfirst: `Sneha and Bishal are expecting a son. There are ${n(NAAM_COUNTS.total)} names to choose from, and we would rather not choose alone.`,
    /**
     * B is Bishal and S is Sneha, and that much is just what the list is. What
     * this line USED to do is explain the third letter — that व is said ब at
     * home, so Vachas is Bachas — and that explanation has been taken out of
     * every static surface on the page.
     *
     * It is the agent's to make, and only in front of a name it is actually
     * holding up in both forms. A joke printed in the furniture before anyone
     * has seen a V name is a joke explained before it is told; the same
     * sentence, said once, about a specific name someone is looking at, is the
     * warmest thing on the page. Same words, and the difference is entirely
     * whether they arrive on cue.
     */
    letters: 'B is Bishal, S is Sneha. The V names are here for a reason of their own.',
    /** Stated once, as the reason two names exist. Three short sentences. */
    nwaran:
      'On the eleventh day, at the nwaran, a priest reads the stars and gives him one name. The name we will actually call him at home is ours to choose. That is the one we are asking about.',
  },

  /**
   * THE APP (docs/design/DESIGN.md §4, P8, P11).
   *
   * /naam is one screen now — a rail, a stream, three slots, a composer — so
   * the words that used to be headings and standfirsts are turns in a
   * conversation. They live here for the same reason as everything else in
   * this file: the app, the no-JS fallback and the form must not drift apart.
   *
   * `greeting` and `invitation` are two turns of one invitation: who is asking
   * and why, then what to do. The greeting is the biggest type on the page
   * (nm-said--lead), so it is kept under thirty words.
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
    /**
     * The folded hands, and they are a separate string rather than a character
     * inside the greeting for two reasons. They are sized independently — the
     * greeting is the largest type on the page and an emoji set to match it
     * reads as a sticker — and they are aria-hidden, because नमस्ते is already
     * the first word a screen reader reaches and "folded hands, namaste" is
     * the same greeting twice.
     *
     * This is the page's one emoji, and the voice note above says there are
     * none. The exception is deliberate: नमस्ते is a gesture before it is a
     * word, and the hands are what makes it one on a screen.
     */
    greetingGlyph: '🙏',
    /**
     * SAID IN THE AFFIRMATIVE. It ended "we would rather not choose it alone",
     * which is the same fact stated as a lack — and a sentence built out of
     * what is missing asks the reader to supply it. Right Speech is not only
     * about avoiding untruth; it is about saying the thing that nourishes, and
     * "find it together" is the identical fact facing forward. Nobody is
     * short-handed, nobody is being rescued, and the reader is being offered
     * something rather than asked to fill a gap.
     */
    greeting:
      'नमस्ते. Sneha and Bishal are expecting a son. He needs a name before he arrives, and we would like to find it together.',
    /**
     * WHAT THE LIST IS, said once and quietly. A visitor who does not know
     * where these names come from cannot tell whether the page is reading a
     * real document or making things up, and that is the one thing this whole
     * build is trying to be trustworthy about. So it is stated plainly, as a
     * fact and not a boast: the count, the two corpora, the three letters.
     *
     * The number is interpolated from naam-facts.ts and never retyped, and the
     * letters are the family's own joke arriving as information — B for Bishal,
     * S for Sneha, V because at home व is said ब.
     */
    /**
     * WHY THERE IS A PAGE AT ALL. A Nepali reader arrives knowing a priest
     * gives the child a name, and until that is answered the whole page is a
     * puzzle — they cannot tell what is being asked of them. It lived only in
     * the no-JS fallback, so the audience it was written for had never read it.
     *
     * Two sentences, not three. The fallback's third — "That is the one we are
     * asking about" — is dropped here because the invitation two blocks later
     * asks it directly, with the question mark, and two sentences doing one job
     * cost the greeting its place on the phone's first screen.
     */
    why: 'On the eleventh day, at the nwaran, a priest reads the stars and gives him one name. The name we call him at home is ours to choose.',
    /** The letters line joins this rather than taking a beat of its own: the
        count, the two corpora and the three letters are one fact. */
    source: `${n(NAAM_COUNTS.total)} names, out of the Vedas and the Sutras — the ones that begin with B, S or V. B is Bishal, S is Sneha; the V names are here for a reason of their own.`,
    /**
     * THE ASK, AND THE PROMISE THAT SOMEONE IS LISTENING. It ended "or start
     * with one of these" and pointed at four pre-written chips; the chips are
     * gone, so the line has to do their teaching itself — a meaning, a sound, a
     * single word is what a person may type, said as an invitation rather than
     * as a syntax.
     *
     * "However it comes to you" is the whole register. There is no correct
     * input here and no wrong way to begin, and telling somebody that before
     * they type is the difference between a question and a test.
     *
     * It does NOT say "our agent will guide you", which is what this line is
     * for. P11 forbids assistant framing on this page and the model itself is
     * instructed never to say what it is, so naming the machinery would break
     * the one illusion the whole build maintains. "We will look through them
     * with you" makes the same promise — someone is on the other side of this,
     * and you will not be left alone with 6,715 names — in the page's voice.
     */
    invitation:
      'What kind of name are you looking for? Say it however it comes to you — a meaning, a sound, a single word. We will look through them with you.',
    /**
     * It said "Names the family keeps coming back to", which was true when the
     * shelf held six seeds and nothing else. It now holds the family's, every
     * approved suggestion, and the visitor's own picks the moment they make
     * them — so a heading naming only the family would be quietly wrong about
     * whose names are on it, and would tell a visitor their leaf belongs to
     * someone else. "People" is all three of them, and it is still warm.
     */
    familyLead: 'Names people keep coming back to',
    /** Over a fresh deal of cards. */
    dealt: 'Keep the ones that sound like him.',
    /**
     * WHAT A TURN YOU HAVE PASSED SAYS. The stream compresses everything before
     * the question you are currently on down to one line, so these are the
     * labels for the turns whose own text is not the summary — a turn that
     * SPOKE (the agent, or you) is summarised by its own first line and needs
     * nothing here.
     *
     * `reopen` is .sr-only on every one of those collapsed controls, because a
     * button named only "Ways to start" says what it is and not what pressing
     * it does. It is the whole accessible name for a keyboard user and it has
     * to carry the verb.
     *
     * Nothing here names the object the stream is drawn as. The page is a mala
     * — a thread with beads on it, worn where you have already passed — and a
     * visitor who owns one recognises it in a second while everyone else sees a
     * well-made progress thread. Explaining it would spend the whole effect.
     */
    bead: {
      reopen: 'Open this again',
      starters: 'Ways to start',
      names: (count: number) => `${count} ${count === 1 ? 'name' : 'names'}, dealt`,
      /** The ask came back with nothing to keep, either way it happened. */
      stuck: 'Nothing came back for that',
      form: 'Who to thank for these',
      sent: 'Sent',
    },
    /**
     * .pulse-line captions, lowercase: they are captions, not labels. They
     * name the two real steps, and they stay literal — the badge says LIVE
     * because a model was asked, and this is the caption that says so while it
     * happens (DESIGN.md §4, P9). A warmer word here would be a nicer lie.
     */
    reading: 'reading the document…',
    asking: 'asking the model…',
    jump: 'Jump to the latest',
    composerLabel: 'Tell us what you have in mind',
    /* Names the group for a screen reader. "Ways to start" rather than
       "suggestions": each one is a whole question, and tapping it asks it. */
    startersLabel: 'Ways to start',
    composerPlaceholder: 'even one word is enough…',
    composerSend: 'Send',
    /**
     * The site footer is display:none here, and the full index used to ride on
     * the tray row at the same visual weight as the three slots. The header nav
     * carries every route except this one, so this is the only one that needed
     * rescuing.
     */
    a11yLink: 'Accessibility',
    /**
     * The sound switch, and its label names the STATE it is in rather than the
     * action it performs — it is an aria-pressed toggle, so the pressed state
     * carries "what happens if I press it" and the word is free to say what is
     * true right now. "Sound off" while silent, "Sound on" while audible; a
     * button reading "Turn sound on" while sound is already on is the usual way
     * this control lies.
     */
    sound: {
      off: 'Sound off',
      on: 'Sound on',
      aria: 'Sound',
    },
    tray: {
      /**
       * Still three, and still stated up front — the slots are drawn empty from
       * the first frame so the shape of the ask is visible before anything is
       * chosen. What changed is that three is a ceiling, not a toll: one is
       * enough to send, and `send.lead` says so the moment a name lands.
       */
      /* "Keep up to three" said the LIMIT and not the purpose — it never
         answered "keep them for what?", which is the only question a
         first-time visitor has about three empty sockets. This says what they
         are. The cap is still on screen: there are three of them, numbered. */
      label: 'The three you send us',
      /** Devanagari numerals. Free, and unmistakably Nepali. */
      ordinals: ['१', '२', '३'],
      empty: (slot: number) => `Slot ${slot} of three, empty`,
      /** The whole slot is the control; there is no grey ✕. */
      taken: (slot: number, name: string) => `Slot ${slot} of three, ${name}. Take it back.`,
      /**
       * AN EMPTY SLOT IS AN OFFER. A name of your own used to be a field inside
       * the send form, which nobody reaches until they have already kept
       * something from the document — so the one person it existed for, the
       * relative who arrived with a name in mind, could not get to it. The tray
       * is where names live; this is how one that is not in the document gets
       * there.
       */
      ownGlyph: '+',
      ownLabel: (slot: number) => `Add a name of your own — slot ${slot} of three`,
      ownPlaceholder: 'a name…',
    },
    /**
     * The last turn: the send fields arrive in the stream on the FIRST kept
     * name, not the third. `submit` lives here rather than in a `tray` block of
     * its own — the old one described a six-pick sidebar that could be cleared,
     * counted and declared full, and every string in it but this one described
     * something the page no longer has. Two blocks called tray, one of them
     * dead, is how the next reader picks the wrong one.
     */
    send: {
      /**
       * WHAT SENDING DOES, SAID BEFORE SENDING. The form states this already,
       * but only once the visitor has committed — the wrong order for the one
       * irreversible thing on the page.
       */
      promise: 'Straight to us. Nothing goes on the wall until we have read it.',
      /**
       * It used to read "That is your three." — true only while the form waited
       * for three, and it no longer does. One name is enough to send, so this
       * has to work when there is exactly one, and it has to keep the other two
       * open without making them a condition. Three is the invitation; it was
       * never meant to be the price of being heard.
       */
      lead: 'Who should we thank for this? Keep more if you like — up to three.',
      picksLabel: 'Sending',
      /** The app's code-switch, glossed by the English right beside it. */
      why: 'Kina? — why this one?',
      submit: 'Send them to us',
      /**
       * The tray's own way out, and it counts what it is about to send so the
       * visitor knows before they leave the tray. It replaces a form that used
       * to open by itself on the first kept name — which read as "that's
       * enough" while somebody was still choosing.
       */
      open: (count: number) => (count === 1 ? 'Send this one →' : `Send these ${count} →`),
      /** Leaves the sheet without sending. Says what it does, not "Cancel". */
      close: 'Not yet',
      /** The scrim's accessible name — same action, reached by clicking away. */
      closeAria: 'Close without sending',
    },
  },

  /** The invitation on the home page, inside CloseSection. Bishal's own voice. */
  invite: {
    line: 'Sneha and I are expecting a son, and we are still choosing his name.',
    cta: 'Help us name him',
  },

  /**
   * The app needs JavaScript; the names and the form below it do not. Without
   * this line a no-JS visitor meets a page with no explanation of why the thing
   * they were sent a link to is not on it.
   */
  noscript: 'The naming app needs JavaScript. The names, their meanings and the form below all work without it.',

  /**
   * What the app says ABOUT a set of names, as opposed to the names themselves.
   * Whittled to three: the wizard's headings, its "show 50 more", its
   * side-by-side lookup and its two loading captions all went with the
   * editorial page, and the app's own captions live under `app` instead.
   */
  results: {
    emptyAsk: 'Nothing came up for that. Try a meaning — light, water, calm — rather than a spelling.',
  },

  /** DESIGN.md §4. Two of the three words; there is no REPLAY on this page. */
  badge: {
    live: 'LIVE',
    local: 'LOCAL',
    liveCaption: 'A real call, just now.',
  },

  /**
   * Honest failure — and honest is not the same as quietly substituted. The
   * model is what the page asked, so when it does not answer the page says one
   * plain line and offers two things: the same ask again, and the document's
   * own list IF the visitor decides they want it. Nothing appears behind their
   * back, which is what keeps DESIGN.md §4 rule 4 (failure is honest, never
   * blank) satisfied without the page pretending the matcher was the answer.
   *
   * So none of these lines may promise names underneath it any more. Each says
   * what happened and stops; `retry` and `escape` are what to do next.
   */
  failure: {
    modelDown: 'The model didn’t answer.',
    modelSlow: 'The model took too long to answer.',
    modelOff: 'There is no model on this build.',
    /** The same sentence, asked again. The visitor retypes nothing. */
    retry: 'Try again',
    dataDown: 'The full list did not load. Reload the page and it will try again.',
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
    /* Uncalled since the card's toggle was removed — kept because the V/B
       distinction is still the page's own premise and the words for it should
       not have to be rewritten when a control for it lands somewhere with the
       room. See the rail comment in NaamCard.tsx. */
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
    /**
     * Two controls take a name back — the slot and the kept card — and only
     * the slot ever said so ("Slot 1 of three, Samaya. Take it back."). The
     * card announced "Kept Samaya", a state and not an action, so nothing told
     * anyone the press would remove it. Deleting the card's toggle is worse: a
     * pressed KEPT button that does nothing is a dead control, and undoing
     * where you did the thing is the most natural undo there is. Both stay and
     * borrow ONE phrase, so they read as one learned sentence.
     */
    takeBack: 'Take it back.',
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
    heading: 'Send us a name',
    standfirst: 'It comes straight to us. Nothing goes on the wall until we have read it.',
    name: {
      label: 'Your name',
      error: 'Add your name, so we know who to thank.',
      tooLong: `Keep it under ${CAPS.name} characters.`,
    },
    relation: {
      label: 'How do we know you?',
      placeholder: 'Choose one',
      options: NAAM_RELATIONS,
      error: 'Choose one — any one is fine.',
    },
    /**
     * Reachable, and only one way: the send turn arrives when the third slot
     * fills, and taking names back afterwards leaves the form on screen with
     * nothing to send. It names the gesture the tray actually has — Keep, and
     * three slots — not the old grid's "tap Pick on any name".
     */
    picks: {
      empty: 'All three are empty again. Keep a name and it lands here.',
    },
    /**
     * The always-usable half of the form. With JavaScript off nothing can fill
     * the hidden picks array, so without this a no-JS visitor could submit the
     * form and send no name at all — the one thing the page is for.
     */
    names: {
      label: 'Or just type a name',
      helper: 'Any spelling. It does not have to be one from the document.',
      placeholder: 'Bishrut, Saurya',
    },
    reason: {
      /** The fallback's code-switch, glossed by the English beside it. */
      label: 'Kina? — one line is plenty',
      helper: 'Optional, and it is the part we will remember.',
      counter: (used: number, max: number) => `${used}/${max}`,
      tooLong: `A bit long. ${CAPS.reason} characters is the limit.`,
    },
    submit: 'Send',
    sending: 'sending…',
    confirmation: {
      heading: 'Dhanyabad.',
      body: 'It is with us now. Nothing goes on the wall until we have read it.',
      again: 'Send another',
      /**
       * The email went and the moderation queue did not take it. Both halves
       * are true and the visitor is told which, rather than being shown "sent"
       * for a write that failed — /api/naam-submit returns `stored` precisely
       * so this can be honest.
       */
    },
    error: {
      network: 'That did not send. Your three are still here, so try once more.',
      server: 'That broke on our side, not yours. Try again in a minute, or write to bishal@vibeset.ai.',
      rateLimited: 'That is a few too many in a row. Give it a minute.',
    },
  },

  /** Caps, so the tray, the form and the endpoint agree on them. Read, never retyped. */
  limits: CAPS,

  wall: {
    heading: 'What other people sent',
    standfirst: 'The ones we have read and put up.',
    /** The six family favorites the wall ships with. */
    seedLabel: 'Ours, to start',
    /** Sits directly under the six seed cards, so it says what is missing. */
    empty: 'Nothing here yet. Yours would be the first.',
    loading: 'loading the wall…',
    failure: 'The wall did not load. Reload the page and it will come back.',
    entry: (name: string, relation: string) => `${name} · ${relation}`,
    /**
     * The tally, in words. The beads on a leaf are aria-hidden decoration and
     * the ink weight is colour, so this is the only place the count actually
     * exists for a screen reader — and it must never be the bare number, which
     * would be read as "Bhaskara 3" with nothing to say what three is.
     */
    support: (count: number) => (count === 1 ? 'chosen by one person' : `chosen by ${count} people`),
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
