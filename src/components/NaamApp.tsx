import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState, useSyncExternalStore } from 'react'
import type { CSSProperties, FormEvent } from 'react'
import NaamCard from './NaamCard'
import NaamWall, { type WallNote } from './NaamWall'
import { askNaam, failureNote, readAsk, withReasons } from '@/lib/naam/ask'
import { useSpeech } from '@/lib/naam/speech'
import { refinements } from '@/lib/naam/refine'
import type { Prefs } from '@/lib/naam/match'
import { NAAM_DEAL_SMALL } from '@/lib/naam/prompt'
import { OPENING_STEPS, useOpening } from '@/lib/naam/opening'
import { NAAM_COPY, NAAM_RELATIONS } from '@/lib/naam/copy'
import type { NaamMatch } from '@/lib/naam/match'
import { hydrateSound, playCue, setSound, soundOff, soundOn, subscribeSound } from '@/lib/naam/sound'
import { NAAM_SEED_ROWS, NAAM_SEED_VOTES } from '@/lib/naam/seeds'
import { pickStarters, type NaamStarter } from '@/lib/naam/starters'
import {
  getDefaultPreferB,
  getEmptyPicks,
  getPicks,
  getPreferB,
  hydrate,
  loadCoreRows,
  loadThesaurus,
  addOwnPick,
  isOwnPick,
  removePick,
  clearPicks,
  subscribe,
  togglePick,
  PICK_MAX,
  type NaamPick,
} from '@/lib/naam/tray'
import { lanternField } from '@/lib/naam/scene/lanterns'
import { naamPreferredDevanagari, naamPreferredForm, type NaamRow } from '@/types/naam'

/**
 * /naam, the whole app in one island (docs/design/DESIGN.md §1 P1, §4, P8,
 * P10, P11 · docs/design/MOTION.md §5).
 *
 * WHY ONE ISLAND: the rail, the stream, the tray and the composer share a
 * single flex chain — the stream's height is whatever the other three leave —
 * and two islands would need a layout channel between them that does not
 * exist. It is `client:load` rather than `client:idle` because on this page
 * the island *is* the page: an idle callback would leave the only control on
 * screen dead for up to two seconds.
 *
 * WHY IT IS NOT A CHAT WIDGET (P11). NaamGuide's header used to say "no
 * bubbles, no avatar, no assistant framing" and then this page grew turns, so
 * the rule is satisfied rather than abandoned: turns are typographic blocks
 * with .sr-only speaker labels — no bubbles, no avatars, no right-aligned
 * user message, which is the single clearest ChatGPT tell. That is also
 * exactly what the accessible markup produces anyway.
 *
 * THE AGENT LEADS. An ask goes straight to the thinking state, and the next
 * thing on screen is the model's reply. There is no provisional local render
 * for it to correct: this page used to rank the document in the browser, deal
 * that immediately under the LOCAL badge and swap it for the model's answer a
 * second later, and the effect was that the model — which was working, and
 * answering — read as absent, because the matcher's list arrived first and the
 * reply looked like an edit to it. The reply is the event now.
 *
 * THE MODEL STILL NEVER NAMES A NAME. readAsk() in src/lib/naam/ask.ts reads
 * the sentence and builds the pool of ids /api/naam-chat may choose from, and
 * every id that comes back is resolved against the local dataset before it
 * renders. A hallucinated name stays structurally impossible; the grounding
 * was always the pool and never the local render.
 *
 * FAILURE IS HONEST, AND IT IS NOT A SILENT SUBSTITUTION (§4 rule 4 — failure
 * is honest, never blank). When the model does not answer, the page says so in
 * one line and offers two things: the same ask again, and — only if the
 * visitor presses it — the matcher's own list, dealt under the LOCAL badge.
 * Nothing happens behind their back, so LOCAL now marks a list somebody asked
 * for rather than one that was substituted without being mentioned.
 *
 * THE STREAM IS STRUNG, NOT LOGGED. A log grows downward with every past turn
 * still fully expanded, so the longer the page is used the more of it is
 * history shouting at the same volume as the thing you are doing — which is
 * exactly why this screen felt overwhelming three turns in. So the stream has a
 * thread running down it and every turn is a bead on that thread, and a bead
 * you have passed is WORN: it shrinks, desaturates, and compresses to one line.
 * Only the exchange you are on is open. Passing is defined by asking — the cut
 * is the last `you` turn, so the opening invitation stays whole until the
 * visitor says something, and each question closes the one before it rather
 * than closing itself mid-answer.
 *
 * The beads are TYPED. A single repeated dot is what makes a timeline read as a
 * log; the agent speaking, you asking, a hand being dealt, the wall, and the
 * send are five different small silhouettes, all drawn in CSS on one <span>
 * (see .nm-bead in naam.astro). The thread's segment above each new bead draws
 * itself over 500ms — the line REACHING is the progress signal, which a node
 * simply appearing is not.
 *
 * A collapsed turn is a real <button> with a real accessible name, never a
 * clickable <div>: it is the page's only re-opening control and axe is the gate
 * that decides. It carries `bead.reopen` in .sr-only text because its visible
 * label says what the turn was and not what pressing it does. Re-opening keeps
 * the turn in view (`justOpened`), because expanding a block above the scroll
 * position would otherwise shove everything the visitor was reading downward —
 * `overflow-anchor: none` is set on the stream, so nothing compensates for it.
 *
 * NOTHING NAMES THE MALA, here or in copy.ts. It is a counting object with worn
 * beads and a thread, a Nepali or Buddhist visitor recognises it immediately,
 * and everyone else sees a progress thread. A glossary would spend the effect.
 *
 * THE MOTION IS THE SIGNATURE, and its numbers live in MOTION.md §5's /naam
 * set rather than in taste: cards are DEALT (90–120ms stagger, 400–450ms
 * each, emphasized-decelerate, seeded ±3° rotation) and Keep is a FLIP arc
 * into the tray (380–480ms, scale finishing at 80% of the path, opacity never
 * below 0.9, 60–120ms of total stillness on landing, then a 160ms settle
 * while the receiving slot recoils). Two hard rules from the card-game
 * research hold here:
 *
 *   1. ANIMATION NEVER GATES INPUT. togglePick() runs on the click, before a
 *      single frame is scheduled, so a second Keep is startable at frame 1 of
 *      the first one's flight. The arc is decoration over state that has
 *      already changed.
 *   2. UN-KEEPING IS AS ANIMATED AS KEEPING. If removal were a small grey ✕
 *      the collection would become a form.
 *
 * Every animation is Web Animations API rather than CSS class-toggling
 * because each one needs measured geometry, and every one is skipped outright
 * under `prefers-reduced-motion` — the state change still happens, instantly.
 *
 * ACCESSIBILITY, the part most likely to fail the gate (P10):
 *   · The stream is <ol role="list" tabIndex={0} aria-label>. role="list" is
 *     explicit because `list-style: none` strips list semantics in Safari, and
 *     tabIndex is required on a scrollable region (axe scrollable-region-
 *     focusable). It is not a <div tabIndex aria-label>, which would be an
 *     aria-prohibited-attr violation.
 *   · NOT role="log". A log region announces everything appended to it,
 *     including the visitor's own echoed message. Instead there is ONE
 *     persistent .sr-only live region, created EMPTY and written only when the
 *     agent says something.
 *   · Each turn carries an .sr-only speaker label.
 *   · Auto-scroll sticks to the bottom only while the visitor is within 100px
 *     of it; past that it stops and offers "jump to the latest" rather than
 *     yanking them down.
 *
 * Teardown: one AbortController per mount, aborted on `astro:before-swap`.
 * Every listener, every fetch and every timer rides its signal.
 *
 * Every visible string comes from src/lib/naam/copy.ts and every match reason
 * from match.ts. Nothing here writes prose.
 */

const C = NAAM_COPY

/**
 * The two badge words this page is allowed (§4). Never a third.
 *
 * LIVE is the ordinary one now. LOCAL is reachable only through the escape
 * button under a failure line, so it labels a deck the visitor asked the
 * matcher for rather than one that quietly replaced the model's.
 */
const BADGE = {
  live: C.badge.live.toLowerCase(),
  local: C.badge.local.toLowerCase(),
} as const

type Badge = keyof typeof BADGE

/* — the motion numbers, in one place so they can be read against MOTION.md — */
const FLIGHT_MS = 430
/** Vlambeer's hitstop: total stillness is what makes the landing an event. */
const HITSTOP_MS = 90
const SETTLE_MS = 160
const RECOIL_MS = 90
const SLOT_SETTLE_MS = 220
const RELEASE_MS = 260
/** Long enough to read the third card as landed before the form arrives. */
/** Ambient motion stays suspended this long after the last keystroke. */
const CALM_MS = 800
/**
 * How long each named step of the wait holds. Long enough to read a short
 * lowercase line without chasing it; much under this and the steps read as
 * flicker and stop being discrete, which is the whole mechanism.
 *
 * MEASURED, so nobody has to guess later: on this dev server the ask comes
 * back in 2.7–3.8s, so a visitor sees step 1 and step 2 and the turn is
 * replaced by the hand before step 3. That is the intended shape — the walk
 * covers the wait it is given and stops — but it does mean the last two lines
 * only appear on a slow round-trip. If they should always be seen, the fix is
 * this number, not the list: four steps need 4×this to all show.
 */
const ASKING_STEP_MS = 1400
const DEAL_EASE = 'cubic-bezier(0.05, 0.7, 0.1, 1)'
const SLOT_EASE = 'cubic-bezier(0.34, 1.56, 0.64, 1)'

type Turn =
  /** `lead` is the biggest type on the page; `quiet` is the smallest. */
  | { id: string; kind: 'agent'; text: string; lead?: boolean; quiet?: boolean }
  | { id: string; kind: 'you'; text: string }
  | { id: string; kind: 'family' }
  | { id: string; kind: 'starters' }
  | { id: string; kind: 'thinking'; caption: string }
  | { id: string; kind: 'names'; matches: readonly NaamMatch[]; badge: Badge; note: string }
  /**
   * The ask produced no names, and the turn says why and what to do. TWO
   * things produce it, which is why it is not called `failed`:
   *
   *   the model did not answer   `note` is the honest line, `retry` is true
   *   it answered and named none `note` is empty — the reply directly above
   *                              already said so in the model's own words, and
   *                              Try again would be offering to redo something
   *                              that worked. Only the escape shows.
   *
   * The second case is common rather than exotic: ask for a quality the
   * document does not have and the model correctly says so and picks nothing.
   * The old local-first render hid that behind eight matcher rows; without
   * this turn it would be a dead end.
   *
   * It carries the ask verbatim because both buttons need it — Try again
   * re-runs that exact sentence and the escape hands it to the matcher — so
   * the visitor never retypes anything.
   */
  | { id: string; kind: 'stuck'; ask: string; note: string; retry: boolean }
  | { id: string; kind: 'form' }
  | { id: string; kind: 'sent'; text: string }

/**
 * One approved suggestion, as /api/naam-wall serves it. Every field is a
 * stranger's typed text and reaches the DOM through JSX interpolation only —
 * never innerHTML — so a payload in `from` arrives as literal characters. The
 * names are re-resolved against the local dataset rather than trusted from the
 * record, so the Devanagari on screen is always the document's.
 */
interface WallEntry {
  id: string
  from: string
  relation: string
  picks: { id: string; spelling: string }[]
}

/**
 * Dust in the lamplight. Fixed positions and irregular periods, declared once
 * at module scope so they are identical on the server and on every re-render —
 * /naam is prerendered and React 19 compares the hydrated tree literally, so a
 * Math.random() here would be a hydration mismatch, and a re-roll on every
 * render would make the specks jump. Eight is enough to read as air; more reads
 * as weather.
 */
const MOTES = [
  { left: 24, bottom: 8, dur: 19, delay: 0 },
  { left: 38, bottom: 2, dur: 24, delay: 3.5 },
  { left: 47, bottom: 12, dur: 17, delay: 7 },
  { left: 56, bottom: 4, dur: 27, delay: 1.5 },
  { left: 63, bottom: 10, dur: 21, delay: 9 },
  { left: 71, bottom: 1, dur: 23, delay: 5 },
  { left: 33, bottom: 6, dur: 29, delay: 12 },
  { left: 52, bottom: 14, dur: 20, delay: 15 },
] as const

let turnSeq = 0
function nextId(): string {
  turnSeq += 1
  return `t${turnSeq}`
}

function reducedMotion(): boolean {
  return typeof window !== 'undefined' && window.matchMedia('(prefers-reduced-motion: reduce)').matches
}

/**
 * ±3°, seeded off the row id so a card lands at the same angle every time it
 * is dealt. A random rotation re-rolls on every re-render, which reads as a
 * glitch rather than as a hand.
 */
function seededTilt(id: string): number {
  let hash = 7
  for (let i = 0; i < id.length; i += 1) hash = (hash * 31 + id.charCodeAt(i)) % 9973
  return Math.round(((hash % 61) - 30) / 10)
}

/**
 * Which bead a turn is strung as. FIVE SILHOUETTES, NOT ONE DOT: the repeated
 * bullet is the single thing that makes a vertical sequence read as a log, and
 * a mala is not one shape repeated either — it has a counting bead, a marker
 * bead and a guru bead, and you find your place on it by feel. Drawn in CSS off
 * this attribute (.nm-bead[data-bead] in naam.astro), so a bead costs one empty
 * <span> and no SVG.
 *
 *   said   the agent spoke — a plain round bead, the ordinary one
 *   asked  you asked — a ring, open, because a question is
 *   dealt  names came — a facet, turned 45°, the one that catches light
 *   note   the wall — square-ish, the shape of the paper it summarises
 *   guru   the send — the large bead a mala is counted from and finished at
 */
function beadOf(turn: Turn): string {
  switch (turn.kind) {
    case 'you':
      return 'asked'
    case 'names':
      return 'dealt'
    case 'family':
      return 'note'
    case 'form':
    case 'sent':
      return 'guru'
    case 'stuck':
      return 'stuck'
    default:
      return 'said'
  }
}

/**
 * The one line a passed turn leaves behind. A turn that SPOKE is summarised by
 * its own words — clamped to one line in CSS rather than truncated here, so
 * nothing is lost when it is opened again — and everything else takes a label
 * from copy.ts. This function writes no prose.
 */
function summaryOf(turn: Turn): string {
  switch (turn.kind) {
    case 'agent':
    case 'you':
      return turn.text
    case 'sent':
      return C.app.bead.sent
    case 'family':
      return C.app.familyLead
    case 'starters':
      return C.app.bead.starters
    case 'names':
      return turn.matches.length > 0 ? C.app.bead.names(turn.matches.length) : C.results.emptyAsk
    case 'stuck':
      return turn.note || C.app.bead.stuck
    case 'form':
      return C.app.bead.form
    // `thinking` is never collapsed — it is transient and always the last turn
    // — but the switch has to be total, and its caption is the honest summary.
    default:
      return turn.caption
  }
}

/**
 * The flying name. Built with createElement + textContent — never innerHTML,
 * and never a clone of the card, because a clone lands at the card's aspect
 * ratio on top of a slot with a different one and the swap is visible. This
 * is sized and styled as the FILLED SLOT, positioned on the slot, and flown
 * backwards from the card: it therefore ends exactly on top of what the slot
 * already renders, so removing it is invisible.
 */
function nameToken(deva: string, latin: string): HTMLElement {
  const token = document.createElement('div')
  token.className = 'nm-token nm-token--flying'
  token.setAttribute('aria-hidden', 'true')
  const script = document.createElement('span')
  script.className = 'nm-token-deva'
  script.lang = 'sa-Deva'
  script.textContent = deva
  const latinEl = document.createElement('span')
  latinEl.className = 'nm-token-latin'
  latinEl.textContent = latin
  token.append(script, latinEl)
  return token
}

function placeOn(token: HTMLElement, rect: DOMRect): void {
  token.style.left = `${rect.left}px`
  token.style.top = `${rect.top}px`
  token.style.width = `${rect.width}px`
  token.style.height = `${rect.height}px`
}

export interface NaamAppProps {
  /**
   * The one family seed that is a real row, resolved from the dataset at build
   * time by naam.astro (which throws if it has gone missing). Passed in rather
   * than looked up here so the family turn is complete on the first frame
   * instead of appearing 1.2 MB later.
   */
  seed: NaamRow
  /**
   * Real, cited, gold rows the page can vouch for — four per letter, chosen at
   * prerender by naam.astro from the same `isNaamGold` filter the no-JS
   * fallback already used. They were being built and then hidden from every
   * visitor who had JavaScript.
   */
  arrival?: readonly NaamRow[]
}

export default function NaamApp({ seed, arrival }: NaamAppProps) {
  const [rows, setRows] = useState<NaamRow[] | null>(null)
  /** Approved suggestions from /api/naam-wall. Empty until one is approved. */
  const [wall, setWall] = useState<readonly WallEntry[]>([])
  const [dataFailed, setDataFailed] = useState(false)
  const [turns, setTurns] = useState<Turn[]>(() => [
    { id: 'greeting', kind: 'agent', text: C.app.greeting, lead: true },
    // TWO TURNS, NOT FOUR. The nwaran explanation and the "6,715 names, out of
    // the Vedas and the Sutras" line were cut on the owner's instruction. What
    // is left is who is asking and what they are asking for — the greeting and
    // the invitation, which were always the two halves of one thought.
    //
    // The provenance did not leave the page with them: every card still carries
    // the document's own gloss in mono, and the no-JS fallback still opens with
    // the count, the corpora and the letters (C.hero.*), which is where that
    // sentence lived before it was ever a turn.
    { id: 'invitation', kind: 'agent', text: C.app.invitation },
  ])
  const [ask, setAsk] = useState('')
  const [asking, setAsking] = useState(false)
  /* THE DIYO IS GONE. It was the page's one permanently-alive element and
     the argument for it was continuity — something burning whether or not you
     touch anything. In place it did not earn that: at tray size it is a small
     SVG running two offset keyframe animations forever, it was the only thing
     still animating on a screen the visitor had stopped looking at, and the
     sky above it already has nine lit lanterns doing the same job at the scale
     the page actually reads at. What it cost is measurable — see the flame's
     share of the frame in the perf notes on this commit.

     Two jobs went with it and both had somewhere better to go: it was the
     thinking state, which the breathing caption already says in words, and it
     was the ember origin, which is now the slot that just filled. */
  /**
   * The rail badge is a CLAIM, not a status light: before anything has been
   * asked there is nothing to claim, so it starts empty and the only word it
   * can ever take is LIVE. It used to initialise to LOCAL — the page therefore
   * announced that the matcher had answered before anybody had asked it a
   * question.
   */
  const [source, setSource] = useState<'live' | ''>('')
  const [announce, setAnnounce] = useState('')
  const [pinned, setPinned] = useState(true)
  const [calm, setCalm] = useState(false)
  const [formShown, setFormShown] = useState(false)
  const [sending, setSending] = useState(false)
  const [sendNote, setSendNote] = useState('')
  /**
   * A name they typed rather than kept. Controlled, because Send has to know
   * whether it is empty: with no picks AND no typed name there is nothing to
   * send, and with either one there is.
   */
  /** Which empty slot is currently accepting a typed name, if any. */
  const [owning, setOwning] = useState<number | null>(null)
  /**
   * CAN THIS BROWSER ACTUALLY DRAW THE FOLDED HANDS?
   *
   * नमस्ते is the first thing anyone reads here, and a system with no emoji
   * font puts a hollow box in front of it — which is worse than no gesture at
   * all. Plenty of Linux desktops ship without one; the machine this was built
   * on has 103 fonts and not one of them has a glyph for it, which is how the
   * tofu got noticed in the first place.
   *
   * Width heuristics are unreliable, so this asks the honest question: emoji
   * fonts are COLOUR fonts, tofu is not. Paint it and look for a pixel whose
   * channels disagree.
   *
   * Starts false and is only ever turned on after mount, which also keeps the
   * server and the first client render identical — a glyph that appears during
   * hydration is a mismatch, and this way it is simply an enhancement that
   * arrives.
   */
  const [canEmoji, setCanEmoji] = useState(false)
  /**
   * Passed turns the visitor has pressed back open. It is never cleared: a bead
   * they chose to re-open stays open, because closing it again on the next ask
   * would be the page overruling a decision it had just been asked to make.
   */
  const [reopened, setReopened] = useState<ReadonlySet<string>>(() => new Set())

  const mountRef = useRef<AbortController | null>(null)
  /**
   * ── LIVE MEANS HYDRATED, NOT LOADED ────────────────────────────────────
   *
   * The composer used to be disabled until the 6,715-name dataset landed,
   * which made the only input on the page dead for 5.6s on fast 4G and past
   * 22s on slow 4G. Ungating it entirely was worse in a way the first
   * measurement missed: this island is server-rendered, so an input with no
   * `disabled` is in the FIRST PAINTED HTML — live-looking, with no
   * JavaScript attached. Typed at 218ms, the keystrokes went into a DOM node
   * React had never seen and Enter did nothing at all. A box that swallows a
   * sentence is worse than a box that says it is not ready.
   *
   * So the gate moved to the thing that actually decides whether typing works:
   * has this component mounted in a browser. `false` on the server and on the
   * first client render, `true` one effect later — which is when the handlers
   * exist. The DATASET no longer gates anything; a sentence typed before it
   * arrives waits inside runAsk.
   */
  const [live, setLive] = useState(false)
  useEffect(() => setLive(true), [])
  const shellRef = useRef<HTMLDivElement | null>(null)
  const valleyRef = useRef<HTMLCanvasElement | null>(null)
  /** The scene loads ~260ms after mount, long after the first notes exist, so
   *  the count is kept in a ref for it to read on arrival — and the handle is
   *  kept so later arrivals can be pushed straight through. */
  const lampCountRef = useRef(0)
  /** Support counts for the sky, held for a scene that has not loaded yet. */
  const lanternCountsRef = useRef<readonly number[]>([])
  /** Parallel note keys, so the field can keep each lantern's state. */
  const lanternKeysRef = useRef<readonly string[]>([])
  const valleyHandleRef = useRef<{
    setLamps(n: number): void
    setLanterns(counts: readonly number[]): void
    setLanternKeys(keys: readonly string[]): void
    setStill(on: boolean): void
  } | null>(null)
  /** React renders this empty and never diffs its children, so it is the one
      safe place to park an imperatively created flying name. */
  const fxRef = useRef<HTMLDivElement | null>(null)
  const streamRef = useRef<HTMLOListElement | null>(null)
  const inputRef = useRef<HTMLInputElement | null>(null)
  const slotRefs = useRef<(HTMLElement | null)[]>([])
  const calmTimer = useRef(0)
  const timers = useRef<Set<number>>(new Set())
  /** The turn the visitor just re-opened, so the layout effect can hold it. */
  const justOpened = useRef<string | null>(null)

  const picks = useSyncExternalStore(subscribe, getPicks, getEmptyPicks)
  const preferB = useSyncExternalStore(subscribe, getPreferB, getDefaultPreferB)
  const pickedIds = useMemo(() => new Set(picks.map((p) => p.id)), [picks])

  const audible = useSyncExternalStore(subscribeSound, soundOn, soundOff)

  /** A timer that cannot outlive the mount. */
  const later = useCallback((fn: () => void, ms: number) => {
    const id = window.setTimeout(() => {
      timers.current.delete(id)
      fn()
    }, ms)
    timers.current.add(id)
  }, [])

  /* — the valley ——————————————————————————————————————————————————— */

  /**
   * ── THE SKY HAS A STOP ON IT ──────────────────────────────────────────────
   *
   * WCAG 2.2.2 (Pause, Stop, Hide) is Level A and it is about MOVING content
   * that starts on its own and does not end: the flags, the lanterns, the dust,
   * the lamps coming on across the valley. prefers-reduced-motion is honoured
   * everywhere on this page and it does not discharge 2.2.2 — it is a setting
   * on somebody's operating system, and this page is read by grandparents on
   * borrowed laptops who have never opened that panel and never will.
   *
   * SEEDED IN A LAYOUT EFFECT, NOT IN useState's INITIALISER.
   *
   * `useState(() => reducedMotion())` is the obvious way to have the control
   * agree with the scene from the first frame, and it was measured doing real
   * damage: this island is client:load, so it is server-rendered, and the
   * server has no media query. Under Playwright with reducedMotion:'reduce'
   * the server sent "sky moving" and the client wanted "sky still" — React 19
   * reported `Hydration failed because the server rendered text didn't match
   * the client. As a result this tree will be regenerated on the client`, which
   * throws away the whole hydrated app and rebuilds it, for precisely the
   * visitor this control exists for.
   *
   * useLayoutEffect runs after hydration commits and BEFORE the browser paints,
   * so nobody ever sees the wrong word — the control still agrees with the
   * scene from the first frame anyone looks at, and the markup matches.
   */
  const [stilled, setStilled] = useState(false)
  useLayoutEffect(() => {
    if (reducedMotion()) setStilled(true)
  }, [])
  /** The scene mounts ~260ms in and cannot be rebuilt to change a boolean, so
   *  the current value has to be readable from inside that async path. */
  const stilledRef = useRef(stilled)
  stilledRef.current = stilled

  useEffect(() => {
    valleyHandleRef.current?.setStill(stilled)
  }, [stilled])

  /**
   * PixiJS is ~117 kB gzipped — larger than every other line of JavaScript on
   * this site put together — so it is imported dynamically, after the page is
   * usable, and never on the critical path. If it never arrives, the CSS
   * gradient behind the canvas is the page and nothing is missing but weather.
   *
   * Under prefers-reduced-motion the scene is built and drawn ONCE. Not
   * skipped: the valley is the room this conversation happens in, and removing
   * it would change what the page is rather than how it moves. What stops is
   * the parallax.
   */
  useEffect(() => {
    const canvas = valleyRef.current
    if (!canvas) return
    let handle: {
      setLamps(n: number): void
      setLanterns(counts: readonly number[]): void
      setLanternKeys(keys: readonly string[]): void
      setStill(on: boolean): void
      resize(): void
      destroy(): void
    } | null = null
    let cancelled = false

    const start = window.setTimeout(async () => {
      try {
        const { createValley } = await import('@/lib/naam/scene/valley')
        if (cancelled) return
        /**
         * READ FROM THE REF, NOT FROM THE MEDIA QUERY. The scene arrives ~260ms
         * after mount and this effect never re-runs — rebuilding the valley to
         * change one boolean would tear down every lantern. So the state the
         * control holds is what the scene is born with, and the control's own
         * effect below cannot help a handle that did not exist when it ran.
         */
        handle = await createValley({ canvas, still: stilledRef.current })
        if (cancelled) {
          handle.destroy()
          return
        }
        valleyHandleRef.current = handle
        // And once more, in case it was pressed while the chunk was in flight.
        handle.setStill(stilledRef.current)
        handle.setLamps(lampCountRef.current)
        handle.setLanternKeys(lanternKeysRef.current)
        handle.setLanterns(lanternCountsRef.current)
      } catch {
        // WebGL blocked, context lost, chunk failed — all the same outcome, and
        // it is not an error state. The gradient below is a complete page.
      }
    }, 260)

    // No pointer listener: the valley does not pan. What moves is the flags and
    // the birds, on their own clock — a window you look through, not a widget
    // that follows the cursor.
    /**
     * A ResizeObserver ON THE CANVAS, not only a window listener.
     *
     * Pixi's docs are explicit that world and screen coordinates diverge "when
     * the canvas is stretched (e.g. via CSS) or rendered at a different
     * resolution than its display size" — and this page depends on those two
     * agreeing exactly: lampSpots() places the glow in canvas coordinates and
     * the DOM button for the same lamp in percentages of the element box. Any
     * divergence and the hover target sits beside the light rather than on it.
     *
     * Today a window listener happens to catch every case, because the canvas
     * is pinned to a viewport-sized shell — verified, dealing nine cards leaves
     * it at 1280x841 with a matching buffer. That is a property of the current
     * layout rather than a guarantee, and the failure mode is silent
     * misalignment rather than an error, which is the kind that ships.
     */
    const onResize = () => handle?.resize()
    const observer = typeof ResizeObserver === 'function' ? new ResizeObserver(onResize) : null
    observer?.observe(canvas)
    // Kept as well: a device-pixel-ratio change (dragging to another monitor)
    // resizes no element at all.
    window.addEventListener('resize', onResize, { passive: true })

    return () => {
      cancelled = true
      window.clearTimeout(start)
      observer?.disconnect()
      window.removeEventListener('resize', onResize)
      valleyHandleRef.current = null
      handle?.destroy()
    }
  }, [])

  /**
   * ── ONE THREAD, AND NO SIMULATION UNDER IT ──────────────────────────────
   *
   * The mala was a small physics canvas hung inside the transcript's scroller,
   * and it was the wrong object in three ways at once.
   *
   * IT COULD NOT CROSS THE BOUNDARY. The canvas lives in the scroller so it
   * scrolls with the text for free — and therefore stops where the scroller
   * stops. The cards below got a CSS thread instead, so the gutter held a
   * hanging arc with varying beads above the divider and a ruler-straight
   * chain below it. Filmed at 3x, they are plainly two different objects.
   *
   * IT ARRIVED LATE AND REPLACED SOMETHING ELSE. The CSS cord draws on the
   * first frame; the canvas mounts around 650ms and takes over. Measured:
   * `data-rope` false at 400ms, true at 700ms. So the strand a visitor sees
   * while the page is still coming up is not the strand they end up with.
   *
   * AND IT KINKED. The rope's polyline resolved into visible zig-zags — slack
   * rope reads as an error rather than as weight.
   *
   * What the object actually is: a thread whose beads mark the turns. That
   * needs no simulation. One element spans the whole gutter, drawn the same
   * way from the first paint, and the marker beads — which are DOM, inside the
   * turns — slide along it as the transcript scrolls, which is what counting a
   * mala looks like.
   *
   * The only thing measured here is where the thread ENDS: the bottom of the
   * hand, which is the last band the strand belongs to. Below that sit the
   * tray and the composer, and a thread running past them into the valley was
   * a real bug once already.
   */
  useLayoutEffect(() => {
    const shell = shellRef.current
    if (!shell) return undefined
    /**
     * X COMES FROM A BEAD, NOT FROM THE INSET ARITHMETIC.
     *
     * The thread has to sit exactly under the marker beads, and those are
     * placed by the turns' own rail column — which is centred on a 720px
     * measure, so its x moves as the window widens. Re-deriving that from the
     * clamp and the rail width worked at 1440 and drifted everywhere else;
     * reading one rendered bead cannot drift, because it IS the thing being
     * aligned to.
     */
    const measure = () => {
      const box = shell.getBoundingClientRect()
      const wrap = shell.querySelector<HTMLElement>('.nm-streamwrap')
      const hand = shell.querySelector<HTMLElement>('.nm-hand')
      const bead = [...shell.querySelectorAll<HTMLElement>('.nm-bead')].find(
        (el) => el.getBoundingClientRect().height > 0,
      )
      if (!wrap) return
      const top = wrap.getBoundingClientRect().top - box.top
      const end = (hand ?? wrap).getBoundingClientRect().bottom - box.top
      shell.style.setProperty('--nm-thread-top', `${Math.round(top)}px`)
      shell.style.setProperty('--nm-thread-h', `${Math.max(0, Math.round(end - top))}px`)
      if (bead) {
        const r = bead.getBoundingClientRect()
        shell.style.setProperty('--nm-thread-x', `${Math.round(r.left + r.width / 2 - box.left)}px`)
      }
    }
    measure()
    const ro = typeof ResizeObserver === 'function' ? new ResizeObserver(measure) : null
    ro?.observe(shell)
    const band = shell.querySelector<HTMLElement>('.nm-hand')
    if (band) ro?.observe(band)
    window.addEventListener('resize', measure)
    return () => {
      ro?.disconnect()
      window.removeEventListener('resize', measure)
    }
  /* `turns.length` covers a new turn arriving; the ResizeObserver covers
     everything else, including the bands resizing on the first ask. */
  }, [turns.length])

  /* — mount ————————————————————————————————————————————————————————— */

  useEffect(() => {
    const ac = new AbortController()
    mountRef.current = ac
    const { signal } = ac
    document.addEventListener('astro:before-swap', () => ac.abort(), { signal })

    hydrate()

    try {
      const canvas = document.createElement('canvas')
      canvas.width = 24
      canvas.height = 24
      const ctx = canvas.getContext('2d', { willReadFrequently: false })
      if (ctx) {
        ctx.font = '20px sans-serif'
        ctx.textBaseline = 'top'
        ctx.fillText(C.app.greetingGlyph, 0, 0)
        const { data } = ctx.getImageData(0, 0, 24, 24)
        for (let i = 0; i < data.length; i += 4) {
          if (data[i + 3] > 0 && (data[i] !== data[i + 1] || data[i + 1] !== data[i + 2])) {
            setCanEmoji(true)
            break
          }
        }
      }
    } catch {
      /* a tainted or unavailable canvas just means no gesture, which is fine */
    }

    // Reads the stored preference only. Nothing can make a sound until the
    // toggle is pressed, which is also the gesture the autoplay policy wants —
    // so a returning visitor with sound on still hears nothing until they act.
    hydrateSound()

    /**
     * Fire-and-forget, deliberately unawaited and never surfaced. It lets
     * "brave" reach Shaura and "calm" reach Shamatha in the FIRST pool, which
     * is what saves the agent a second Bedrock round trip. If it is slow or
     * absent, readAsk() reads `{}` and retrieval is exactly what it was before
     * the table existed — so there is nothing here worth blocking a render on,
     * and nothing worth telling anyone about.
     */
    loadThesaurus()

    // Fetched here rather than behind onIdle(): the composer cannot compute a
    // pool without the dataset, and on this page the composer is the only
    // affordance on screen. naam.astro preloads the same URL.
    loadCoreRows()
      .then((loaded) => {
        if (!signal.aborted) setRows(loaded)
      })
      .catch(() => {
        if (!signal.aborted) setDataFailed(true)
      })

    /**
     * What other people have already sent, once Bishal has approved it. Without
     * this the moderation flow is a dead end: a suggestion is stored, emailed,
     * approved into wall.json — and displayed nowhere. The endpoint answers
     * `{ entries: [] }` on any failure rather than erroring, so a silent catch
     * is the whole error path; an empty wall and an unreachable one look the
     * same on purpose, because neither is worth a message to a visitor who came
     * to name a baby.
     */
    fetch('/api/naam-wall', { signal })
      .then((res) => (res.ok ? res.json() : { entries: [] }))
      .then((data: { entries?: WallEntry[] }) => {
        if (!signal.aborted && Array.isArray(data.entries)) setWall(data.entries)
      })
      .catch(() => {
        /* the family strip still has its seeds */
      })

    /**
     * iOS never resizes the layout viewport for the soft keyboard, so a
     * 100svh app would put the composer underneath it. visualViewport is the
     * only thing that reports the covered height; Android is handled by
     * `interactive-widget=resizes-content` in the layout's viewport meta and
     * reports 0 here.
     */
    const vv = window.visualViewport
    if (vv) {
      const syncInset = () => {
        const covered = Math.max(0, window.innerHeight - vv.height - vv.offsetTop)
        document.documentElement.style.setProperty('--kb-inset', `${Math.round(covered)}px`)
      }
      vv.addEventListener('resize', syncInset, { signal })
      vv.addEventListener('scroll', syncInset, { signal })
      syncInset()
    }

    const held = timers.current
    signal.addEventListener('abort', () => {
      for (const id of held) window.clearTimeout(id)
      held.clear()
      window.clearTimeout(calmTimer.current)
      document.documentElement.style.removeProperty('--kb-inset')
    })

    return () => ac.abort()
  }, [])

  /* — the stream sticks to the bottom, but only if you are already there —— */

  /**
   * Has the visitor said anything yet? Sticking to the bottom is correct once
   * there is a conversation to follow and WRONG before there is one. The
   * opening turns — the invitation, the wall, the chips — are taller than the
   * stream, so a page that scrolled on mount opened halfway down its own
   * greeting: measured at 1280×800, "नमस्ते. Sneha and Bishal are" was sliced
   * across the middle by the header and the first thing fully on screen was
   * the starter chips. The invitation is the one thing a visitor must read, so
   * the stream stays at the top until they have actually asked something.
   */
  const asked = useMemo(() => turns.some((turn) => turn.kind === 'you'), [turns])

  /** The sentence the visitor last sent, and what it parsed to. */
  const [lastAsk, setLastAsk] = useState<{ text: string; prefs: Prefs } | null>(null)

  /**
   * SPEECH, AND IT ONLY EVER FILLS THE BOX. It calls setAsk and stops there —
   * submitting on a final transcript would act on a sentence nobody has read,
   * and the measurement above says a mis-heard THEME word silently returns a
   * different set rather than failing. Visible and editable is the whole
   * mitigation.
   *
   * Capability-detected at mount, not at module scope: this component
   * server-renders, and `window` is not there when it does.
   */
  const speech = useSpeech((text) => {
    setAsk(text)
    opening.rush()
    inputRef.current?.focus()
  })

  /** The arrival shelf as matches, so it renders through the same dealCards
      path as a real hand. score/reasons are empty because nothing scored it —
      it is the document's own selection, not an answer to a question. */
  const arrivalHand = useMemo<readonly NaamMatch[]>(
    () => (arrival ?? []).map((row) => ({ row, score: 0, reasons: [] })),
    [arrival],
  )

  /**
   * THE OPENING ARRIVES ONE BLOCK AT A TIME. Owned by a hook rather than by
   * three more useState here — see src/lib/naam/opening.ts for why the CSS
   * delays it replaces could not express "finish early but still finish".
   *
   * CAPTURED ONCE, NOT DERIVED FROM `asked`. Passing `!asked` looked right and
   * reintroduced the exact bug this replaced: `asked` flips true the instant
   * somebody submits, the hook was told to stop staging, and every remaining
   * block un-waited on the same frame. Measured — interrupting after one block
   * jumped straight to all of them in 2ms, which is the pop, not a rush.
   *
   * What the hook actually needs to know is whether this page STARTED on the
   * invitation, which is a fact about the first frame and cannot change
   * afterwards. A returning visitor mid-conversation still skips it.
   */
  const [staged] = useState(() => turns.every((turn) => turn.kind !== 'you'))
  const opening = useOpening(staged)

  useLayoutEffect(() => {
    const el = streamRef.current
    if (!el || !pinned || !asked) return
    el.scrollTo({ top: el.scrollHeight, behavior: reducedMotion() ? 'auto' : 'smooth' })
  }, [turns, pinned, asked])

  /**
   * The scroll above fires when a TURN arrives. The thing that moves the
   * bottom of the transcript, though, is the HAND arriving underneath it — on
   * a phone the cards take their height out of the stream, so the box shrinks
   * a moment after the last effect has already run and the transcript is left
   * sitting short of its own end. Measured at 412x915: the last line stopped
   * 164px above the cards on some runs and 53px on others, which is what a
   * race looks like from the outside.
   *
   * So the pin is re-asserted whenever the box or its contents change size,
   * not only when the conversation does. `auto`, never smooth: this is a
   * correction to a layout that has already happened, and animating it would
   * read as the page scrolling itself for no reason.
   */
  useEffect(() => {
    const el = streamRef.current
    if (!el || typeof ResizeObserver === 'undefined') return
    const settle = () => {
      /**
       * AND NOT DURING THE OPENING. The scroll effect above this one is gated
       * on `asked`; this one was not, so every block of the opening resized the
       * box and re-pinned it to the bottom. With a fourth block that puts the
       * greeting at a CONSTANT top:-110px on a phone — never scrolling away and
       * never on screen, so नमस्ते and the reason the page exists are unread at
       * every moment of the entry.
       *
       * There is nothing newer to follow before the first ask: the opening is a
       * sequence that reads top-down. Pinning is right after it, wrong during.
       */
      if (!pinned || !asked) return
      el.scrollTop = el.scrollHeight
    }
    const ro = new ResizeObserver(settle)
    ro.observe(el)
    for (const child of el.children) ro.observe(child)
    return () => ro.disconnect()
  }, [pinned, turns, asked])

  /**
   * ONE SETTLE, WHEN THE OPENING FINISHES — the other half of the fix above.
   * Not pinning during the opening puts नमस्ते on screen and costs the
   * invitation: four blocks do not fit a phone viewport, so "What kind of name
   * are you looking for?" ends up a fifth visible at rest.
   *
   * Both are satisfiable because they are not simultaneous. Each block is fully
   * on screen as it lands, the greeting alone for the first beat and readable
   * for about four seconds; when the last block arrives the page settles onto
   * the ask, which is what it should be resting on. Smooth, unlike the
   * ResizeObserver's correction: this one IS the page moving, deliberately.
   */
  useEffect(() => {
    if (!opening.done || asked) return
    const el = streamRef.current
    if (!el) return
    /**
     * AS FAR AS THE INVITATION, AND NOT ONE PIXEL FURTHER.
     *
     * This scrolled to `scrollHeight`, and that was right when the opening was
     * four blocks: they could not fit, so resting on the last of them was the
     * best available answer. The opening is two blocks now, and at rest on a
     * 412x915 phone they fit almost exactly — greeting 24->187, invitation
     * 187->345, in a 348px band. Scrolling to the end no longer reveals the
     * invitation; it scrolls PAST it to uncover the wall row underneath, which
     * is a third .nm-turn on this layout and 103px tall. Measured: scrollTop
     * 118 of 118, and the first words on the phone were "name before he
     * arrives, and we would like to find it together" — नमस्ते, and the fact
     * that this page is asking on behalf of two named people, both above the
     * top edge and never read.
     *
     * `[data-turn]` is the opening's own turns and nothing else: the wall row
     * carries no such attribute, which is exactly the distinction that was
     * missing. So the page settles onto the bottom of the last thing it SAID,
     * and if that already fits it does not move at all.
     */
    const spoken = el.querySelectorAll<HTMLElement>('[data-turn]')
    const last = spoken[spoken.length - 1]
    if (!last) return
    const target = Math.max(0, last.offsetTop + last.offsetHeight - el.clientHeight)
    if (Math.abs(target - el.scrollTop) < 2) return
    el.scrollTo({ top: target, behavior: reducedMotion() ? 'auto' : 'smooth' })
  }, [opening.done, asked])

  /**
   * AND ALWAYS AFTER A SEND, whether or not anybody typed.
   *
   * Every other scroll on this page is gated on `asked`, which is true only
   * once there is a 'you' turn. A visitor who keeps three names straight off
   * the arrival shelf and sends them never types anything — so `asked` stays
   * false, the settle never ran, and their "Dhanyabad." landed 135px below the
   * fold. Measured: stream 345px tall, scrollTop 382 of a 862px scrollHeight,
   * the confirmation ending 15px past the bottom edge.
   *
   * That is the most likely path for the reader this page was built for, and it
   * was the one path where the thank-you could not be read.
   */
  const lastKind = turns[turns.length - 1]?.kind
  useEffect(() => {
    if (lastKind !== 'sent') return
    const el = streamRef.current
    el?.scrollTo({ top: el.scrollHeight, behavior: reducedMotion() ? 'auto' : 'smooth' })
  }, [lastKind])

  const onStreamScroll = useCallback(() => {
    const el = streamRef.current
    if (!el) return
    setPinned(el.scrollHeight - el.scrollTop - el.clientHeight < 100)
  }, [])

  const jumpToLatest = useCallback(() => {
    const el = streamRef.current
    setPinned(true)
    el?.scrollTo({ top: el.scrollHeight, behavior: reducedMotion() ? 'auto' : 'smooth' })
  }, [])

  /* — a passed bead, pressed back open ————————————————————————————————— */

  const reopen = useCallback((id: string) => {
    justOpened.current = id
    setReopened((prev) => new Set(prev).add(id))
  }, [])

  /**
   * Hold the turn that was just re-opened. A collapsed turn is one line and an
   * open one can be a whole hand of cards, so expanding one above the scroll
   * position pushes everything below it down by a few hundred pixels — and the
   * stream sets `overflow-anchor: none` (it has to, or the anchor fights the
   * scroll-to-bottom on every insertion), so nothing compensates for that. The
   * visitor pressed the bead to read it; this is what keeps it where they can.
   *
   * `block: 'nearest'` does nothing when the turn is already fully visible,
   * which is the common case, so this is not a jump on every press. It is
   * deliberately separate from the pin effect above: that one fires on `turns`
   * and would scroll to the BOTTOM, which is the opposite of what is wanted.
   */
  useLayoutEffect(() => {
    const id = justOpened.current
    if (!id) return
    justOpened.current = null
    const el = streamRef.current?.querySelector(`[data-turn="${CSS.escape(id)}"]`)
    el?.scrollIntoView({ block: 'nearest', behavior: 'auto' })
  }, [reopened])

  /* — the composer suspends ambient motion while it is being used ————— */

  const bumpCalm = useCallback(() => {
    setCalm(true)
    window.clearTimeout(calmTimer.current)
    calmTimer.current = window.setTimeout(() => {
      if (document.activeElement !== inputRef.current) setCalm(false)
    }, CALM_MS)
  }, [])

  /* — the wait, walked ———————————————————————————————————————————————— */

  /**
   * The thinking caption steps through `C.app.askingSteps` while an ask is
   * out. Ziat et al., Scientific Reports 2022: hold elapsed time constant and
   * split the wait into more discrete steps, and people rate progress as
   * faster and UNDERESTIMATE how long they waited. The request is unchanged;
   * only what the visitor is told about it is.
   *
   * IT STOPS ON THE LAST STEP. A caption that cycles back to step one says the
   * request restarted, and it did not — so the interval clears itself the
   * moment it has nothing further that is true to say, and the last line holds
   * for however long the model takes. That is the honest shape of this wait:
   * four things are known to happen, and then it is out of our hands.
   *
   * Keyed on `asking`, so the walk starts with the ask and is torn down by the
   * same state change that removes the thinking turn.
   */
  useEffect(() => {
    if (!asking) return
    let step = 0
    const timer = window.setInterval(() => {
      step += 1
      if (step >= C.app.askingSteps.length - 1) window.clearInterval(timer)
      const caption = C.app.askingSteps[Math.min(step, C.app.askingSteps.length - 1)]
      setTurns((prev) =>
        prev.map((turn) => (turn.kind === 'thinking' ? { ...turn, caption } : turn)),
      )
    }, ASKING_STEP_MS)
    return () => window.clearInterval(timer)
  }, [asking])

  /* — one ask ————————————————————————————————————————————————————————— */

  /**
   * The model half of an ask, and the only thing on this page that deals cards
   * unasked. Thinking state, then the reply. Nothing renders in between,
   * because anything that did would be the page answering its own question.
   *
   * `replaceId` is the stuck turn being retried. It is dropped in the same
   * update that puts the thinking state in, so Try again reuses the place in
   * the stream the failure was occupying instead of stacking a second attempt
   * underneath a line that is no longer true.
   */
  const runModel = useCallback(
    // `given` is the dataset a caller already has in hand. A sentence typed
    // before the download finished resolves it and passes it straight through,
    // rather than waiting a render for `rows` state to catch up.
    async (value: string, replaceId?: string, given?: NaamRow[]) => {
      const dataset = given ?? rows
      const mount = mountRef.current
      if (!dataset || !mount || asking || value.length === 0) return

      const { prefs, poolIds: allIds, near } = readAsk(value, dataset)
      // Kept so the refinement chips can add a clause to THIS question rather
      // than replacing it — see src/lib/naam/refine.ts for why that matters.
      setLastAsk({ text: value, prefs })
      /**
       * THE TRAY TELLS THE POOL WHAT IT ALREADY HOLDS. Measured: keep Bhagin,
       * then ask "names that mean fortunate or blessed", and Bhagin comes back
       * in the deal — a card you already own, spending one of eight slots and
       * quietly saying the page was not listening.
       *
       * Filtered HERE, at the call site, and deliberately not inside readAsk or
       * retrieve. The pool builder's job is "what does the document have for
       * this sentence", which is a question about the document; "and what does
       * this visitor already have" is a question about this session. Keeping
       * them apart is also what keeps the eval honest — scripts/naam/eval
       * scores match.retrieve directly, so nothing it measures moves.
       *
       * The floor matters. With three kept and a thin pool, subtracting could
       * empty it, and an empty pool renders as a failure rather than as a short
       * answer — so the unfiltered list is restored if filtering would leave
       * too little to deal from.
       */
      const poolIds = (() => {
        if (pickedIds.size === 0) return allIds
        const fresh = allIds.filter((id) => !pickedIds.has(id))
        return fresh.length >= Math.min(NAAM_DEAL_SMALL, allIds.length) ? fresh : allIds
      })()
      const thinkingId = nextId()
      setAsking(true)
      setTurns((prev) => [
        ...(replaceId ? prev.filter((turn) => turn.id !== replaceId) : prev),
        { id: thinkingId, kind: 'thinking', caption: C.app.askingSteps[0] },
      ])
      // The live region gets ONE line, not four. A screen reader reading a
      // caption that rewrites itself every 1.4s would be told the same wait
      // four times; the steps are a visual reassurance and this is the fact.
      setAnnounce(C.app.asking)

      const result = await askNaam({
        ask: value,
        poolIds,
        rows: dataset,
        // Only the words they typed. The rows nearest to them are already at
        // the head of poolIds, so this tells the model what is MISSING, which
        // is the one thing the pool cannot say on its own.
        absent: near.map((miss) => miss.typed),
        signal: mount.signal,
      })
      // askNaam never throws — an abort resolves to `unreachable` — so the
      // caller is the one that must not touch state on a dead mount.
      if (mount.signal.aborted) return
      setAsking(false)

      if (result.kind !== 'live') {
        const note = result.kind === 'degraded' ? failureNote(result.reason) : C.failure.modelDown
        setTurns((prev) => [
          ...prev.filter((turn) => turn.id !== thinkingId),
          { id: nextId(), kind: 'stuck', ask: value, note, retry: true },
        ])
        setAnnounce(note)
        return
      }

      setSource('live')
      // The model can answer and name nothing we could resolve — asked for a
      // quality the document does not have, it says so and picks nothing, and
      // that is the right answer rather than a failure. The reply stands on
      // its own; the page is not obliged to produce cards for every turn.
      const matches = result.rows.length > 0 ? withReasons(result.rows, prefs) : []
      setTurns((prev) => {
        const dealt: Turn[] = [{ id: nextId(), kind: 'agent', text: result.reply }]
        if (matches.length === 0) {
          // No Try again here: the model DID answer. What it left behind is a
          // turn with no names in it, so the only thing worth offering is the
          // document — and only to someone who asks for it.
          dealt.push({ id: nextId(), kind: 'stuck', ask: value, note: '', retry: false })
        } else {
          // Said once, over the first hand. It is a standing instruction about
          // how the tray works, and under every subsequent deal — each of which
          // the model has already framed in its own words — it would be chrome.
          if (!prev.some((turn) => turn.kind === 'names')) {
            dealt.push({ id: nextId(), kind: 'agent', text: C.app.dealt })
          }
          dealt.push({ id: nextId(), kind: 'names', matches, badge: 'live', note: C.badge.liveCaption })
        }
        return [...prev.filter((turn) => turn.id !== thinkingId), ...dealt]
      })
      setAnnounce(result.reply)
    },
    /* pickedIds belongs here, and its absence was the whole bug. It is a
       useMemo over `picks`, so it is stable between keeps and changes exactly
       when the tray does — without it runModel closed over the empty Set from
       the first render and the filter above could never fire. Measured before:
       keep Bhagin, ask, Bhagin comes back. */
    [asking, rows, pickedIds],
  )

  /**
   * What the visitor said, then the model's turn.
   *
   * ── IT NO LONGER REFUSES A SENTENCE TYPED TOO EARLY ─────────────────────
   *
   * The composer used to be `disabled` until the 6,715-name dataset had
   * landed, because the client builds the pool of ids the model may choose
   * from and cannot do that without it. Honest, and measured on a 412px phone
   * at 4x CPU it was awful: the only input on the page was dead for 5.6s on
   * fast 4G and had not come alive after 22 seconds on slow 4G. The page said
   * "reading the document…" and a visitor who wanted to type simply could not.
   *
   * The premise was wrong. The dataset is needed to SEND, not to TYPE, and
   * somebody typing a sentence takes seconds — which is exactly the window the
   * download needs. So the box is live from the first frame and the sentence
   * waits here instead: `loadCoreRows()` hands back the same cached promise the
   * boot fetch is already waiting on, so this joins that request rather than
   * starting another.
   *
   * The turn is pushed BEFORE the await, so what they typed is on screen the
   * instant they press enter and the thinking state carries the wait — the
   * page reacts to the person, then to the network.
   */
  const runAsk = useCallback(
    (text: string) => {
      const value = text.trim()
      if (asking || value.length === 0) return

      setTurns((prev) => [...prev, { id: nextId(), kind: 'you', text: value }])
      if (rows) {
        void runModel(value)
        return
      }
      void loadCoreRows().then(
        (loaded) => {
          setRows(loaded)
          void runModel(value, undefined, loaded)
        },
        () => setDataFailed(true),
      )
    },
    [asking, rows, runModel],
  )

  /**
   * THREE WAYS IN, chosen after mount.
   *
   * A blank box with a blinking cursor asks the visitor to have already decided
   * what kind of name they want, which is the one thing they came here without.
   * These are three different questions, not three suggestions — tapping one
   * sends it as the visitor's own turn, so the thread reads the same whether it
   * was typed or tapped.
   *
   * They stay for the whole conversation rather than only the first turn: the
   * second question is as hard to think of as the first, and a row that
   * disappears the moment you use it punishes the visitor who found it useful.
   * They REFRESH instead — `usedStarters` keeps what has already been asked, so
   * the row never offers back a question answered directly above it.
   *
   * Picked in an effect and not during render: the island is `client:load`, and
   * a random choice made while rendering would not match the server's HTML.
   */
  /* THE DOODLE IS GONE, and the arrival shelf is why. It was a pen-stroke
     curving from the last line of the invitation down to the composer, built
     for a desktop layout where those two had 255px of nothing between them —
     "a line curving down to the box is the gesture a person makes when they
     point at something across a table."

     The shelf now occupies that table. Measured at 1440: the stroke ran
     582->720 inside a hand running 401->732, so 138px of it was drawn straight
     across the cards and the gesture meaning "the box is down there" landed as
     a stray pink mark on the Svaraj card. It pointed across a gap that no
     longer exists. Twelve real names with the composer directly beneath them
     say where to go without drawing on anything. */

  /**
   * ── THE SKY ANSWERS THE ASK ─────────────────────────────────────────────
   *
   * The lantern field is held until here (see `release()` in lanterns.ts). It
   * comes up when the invitation has finished asking — not before, because
   * lanterns rising while the page is still explaining itself are scenery
   * competing with the words, and the same lanterns rising after the line
   * "what kind of name are you looking for?" are the reply to it (L3).
   *
   * SKY_BEAT is a rest, not a delay. The third block takes 520ms to arrive and
   * then the page is quiet for a moment before anything answers — that pause
   * is what makes the sky read as a response rather than as the next item in a
   * queue. Filling it would be the exact failure mode this objective warns
   * about: more motion, less room (L1).
   *
   * `opening.done` is already true on the first frame for a returning visitor
   * mid-conversation, so their sky comes up immediately, which is right — they
   * are not being invited, they are coming back.
   */
  useEffect(() => {
    if (!opening.done) return undefined
    const SKY_BEAT = 500
    const id = window.setTimeout(() => lanternField().release(), SKY_BEAT)
    return () => window.clearTimeout(id)
  }, [opening.done])

  const [starters, setStarters] = useState<NaamStarter[]>([])
  const usedStarters = useRef<Set<string>>(new Set())
  useEffect(() => {
    setStarters(pickStarters(3))
  }, [])

  const submitAsk = useCallback(() => {
    // Typing is consent to get on with it — the rest of the opening arrives
    // fast and in order rather than snapping to its end state.
    opening.rush()
    const value = ask.trim()
    if (value.length === 0) return
    setAsk('')
    runAsk(value)
  }, [ask, runAsk])

  /* — Keep, and take back, which is animated exactly as hard ——————————— */

  /**
   * Taking a name back. The slot that empties is always the LAST filled one,
   * because the tray is an ordered list and everything after the gap shuffles
   * down — so that is the slot that reacts, and the name dissolves upward out
   * of the slot it was sitting in.
   */
  const release = useCallback((deva: string, latin: string, from: DOMRect, lastIndex: number) => {
    const host = fxRef.current
    const slot = slotRefs.current[lastIndex]
    if (reducedMotion() || !host) return
    const token = nameToken(deva, latin)
    placeOn(token, from)
    host.append(token)
    token
      .animate(
        [
          { transform: 'translate(0px, 0px) scale(1)', opacity: 1 },
          { transform: 'translate(0px, -14px) scale(1.12)', opacity: 0 },
        ],
        { duration: RELEASE_MS, easing: DEAL_EASE, fill: 'forwards' },
      )
      .finished.then(() => token.remove())
      .catch(() => token.remove())
    slot?.animate([{ transform: 'scale(1.04, 0.94)' }, { transform: 'scale(1, 1)' }], {
      duration: RECOIL_MS + SLOT_SETTLE_MS,
      easing: SLOT_EASE,
    })
  }, [])

  const takeBack = useCallback(
    (pick: NaamPick, index: number) => {
      const slot = slotRefs.current[index]
      const token = slot?.querySelector<HTMLElement>('.nm-token')
      const deva = token?.querySelector('.nm-token-deva')?.textContent ?? ''
      const rect = slot?.getBoundingClientRect()
      removePick(pick.id)
      if (rect) release(deva, pick.spelling, rect, picks.length - 1)
    },
    [picks.length, release],
  )

  const keep = useCallback(
    (row: NaamRow) => {
      const already = pickedIds.has(row.id)
      const index = picks.length
      /**
       * Does THIS press close the third slot? Read from the action rather than
       * derived from picks.length, and that distinction is a bug I shipped and
       * caught: as an effect watching the count, it fired again on every page
       * load with a full tray, because picks are restored from localStorage and
       * an empty-then-three rehydration looks exactly like a fill. A returning
       * visitor got the bowl struck and a dozen embers for doing nothing. A
       * celebration belongs to the press that earned it.
       */
      const fills = !already && index === PICK_MAX - 1
      const slot = slotRefs.current[index]
      /**
       * SEARCHED FROM THE SHELL, not from the stream. This read streamRef until
       * the hand moved onto the tray, and then it silently found nothing: `from`
       * came back undefined, the guard below returned early, and the card
       * flight — the one piece of motion this page is built around — stopped
       * happening at all. Keeping still worked, so nothing looked broken; the
       * name simply appeared in the slot with no arc, no hitstop and no recoil.
       *
       * .nm-shell is the right scope rather than `document`: the no-JS fallback
       * renders twelve more [data-nm-card] nodes and hides them with CSS, and
       * measuring a display:none card gives an all-zero rect, which would fly
       * every name from the top-left corner of the window.
       */
      const card = shellRef.current?.querySelector<HTMLElement>(
        `[data-nm-card="${typeof CSS !== 'undefined' && CSS.escape ? CSS.escape(row.id) : row.id}"]`,
      )
      const from = card?.getBoundingClientRect()
      const to = slot?.getBoundingClientRect()

      const deva = naamPreferredDevanagari(row, preferB)
      const latin = naamPreferredForm(row, preferB)

      // THE NAME IS KEPT THE INSTANT IT IS CLICKED. Everything below this line
      // is decoration over state that has already changed, which is what makes
      // a second Keep startable at frame 1 of the first one's flight.
      togglePick(row)

      // A second click on a kept card takes it back, from wherever it sits.
      if (already) {
        const seat = slotRefs.current[picks.findIndex((p) => p.id === row.id)]
        const rect = seat?.getBoundingClientRect()
        if (rect) release(deva, latin, rect, picks.length - 1)
        return
      }
      if (reducedMotion() || !from || !to || !slot || !fxRef.current || index >= PICK_MAX) return

      const token = nameToken(deva, latin)
      placeOn(token, to)
      fxRef.current.append(token)

      /**
       * The slot is ALREADY FILLED at this point — state never waits for an
       * animation — so its seated name would sit under the flying one for the
       * whole arc and the two would read as one object smeared. Hiding it
       * until the token lands means the name appears in exactly the frame the
       * token is removed. Set imperatively rather than through state: React
       * does not own this attribute and will not clobber it, and a re-render
       * mid-flight must not be able to reveal the seat early.
       */
      slot.dataset.landing = 'true'
      const seat = () => {
        delete slot.dataset.landing
      }

      const lift = Math.min(2.4, Math.max(1.05, from.width / to.width))
      const dx = from.left + from.width / 2 - (to.left + to.width / 2)
      const dy = from.top + from.height / 2 - (to.top + to.height / 2)

      const flight = token.animate(
        [
          { transform: `translate(${dx}px, ${dy}px) scale(${lift})`, opacity: 1, offset: 0 },
          {
            transform: `translate(${dx * 0.54}px, ${dy * 0.54 - 22}px) rotate(-2.4deg) scale(${1 + (lift - 1) * 0.5})`,
            opacity: 0.97,
            offset: 0.46,
          },
          // Scale finishes at 80% of the path: it ARRIVES small rather than
          // shrinking as it lands.
          { transform: `translate(${dx * 0.2}px, ${dy * 0.2}px) scale(1)`, opacity: 0.94, offset: 0.8 },
          { transform: 'translate(0px, 0px) scale(1)', opacity: 0.9, offset: 1 },
        ],
        { duration: FLIGHT_MS, easing: DEAL_EASE, fill: 'forwards' },
      )

      flight.finished
        .then(() => {
          // Total stillness, then the receiver reacts — not just the card.
          later(() => {
            token.remove()
            seat()
            // CLAY, on the frame the seat appears — the sound is the contact,
            // so it belongs after the hitstop with the recoil, not at the end
            // of the flight when the token is still in the air.
            playCue('land')
            // …and if that was the third, the bowl and the embers ride the same
            // frame. The ring starts under the tock rather than after it, which
            // is what makes three names read as one completed thing instead of
            // two events in a row.
            if (fills) {
              playCue('complete')
              embers(slot)
            }
            slot.animate(
              [
                { transform: 'scale(1, 1)', offset: 0 },
                { transform: 'scale(0.96, 1.03)', offset: RECOIL_MS / (RECOIL_MS + SLOT_SETTLE_MS) },
                { transform: 'scale(1, 1)', offset: 1 },
              ],
              { duration: RECOIL_MS + SLOT_SETTLE_MS, easing: SLOT_EASE },
            )
            slot.querySelector('.nm-token')?.animate(
              [
                { opacity: 0.6, transform: 'scale(0.94)' },
                { opacity: 1, transform: 'none' },
              ],
              {
                duration: SETTLE_MS,
                easing: DEAL_EASE,
              },
            )
          }, HITSTOP_MS)
        })
        .catch(() => {
          // Cancelled by a teardown. The pick is already in the store, so the
          // only thing that must not survive is the hidden seat.
          token.remove()
          seat()
        })
    },
    [later, picks, pickedIds, preferB, release],
  )

  /**
   * THE SEND ARRIVES ON THE FIRST NAME, not the third.
   *
   * Three is the invitation and it stays the invitation — three slots, drawn
   * from the first frame, and the tray says so. But requiring three was the
   * page setting a price on being heard: someone who has one name they love
   * and no opinion on a second had no way to tell us, and the most likely
   * thing they do about a form that will not open is close the tab.
   *
   * So one is enough to send, and the other two slots stay open behind the
   * form — the endowed-progress shape is intact, and the visitor keeps
   * choosing if they want to. The form's own lead does the asking now
   * (`send.lead`), which is why that string had to stop saying "that is your
   * three": it fires when there is one.
   */
  /**
   * OPENED BY THE VISITOR, never by the count.
   *
   * The form used to arrive on its own the moment a name was kept, and it was
   * in the way: it filled the conversation column with fields while somebody
   * was still choosing, and it read as "that's enough, hand them over" after
   * exactly one pick. Three slots that invite three names and a form that opens
   * on the first are arguing with each other.
   *
   * So the tray offers `send.open` once there is anything to send, and pressing
   * it is what brings the form. Keeping goes on happening either way.
   */
  /** The dialog's own box, and whatever had focus before it opened. */
  const overlayRef = useRef<HTMLDivElement | null>(null)
  const formOpener = useRef<HTMLElement | null>(null)

  const openForm = useCallback(() => {
    if (formShown) return
    // Remembered before the dialog steals focus, so Escape can hand it back.
    formOpener.current = document.activeElement as HTMLElement | null
    setFormShown(true)
    setAnnounce(C.app.send.lead)
  }, [formShown])

  /**
   * ─── THE THREE GO UP ───────────────────────────────────────────────────
   *
   * On a successful send each kept name lifts out of its slot and rises to the
   * lantern that carries it, then fades into it.
   *
   * WHAT THIS IS NOT DOING: creating the lantern. A kept name is already a
   * vote — `notes` walks `picks` — so by the time this runs the sky has
   * already either grown a new lantern for a name nobody had chosen or added a
   * voice to one that was up there. The merge and the arrival are data; this
   * is the gesture that shows them happening, and if it never ran the state
   * would still be right.
   *
   * It flies to the lantern's CURRENT box, read at the moment of release,
   * because the lanterns drift — a target measured a second earlier is a
   * target that has moved.
   */
  const releaseToSky = useCallback(
    (sent: readonly { id: string; spelling: string }[]) => {
      const host = fxRef.current
      if (!host || reducedMotion()) return

      sent.forEach((pick, i) => {
        const slot = slotRefs.current[i]
        const from = slot?.getBoundingClientRect()
        // A lantern is matched by the row id the wall keys its notes on.
        const lantern = document.querySelector<HTMLElement>(
          `.nm-lamp[data-key="${CSS.escape(pick.id)}"]`,
        )
        const to = lantern?.getBoundingClientRect()
        if (!from || !to || from.width === 0) return

        const deva = slot?.querySelector('.nm-token-deva')?.textContent ?? ''
        const token = nameToken(deva, pick.spelling)
        placeOn(token, from)
        host.append(token)

        const dx = to.left + to.width / 2 - (from.left + from.width / 2)
        const dy = to.top + to.height / 2 - (from.top + from.height / 2)
        token
          .animate(
            [
              { transform: 'translate(0, 0) scale(1)', opacity: 1 },
              // Up and out first, the way something buoyant leaves a hand,
              // then across to the lantern rather than straight at it.
              { transform: `translate(${dx * 0.3}px, ${dy * 0.55 - 26}px) scale(0.9)`, opacity: 1, offset: 0.45 },
              { transform: `translate(${dx}px, ${dy}px) scale(0.55)`, opacity: 0 },
            ],
            { duration: 900, delay: i * 130, easing: 'cubic-bezier(0.22, 0.7, 0.24, 1)', fill: 'forwards' },
          )
          .finished.then(
            () => token.remove(),
            () => token.remove(),
          )
      })
    },
    [],
  )

  const closeForm = useCallback(() => setFormShown(false), [])

  /**
   * FOCUS GOES BACK, AFTER THE SHEET IS ACTUALLY GONE.
   *
   * Two things had to be true and neither was obvious from the code:
   *
   * · The stored node may no longer exist. Keeping a name re-renders the tray,
   *   so the button that opened the sheet can be a different element by the
   *   time it closes, and focus() on the detached one does nothing.
   * · Restoring inside the close handler is too early. React has not removed
   *   the dialog yet, focus is still inside it, and the removal that follows
   *   drops focus to <body> — which is precisely what it measured as.
   *
   * So it runs in an effect, after the render that unmounts the sheet, and
   * falls back to the live send button when the remembered one has gone.
   */
  const wasFormShown = useRef(false)
  useEffect(() => {
    if (wasFormShown.current && !formShown) {
      const remembered = formOpener.current
      const target =
        remembered && remembered.isConnected
          ? remembered
          : shellRef.current?.querySelector<HTMLElement>('.nm-tray-send')
      target?.focus?.()
    }
    wasFormShown.current = formShown
  }, [formShown])

  /**
   * MODAL BEHAVIOUR, written out rather than imported.
   *
   * Escape closes. Tab cycles inside the sheet and cannot leave it — the whole
   * point of a modal is that the page behind it is unreachable, and a dialog
   * you can tab out of is a dialog that strands you in a room you cannot see.
   * Focus moves to the first field on open.
   */
  useEffect(() => {
    if (!formShown) return undefined
    const sheet = overlayRef.current
    if (!sheet) return undefined

    const focusable = () =>
      [...sheet.querySelectorAll<HTMLElement>('a[href], button, input, select, textarea')].filter(
        (el) => !el.hasAttribute('disabled') && el.offsetParent !== null,
      )

    focusable()[0]?.focus()

    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.preventDefault()
        closeForm()
        return
      }
      if (event.key !== 'Tab') return
      const items = focusable()
      if (items.length === 0) return
      const first = items[0]
      const last = items[items.length - 1]
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault()
        last.focus()
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault()
        first.focus()
      }
    }

    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [closeForm, formShown])

  /**
   * Two posts, and they say different things. Netlify Forms is the durable
   * path — it emails us — and /api/naam-submit is the moderation queue. The
   * visitor is told which half happened rather than being shown "sent" for a
   * write that failed. Neither rides the mount signal: someone who submits and
   * immediately clicks a nav link would otherwise have their suggestion
   * aborted in flight and never know.
   */
  const sendPicks = useCallback(
    async (event: FormEvent<HTMLFormElement>) => {
      event.preventDefault()
      if (sending || picks.length === 0) return
      const form = event.currentTarget
      const data = new FormData(form)
      const from = String(data.get('from') ?? '').trim()
      const relation = String(data.get('relation') ?? '')
      const reason = String(data.get('reason') ?? '')
      // The name they typed themselves, if any. It has always been carried by
      // both endpoints; the app was the only surface that never collected it
      // and posted an empty string every time.
      /**
       * TYPED NAMES TRAVEL IN `names`, CITED ONES IN `picks`, and the split is
       * the server's rule rather than a preference: /api/naam-submit checks
       * every submitted pick against the document's real rows and drops what
       * does not resolve. An own-name in `picks` would simply vanish. `names`
       * is the field the no-JS form has always used for exactly this.
       */
      const cited = picks.filter((pick) => !isOwnPick(pick.id))
      const names = picks
        .filter((pick) => isOwnPick(pick.id))
        .map((pick) => pick.spelling)
        .join(', ')
      setSending(true)
      setSendNote('')

      /**
       * ── ONE POST, AND IT IS THE ONE THAT KEEPS THE RECORD ────────────────
       *
       * There were two: Netlify Forms, which emailed us, and /api/naam-submit,
       * which is the moderation queue. Forms is gone entirely.
       *
       * It was not a trade. It never worked on this site and api/naam-submit.ts
       * documents why — the adapter claims '/*' with preferStatic, preferStatic
       * only covers GET, so the POST lands in the SSR function which renders a
       * page and returns 200 while Forms never sees it, verified against
       * production on four paths. What it actually bought was a second way for
       * a submission to half-succeed, an outcome the visitor had to be told
       * about in its own sentence, and a send that could not be exercised
       * anywhere but production — the reason the launch animation below had
       * never once been watched end to end.
       *
       * The notification the email was for still happens: the endpoint posts to
       * Slack with a one-click approve link.
       */
      const response = await fetch('/api/naam-submit', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ from, relation, reason, names, picks: cited }),
        keepalive: true,
      }).catch(() => null)

      const rateLimited = response?.status === 429
      let queued = false
      if (response && response.ok) {
        try {
          const body = (await response.json()) as { ok?: unknown; stored?: unknown }
          queued = body.ok === true && body.stored === true
        } catch {
          /* a body we cannot read is not a stored suggestion */
        }
      }

      // One path means one failure, said plainly, with the typing still in the
      // form behind it.
      if (!queued && !rateLimited) {
        setSending(false)
        setSendNote(C.form.error.network)
        setAnnounce(C.form.error.network)
        return
      }

      const line = rateLimited ? C.form.error.rateLimited : C.form.confirmation.body
      setSending(false)
      // Read the slots BEFORE the sheet closes and anything re-renders.
      const rising = picks.map((pick) => ({ id: pick.id, spelling: pick.spelling }))
      // The sheet closes and the outcome goes to the conversation, which is
      // where the rest of the exchange already lives.
      setFormShown(false)
      later(() => releaseToSky(rising), 120)
      setTurns((prev) => [...prev, { id: nextId(), kind: 'sent', text: line }])
      /**
       * THE TRAY EMPTIES, BECAUSE THE NAMES HAVE GONE.
       *
       * They did not before, and the culminating moment on the page read as
       * though nothing had happened: "Dhanyabad." arrived in the thread while
       * the three slots still held the same three names and the button still
       * offered "Send these 3 →". A page that keeps offering to send what it
       * has already sent is telling the visitor it did not hear them.
       *
       * `rising` was captured above, so the lanterns still carry these exact
       * names up — which is where they visibly go. Emptying the slots is the
       * other half of that gesture rather than a reset.
       *
       * Not on a rate-limit: nothing was queued, so the three are still theirs
       * and error.rateLimited says to wait a moment and try again.
       */
      if (!rateLimited) clearPicks()
      setAnnounce(line)
    },
    [later, picks, releaseToSky, sending],
  )

  /* — render ————————————————————————————————————————————————————————— */

  /**
   * True on the server, where `rows` is null and always will be, and true in
   * the browser until the dataset lands. Every control gates on it — the
   * repo's `notReady` convention — because a control that cannot do its job
   * is honest as disabled and dishonest as a dead enabled one.
   */
  const notReady = rows === null
  const family = useMemo(() => [seed, ...NAAM_SEED_ROWS], [seed])
  const slots = useMemo(() => Array.from({ length: PICK_MAX }, (_, i) => picks[i] ?? null), [picks])

  /**
   * THE HAND — the most recent names the agent dealt, and the reason the tray
   * is a tray rather than a footer.
   *
   * These cards used to render inside their turn in the stream, and that is
   * what made the model read as absent even after it started leading. Measured
   * at 1280×800 the moment an answer arrived: the stream is 439px tall, a hand
   * of cards is ~370 of it, and the model's own sentence — the whole point of
   * having an agent — sat at top:-180, entirely above the fold. At 390 it was
   * 581px above it. No scroll position could have shown both, because reply
   * (82) + line (54) + cards (370) does not fit in 439 at any offset.
   *
   * So the hand moved out of the conversation and onto the tray, where the
   * three slots already are. That is also the truer object: a mala is a thread
   * you pass beads along, and a thali is a plate things are laid on and chosen
   * from. Words belong on the thread; names being weighed belong on the plate.
   * The conversation column now only ever holds language, so nothing the agent
   * says can be pushed off the screen by cards again.
   *
   * A REOPENED past hand still renders its cards inline in the stream — it is
   * a record of what was dealt then, not what is on the tray now, and the two
   * must not be confused. Only the current hand is on the plate.
   */
  const hand = useMemo(() => {
    for (let i = turns.length - 1; i >= 0; i--) {
      const turn = turns[i]
      if (turn.kind === 'names' && turn.matches.length > 0) return turn
    }
    return null
  }, [turns])

  /**
   * What to offer under the deal. Derived from the rows that came back and the
   * question that produced them, so every chip continues the visitor's own
   * sentence — never starts a new one.
   */
  const refineChips = useMemo(() => {
    if (!hand || !lastAsk) return []
    const freshest = picks.length > 0 ? picks[picks.length - 1] : null
    return refinements(
      hand.matches.map((m) => m.row),
      lastAsk.prefs,
      lastAsk.text,
      freshest ? { latin: freshest.spelling ?? freshest.id } : null,
    )
  }, [hand, lastAsk, picks])

  /**
   * PAPER, once per card, on the beat the card actually arrives. The deal is
   * staggered 100ms apart and each card takes 420ms, so a tick at i*100+360
   * lands under the card rather than under the gesture that requested it — the
   * difference between hearing three cards put down and hearing one click
   * repeated. Nothing plays unless the visitor turned sound on.
   */
  const dealtId = useRef<string | null>(null)
  useEffect(() => {
    if (!hand) {
      dealtId.current = null
      return
    }
    if (dealtId.current === hand.id) return
    dealtId.current = hand.id
    hand.matches.forEach((_, i) => later(() => playCue('deal'), i * 100 + 360))
  }, [hand, later])

  /**
   * EMBERS, not confetti — and this is the reason there is no confetti library
   * here at all. `canvas-confetti` is 6 kB for a shape the page would have to
   * argue with: paper squares raining down belong to a birthday, and every
   * light on this screen is a flame inside paper.
   *
   * They used to lift off the diyo. With the lamp gone they lift off the SLOT
   * THAT JUST FILLED, which is the better origin anyway: the spark leaves the
   * name the visitor just placed, rather than a decoration standing beside it,
   * and it rises toward the lanterns that name is about to join.
   *
   * Math.random is safe here in a way it is not in render: this runs from a
   * click, long after hydration, so there is no server frame to disagree with.
   */
  const embers = useCallback((from: HTMLElement | null) => {
    const host = fxRef.current
    if (!host || !from || reducedMotion()) return
    const lit = from.getBoundingClientRect()
    const box = host.getBoundingClientRect()
    for (let i = 0; i < 12; i++) {
      const spark = document.createElement('span')
      spark.className = 'nm-ember'
      spark.style.left = `${lit.left - box.left + lit.width * (0.34 + Math.random() * 0.32)}px`
      spark.style.top = `${lit.top - box.top + lit.height * 0.2}px`
      host.append(spark)
      const dx = (Math.random() - 0.5) * 56
      const dy = -(58 + Math.random() * 92)
      const rise = spark.animate(
        [
          { transform: 'translate(0, 0) scale(0.6)', opacity: 0 },
          { transform: `translate(${dx * 0.38}px, ${dy * 0.34}px) scale(1)`, opacity: 1, offset: 0.24 },
          { transform: `translate(${dx}px, ${dy}px) scale(0.42)`, opacity: 0 },
        ],
        { duration: 900 + Math.random() * 520, easing: 'cubic-bezier(0.16, 0.7, 0.3, 1)', delay: i * 34 },
      )
      const clear = () => spark.remove()
      rise.finished.then(clear, clear)
    }
  }, [])

  /** One deal, rendered the same whether it is on the tray or reopened inline. */
  const dealCards = (matches: readonly NaamMatch[]) => (
    <div className="nm-deal">
      {matches.map((match, i) => (
        <div
          className="nm-dealt"
          key={match.row.id}
          style={{ '--i': i, '--tilt': `${seededTilt(match.row.id)}deg` } as CSSProperties}
        >
          <NaamCard
            row={match.row}
            preferB={preferB}
            picked={pickedIds.has(match.row.id)}
            trayFull={picks.length >= PICK_MAX && !pickedIds.has(match.row.id)}
            onPick={() => keep(match.row)}
          />
        </div>
      ))}
    </div>
  )

  /**
   * The wall's notes: the family's own six first, then every approved
   * suggestion. The approved ones carry a signature because that is the warm
   * part — you are choosing next to people who already chose — and the ones
   * that resolve to a real row get their Devanagari from the dataset rather
   * than from the stored spelling, so the script on the wall is always the
   * document's even when the spelling somebody typed is not.
   *
   * The tilt is hashed here rather than inside NaamWall because seededTilt is
   * already this file's function, is already what gives a dealt card its angle,
   * and hashing in two places is how two surfaces end up disagreeing about what
   * ±3° means.
   */
  const notes = useMemo<readonly WallNote[]>(() => {
    /**
     * ONE LEAF PER NAME, and the repetition becomes the tally.
     *
     * This used to emit a leaf per (sender × pick), so a name two relatives
     * both loved sat on the shelf twice and read as two different names —
     * throwing away the single most useful thing the wall knows. Grouped by
     * id, the same data answers the question a family actually has: which
     * names are gathering support.
     */
    const byName = new Map<string, WallNote>()
    /**
     * `weight` is how many voices this addition represents. The wall and the
     * visitor's own picks are worth one each — one person, one name. The family
     * seeds carry their own count, because that list is one ROW per name and
     * says nothing on its own about how many people have said it.
     */
    const add = (key: string, note: Omit<WallNote, 'count'>, weight = 1) => {
      const found = byName.get(key)
      if (found) {
        found.count += weight
        // Two supporters and a single signature would credit one of them for
        // both, so the attribution drops the moment it stops being true.
        found.who = undefined
        found.mine = found.mine || note.mine
        return
      }
      byName.set(key, { ...note, count: weight })
    }

    for (const row of family) {
      add(
        row.id,
        {
          key: row.id,
          deva: naamPreferredDevanagari(row, preferB),
          latin: naamPreferredForm(row, preferB),
        },
        NAAM_SEED_VOTES[row.id] ?? 1,
      )
    }
    for (const entry of wall) {
      for (const pick of entry.picks) {
        const row = rows?.find((r) => r.id === pick.id)
        add(pick.id, {
          key: pick.id,
          deva: row ? naamPreferredDevanagari(row, preferB) : '',
          latin: row ? naamPreferredForm(row, preferB) : pick.spelling,
          who: entry.relation ? C.wall.entry(entry.from, entry.relation) : entry.from,
        })
      }
    }
    /**
     * THE VISITOR'S OWN CHOICES GO ON THE SAME SHELF, immediately, before
     * anything is sent. That is the whole point of moving the wall here: you
     * are not filling in a form beside other people's names, you are adding to
     * the same shelf they are on and watching it change as you choose. A name
     * the family already keeps simply gains a bead rather than appearing
     * twice — which is exactly what it means for you to agree with them.
     */
    for (const pick of picks) {
      const row = rows?.find((r) => r.id === pick.id)
      add(pick.id, {
        key: pick.id,
        deva: row ? naamPreferredDevanagari(row, preferB) : '',
        latin: row ? naamPreferredForm(row, preferB) : pick.spelling,
        mine: true,
      })
    }

    /**
     * Most-supported first, and YOURS first among equals. The shelf sorts
     * itself as the visitor chooses, so agreeing with a name the family already
     * keeps visibly carries it up the list — and a name only you have chosen
     * still rises above the ones sitting at the same count. That second clause
     * is not a nicety: without it, keeping two names appended them below the
     * fold of a scrolling shelf and the thing the visitor had just done left no
     * mark on the surface it was supposed to be adding to.
     */
    return [...byName.values()].sort(
      (a, b) => b.count - a.count || (b.mine ? 1 : 0) - (a.mine ? 1 : 0) || a.latin.localeCompare(b.latin),
    )
  }, [family, picks, preferB, rows, wall])

  /**
   * TAPPING A LANTERN KEEPS THAT NAME, and adds a voice to it.
   *
   * The lanterns were focusable buttons that did nothing — a control that
   * announces itself to a screen reader and then ignores the press. Now the
   * sky is an input as well as a readout: agreeing with a name the family is
   * already holding is the commonest thing a relative will want to do, and it
   * was the one thing the page had no gesture for.
   *
   * The vote is what moves it. Support drives depth, so a name gaining a voice
   * drifts nearer over the next few seconds rather than jumping — which is the
   * feedback, and it needs no toast to say so.
   *
   * KEEPING IS THE VOTE. The first cut added a separate tally on top of the
   * pick, and the count went from one to three on a single press: `notes`
   * already walks `picks` and credits each one. Agreeing with a name and
   * keeping it are the same act here, so they must not be counted twice — and
   * taking it back correctly removes the voice again.
   */
  /**
   * THE SKY ASKS, IT DOES NOT VOTE. This called keep(), which made the scene
   * the third way to do the one thing a card does — measured at four entry
   * points into keep(), against a rule whose whole test is that the count goes
   * DOWN. A lantern now puts its name to the agent, so the gesture feeds the
   * one path instead of forking it, and the deal it produces is something you
   * can keep from in the ordinary way.
   *
   * The older note here said "keeping is the vote" — that a press must not be
   * counted twice because `notes` already walks `picks`. That problem stops
   * existing: asking credits nothing, so there is no second tally to collide.
   */
  const askAboutName = useCallback(
    (key: string, fallbackLatin?: string) => {
      const row = family.find((candidate) => candidate.id === key) ?? rows?.find((r) => r.id === key)
      const name = row ? naamPreferredForm(row, preferB) : fallbackLatin
      if (name) runAsk(C.app.askLike(name))
    },
    [family, rows, runAsk, preferB],
  )


  /**
   * The chosen names go to the scene twice over: a lamp count for the town, and
   * the support COUNTS for the sky, where depth encodes how many people chose
   * each one. The refs cover the case where notes exist before the scene has
   * loaded; the handle covers every change after it has.
   */
  const lanternCounts = useMemo(() => notes.map((note) => note.count), [notes])
  const lanternKeys = useMemo(() => notes.map((note) => note.key), [notes])
  useEffect(() => {
    lampCountRef.current = notes.length
    lanternCountsRef.current = lanternCounts
    lanternKeysRef.current = lanternKeys
    valleyHandleRef.current?.setLamps(notes.length)
    // Keys first: setLanterns is what triggers the rebuild, and the rebuild
    // needs them to match bodies to the names they already belong to.
    valleyHandleRef.current?.setLanternKeys(lanternKeys)
    valleyHandleRef.current?.setLanterns(lanternCounts)
  }, [notes.length, lanternCounts, lanternKeys])

  /**
   * The cut between what you have passed and what you are on: the last thing
   * YOU said. Everything above it is worn down to a bead and a line; everything
   * from it down stays open. Asking is what makes a turn past — not time, and
   * not the arrival of the next turn — so the four turns of the opening
   * invitation stay whole until the visitor actually says something, and an
   * answer never collapses halfway through being read.
   */
  let passed = -1
  for (let i = turns.length - 1; i >= 0; i -= 1) {
    if (turns[i].kind === 'you') {
      passed = i
      break
    }
  }

  const turnBody = (turn: Turn) => {
    switch (turn.kind) {
      case 'agent':
        return (
          <p className={turn.lead ? 'nm-said nm-said--lead' : turn.quiet ? 'nm-said nm-said--quiet' : 'nm-said'}>
            {turn.lead && canEmoji && (
              <span className="nm-namaste" aria-hidden="true">
                {C.app.greetingGlyph}
              </span>
            )}
            {turn.text}
          </p>
        )

      case 'you':
        return <p className="nm-said nm-said--you">{turn.text}</p>

      case 'family':
        return null

      /**
       * THE PRE-WRITTEN CHIPS ARE GONE. Four ready-made sentences under an
       * invitation quietly reframe it as multiple choice: the question stops
       * being "what kind of name are you looking for" and becomes "pick one of
       * these", and the answers a visitor gives when they are handed options
       * are not the ones they arrived with. The invitation now teaches what to
       * say in its own words — a meaning, a sound, a single word — and the
       * arrival shelf shows what one looks like.
       *
       * The turn kind survives so a session stored before this change still
       * renders; it simply draws nothing.
       */
      case 'starters':
        return null

      case 'thinking':
        return (
          /* THE CAPTION IS THE LOADING STATE, on its own now.
             It used to share the job with the diyo, which leaned and burned
             hotter for the whole turn — so the .pulse-line was dropped from
             here to stop two things saying the same word. With the lamp gone
             the caption keeps the job and keeps breathing; it says in words
             what the flame said in light, which is the more legible of the
             two on a phone anyway. Decorative; the announcement itself rides
             the live region. */
          <div className="nm-thinking" aria-hidden="true">
            <p className="pulse-caption nm-caption">{turn.caption}</p>
          </div>
        )

      case 'names':
        return (
          <>
            {turn.matches.length === 0 ? (
              <p className="nm-said">{C.results.emptyAsk}</p>
            ) : (
              /* The CURRENT hand is on the tray, so here it is only its
                 provenance line — which belongs with the conversation anyway,
                 because "a real call, just now" is something the page is
                 telling you, not a property of a card. A reopened OLDER hand
                 still draws its cards, since that turn is a record of what was
                 dealt then rather than what is on the plate now. */
              hand?.id !== turn.id && dealCards(turn.matches)
            )}
            {/* THE PER-TURN PROVENANCE LINE IS GONE. "● live — A real call,
                just now." sat under every hand restating what the rail already
                says permanently, in a page whose whole argument is that the
                agent is simply talking to you. Announcing the transport under
                each answer is the page auditing itself out loud, and it read as
                machinery in the middle of a conversation.

                DESIGN.md §4 is still satisfied and this is not a quiet
                downgrade of it: the honesty vocabulary must be VISIBLE, not
                repeated per turn, and .nm-badge--rail carries `live` in the
                topbar for as long as the model is answering. The one place the
                word still earns a line of its own is a failure, where LOCAL
                marks a deck the visitor pressed a button to see — and that
                turn prints it. */}
          </>
        )

      /**
       * Nothing came back with names in it, said plainly and with somewhere to
       * go. The difference between the two controls is the whole point: Try
       * again is the primary one, and the matcher is an OPT-IN behind the
       * quiet one — a fallback the visitor was never offered would be the page
       * deciding on their behalf that a different system's answer would do.
       *
       * The note is omitted when the model answered and simply named nothing:
       * its own reply is immediately above, and repeating the point in the
       * page's error voice would contradict a turn that went fine.
       *
       * Both reuse the starter-chip classes rather than growing a pair of
       * one-off buttons: .nm-chip is already the page's small affirmative pill
       * and .nm-chip-hide its quiet text control, which is exactly the weight
       * these two want. The label-mono classes are deliberately NOT on the
       * escape — they uppercase in CSS, and a shouted mono sentence is a
       * system message where this page's most human line has to sit.
       */
      case 'stuck':
        return (
          <>
            {turn.note && <p className="nm-said nm-said--note">{turn.note}</p>}
            {/* The two data attributes are the failure drill's handles
                (scripts/verify/run.mjs). It cannot key off the classes —
                .nm-chip and .nm-chip-hide are the starter row's too — and
                keying off the words would tie the gate to copy.ts, which is
                rewritten far more often than this markup. */}
            <div className="nm-chips">
              {turn.retry && (
                <button
                  type="button"
                  className="nm-chip"
                  data-nm-retry=""
                  disabled={notReady || asking}
                  onClick={() => void runModel(turn.ask, turn.id)}
                >
                  {C.failure.retry}
                </button>
              )}
              {/* NO "SHOW ME WHAT THE DOCUMENT HAS ANYWAY". It was the honest
                  way to offer the matcher's own list when the model could not
                  answer, and it was still an invitation to give up on the
                  conversation and read a ranked dump of 6,715 rows. Nobody
                  naming a child wants a list; they want the next question. So
                  the failure offers the one thing that actually helps — try
                  again — and the agent's job is to make the retry worth it. */}
            </div>
          </>
        )

      case 'form':
        // The form is no longer a turn: it opens as a centred dialog (see
        // .nm-overlay below). The turn kind survives only so an interrupted
        // session restored from storage still has something to key off.
        return null

      case 'sent':
        return (
          <>
            <p className="nm-said nm-said--lead">{C.form.confirmation.heading}</p>
            <p className="nm-said">{turn.text}</p>
          </>
        )
    }
  }

  /**
   * ONE <li> PER TURN, and it is built here rather than nine times inside the
   * switch: the bead, the thread and the speaker label are the same three
   * things on every turn, and the moment they are copied into each case they
   * start drifting. The switch above is now only what a turn IS.
   *
   * A passed turn renders as a button and NOT as the body. Rendering both and
   * hiding one with CSS would leave every card in every past hand focusable,
   * announced and clickable, which is the log we are escaping with a
   * `display: none` painted over it.
   *
   * The FORM never collapses even when passed. It is the only turn holding
   * input the visitor has half-filled, and closing it out from under them to
   * make the thread tidier would lose typing.
   */
  const renderTurn = (turn: Turn, index: number) => {
    const body = turnBody(turn)
    if (body === null) return null
    const past = index < passed && turn.kind !== 'form' && !reopened.has(turn.id)
    return (
      <li
        className="nm-turn"
        key={turn.id}
        data-turn={turn.id}
        data-past={past ? 'true' : undefined}
        /* THE OPENING'S OWN BLOCKS ARRIVE IN ORDER. `data-wait` marks a block
           whose turn has not come yet; CSS holds it at zero opacity and lets it
           transition in when the attribute clears. A transition rather than an
           animation, because the arrival has to work whether it comes on the
           slow beat or the rushed one — an animation-delay cannot be changed
           after it has started, which is exactly why the CSS version could not
           be interrupted. Only the opening is staged; every later turn arrives
           the moment it exists. */
        data-wait={index < OPENING_STEPS && index >= opening.stage ? 'true' : undefined}
      >
        <span className="sr-only">{turn.kind === 'you' ? C.app.speakerYou : C.app.speakerAgent}</span>
        {/* The thread and the bead. Decorative in the strict sense — the turn
            below says everything this says, and a screen reader announcing
            "bead" on every entry would be nine words of furniture per turn. */}
        <span className="nm-turn-rail" aria-hidden="true">
          <span className="nm-bead" data-bead={beadOf(turn)}></span>
        </span>
        {past ? (
          <button type="button" className="nm-turn-shut" onClick={() => reopen(turn.id)}>
            <span className="nm-turn-shut-line">{summaryOf(turn)}</span>
            <span className="sr-only">{C.app.bead.reopen}</span>
          </button>
        ) : (
          <div className="nm-turn-body">{body}</div>
        )}
      </li>
    )
  }

  return (
    /**
     * THE ROOM HAS A STATE, and these four attributes are it. Everything
     * ambient on this page — the light pool, its warmth, the dust, the focus —
     * is CSS reading them, so the negative space stops being a painted
     * rectangle and starts answering what the visitor is doing. A game does
     * not hold still between inputs; neither should the empty half of this
     * screen.
     *
     *   data-calm     typing — the dust settles, the flame does not
     *   data-thinking the ask is out — the light draws in and waits
     *   data-kept     0–3 — the ground warms a step per name kept
     *   data-first    nothing asked yet — everything not yet usable stands down
     */
    <div
      className="nm-shell"
      ref={shellRef}
      data-calm={calm ? 'true' : undefined}
      data-thinking={asking ? 'true' : undefined}
      data-kept={picks.length}
      /* The three have gone. The shell uses this to give the thank-you the
         room the result no longer needs. */
      data-sent={turns[turns.length - 1]?.kind === 'sent' ? 'true' : undefined}
      data-first={!asked ? 'true' : undefined}
      /* data-still  the sky is stopped — WCAG 2.2.2. JS gates the canvas and
         the label loop; this is how the CSS keyframes hear about it, because
         no amount of cancelAnimationFrame touches a @keyframes. */
      data-still={stilled ? 'true' : undefined}
      /* Typing before the first ask: the opening lifts and thins out, because
         the invitation has done its job the moment somebody answers it. */
    >
      {/* The room the lamp lights, and the dust in it. Both are behind
          everything (z-index 0, pointer-events none) and both are decoration
          in the strict sense — aria-hidden, no content, nothing to read. They
          exist because the page was inert between clicks, which is the whole
          reason it read as a document. */}
      {/* THE VALLEY. Dusk over the Kathmandu valley, veiled back to paper on
          the left so you are inside a room looking out rather than reading
          over a sky. aria-hidden and pointer-events:none: it carries no
          information and takes no input, so axe still sees a complete page and
          every name above it stays selectable. A CSS gradient sits behind it in
          naam.astro, so a blocked or failed WebGL context looks deliberate
          instead of blank. */}
      <canvas className="nm-valley" ref={valleyRef} aria-hidden="true" />
      <div className="nm-room" aria-hidden="true" />
      <div className="nm-motes" aria-hidden="true">
        {MOTES.map((m, i) => (
          <span
            key={i}
            className="nm-mote"
            style={
              {
                left: `${m.left}%`,
                bottom: `${m.bottom}%`,
                animationDuration: `${m.dur}s`,
                animationDelay: `${m.delay}s`,
              } as CSSProperties
            }
          />
        ))}
      </div>

      {/* THE BAR IS GONE. It held the wordmark, the व/ब toggle, the live
          badge, the sound switch and one link — five pieces of chrome across
          the top of a page whose whole argument is that you are in a room
          having a conversation, and it was the first thing a visitor's eye had
          to get past to reach the invitation. On a phone it also spent 44px of
          a budget the send button needed.

          Where each piece went, and why none of them were simply dropped:
            · the wordmark  → .sr-only <h1>. The page still needs exactly one
              heading for a screen reader and for the document outline; it does
              not need to print its own name at a visitor who just clicked a
              link that said naam.
            · व / ब         → deleted outright. Every card that HAS a B-form
              already carries its own व/ब pill, and that pill sets the same
              page-wide preference. A global copy of a control that is already
              on the object it affects is a second way to do one thing.
            · live · sound · accessibility → .nm-utility, hairline-small and
              bottom-aligned with the composer. `live` stays because DESIGN.md
              §4 requires the honesty word to be VISIBLE, and this satisfies it
              in the place a visitor looks after reading an answer rather than
              before. */}
      <h1 className="sr-only">{C.app.heading}</h1>

      <div className="nm-streamwrap">
        {/* Both rules are wrong for a scrollable transcript, and axe is the
            gate that decides (P10): role="list" is restorative because
            `list-style: none` strips list semantics in Safari, and tabIndex is
            REQUIRED — a scroll container that keyboard users cannot reach is
            axe's scrollable-region-focusable. The alternative axe rejects is a
            <div tabIndex aria-label>, which is aria-prohibited-attr. */}
        {/* eslint-disable-next-line jsx-a11y/no-redundant-roles */}
        <ol
          className="nm-stream"
          role="list"
          // eslint-disable-next-line jsx-a11y/no-noninteractive-tabindex
          tabIndex={0}
          aria-label={C.app.streamLabel}
          ref={streamRef}
          onScroll={onStreamScroll}
        >
          {turns.map(renderTurn)}
          {/* Strung like every other turn — it is the last bead on the thread
              when it happens, and a line sitting off the thread would read as
              chrome rather than as something the page said. */}
          {dataFailed && (
            <li className="nm-turn" key="data-failed">
              <span className="sr-only">{C.app.speakerAgent}</span>
              <span className="nm-turn-rail" aria-hidden="true">
                <span className="nm-bead" data-bead="stuck"></span>
              </span>
              <div className="nm-turn-body">
                <p className="nm-said">{C.failure.dataDown}</p>
              </div>
            </li>
          )}

          {/* ── THE FAMILY'S CHOICES, ON A PHONE ──────────────────────────────
              On the wide layout these hang in the sky as lanterns. On a phone
              there is no sky: measured at 375x667, the largest empty strip in
              the whole frame is 3.9% of its height, and floating them anyway
              put six names across "What kind of name are you looking for?".

              But the transcript SCROLLS, and content inside it costs no layout
              at all. And these names are conversation — what the family is
              already holding — rather than scenery, so the stream is where
              they belong when there is no world to put them in.

              Only before the first question. After that the conversation is
              the content, and this would be the page talking over it. */}
          {/* ── AND IT WAITS FOR THE ASK, LIKE THE SKY DOES ──────────────────
              On the stacked layout this row IS the sky: the family's names,
              in the thread, because there is no world to hang them in. So it
              answers the same line the lanterns answer (L3, L5) — and it was
              not, it was sitting there from 449ms, three seconds before the
              page asked for anything.

              Most of this family reads on a phone. That makes this the version
              of the entry that matters most, and it had no entry at all — it
              inherited the desktop's content without the desktop's
              choreography, which is exactly what L5 exists to prevent. */}
          {/* MOUNTED FROM THE FIRST FRAME, REVEALED ON THE ASK. Gating the
              whole row on `opening.done` gave it the right timing and the
              wrong mechanism: the stream is vertically centred while nothing
              has been asked, so a row appearing at 3.6s grew the group and
              pushed the first line down 131px in one frame. The arrival has to
              be in the ink, never in the layout — same as the starters row,
              and for the same reason. */}
          {!asked && notes.length > 0 && (
            <li className="nm-turn nm-wallrow" data-on={opening.done ? 'true' : undefined}>
              <span className="nm-turn-rail" aria-hidden="true">
                {/* `note`, which already exists and already means exactly
                    this — a bead for the family's notes. Inventing a `wall`
                    state gave it no styling at all and it rendered bare
                    white, the one bead on the thread with no material. */}
                <span className="nm-bead" data-bead="note"></span>
              </span>
              <div className="nm-turn-body">
                <p className="nm-wallrow-lead label-mono label-mono--sm">{C.app.familyLead}</p>
                {/* Horizontally scrollable: a phone has width to spare and no
                    height at all, so the overflow runs sideways. */}
                {/* Buttons, like the lanterns they stand in for. Tapping one
                    keeps that name and adds a voice to it — the same gesture
                    on both layouts, because a phone visitor agreeing with a
                    name the family already holds is doing exactly what the
                    desktop visitor does when they tap a lantern. */}
                <ul className="nm-wallrow-list">
                  {notes.map((note) => (
                    <li key={note.key}>
                      <button
                        type="button"
                        className="nm-wallrow-item"
                        onClick={() => askAboutName(note.key, note.latin)}
                      >
                        {note.deva && (
                          <span className="nm-wallrow-deva" lang="sa-Deva">
                            {note.deva}
                          </span>
                        )}
                        <span className="nm-wallrow-latin">{note.latin}</span>
                        <span className="sr-only">
                          {C.wall.askAria(note.latin)}. {C.wall.support(note.count)}
                        </span>
                      </button>
                    </li>
                  ))}
                </ul>
              </div>
            </li>
          )}

          {/* The tail anchor that used to live here is gone with the rope.
              It existed so the simulated strand had one more `.nm-bead` to
              reach, which is how it was made to touch the boundary; a thread
              that spans both bands by construction needs no such thing, and
              the bead it drew marked nothing. */}
        </ol>

        {!pinned && (
          <button type="button" className="nm-jump label-mono label-mono--sm" onClick={jumpToLatest}>
            {C.app.jump}
          </button>
        )}
      </div>

      {/* THE THREE SLOTS ARE ONE SHAPE, three parts missing. The container
          draws a single ring and every empty slot erases its segment of it, so
          filling the middle one closes the middle third and changes what the
          other two look like — Gestalt closure, which three separate grey
          rectangles cannot do. There is no "1/3 KEPT" counter: the shape
          already says it. */}
      <section className="nm-tray" aria-label={C.app.tray.label}>
        {/* The plate. At ≥1100px this is the right-hand column and the hand
            lies across the top of it; below that it is a band above the slots
            and the cards scroll sideways through it. Either way the names you
            are weighing sit next to the three you have kept, which is the
            comparison the page exists to support and which the old layout put
            a scroll apart. */}
        {/* THE SHELF, and it is on this side for a reason. In the conversation
            it was a thing that had already happened, scrolled past within two
            turns and never seen again. Here it is the surface the visitor is
            adding to: their kept names land on the same shelf the family's sit
            on, a name they agree with gains a bead instead of appearing twice,
            and the order re-sorts under them as they choose. The right column
            stops being an output and starts being the thing that changes. */}
        <div className="nm-shelfwrap">
          <NaamWall notes={notes} onAsk={askAboutName} still={stilled} />
        </div>

        {/* The lamp sits WITH the slots, not in the rail corner where it
            started. Two reasons, and the first is the important one: it is the
            page's light source, and the ground's radial pool is centred at 50%
            — a lamp at 3% lighting a room from the middle is a lie you can see.
            The second is that at rail size it rendered ~15px, clipped by the
            site header, with a ±1.4° sway that is sub-pixel at that scale:
            technically animating and visually inert, which is the exact failure
            it exists to prevent. Here it is unclipped and it lights the three
            beads it sits above.

            The site index that used to live on this row is gone. It was chrome
            at the same visual weight as the thing the page is actually asking
            for; the header nav carries four of those five and the fifth is now
            one quiet link in the rail. */}
        <div className="nm-tray-head">
          <p className="label-mono label-mono--sm nm-quiet-label">
            {lastKind === 'sent' && picks.length === 0 ? C.app.tray.labelSent : C.app.tray.label}
          </p>
        </div>

        {/* eslint-disable-next-line jsx-a11y/no-redundant-roles -- restorative, see above */}
        <ol className="nm-slots" role="list">
          {slots.map((pick, index) => (
            // Keyed by position, never by pick id: a slot is a place in the
            // tray, not an identity, and re-keying it on every fill would
            // hand the flight a destination React had already replaced.
            <li
              className="nm-slot"
              key={`slot-${index}`}
              data-filled={pick ? 'true' : undefined}
              ref={(node) => {
                slotRefs.current[index] = node
              }}
            >
              {pick ? (
                <button type="button" className="nm-slot-take" onClick={() => takeBack(pick, index)}>
                  <span className="nm-token" aria-hidden="true">
                    <span className="nm-token-deva" lang="sa-Deva">
                      {seatedDeva(rows, pick, preferB)}
                    </span>
                    <span className="nm-token-latin">{seatedLatin(rows, pick, preferB)}</span>
                  </span>
                  <span className="sr-only">{C.app.tray.taken(index + 1, seatedLatin(rows, pick, preferB))}</span>
                </button>
              ) : owning === index ? (
                /* Typing straight into the slot. It commits on Enter or on
                   blur and cancels on Escape, so there is no confirm button in
                   a 90px box. */
                <input
                  className="nm-slot-input"
                  /* Focused on mount rather than with autoFocus: same effect,
                     and it is correct here because the visitor pressed this
                     slot in order to type in it. */
                  ref={(node) => node?.focus()}
                  type="text"
                  maxLength={C.limits.name}
                  placeholder={C.app.tray.ownPlaceholder}
                  aria-label={C.app.tray.ownLabel(index + 1)}
                  onBlur={(event) => {
                    addOwnPick(event.target.value)
                    setOwning(null)
                  }}
                  onKeyDown={(event) => {
                    if (event.key === 'Enter') {
                      event.preventDefault()
                      event.currentTarget.blur()
                    }
                    if (event.key === 'Escape') {
                      event.currentTarget.value = ''
                      event.currentTarget.blur()
                    }
                  }}
                />
              ) : (
                /* AN EMPTY SLOT IS THE OFFER, not a placeholder. It used to be
                   inert type, and the way to contribute a name the document
                   does not have was a field buried in the send form — which
                   nobody reaches until they have already kept something. The
                   slot is where names live, so it is where a name of your own
                   goes in. */
                <button type="button" className="nm-slot-empty" onClick={() => setOwning(index)}>
                  <span className="nm-slot-ord" aria-hidden="true" lang="sa-Deva">
                    {C.app.tray.ordinals[index]}
                  </span>
                  <span className="nm-slot-add" aria-hidden="true">
                    {C.app.tray.ownGlyph}
                  </span>
                  <span className="sr-only">{C.app.tray.ownLabel(index + 1)}</span>
                </button>
              )}
            </li>
          ))}
        </ol>

        {/* The way out of the tray. It appears only when there is something to
            send — before that it would be a control for nothing, and after the
            form opens it would be a second copy of a button already on screen.

            ITS SPACE IS RESERVED FROM THE START, and that is not cosmetic. The
            tray is vertically centred, so a child arriving re-centres the group
            above it: measured, the first Keep moved slot 1 up by 27px WHILE the
            name was flying into it, because the flight is aimed at the slot's
            box at the moment of the click and the button mounted mid-arc. The
            token then landed 27px below where its name appeared. Slots 2 and 3
            were reported as fine and measured as fine — by then the button
            already existed and nothing moved. */}
        <div className="nm-tray-send-slot">
          {picks.length > 0 && !formShown && (
            <button type="button" className="nm-tray-send" onClick={openForm}>
              {C.app.send.open(picks.length)}
            </button>
          )}
          {/* A line about what sending does has to sit WITH the thing that
              sends. In a flex row it rendered 250px to the button's right,
              over the sunset, at 3.45:1 — correct copy in the correct
              component and nowhere near the decision. */}
          {/* AND IT IS HERE BEFORE THERE IS ANYTHING TO SEND. This slot is
              reserved so the send button can appear without shoving the slots
              above it, which is right — but until a name is kept the reserve
              was 86px of empty valley on a 915px phone, between the three
              slots and the composer. Measured: slots end at 640, composer
              starts at 744.

              The promise is true before the first pick and after it — it says
              where the names go, under a heading that already says which
              names. So the space that was scenery now answers the question
              somebody has while they are deciding, and nothing moves when the
              button arrives above it. */}
          {!formShown && (
            <p className="nm-tray-promise label-mono label-mono--sm">{C.app.send.promise}</p>
          )}
        </div>
      </section>


      {/* ── THE SEND FORM, AS A DIALOG ────────────────────────────────────────
          It used to arrive as another turn in the conversation, which put a
          name field, a relation select and a reason box into a scroller whose
          job is reading — and on a phone it pushed the three names it is about
          off the screen entirely.

          Sending is a decision, not a remark, so it gets the page's attention:
          a centred sheet over a dimmed room. Real dialog semantics rather than
          a styled div — labelled, modal, Escape closes it, focus is trapped
          while it is open and returned to the button that opened it, because
          the alternative is a keyboard user tabbing into a page they cannot
          see. */}
      {formShown && (
        <div className="nm-overlay">
          {/* The scrim is a BUTTON, not a div with a click handler. Clicking
              away is a real action and it needs a real control with a name —
              a listener on a div is unreachable by keyboard and silent to a
              screen reader. It is behind the sheet and aria-hidden from the
              reading order, because Escape and "Not yet" already say this. */}
          <button
            type="button"
            className="nm-overlay-scrim"
            aria-label={C.app.send.closeAria}
            tabIndex={-1}
            onClick={closeForm}
          />
          <div
            className="nm-overlay-sheet"
            role="dialog"
            aria-modal="true"
            aria-labelledby="nm-send-title"
            ref={overlayRef}
          >
            <p className="nm-said nm-said--lead" id="nm-send-title">
              {C.app.send.lead}
            </p>
            <form className="nm-send" onSubmit={(event) => void sendPicks(event)}>
              <p className="label-mono label-mono--sm nm-quiet-label">{C.app.send.picksLabel}</p>
              <p className="nm-send-picks">{picks.map((pick) => pick.spelling).join(' · ') || C.form.picks.empty}</p>

              <div className="nm-send-row">
                <span className="nm-field">
                  <label className="label-mono label-mono--sm" htmlFor="nma-from">
                    {C.form.name.label}
                  </label>
                  <input id="nma-from" name="from" type="text" required maxLength={C.limits.name} autoComplete="name" />
                </span>
                <span className="nm-field">
                  <label className="label-mono label-mono--sm" htmlFor="nma-relation">
                    {C.form.relation.label}
                  </label>
                  <select id="nma-relation" name="relation" required defaultValue="">
                    <option value="">{C.form.relation.placeholder}</option>
                    {NAAM_RELATIONS.map((relation) => (
                      <option value={relation} key={relation}>
                        {relation}
                      </option>
                    ))}
                  </select>
                </span>
              </div>

              {/* The "or just type a name" field used to sit here. It has moved to
                  the empty slots, where it belongs: a name is a name whether it
                  came out of the document or out of somebody's head, and the tray
                  is where names go. A field in the form asked the visitor to
                  switch surfaces to do the same thing twice. */}

              <span className="nm-field">
                <label className="label-mono label-mono--sm" htmlFor="nma-reason">
                  {C.app.send.why}
                </label>
                <textarea id="nma-reason" name="reason" rows={2} maxLength={C.limits.reason}></textarea>
              </span>

              <button type="submit" className="nm-send-go" disabled={sending || picks.length === 0}>
                {C.app.send.submit}
              </button>

              {sending && (
                <div className="nm-thinking" aria-hidden="true">
                  <div className="pulse-line"></div>
                  <p className="pulse-caption nm-caption">{C.form.sending}</p>
                </div>
              )}
              {sendNote && <p className="nm-said nm-said--note">{sendNote}</p>}
            </form>
            <button type="button" className="nm-overlay-close" onClick={closeForm}>
              {C.app.send.close}
            </button>
          </div>
        </div>
      )}

      {/* ── THE HAND, ON THE CHAT SIDE ────────────────────────────────────────
          Asked for, and it comes with a caveat this file already recorded: the
          cards used to live INSIDE the conversation and were moved out because
          they did not fit. Measured at 1280x800, the stream is 439px, a hand is
          ~370 of it, and the model's own sentence ended up at top:-180 — above
          the fold, unreachable at any scroll offset.

          So they are on this side but NOT in the scroller. The conversation
          keeps its own box and the hand gets its own band beneath it, the way
          the tray used to work — which means a reply can never be pushed off
          the screen by cards again, and the names being weighed still sit in
          the column the visitor is reading and typing in.

          The room comes from the tray: with the cards gone it holds only the
          lamp and the three slots, so it gives back the height this needs. */}
      {hand ? (
        <div className="nm-hand">
          {dealCards(hand.matches)}
          {/* UNDER THE RESULT, because a control that acts on something belongs
              beside it. Hidden while a turn is in flight so a tap cannot queue
              a second request on top of the first. */}
          {refineChips.length > 0 && !asking && (
            <ul className="nm-refine" aria-label={C.app.refine.label}>
              {refineChips.map((chip) => (
                <li key={chip.id}>
                  <button
                    type="button"
                    className="nm-refine-chip"
                    onClick={() => runAsk(chip.prompt)}
                  >
                    {chip.label}
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>
      ) : (
        /* THE DOCUMENT, OPENED. Same band, same cards, same Keep — no new
           component and no new geometry, so the measured phone layout (two
           whole cards plus the edge of a third, naam.astro's 10.25rem) holds
           by construction.

           Gated on opening.done for the reason L3 gives about the sky: this
           ANSWERS the invitation, it does not accompany it. Arriving while
           "What kind of name are you looking for?" is still being said would
           answer a question the page has not finished asking.

           It is a shelf, not a hand. Twelve in a rail that shows two at a time
           reads as "swipe through the document"; three would read as "here are
           your three", which is a complete task — and forty guests sending the
           same three names is the one way this page fails at its actual job. */
        /* NO GATE. It was gated on opening.done — first keepable name at
           4,818ms, measured — then on the source line, 3,612ms. Both were the
           wrong shape of answer.

           These cards are SERVER-RENDERED. Gating them behind a JS clock hides
           markup that is already in the document on first paint, which is the
           exact fault this shelf was built to fix: naam.astro was computing
           twelve cited cards and then setting display:none on them. Delaying
           them is the same mistake wearing a timer.

           And it is consistent with everything else in this band: the three
           empty slots and the composer are both on screen from the first frame.
           A band of names is not more of an interruption than an empty tray.

           The opening still stages the WORDS, which is where the sequence
           lives (L1, L2). The shelf is furniture, like the tray — it does not
           speak, so it does not need a turn. */
        arrivalHand.length > 0 && (
          <div className="nm-hand" data-arrival="true">
            {/* NO LABEL OVER THE SHELF. "A few from the document" named what
                the cards already are — twelve cited names, each with its
                meaning printed on it — and it was the only heading in a column
                that otherwise just speaks. Cut on the owner's instruction. */}
            {dealCards(arrivalHand)}
          </div>
        )
      )}

      {/* LAST IN-FLOW ITEM OF THE COLUMN, never position: fixed — a fixed
          composer sits against the layout viewport and ends up underneath the
          iOS keyboard permanently. */}
      <div className="nm-composer" data-typing={ask.trim().length > 0 ? 'true' : undefined}>
        {/* Present for the whole conversation, refreshing as they are used.
            THE SLOT IS ALWAYS HERE, THE CHIPS ARE NOT. The starters are picked
            at random on mount, so they cannot be server-rendered without a
            hydration mismatch — which meant the row appeared 364ms in on
            desktop and 1269ms in on a phone and shoved the composer down 59px
            in a single frame. The slot is plain markup with the height of one
            chip row, so it is in the layout from the first paint and the chips
            fade into a space that was already theirs. */}
        {/* ── ONE ROW, TWO OCCUPANTS ─────────────────────────────────────────
            The status ("reading the document…") used to sit BELOW the input and
            collapse when the document arrived, on a `grid-template-rows`
            transition — which is a layout animation by definition, and it ran
            during boot, the worst moment to reflow a column. Measured at 4x on
            a 412px phone: 44.7ms of layout and 206ms of style recalculation for
            one 420ms collapse.

            The two things were always the same moment anyway: the status says
            what is happening, the chips say what you can do, and the second
            replaces the first. So they share the row that was already reserved
            above the input, crossfading on opacity alone. The composer's height
            never changes, so there is no layout to animate. */}
        {/* AND THE ROW STAYS EMPTY RATHER THAN CLOSING, which was measured
            and reverted. Once the refinements show, this slot can never be
            filled again, so collapsing it looked like free space — 44px of
            nothing between the chips and the box. Collapsing it moved the
            COMPOSER 58px and the hand 47px on the first deal. The composer and
            the tray are the two things that hold still through every
            interaction on this page; a visitor navigates by them. Dead space
            above the box costs less than the box moving under a thumb. */}
        <div className="nm-starters-slot">
        <p className="nm-cue-reading label-mono label-mono--sm" data-on={notReady && !dataFailed ? 'true' : undefined} aria-hidden="true">
          {C.app.reading}
        </p>
        {/* THE EXAMPLES WAIT FOR THE QUESTION. These three chips are ways of
            answering "what kind of name are you looking for?" — and that line
            is the invitation's THIRD block, which does not land until ~3.7s.
            Filmed, they were on screen at 700ms: three answers offered three
            seconds before anything had been asked, which reads as a menu rather
            than as help (L1, L3).

            They arrive with the ask now, in the same beat — a question and its
            examples are one thought, not two events. The row's height is still
            reserved from the first frame, so nothing moves when they land. */}
        {/* ONE CHIP RAIL AT A TIME, NEVER TWO. After a deal the refine chips
            sit above this row and these examples sat under them: two rows of
            identical pills, 60px apart, one meaning "ask me something like
            this" and the other meaning "narrow what you just got". Measured at
            1440: .nm-refine at y=682 with four chips, .nm-starters at y=742
            with three. Nothing on either row says which is which.

            The examples teach how to answer the invitation, and once a visitor
            has answered it they have been taught. From then on the refinements
            are the tool, because they compose onto the sentence that person
            actually wrote. */}
        {starters.length > 0 && opening.done && refineChips.length === 0 && (
          <ul className="nm-starters" aria-label={C.app.startersLabel} data-on={notReady ? undefined : 'true'}>
            {starters.map((starter) => (
              <li key={starter.id}>
                <button
                  type="button"
                  className="nm-starter"
                  onClick={() => {
                    opening.rush()
                    setAsk('')
                    usedStarters.current.add(starter.id)
                    setStarters(pickStarters(3, usedStarters.current))
                    runAsk(starter.prompt)
                  }}
                >
                  {starter.label}
                </button>
              </li>
            ))}
          </ul>
        )}
        </div>
        <div className="nm-composer-in">
          {/* Only where the browser actually has it. No shim, no upsell, and
              nothing at all to see where it is unsupported. */}
          {speech.available && (
            <button
              type="button"
              className="nm-speak"
              aria-pressed={speech.listening}
              disabled={!live || asking}
              onClick={() => speech.toggle()}
            >
              <span className="nm-speak-glyph" aria-hidden="true" data-on={speech.listening ? 'true' : undefined} />
              {/* THE WORD IS ON THE PAGE. An unlabelled glyph for an action is
                  the exact fault L10 was earned on — the Keep control spent a
                  session as a 14px ring with its verb in an .sr-only span, and
                  no sighted visitor ever saw it. Same corner, same footprint,
                  carrying its own word. */}
              <span className="nm-speak-word label-mono label-mono--sm" aria-hidden="true">
                {speech.listening ? C.app.speak.listening : C.app.speak.idle}
              </span>
              <span className="sr-only">{speech.listening ? C.app.speak.listening : C.app.speak.idle}</span>
            </button>
          )}
          <label className="sr-only" htmlFor="nma-ask">
            {C.app.composerLabel}
          </label>
          <input
            id="nma-ask"
            ref={inputRef}
            className="nm-composer-input"
            type="text"
            value={ask}
            placeholder={live ? C.app.composerPlaceholder : C.app.reading}
            autoComplete="off"
            maxLength={400}
            disabled={!live}
            onFocus={bumpCalm}
            onBlur={bumpCalm}
            onChange={(event) => {
              setAsk(event.target.value)
              bumpCalm()
              /* THE FIRST KEYSTROKE ENDS THE PERFORMANCE, not the first submit.
                 `rush()` was wired to submitAsk, under a comment reading
                 "typing is consent to get on with it" — which is what it should
                 do and was not what it did. Measured: typing at 700ms left the
                 blocks arriving at 3773ms, exactly as if nobody had touched
                 anything, so someone composing a sentence did it with the page
                 still performing in their peripheral vision.

                 Somebody who has started typing has told you they are ready.
                 Idempotent, so every later keystroke is free (L4). */
              opening.rush()
            }}
            onKeyDown={(event) => {
              if (event.key !== 'Enter') return
              event.preventDefault()
              submitAsk()
            }}
          />
          <button
            type="button"
            className="nm-composer-go"
            /* Not gated on the dataset — Enter is not either, and a send
               button that refuses a sentence the Enter key accepts is two
               different answers to the same question. runAsk waits for the
               rows itself. */
            disabled={!live || asking || ask.trim().length === 0}
            onClick={submitAsk}
          >
            <span aria-hidden="true">↑</span>
            <span className="sr-only">{C.app.composerSend}</span>
          </button>
        </div>
        {/* What the bar used to hold, at the weight it actually deserves: one
            hairline row under the composer, which is where a visitor's eye
            already is. `live` is not decoration — DESIGN.md §4 requires the
            honesty word to be visible whenever a model produced what is on
            screen, and .nm-badge--rail keeps its class so the verification
            drill still reads it. */}
        <div className="nm-utility">
          <span
            className={source ? 'nm-badge nm-badge--rail label-mono label-mono--sm' : 'nm-badge--rail'}
            data-source={source || undefined}
          >
            {source ? BADGE[source] : ''}
          </span>

          {/* Default off, and pressing it is BOTH the preference and the
              gesture the autoplay policy requires — so there is no second
              "enable audio" step and nothing makes a noise at somebody who did
              not ask. The label says which state it is in; aria-pressed carries
              what pressing it will do. */}
          <button
            type="button"
            className="nm-sound label-mono label-mono--sm"
            aria-pressed={audible}
            onClick={() => {
              const next = !audible
              setSound(next)
              // Struck on the way ON only, and it is the sample: this is what
              // you have just switched on, at the volume it will be.
              if (next) playCue('land')
            }}
          >
            <span className="nm-sound-glyph" aria-hidden="true" data-on={audible ? 'true' : undefined} />
            <span className="nm-sound-word">{audible ? C.app.sound.on : C.app.sound.off}</span>
          </button>

          {/* WCAG 2.2.2, and it sits next to the sound switch because it is the
              same kind of thing: an ambient channel this page turns on by
              itself, with the visitor's word on whether it stays on. Same
              grammar too — the label names the state, aria-pressed carries what
              pressing will do. */}
          <button
            type="button"
            className="nm-sound label-mono label-mono--sm"
            aria-pressed={!stilled}
            onClick={() => setStilled((was) => !was)}
          >
            {/* Same chip, rotated — the word is clipped below 600px and two
                identical dots in a row are two controls nobody can tell
                apart. See .nm-sky-glyph in naam.astro. */}
            <span
              className="nm-sound-glyph nm-sky-glyph"
              aria-hidden="true"
              data-on={!stilled ? 'true' : undefined}
            />
            <span className="nm-sound-word">{stilled ? C.app.motion.still : C.app.motion.moving}</span>
          </button>

          <a className="nm-rail-a11y label-mono label-mono--sm" href="/accessibility-statement">
            {C.app.a11yLink}
          </a>
        </div>
      </div>

      {/* ── THE THREAD ────────────────────────────────────────────────────
          Last in the shell, and that placement is the whole reason it works.
          The card band carries the column's paper and is a positioned sibling
          of the transcript, so a thread drawn inside the transcript was simply
          painted over below the divider — which is exactly how the strand came
          to stop at the boundary. Drawn after both bands it crosses them, and
          `z-index: 0` keeps it under the marker beads, which is the right way
          round: the beads are ON the string. */}
      <div className="nm-thread" aria-hidden="true" />

      {/* Flying names live here. It is rendered with no children, so React
          never diffs what is inside it. */}
      <div className="nm-fx" ref={fxRef} aria-hidden="true"></div>

      {/* The one live region, created EMPTY and written only when the agent
          says something. NOT role="log" on the stream: a log announces
          everything appended to it, including the visitor's own message read
          back to them. */}
      <p className="sr-only" aria-live="polite" aria-atomic="true">
        {announce}
      </p>
    </div>
  )
}

/**
 * A seated name follows the page-wide व/ब preference, script and spelling
 * together — a slot reading वस्तु over "Bastu" is what tells the one reader who
 * can read it that the swap is not happening. `pick.spelling` is the spelling
 * at the moment of the pick and is what gets SUBMITTED; it is also the only
 * thing there is to show before the dataset lands.
 */
function seatedLatin(rows: NaamRow[] | null, pick: NaamPick, preferB: boolean): string {
  const row = rows?.find((r) => r.id === pick.id)
  return row ? naamPreferredForm(row, preferB) : pick.spelling
}

function seatedDeva(rows: NaamRow[] | null, pick: NaamPick, preferB: boolean): string {
  const row = rows?.find((r) => r.id === pick.id)
  return row ? naamPreferredDevanagari(row, preferB) : ''
}
