import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState, useSyncExternalStore } from 'react'
import type { CSSProperties, FormEvent } from 'react'
import NaamCard from './NaamCard'
import NaamDiyo, { type DiyoState } from './NaamDiyo'
import NaamWall, { type WallNote } from './NaamWall'
import { askNaam, failureNote, readAsk, withReasons } from '@/lib/naam/ask'
import { NAAM_COPY, NAAM_RELATIONS } from '@/lib/naam/copy'
import type { NaamMatch } from '@/lib/naam/match'
import { hydrateSound, playCue, setSound, soundOff, soundOn, subscribeSound } from '@/lib/naam/sound'
import { NAAM_SEED_ROWS } from '@/lib/naam/seeds'
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
  subscribe,
  togglePick,
  toggleSwap,
  PICK_MAX,
  type NaamPick,
} from '@/lib/naam/tray'
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
const FORM_DELAY_MS = 620
/** Ambient motion stays suspended this long after the last keystroke. */
const CALM_MS = 800
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
}

export default function NaamApp({ seed }: NaamAppProps) {
  const [rows, setRows] = useState<NaamRow[] | null>(null)
  /** Approved suggestions from /api/naam-wall. Empty until one is approved. */
  const [wall, setWall] = useState<readonly WallEntry[]>([])
  const [dataFailed, setDataFailed] = useState(false)
  const [turns, setTurns] = useState<Turn[]>(() => [
    { id: 'greeting', kind: 'agent', text: C.app.greeting, lead: true },
    // What the list actually is — the count, the two corpora, the three
    // letters — stated once, quietly, before anything is asked. Without it a
    // visitor cannot tell whether this page is reading a real document or
    // inventing names, which is the one thing it must never be ambiguous about.
    { id: 'source', kind: 'agent', text: C.app.source, quiet: true },
    { id: 'invitation', kind: 'agent', text: C.app.invitation },
  ])
  const [ask, setAsk] = useState('')
  const [asking, setAsking] = useState(false)
  /**
   * The lamp reacts rather than reporting: it leans while the model is out and
   * flares once when a name seats. `flare` is transient and NaamDiyo clears it
   * through onFlareEnd, so a second Keep mid-flare restarts the animation
   * instead of being swallowed.
   */
  const [flare, setFlare] = useState(0)
  const diyo: DiyoState = flare > 0 ? 'flare' : asking ? 'thinking' : 'idle'
  const endFlare = useCallback(() => setFlare(0), [])
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
  const shellRef = useRef<HTMLDivElement | null>(null)
  const valleyRef = useRef<HTMLCanvasElement | null>(null)
  /** The scene loads ~260ms after mount, long after the first notes exist, so
   *  the count is kept in a ref for it to read on arrival — and the handle is
   *  kept so later arrivals can be pushed straight through. */
  const lampCountRef = useRef(0)
  const valleyHandleRef = useRef<{ setLamps(n: number): void } | null>(null)
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
    let handle: { setLamps(n: number): void; resize(): void; destroy(): void } | null = null
    let cancelled = false
    const still = reducedMotion()

    const start = window.setTimeout(async () => {
      try {
        const { createValley } = await import('@/lib/naam/scene/valley')
        if (cancelled) return
        handle = await createValley({ canvas, still })
        if (cancelled) {
          handle.destroy()
          return
        }
        valleyHandleRef.current = handle
        handle.setLamps(lampCountRef.current)
      } catch {
        // WebGL blocked, context lost, chunk failed — all the same outcome, and
        // it is not an error state. The gradient below is a complete page.
      }
    }, 260)

    // No pointer listener: the valley does not pan. What moves is the flags and
    // the birds, on their own clock — a window you look through, not a widget
    // that follows the cursor.
    const onResize = () => handle?.resize()
    window.addEventListener('resize', onResize, { passive: true })

    return () => {
      cancelled = true
      window.clearTimeout(start)
      window.removeEventListener('resize', onResize)
      valleyHandleRef.current = null
      handle?.destroy()
    }
  }, [])

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

  useLayoutEffect(() => {
    const el = streamRef.current
    if (!el || !pinned || !asked) return
    el.scrollTo({ top: el.scrollHeight, behavior: reducedMotion() ? 'auto' : 'smooth' })
  }, [turns, pinned, asked])

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
    async (value: string, replaceId?: string) => {
      const dataset = rows
      const mount = mountRef.current
      if (!dataset || !mount || asking || value.length === 0) return

      const { prefs, poolIds, near } = readAsk(value, dataset)
      const thinkingId = nextId()
      setAsking(true)
      setTurns((prev) => [
        ...(replaceId ? prev.filter((turn) => turn.id !== replaceId) : prev),
        { id: thinkingId, kind: 'thinking', caption: C.app.asking },
      ])
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
    [asking, rows],
  )

  /** What the visitor said, then the model's turn. */
  const runAsk = useCallback(
    (text: string) => {
      const value = text.trim()
      if (!rows || asking || value.length === 0) return

      setTurns((prev) => [...prev, { id: nextId(), kind: 'you', text: value }])
      void runModel(value)
    },
    [asking, rows, runModel],
  )

  const submitAsk = useCallback(() => {
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

      // The lamp answers the touch. Bumped on every keep and unkeep, before any
      // frame is scheduled, so it fires even when the flight itself is skipped
      // under reduced motion — the flare is CSS and that media query stills it.
      setFlare((n) => n + 1)

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
              embers()
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
  const openForm = useCallback(() => {
    if (formShown) return
    setFormShown(true)
    later(
      () => {
        setTurns((prev) => [
          ...prev,
          { id: nextId(), kind: 'agent', text: C.app.send.lead },
          { id: nextId(), kind: 'form' },
        ])
        setAnnounce(C.app.send.lead)
      },
      reducedMotion() ? 0 : FORM_DELAY_MS,
    )
  }, [formShown, later])

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

      const encoded = new URLSearchParams({
        'form-name': 'naam-suggestion',
        from,
        relation,
        picks: JSON.stringify(cited),
        names,
        reason,
      })

      /**
       * TWO POSTS, AND NEITHER IS ALLOWED TO CANCEL THE OTHER.
       *
       * This used to send the Netlify Forms post first and RETURN on any
       * non-2xx from it, so a failure of the best-effort email path threw away
       * a suggestion the durable queue would have accepted. The dependency was
       * exactly backwards: Forms is the notification, /api/naam-submit is the
       * record. Caught locally, where `netlify serve` answers POST / with 405
       * and every submission silently aborted before reaching the queue at all.
       *
       * They run together now and the visitor is only told it failed if BOTH
       * did. `Promise.allSettled`, not `all`: one rejecting must not take the
       * other's result with it.
       */
      const [mailed, stored] = await Promise.allSettled([
        fetch('/', {
          method: 'POST',
          headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
          body: encoded.toString(),
          keepalive: true,
        }),
        fetch('/api/naam-submit', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ from, relation, reason, names, picks: cited }),
          keepalive: true,
        }),
      ])

      const emailed = mailed.status === 'fulfilled' && mailed.value.ok
      let queued = false
      let rateLimited = false
      if (stored.status === 'fulfilled') {
        rateLimited = stored.value.status === 429
        try {
          const body = (await stored.value.json()) as { ok?: unknown; stored?: unknown }
          queued = body.ok === true && body.stored === true
        } catch {
          /* a body we cannot read is not a stored suggestion */
        }
      }

      // Nothing got through at all — the only case worth stopping for.
      if (!emailed && !queued && !rateLimited) {
        setSending(false)
        setSendNote(C.form.error.network)
        setAnnounce(C.form.error.network)
        return
      }

      const line = rateLimited
        ? C.form.error.rateLimited
        : queued
          ? C.form.confirmation.body
          : C.form.confirmation.emailOnly
      setSending(false)
      setTurns((prev) => prev.map((t) => (t.kind === 'form' ? { id: t.id, kind: 'sent', text: line } : t)))
      setAnnounce(line)
    },
    [picks, sending],
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
   * argue with: paper squares raining down belong to a birthday, and the one
   * light source on this screen is a flame. So the third name lifts a dozen
   * sparks off the lamp instead. They come from the diyo's own rect, they rise
   * and cool, and they are warm-only. Twelve of them, once.
   *
   * Math.random is safe here in a way it is not in render: this runs from a
   * click, long after hydration, so there is no server frame to disagree with.
   */
  const embers = useCallback(() => {
    const host = fxRef.current
    const lamp = shellRef.current?.querySelector('.nm-diyo')
    if (!host || !lamp || reducedMotion()) return
    const lit = lamp.getBoundingClientRect()
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
            onSwap={toggleSwap}
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
    const add = (key: string, note: Omit<WallNote, 'count'>) => {
      const found = byName.get(key)
      if (found) {
        found.count += 1
        // Two supporters and a single signature would credit one of them for
        // both, so the attribution drops the moment it stops being true.
        found.who = undefined
        found.mine = found.mine || note.mine
        return
      }
      byName.set(key, { ...note, count: 1 })
    }

    for (const row of family) {
      add(row.id, {
        key: row.id,
        deva: naamPreferredDevanagari(row, preferB),
        latin: naamPreferredForm(row, preferB),
      })
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

  /** One lamp per name, pushed to the scene whenever the family's list changes.
   *  The ref covers the case where the notes exist before the scene has loaded;
   *  the handle covers every change after it has. */
  useEffect(() => {
    lampCountRef.current = notes.length
    valleyHandleRef.current?.setLamps(notes.length)
  }, [notes.length])

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
       * doodle points at where to say it.
       *
       * The turn kind survives so a session stored before this change still
       * renders; it simply draws nothing.
       */
      case 'starters':
        return null

      case 'thinking':
        return (
          /* THE LAMP IS THE LOADING STATE. NaamDiyo is already leaning and
             burning hotter for the whole of this turn (§4c), and a travelling
             spike on a 2px rule beside it would be a second thing saying the
             same word — so the .pulse-line is gone from here and the caption
             stays, breathing on its own. It survives where nothing else is
             animating: the composer's "reading the document…" and the send
             form's "sending…", neither of which the flame is reporting on.
             Decorative; the announcement rides the live region. */
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
        return (
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
        )

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
      <li className="nm-turn" key={turn.id} data-turn={turn.id} data-past={past ? 'true' : undefined}>
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
      data-first={!asked ? 'true' : undefined}
      /* Typing before the first ask: the opening lifts and thins out, because
         the invitation has done its job the moment somebody answers it. */
      data-composing={!asked && ask.trim().length > 0 ? 'true' : undefined}
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
        </ol>

        {/* THE DOODLE. Drawn rather than written: a curve from the last line of
            the invitation down to the composer, with an arrowhead. It draws
            itself once on arrival, sways very slightly while it waits, and
            leaves the moment the visitor starts typing — a hint that outstays
            its answer is a nag. aria-hidden and pointer-events:none; the
            composer's own label is what a screen reader follows. */}
        {!asked && (
          <svg
            className="nm-doodle"
            viewBox="0 0 64 210"
            preserveAspectRatio="xMidYMax meet"
            aria-hidden="true"
            focusable="false"
          >
            {/* pathLength="100" normalises the dash maths, so the draw-on needs
                no getTotalLength() call and survives any edit to the curve or
                the viewBox. The first cut hard-coded a dasharray of 132 and
                would have silently half-drawn the moment the path changed. */}
            {/* IT WANDERS ON THE WAY DOWN. A clean arc from A to B is a
                connector; three changes of mind between them is somebody's
                hand. The curve overshoots left, right, left again and settles —
                the squiggles are not decoration, they are the reason it reads as
                drawn rather than as drawn-by-software.

                pathLength="100" normalises the dash maths, so the draw-on needs
                no getTotalLength() and survives any edit to this curve. It draws
                from the START of the path, which is the top — so the stroke
                travels the way a hand would, down toward the box. */}
            <path
              className="nm-doodle-line"
              d="M34 6C14 30 54 42 30 64C8 86 50 100 28 124C12 148 46 160 33 191"
              pathLength="100"
              fill="none"
              strokeLinecap="round"
            />
            {/* The two barbs are separate paths on separate beats, after the
                shaft has arrived. One path drawing shaft-and-head in a single
                continuous sweep is a stroke no hand can make, and the eye knows
                it even when it cannot say why. */}
            <path className="nm-doodle-barb" d="M23 178L33 191" pathLength="100" fill="none" strokeLinecap="round" />
            <path
              className="nm-doodle-barb nm-doodle-barb--b"
              d="M44 180L33 191"
              pathLength="100"
              fill="none"
              strokeLinecap="round"
            />
          </svg>
        )}

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
        {hand && <div className="nm-hand">{dealCards(hand.matches)}</div>}

        {/* THE SHELF, and it is on this side for a reason. In the conversation
            it was a thing that had already happened, scrolled past within two
            turns and never seen again. Here it is the surface the visitor is
            adding to: their kept names land on the same shelf the family's sit
            on, a name they agree with gains a bead instead of appearing twice,
            and the order re-sorts under them as they choose. The right column
            stops being an output and starts being the thing that changes. */}
        <div className="nm-shelfwrap">
          <NaamWall notes={notes} />
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
          <NaamDiyo state={diyo} onFlareEnd={endFlare} />
          <p className="label-mono label-mono--sm nm-quiet-label">{C.app.tray.label}</p>
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

        {/* The way out of the tray, and it appears only when there is something
            to send. Before that it would be a control for nothing; after the
            form is open it would be a second copy of a button already on
            screen. */}
        {picks.length > 0 && !formShown && (
          <button type="button" className="nm-tray-send" onClick={openForm}>
            {C.app.send.open(picks.length)}
          </button>
        )}
      </section>

      {/* LAST IN-FLOW ITEM OF THE COLUMN, never position: fixed — a fixed
          composer sits against the layout viewport and ends up underneath the
          iOS keyboard permanently. */}
      <div className="nm-composer" data-typing={ask.trim().length > 0 ? 'true' : undefined}>
        <div className="nm-composer-in">
          <label className="sr-only" htmlFor="nma-ask">
            {C.app.composerLabel}
          </label>
          <input
            id="nma-ask"
            ref={inputRef}
            className="nm-composer-input"
            type="text"
            value={ask}
            placeholder={notReady ? C.app.reading : C.app.composerPlaceholder}
            autoComplete="off"
            maxLength={400}
            disabled={notReady}
            onFocus={bumpCalm}
            onBlur={bumpCalm}
            onChange={(event) => {
              setAsk(event.target.value)
              bumpCalm()
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
            disabled={notReady || asking || ask.trim().length === 0}
            onClick={submitAsk}
          >
            <span aria-hidden="true">↑</span>
            <span className="sr-only">{C.app.composerSend}</span>
          </button>
        </div>
        {notReady && !dataFailed && (
          <div className="nm-thinking nm-thinking--composer" aria-hidden="true">
            <div className="pulse-line"></div>
            <p className="pulse-caption nm-caption">{C.app.reading}</p>
          </div>
        )}

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

          <a className="nm-rail-a11y label-mono label-mono--sm" href="/accessibility-statement">
            {C.app.a11yLink}
          </a>
        </div>
      </div>

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
