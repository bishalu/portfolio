import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState, useSyncExternalStore } from 'react'
import type { CSSProperties, FormEvent } from 'react'
import NaamCard from './NaamCard'
import { askNaam, failureNote, mergeNamed, withReasons, RESULT_MAX } from '@/lib/naam/ask'
import { NAAM_COPY, NAAM_RELATIONS } from '@/lib/naam/copy'
import { normalizePrefs, parseFreeText, rankRelaxed, type NaamMatch } from '@/lib/naam/match'
import { NAAM_SEED_ROWS } from '@/lib/naam/seeds'
import {
  getDefaultPreferB,
  getEmptyPicks,
  getPicks,
  getPreferB,
  hydrate,
  loadCoreRows,
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
 * THE MODEL NEVER NAMES A NAME. Every ask is ranked locally first and rendered
 * immediately with the LOCAL badge (§4: real algorithm, real data, in your
 * browser); src/lib/naam/ask.ts then asks /api/naam-chat to reorder that pool
 * and every id it returns is resolved against the local dataset. So a
 * hallucinated name is structurally impossible, and when Bedrock is off, slow
 * or broken the local names simply stand with an honest line under them
 * (§4 rule 4 — failure is honest, never blank).
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

/** The two badge words this page is allowed (§4). Never a third. */
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
  | { id: string; kind: 'agent'; text: string; lead?: boolean }
  | { id: string; kind: 'you'; text: string }
  | { id: string; kind: 'family' }
  | { id: string; kind: 'starters' }
  | { id: string; kind: 'thinking'; caption: string }
  | { id: string; kind: 'names'; matches: readonly NaamMatch[]; badge: Badge; note: string }
  | { id: string; kind: 'form' }
  | { id: string; kind: 'sent'; text: string }

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
  const [dataFailed, setDataFailed] = useState(false)
  const [turns, setTurns] = useState<Turn[]>(() => [
    { id: 'greeting', kind: 'agent', text: C.app.greeting, lead: true },
    { id: 'family', kind: 'family' },
    { id: 'invitation', kind: 'agent', text: C.app.invitation },
    { id: 'starters', kind: 'starters' },
  ])
  const [starters, setStarters] = useState<readonly string[]>(C.app.starters)
  const [ask, setAsk] = useState('')
  const [asking, setAsking] = useState(false)
  const [source, setSource] = useState<Badge>('local')
  const [announce, setAnnounce] = useState('')
  const [pinned, setPinned] = useState(true)
  const [calm, setCalm] = useState(false)
  const [formShown, setFormShown] = useState(false)
  const [sending, setSending] = useState(false)
  const [sendNote, setSendNote] = useState('')

  const mountRef = useRef<AbortController | null>(null)
  const shellRef = useRef<HTMLDivElement | null>(null)
  /** React renders this empty and never diffs its children, so it is the one
      safe place to park an imperatively created flying name. */
  const fxRef = useRef<HTMLDivElement | null>(null)
  const streamRef = useRef<HTMLOListElement | null>(null)
  const inputRef = useRef<HTMLInputElement | null>(null)
  const slotRefs = useRef<(HTMLElement | null)[]>([])
  const calmTimer = useRef(0)
  const timers = useRef<Set<number>>(new Set())

  const picks = useSyncExternalStore(subscribe, getPicks, getEmptyPicks)
  const preferB = useSyncExternalStore(subscribe, getPreferB, getDefaultPreferB)
  const pickedIds = useMemo(() => new Set(picks.map((p) => p.id)), [picks])

  /** A timer that cannot outlive the mount. */
  const later = useCallback((fn: () => void, ms: number) => {
    const id = window.setTimeout(() => {
      timers.current.delete(id)
      fn()
    }, ms)
    timers.current.add(id)
  }, [])

  /* — mount ————————————————————————————————————————————————————————— */

  useEffect(() => {
    const ac = new AbortController()
    mountRef.current = ac
    const { signal } = ac
    document.addEventListener('astro:before-swap', () => ac.abort(), { signal })

    hydrate()

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

  useLayoutEffect(() => {
    const el = streamRef.current
    if (!el || !pinned) return
    el.scrollTo({ top: el.scrollHeight, behavior: reducedMotion() ? 'auto' : 'smooth' })
  }, [turns, pinned])

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

  /* — the composer suspends ambient motion while it is being used ————— */

  const bumpCalm = useCallback(() => {
    setCalm(true)
    window.clearTimeout(calmTimer.current)
    calmTimer.current = window.setTimeout(() => {
      if (document.activeElement !== inputRef.current) setCalm(false)
    }, CALM_MS)
  }, [])

  /* — one ask ————————————————————————————————————————————————————————— */

  const runAsk = useCallback(
    async (text: string, typed: boolean) => {
      const dataset = rows
      const value = text.trim()
      const mount = mountRef.current
      if (!dataset || !mount || asking || value.length === 0) return

      // Chips are a way in, not a script. Once the visitor has said something
      // of their own the starters have done their job and leave.
      if (typed) setStarters([])
      else setStarters((prev) => prev.filter((chip) => chip !== text))

      const parsed = parseFreeText(value, dataset)
      const prefs = normalizePrefs(parsed.prefs)
      const named = parsed.compare.length > 0 ? parsed.compare : parsed.lookups
      const { matches: ranked, relaxed } = rankRelaxed(dataset, prefs, RESULT_MAX)
      const local = mergeNamed(named, ranked, prefs)

      const namesId = nextId()
      const thinkingId = nextId()
      setAsking(true)
      setSource('local')
      setTurns((prev) => [
        ...prev,
        { id: nextId(), kind: 'you', text: value },
        { id: nextId(), kind: 'agent', text: C.app.dealt },
        {
          id: namesId,
          kind: 'names',
          matches: local,
          badge: 'local',
          note: relaxed ? C.results.relaxed : C.badge.localCaption,
        },
        { id: thinkingId, kind: 'thinking', caption: C.app.asking },
      ])
      setAnnounce(local.length === 0 ? C.results.emptyAsk : C.badge.localCaption)

      const result = await askNaam({ ask: value, rows: dataset, signal: mount.signal })
      // askNaam never throws — an abort resolves to `unreachable` — so the
      // caller is the one that must not touch state on a dead mount.
      if (mount.signal.aborted) return

      let badge: Badge = 'local'
      // Annotated, or NAAM_COPY's `as const` narrows it to the one line it was
      // initialised with and the other three branches stop compiling.
      let note: string = C.failure.modelDown
      let reply = ''
      let matches: readonly NaamMatch[] | null = null

      if (result.kind === 'live') {
        badge = 'live'
        note = relaxed ? `${C.badge.liveCaption} ${C.results.relaxed}` : C.badge.liveCaption
        reply = result.reply
        // `rows` can be empty on a live answer — the model replied but named
        // nothing we could resolve. The local matches stay on screen.
        if (result.rows.length > 0) matches = withReasons(result.rows, prefs)
      } else if (result.kind === 'degraded') {
        note = failureNote(result.reason)
      }

      setSource(badge)
      setTurns((prev) => {
        const out: Turn[] = []
        for (const turn of prev) {
          if (turn.id === thinkingId) continue
          if (turn.id === namesId && turn.kind === 'names') {
            // The model's framing reads before the cards it is framing, so it
            // is inserted at the names turn rather than appended at the end.
            if (reply) out.push({ id: nextId(), kind: 'agent', text: reply })
            out.push({ ...turn, badge, note, matches: matches ?? turn.matches })
            continue
          }
          out.push(turn)
        }
        return out
      })
      setAnnounce(reply || note)
      setAsking(false)
    },
    [asking, rows],
  )

  const submitAsk = useCallback(() => {
    const value = ask.trim()
    if (value.length === 0) return
    setAsk('')
    void runAsk(value, true)
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
      const slot = slotRefs.current[index]
      const card = streamRef.current?.querySelector<HTMLElement>(
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

  /* — the last turn arrives when the hand is complete ————————————————— */

  useEffect(() => {
    if (formShown || picks.length < PICK_MAX) return
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
  }, [formShown, later, picks.length])

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
      setSending(true)
      setSendNote('')

      const encoded = new URLSearchParams({
        'form-name': 'naam-suggestion',
        from,
        relation,
        picks: JSON.stringify(picks),
        names: '',
        reason,
      })

      try {
        const res = await fetch('/', {
          method: 'POST',
          headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
          body: encoded.toString(),
          keepalive: true,
        })
        if (!res.ok) throw new Error(String(res.status))
      } catch {
        setSending(false)
        setSendNote(C.form.error.network)
        setAnnounce(C.form.error.network)
        return
      }

      let queued = false
      let rateLimited = false
      try {
        const res = await fetch('/api/naam-submit', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ from, relation, reason, names: '', picks }),
          keepalive: true,
        })
        rateLimited = res.status === 429
        const body = (await res.json()) as { ok?: unknown; stored?: unknown }
        queued = body.ok === true && body.stored === true
      } catch {
        /* the email path already succeeded */
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

  const renderTurn = (turn: Turn) => {
    switch (turn.kind) {
      case 'agent':
        return (
          <li className="nm-turn" key={turn.id}>
            <span className="sr-only">{C.app.speakerAgent}</span>
            <p className={turn.lead ? 'nm-said nm-said--lead' : 'nm-said'}>{turn.text}</p>
          </li>
        )

      case 'you':
        return (
          <li className="nm-turn nm-turn--you" key={turn.id}>
            <span className="sr-only">{C.app.speakerYou}</span>
            <p className="nm-said nm-said--you">{turn.text}</p>
          </li>
        )

      case 'family':
        return (
          <li className="nm-turn" key={turn.id}>
            <span className="sr-only">{C.app.speakerAgent}</span>
            <p className="label-mono label-mono--sm nm-quiet-label">{C.app.familyLead}</p>
            {/* eslint-disable-next-line jsx-a11y/no-redundant-roles -- `list-style: none`
                strips list semantics in Safari/VoiceOver, so the role is restorative
                rather than redundant. Every list on this page is unstyled. */}
            <ul className="nm-family" role="list">
              {family.map((row) => (
                <li className="nm-family-item" key={row.id}>
                  <span className="nm-family-deva" lang="sa-Deva">
                    {naamPreferredDevanagari(row, preferB)}
                  </span>
                  <span className="nm-family-latin">{naamPreferredForm(row, preferB)}</span>
                </li>
              ))}
            </ul>
          </li>
        )

      case 'starters':
        if (starters.length === 0) return null
        return (
          <li className="nm-turn" key={turn.id}>
            <span className="sr-only">{C.app.speakerAgent}</span>
            <div className="nm-chips">
              {starters.map((chip) => (
                <button
                  type="button"
                  className="nm-chip"
                  key={chip}
                  disabled={notReady || asking}
                  onClick={() => void runAsk(chip, false)}
                >
                  {chip}
                </button>
              ))}
              <button type="button" className="nm-chip-hide label-mono label-mono--sm" onClick={() => setStarters([])}>
                {C.app.dismissStarters}
              </button>
            </div>
          </li>
        )

      case 'thinking':
        return (
          <li className="nm-turn" key={turn.id}>
            <span className="sr-only">{C.app.speakerAgent}</span>
            {/* P8: a travelling spike on a 2px rule and a mono caption. Never a
                spinner. Decorative — the announcement rides the live region. */}
            <div className="nm-thinking" aria-hidden="true">
              <div className="pulse-line"></div>
              <p className="pulse-caption nm-caption">{turn.caption}</p>
            </div>
          </li>
        )

      case 'names':
        return (
          <li className="nm-turn" key={turn.id}>
            <span className="sr-only">{C.app.speakerAgent}</span>
            {turn.matches.length === 0 ? (
              <p className="nm-said">{C.results.emptyAsk}</p>
            ) : (
              <div className="nm-deal">
                {turn.matches.map((match, i) => (
                  <div
                    className="nm-dealt"
                    key={match.row.id}
                    style={{ '--i': i, '--tilt': `${seededTilt(match.row.id)}deg` } as CSSProperties}
                  >
                    <NaamCard
                      row={match.row}
                      preferB={preferB}
                      reasons={match.reasons}
                      picked={pickedIds.has(match.row.id)}
                      trayFull={picks.length >= PICK_MAX && !pickedIds.has(match.row.id)}
                      onSwap={toggleSwap}
                      onPick={() => keep(match.row)}
                    />
                  </div>
                ))}
              </div>
            )}
            <p className="nm-turn-note">
              <span className="nm-badge label-mono label-mono--sm" data-source={turn.badge}>
                {BADGE[turn.badge]}
              </span>
              <span className="nm-turn-note-text">{turn.note}</span>
            </p>
          </li>
        )

      case 'form':
        return (
          <li className="nm-turn" key={turn.id}>
            <span className="sr-only">{C.app.speakerAgent}</span>
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

              <span className="nm-field">
                <label className="label-mono label-mono--sm" htmlFor="nma-reason">
                  {C.app.send.why}
                </label>
                <textarea id="nma-reason" name="reason" rows={2} maxLength={C.limits.reason}></textarea>
              </span>

              <button type="submit" className="nm-send-go" disabled={sending || picks.length === 0}>
                {C.tray.send}
              </button>

              {sending && (
                <div className="nm-thinking" aria-hidden="true">
                  <div className="pulse-line"></div>
                  <p className="pulse-caption nm-caption">{C.form.sending}</p>
                </div>
              )}
              {sendNote && <p className="nm-said nm-said--note">{sendNote}</p>}
            </form>
          </li>
        )

      case 'sent':
        return (
          <li className="nm-turn" key={turn.id}>
            <span className="sr-only">{C.app.speakerAgent}</span>
            <p className="nm-said nm-said--lead">{C.form.confirmation.heading}</p>
            <p className="nm-said">{turn.text}</p>
          </li>
        )
    }
  }

  return (
    <div className="nm-shell" ref={shellRef} data-calm={calm ? 'true' : undefined}>
      {/* .nm-topbar, not .nm-rail: NaamCard's sound rail already owns that
          class name and this page renders both. */}
      <div className="nm-topbar">
        <h1 className="nm-brand">{C.app.heading}</h1>

        <button type="button" className="nm-swap-rail" aria-pressed={preferB} onClick={toggleSwap}>
          <span aria-hidden="true">{C.card.swapGlyph}</span>
          <span className="sr-only">{C.card.swapAria}</span>
        </button>

        <span className="nm-badge nm-badge--rail label-mono label-mono--sm" data-source={source}>
          {BADGE[source]}
        </span>
      </div>

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
          {dataFailed && (
            <li className="nm-turn" key="data-failed">
              <span className="sr-only">{C.app.speakerAgent}</span>
              <p className="nm-said">{C.failure.dataDown}</p>
            </li>
          )}
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
        <div className="nm-tray-head">
          <p className="label-mono label-mono--sm nm-quiet-label">{C.app.tray.label}</p>
          {/* The site footer is display:none on this page, so its index rides
              here — /naam must keep an inbound path to every route it had, the
              accessibility statement above all. */}
          <nav className="nm-index label-mono label-mono--sm" aria-label={C.app.indexLabel}>
            <a href="/">bishal.ai</a>
            <a href="/about">About</a>
            <a href="/research">Research</a>
            <a href="/notes/choon">Case note</a>
            <a href="/accessibility-statement">Accessibility</a>
          </nav>
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
              ) : (
                <span className="nm-slot-empty">
                  <span className="nm-slot-ord" aria-hidden="true" lang="sa-Deva">
                    {C.app.tray.ordinals[index]}
                  </span>
                  <span className="sr-only">{C.app.tray.empty(index + 1)}</span>
                </span>
              )}
            </li>
          ))}
        </ol>
      </section>

      {/* LAST IN-FLOW ITEM OF THE COLUMN, never position: fixed — a fixed
          composer sits against the layout viewport and ends up underneath the
          iOS keyboard permanently. */}
      <div className="nm-composer">
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
