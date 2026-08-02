/**
 * The browser-side state /naam holds outside React (docs/design/DESIGN.md §4,
 * P8, P10).
 *
 * WHY one module instead of three: the page has exactly three pieces of state
 * that outlive a single component — the picks tray, the व/ब display
 * preference, and the parsed dataset. It had three readers when the page was
 * an editorial document (two islands and a vanilla script upgrading the
 * server-rendered cards); the rebuild left one, NaamApp.tsx.
 *
 * SO WHY IS THIS STILL A STORE. Two reasons that did not go away with the
 * second island. The dataset is a 1.2 MB parse that must survive an unmount
 * and must not be re-fetched, and the tray is sessionStorage-backed, so the
 * state has a lifetime the component does not. Module scope plus a
 * subscription is what `useSyncExternalStore` wants; folding it into a hook
 * would make both of those the component's problem.
 *
 * THREE THINGS TO KNOW BEFORE EDITING:
 *
 *   1. The initial state is a CONSTANT. sessionStorage is not read until
 *      hydrate() is called from an effect, because /naam is prerendered and
 *      React 19 compares the hydrated tree literally — a storage read during
 *      the first render would produce a mismatch on any visitor who had
 *      already picked something.
 *
 *   2. The swap is ONE preference for the whole page, not one per card. That
 *      is what src/types/naam.ts models (`naamPreferredForm(row, preferB)`
 *      takes a single boolean) and it is what makes the control on the first
 *      tick of every V name legible: flipping it anywhere re-letters every V
 *      name everywhere. `row.latin` is never mutated — this is a display
 *      choice, and the spelling the visitor was looking at when they picked a
 *      name is recorded on the pick itself.
 *
 *   3. The dataset is fetched once per page load and cached in module scope,
 *      so a back-navigation re-renders from memory instead of re-parsing
 *      1.1 MB of JSON. names-rest.json is only ever fetched behind an explicit
 *      request (DESIGN.md P8: never render 2,000 rows, never spin).
 */
import { NAAM_COPY } from './copy'
import { naamPreferredForm, type NaamRow } from '@/types/naam'

/** A name the visitor put in the tray, with the spelling they were shown. */
export interface NaamPick {
  id: string
  /** naamPreferredForm() at the moment of the pick — what they are recommending. */
  spelling: string
}

type Listener = () => void

/**
 * v2 because the cap went from six to three. `hydrate()` would have truncated a
 * stored six to the first three, which is safe but silent — and on a page whose
 * whole signature is a tray filling up, a visitor would have lost three slots
 * with no explanation, chosen by insertion order rather than by preference.
 * A new key ignores stale trays outright. This is sessionStorage, so the
 * affected group is only people mid-visit across the deploy.
 */
const PICKS_KEY = 'naam.picks.v2'
const SWAP_KEY = 'naam.swap.v1'

/** The cap the form and the endpoint both agree on. */
export const PICK_MAX = NAAM_COPY.limits.picks

/** Stable empty reference — useSyncExternalStore compares snapshots by identity. */
const NO_PICKS: readonly NaamPick[] = Object.freeze([])

let picks: readonly NaamPick[] = NO_PICKS
let preferB = true
let hydrated = false

const listeners = new Set<Listener>()

function emit(): void {
  for (const fn of [...listeners]) fn()
}

/** Subscribe to picks + swap. One channel; both are cheap to re-render. */
export function subscribe(fn: Listener): () => void {
  listeners.add(fn)
  return () => {
    listeners.delete(fn)
  }
}

/* ────────────────────────────────────────────────────────────────────────────
   PERSISTENCE — after mount only. See note 1 above.
   ──────────────────────────────────────────────────────────────────────────── */

/**
 * Read what this device already had. Safe to call any number of times; the
 * first call wins. Never throws: sessionStorage is unavailable in some privacy
 * modes and a missing tray is not worth a broken page.
 */
export function hydrate(): void {
  if (hydrated) return
  hydrated = true
  let changed = false
  try {
    const rawSwap = sessionStorage.getItem(SWAP_KEY)
    if (rawSwap === '0' || rawSwap === '1') {
      const next = rawSwap === '1'
      if (next !== preferB) {
        preferB = next
        changed = true
      }
    }
    const rawPicks = sessionStorage.getItem(PICKS_KEY)
    if (rawPicks) {
      const parsed: unknown = JSON.parse(rawPicks)
      const clean = Array.isArray(parsed) ? parsed.filter(isPick).slice(0, PICK_MAX) : []
      if (clean.length > 0) {
        picks = Object.freeze(clean)
        changed = true
      }
    }
  } catch {
    /* no storage, no tray — the page still works */
  }
  if (changed) emit()
}

function isPick(value: unknown): value is NaamPick {
  if (!value || typeof value !== 'object') return false
  const v = value as Record<string, unknown>
  return typeof v.id === 'string' && typeof v.spelling === 'string' && v.id.length < 64
}

function persist(): void {
  try {
    sessionStorage.setItem(PICKS_KEY, JSON.stringify(picks))
    sessionStorage.setItem(SWAP_KEY, preferB ? '1' : '0')
  } catch {
    /* see hydrate() */
  }
}

/* ────────────────────────────────────────────────────────────────────────────
   THE TRAY
   ──────────────────────────────────────────────────────────────────────────── */

export function getPicks(): readonly NaamPick[] {
  return picks
}

/** The server snapshot, and the client's first snapshot before hydrate(). */
export function getEmptyPicks(): readonly NaamPick[] {
  return NO_PICKS
}

export function isPicked(id: string): boolean {
  return picks.some((p) => p.id === id)
}

/** true when the tray is at the cap and `id` is not already in it. */
export function isTrayFull(id: string): boolean {
  return picks.length >= PICK_MAX && !isPicked(id)
}

/** Add or remove. Records the spelling currently on display (see note 2). */
export function togglePick(row: NaamRow): void {
  if (isPicked(row.id)) {
    picks = Object.freeze(picks.filter((p) => p.id !== row.id))
  } else {
    if (picks.length >= PICK_MAX) return
    picks = Object.freeze([...picks, { id: row.id, spelling: naamPreferredForm(row, preferB) }])
  }
  persist()
  emit()
}

export function removePick(id: string): void {
  if (!isPicked(id)) return
  picks = Object.freeze(picks.filter((p) => p.id !== id))
  persist()
  emit()
}

export function clearPicks(): void {
  if (picks.length === 0) return
  picks = NO_PICKS
  persist()
  emit()
}

/* ────────────────────────────────────────────────────────────────────────────
   THE SWAP — one preference, page-wide. See note 2.
   ──────────────────────────────────────────────────────────────────────────── */

export function getPreferB(): boolean {
  return preferB
}

/** The constant the server rendered with, and the client's first snapshot. */
export function getDefaultPreferB(): boolean {
  return true
}

export function toggleSwap(): void {
  preferB = !preferB
  persist()
  emit()
}

/* ────────────────────────────────────────────────────────────────────────────
   THE DATASET — fetched once, cached in module scope. See note 3.
   ──────────────────────────────────────────────────────────────────────────── */

/**
 * Run `fn` when the browser is not busy, with a setTimeout fallback. Returns
 * its own canceller so a component that unmounts before the callback fires
 * does not schedule work for a page that is gone.
 */
export function onIdle(fn: () => void): () => void {
  if (typeof window === 'undefined') return () => {}
  const ric = window.requestIdleCallback
  if (typeof ric === 'function') {
    const handle = ric(fn, { timeout: 2000 })
    return () => window.cancelIdleCallback?.(handle)
  }
  const handle = window.setTimeout(fn, 200)
  return () => window.clearTimeout(handle)
}

let corePromise: Promise<NaamRow[]> | null = null
let restPromise: Promise<NaamRow[]> | null = null
let coreRows: NaamRow[] | null = null
let allRows: NaamRow[] | null = null

/** Whatever is already parsed, without starting a fetch. */
export function peekRows(): NaamRow[] | null {
  return allRows ?? coreRows
}

/** True once names-rest.json has been pulled in. */
export function hasAllRows(): boolean {
  return allRows !== null
}

async function fetchRows(url: string): Promise<NaamRow[]> {
  const res = await fetch(url)
  if (!res.ok) throw new Error(`${url} ${res.status}`)
  const data: unknown = await res.json()
  if (!Array.isArray(data)) throw new Error(`${url} is not an array`)
  return data as NaamRow[]
}

/**
 * The 2,098 core rows. A failed load clears the cache so the next mount
 * retries rather than inheriting a rejected promise.
 *
 * NO SIGNAL. This promise is shared by three consumers — both islands and the
 * page's enhancement script — and it used to be bound to whichever of them
 * called first. That is a latent bug with a nasty shape: today all three abort
 * together on `astro:before-swap`, so nothing shows, but the day one of them
 * aborts alone the other two inherit a rejected promise and render
 * "the full list did not load" for a load that was fine. A shared cache cannot
 * take one caller's lifetime. Every consumer already re-checks its own signal
 * before using the result, which is the part that actually matters.
 */
export function loadCoreRows(): Promise<NaamRow[]> {
  if (!corePromise) {
    corePromise = fetchRows('/naam/names-core.json')
      .then((rows) => {
        coreRows = rows
        return rows
      })
      .catch((err) => {
        corePromise = null
        throw err
      })
  }
  return corePromise
}

/** Core plus the other 4,617. Only ever called from an explicit control. */
export function loadAllRows(): Promise<NaamRow[]> {
  if (!restPromise) {
    restPromise = Promise.all([loadCoreRows(), fetchRows('/naam/names-rest.json')])
      .then(([core, rest]) => {
        allRows = [...core, ...rest]
        return allRows
      })
      .catch((err) => {
        restPromise = null
        throw err
      })
  }
  return restPromise
}
