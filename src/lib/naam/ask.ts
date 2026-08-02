/**
 * The free-form path's client half (docs/design/DESIGN.md §4, P9).
 *
 * WHY this is a module and not a function inside the island: the badge on
 * /naam is decided here, and the decision is worth more than the UI that shows
 * it. The browser's matcher picks a pool, this asks /api/naam-chat to reorder
 * and frame it, and every id that comes back is resolved against the local
 * dataset — so the island renders rows it already had, never model text. A
 * hallucinated name is structurally impossible rather than merely unlikely, and
 * when Bedrock is off, slow or broken the caller still has the matcher's own
 * list to show. That is the whole reason the page may wear LIVE at all, and it
 * should not be re-implemented once per surface.
 *
 * THE RETURN IS A VERDICT, NOT THE ENDPOINT'S BODY. /api/naam-chat answers
 * HTTP 200 for every outcome, degraded ones included (§4 rule 4: the LOCAL
 * branch is a signal, not an exception) — so a caller reading `res.ok` learns
 * nothing. NaamAskResult says which of the three things happened:
 *
 *   live         the model answered, and `rows` are OUR rows, resolved by id
 *   degraded     the endpoint said so, and `reason` picks the honest line
 *   unreachable  the fetch never completed — offline, aborted, or timed out
 *
 * `unreachable` is only ever a genuine network or abort failure, because the
 * route never returns a 5xx. All three are answerable: the caller has already
 * ranked the document locally before calling, so none of them is an error
 * state and none of them is empty.
 *
 * Teardown: pass the mount's AbortSignal. A per-request controller is chained
 * off it so the CHAT_TIMEOUT_MS deadline can fire without tearing the island
 * down, and both are cleaned up in `finally` — an unremoved abort listener on a
 * long-lived mount signal is a leak that survives every view transition.
 *
 * Nothing here writes prose: reason strings come from match.ts, visible lines
 * from copy.ts.
 */
import { NAAM_COPY } from './copy'
import { normalizePrefs, parseFreeText, pool, scoreName, type NaamMatch, type Prefs } from './match'
import type { NaamRow } from '@/types/naam'

const C = NAAM_COPY

/** Eight cards is what a stream turn can hold before it becomes a list. */
export const RESULT_MAX = 8
/** What the model is allowed to talk about. The route caps its own side at 60. */
export const POOL_SIZE = 40
/**
 * Longer than the route's own 11s budget by a second, so the server's honest
 * `timeout` reason wins the race and the visitor is told which end was slow.
 */
export const CHAT_TIMEOUT_MS = 12_000

/**
 * What the model call turned out to be. `rows` are rows of the local dataset,
 * looked up by id — never anything the model spelled.
 */
export type NaamAskResult =
  | { kind: 'live'; reply: string; rows: NaamRow[] }
  | { kind: 'degraded'; reason: string }
  | { kind: 'unreachable' }

/**
 * Ask the model about one sentence.
 *
 * Reads the sentence with the same parser the local path uses, hands
 * /api/naam-chat a pool of ids and the raw ask, and resolves whatever ids come
 * back against `rows`. The caller keeps the local answer it already computed
 * and replaces it only on `live`.
 */
export async function askNaam(args: {
  ask: string
  rows: readonly NaamRow[]
  signal: AbortSignal
}): Promise<NaamAskResult> {
  const { ask, rows, signal } = args
  const text = ask.trim()
  if (!text || rows.length === 0) return { kind: 'unreachable' }

  // parseFreeText already normalizes what it read; the second pass is
  // idempotent and keeps this module's guarantee to pool() its own, rather
  // than borrowed from a function it does not control.
  const prefs = normalizePrefs(parseFreeText(text, rows).prefs)
  const poolIds = pool(rows, prefs, POOL_SIZE).map((row) => row.id)
  if (poolIds.length === 0) return { kind: 'unreachable' }

  const request = new AbortController()
  const onOuterAbort = () => request.abort()
  if (signal.aborted) request.abort()
  else signal.addEventListener('abort', onOuterAbort, { once: true })
  const timer = setTimeout(() => request.abort(), CHAT_TIMEOUT_MS)

  try {
    const res = await fetch('/api/naam-chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ask: text, poolIds }),
      signal: request.signal,
    })
    const data: unknown = await res.json()
    const body = (data ?? {}) as { degraded?: boolean; reason?: unknown; reply?: unknown; pickIds?: unknown }

    if (body.degraded !== true && typeof body.reply === 'string' && body.reply.trim().length > 0) {
      const byId = new Map(rows.map((row) => [row.id, row]))
      const picked = (Array.isArray(body.pickIds) ? body.pickIds : [])
        .map((id) => (typeof id === 'string' ? byId.get(id) : undefined))
        .filter((row): row is NaamRow => Boolean(row))
      return { kind: 'live', reply: body.reply.trim(), rows: picked }
    }

    // `reason` is whatever crossed the wire, so it is narrowed here rather
    // than trusted. failureNote() maps anything it does not know to modelDown.
    return { kind: 'degraded', reason: typeof body.reason === 'string' ? body.reason : 'error' }
  } catch {
    /* aborted, offline, or no endpoint at all — the matcher already answered */
    return { kind: 'unreachable' }
  } finally {
    clearTimeout(timer)
    signal.removeEventListener('abort', onOuterAbort)
  }
}

/* ────────────────────────────────────────────────────────────────────────────
   helpers — every reason string still comes out of the matcher
   ──────────────────────────────────────────────────────────────────────────── */

export function withReasons(rows: readonly NaamRow[], prefs: Prefs): NaamMatch[] {
  return rows.map((row) => {
    const { score, reasons } = scoreName(row, prefs)
    return { row, score: Number.isFinite(score) ? score : 0, reasons }
  })
}

/** A name the visitor asked about outranks anything the wish alone found. */
export function mergeNamed(named: readonly NaamRow[], ranked: readonly NaamMatch[], prefs: Prefs): NaamMatch[] {
  if (named.length === 0) return [...ranked]
  const seen = new Set(named.map((row) => row.id))
  return [...withReasons(named, prefs), ...ranked.filter((m) => !seen.has(m.row.id))].slice(0, RESULT_MAX)
}

export function failureNote(reason: unknown): string {
  if (reason === 'timeout') return C.failure.modelSlow
  if (reason === 'unset') return C.failure.modelOff
  return C.failure.modelDown
}
