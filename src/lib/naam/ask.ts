/**
 * The free-form path's client half (docs/design/DESIGN.md §4, P9).
 *
 * WHY this is a module and not a function inside the island: the badge on
 * /naam is decided here, and the decision is worth more than the UI that shows
 * it. This module does two separable jobs and it is worth naming them apart:
 *
 *   readAsk()   the MATCHER's job. Read the sentence, normalize the wish, and
 *               build the pool of ids the model is allowed to choose from.
 *   askNaam()   the MODEL's job. POST that pool to /api/naam-chat and resolve
 *               every id that comes back against the local dataset.
 *
 * THE AGENT LEADS, AND NOTHING RENDERS BEFORE IT. This module used to promise
 * the caller "you have already ranked the document locally, so none of these
 * outcomes is empty" — and the island duly rendered the matcher's own list the
 * instant it was asked, then swapped it for the model's. The model was doing
 * real work and the page read as though it were not, because the matcher's
 * answer arrived first and the model's looked like a correction to it. So the
 * local ranking is gone from the ask path: readAsk() builds a pool, askNaam()
 * asks, and the reply is the event.
 *
 * GROUNDING IS UNCHANGED, and it never lived in the local render anyway. It
 * lives in the pool: the model is handed ids, it answers with ids, and the
 * island renders rows it already had. A hallucinated name stays structurally
 * impossible rather than merely unlikely, which is the whole reason the page
 * may wear LIVE at all.
 *
 * THE RETURN IS A VERDICT, NOT THE ENDPOINT'S BODY. /api/naam-chat answers
 * HTTP 200 for every outcome, degraded ones included — so a caller reading
 * `res.ok` learns nothing. NaamAskResult says which of the three things
 * happened:
 *
 *   live         the model answered, and `rows` are OUR rows, resolved by id
 *   degraded     the endpoint said so, and `reason` picks the honest line
 *   unreachable  the fetch never completed — offline, aborted, or timed out
 *
 * `unreachable` is only ever a genuine network or abort failure, because the
 * route never returns a 5xx. The two failing branches are not fallbacks now:
 * the caller says so plainly and offers to run the matcher, and a visitor who
 * wants the document's own answer asks for it (§4 rule 4 — failure is honest,
 * never blank; it does not say honest means silently substituted).
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
import {
  normalizePrefs,
  parseFreeText,
  passesHardFilters,
  pool,
  rankRelaxed,
  relatedTo,
  retrieve,
  scoreName,
  type NaamHit,
  type NaamMatch,
  type NaamNearMiss,
  type Prefs,
} from './match'
import type { NaamRow } from '@/types/naam'
/**
 * Everyday English → this lexicon's own vocabulary, built offline by
 * scripts/naam/build-thesaurus.mjs and fetched alongside the rows.
 *
 * READ SYNCHRONOUSLY AND ALLOWED TO BE EMPTY. readAsk() is synchronous and runs
 * the moment someone submits; the table is requested at mount, beside a 1.22 MB
 * row payload the page cannot work without, so in practice it is always there.
 * When it is not, retrieval simply behaves as it did before it existed — the
 * gap goes uncrossed, nothing breaks, and the agent's own search still covers
 * it. That is why this is a getter and not a required argument.
 */
import { currentThesaurus } from './tray'

const C = NAAM_COPY

/** Eight cards is what a stream turn can hold before it becomes a list. */
export const RESULT_MAX = 8
/** What the model is allowed to talk about. The route caps its own side at 60. */
export const POOL_SIZE = 40
/**
 * Neighbours per name the visitor mentioned. Six is enough for the model to
 * have a real choice among names like that one, and small enough that two
 * mentioned names cannot crowd out the pool the wish asked for.
 */
const RELATED_PER_NAME = 6
/**
 * How many rows the meaning search may put at the head of the pool. Generous,
 * because this is the part that actually answers the question — but well under
 * POOL_SIZE so a wish with both a meaning and a shape ("a short name about the
 * moon") still leaves room for the shape-ranked candidates behind it.
 */
const RETRIEVE_MAX = 22
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

/** What the matcher read out of one sentence, before anyone calls a model. */
export interface NaamRead {
  /** The wish, normalized. What the model's picks are scored against. */
  prefs: Prefs
  /** Names the visitor put on the table themselves — a lookup or a comparison. */
  named: NaamRow[]
  /**
   * What searching the document's own meanings turned up, with the terms each
   * row matched. Empty means the words they used appear in no definition at
   * all — which is the signal that the concept needs rethinking rather than
   * that the document has nothing.
   */
  hits: NaamHit[]
  /**
   * A name they typed that this document does not contain, with the nearest
   * rows it does. Carried through to the model so it can say so plainly rather
   * than answering a question nobody asked.
   */
  near: NaamNearMiss[]
  /** The ids the model may choose from. Named rows first, then the pool. */
  poolIds: string[]
  /**
   * The wish intersected to nothing and the ladder gave a rung back, so the
   * candidates answer most of what was asked rather than all of it. Said out
   * loud by the caller; never quietly swallowed.
   */
  relaxed: boolean
}

/**
 * Read one sentence and decide what the model is allowed to talk about.
 *
 * THE NAMED ROWS GO IN FIRST, and that is not a nicety. "what does Snehaja
 * mean" parses to a lookup with almost no wish attached, so pool() ranks it by
 * the standing quality prior and the one name actually asked about can fall
 * outside the top forty — in which case the model is structurally unable to
 * answer the question it was asked. The old local-first render papered over
 * this by merging the lookup back in on the client after the fact.
 *
 * THE LADDER IS THE MATCHER'S, NOT A SECOND ONE. `letters`, `syllables` and
 * `easySay` are hard filters, so a reasonable wish — one syllable, V, easy for
 * a cousin to say — can intersect to nothing and hand the model an empty pool,
 * which reads to a visitor as "the model didn't answer" when the truth is that
 * nobody asked it anything. rankRelaxed() gives the constraints back one at a
 * time in the order a person would, and reports that it did.
 */
export function readAsk(text: string, rows: readonly NaamRow[]): NaamRead {
  const parsed = parseFreeText(text, rows)
  // parseFreeText already normalizes what it read; the second pass is
  // idempotent and keeps this module's guarantee to pool() its own, rather
  // than borrowed from a function it does not control.
  const prefs = normalizePrefs(parsed.prefs)
  const named = parsed.compare.length > 0 ? parsed.compare : parsed.lookups

  let candidates = pool(rows, prefs, POOL_SIZE)
  let relaxed = false
  if (candidates.length === 0) {
    const fallback = rankRelaxed(rows, prefs, POOL_SIZE)
    candidates = fallback.matches.map((match) => match.row)
    relaxed = fallback.relaxed
  }

  /**
   * NAMING A NAME IS THE STRONGEST SIGNAL A VISITOR EVER GIVES, and until now
   * it bought them only that one row. "I like Bhaskara" is a statement about a
   * whole shape — a meaning, a length, a sound — so the names STANDING NEXT TO
   * the ones they named now go into the pool right behind them, ahead of the
   * general ranking. Same for a near miss: if they typed a name this document
   * does not have, the closest things it does have are what the answer is made
   * of, and "Sanskar" — one of the family's own favourites, and not a row —
   * is exactly that case.
   *
   * Order is the whole of it. The pool is capped, the model reads it in order
   * and may choose only from it, so what goes in first is what the visitor
   * gets an answer about.
   */
  const neighbours = named.flatMap((row) => relatedTo(row, rows, RELATED_PER_NAME))
  const nearRows = parsed.near.flatMap((miss) => miss.rows)

  /**
   * THE MEANINGS COME FIRST, and this is the change that matters.
   *
   * Everything above answers "what SHAPE of name did they ask for" — a letter,
   * a length, a name they already like. None of it answers "what did they ask
   * it to MEAN", because that used to be handled by mapping the sentence onto
   * nineteen hand-tagged themes, and a concept outside those nineteen produced
   * no signal whatsoever. The pool then fell back to its quality-ranked default
   * and the model was handed forty names that had nothing to do with the
   * question. "Moon" returned none of the thirty-six rows whose meaning
   * mentions the moon.
   *
   * retrieve() reads the document's own definitions instead. It goes ahead of
   * the ranked candidates because a name that MEANS what somebody asked for is
   * a better answer than a name that merely scores well.
   *
   * The visitor's shape constraints still apply to it — asking for a short moon
   * name should not return a four-syllable one — but only while they leave
   * something behind. A hard filter that empties the retrieval is a filter that
   * has thrown away the only rows that answered the question, so it is dropped
   * and the meaning wins.
   */
  const hits = retrieve(rows, text, RETRIEVE_MAX, currentThesaurus())
  const shaped = hits.filter((hit) => passesHardFilters(hit.row, prefs))
  const meant = (shaped.length > 0 ? shaped : hits).map((hit) => hit.row)

  const poolIds = [
    ...new Set([...named, ...nearRows, ...meant, ...neighbours, ...candidates].map((row) => row.id)),
  ].slice(0, POOL_SIZE)
  return { prefs, named, near: parsed.near, hits, poolIds, relaxed }
}

/**
 * Ask the model about one sentence.
 *
 * Hands /api/naam-chat the pool readAsk() built and the raw ask, and resolves
 * whatever ids come back against `rows`. Nothing is on screen behind this call
 * but the thinking state, so the answer is the first thing the visitor reads.
 */
export async function askNaam(args: {
  ask: string
  poolIds: readonly string[]
  rows: readonly NaamRow[]
  /** Names they typed that the document does not have, so the model can say so. */
  absent?: readonly string[]
  signal: AbortSignal
}): Promise<NaamAskResult> {
  const { ask, poolIds, rows, absent = [], signal } = args
  const text = ask.trim()
  if (!text || poolIds.length === 0) return { kind: 'unreachable' }

  const request = new AbortController()
  const onOuterAbort = () => request.abort()
  if (signal.aborted) request.abort()
  else signal.addEventListener('abort', onOuterAbort, { once: true })
  const timer = setTimeout(() => request.abort(), CHAT_TIMEOUT_MS)

  try {
    const res = await fetch('/api/naam-chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ask: text, poolIds: [...poolIds], absent: [...absent] }),
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
    /* aborted, offline, or no endpoint at all — and the caller says so */
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
