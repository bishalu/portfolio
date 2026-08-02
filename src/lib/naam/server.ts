/**
 * The four /naam endpoints' shared plumbing (docs/design/DESIGN.md §4, P8, P9).
 *
 * WHY one module: submit, approve and wall all need the same Blobs store, the
 * same rate limiter and the same JSON envelope, and chat and submit both need
 * to resolve a row id against the dataset. Four copies of a store helper is
 * how the wrong one gets fixed.
 *
 * THE STORE HELPER IS INVERTED ON PURPOSE. netlify/functions/live-signals.ts
 * does it the other way round — explicit `{ siteID, token }` first, ambient
 * `getStore(name)` only as a fallback — and that is backwards. Passing
 * credentials builds a client from scratch and discards `uncachedEdgeURL` from
 * the ambient Netlify context, which is the only path that can serve
 * `consistency: 'strong'`. So: ask for ambient first, and fall back to
 * credentials only when there is genuinely no context
 * (MissingBlobsEnvironmentError), which is the local-CLI case. The wall reads
 * its own writes; that requires strong consistency; that requires ambient.
 *
 * THE DATASET IS FETCHED, NOT BUNDLED. `public/naam/names-core.json` is 1.2 MB
 * and `names-rest.json` another 2.5 MB. Importing either into the SSR bundle
 * would put it in every route's module graph and hand `tsc` a 2,098-element
 * literal to infer. The Lambda fetches them from the site's own CDN instead,
 * once per warm instance, and `names-rest.json` only when a submitted id
 * missed the shortlist — which is rare, because the guided path only ever
 * serves shortlist rows.
 *
 * IT FETCHES THEM FROM A CONSTANT ORIGIN, NEVER FROM `request.url`. Every one
 * of these endpoints used to derive its own origin from the incoming request,
 * which put the Host header in charge of where "the document" comes from. A
 * forged Host made the Lambda fetch the dataset from the attacker's server,
 * cached that in module scope for the life of the warm instance, validated the
 * attacker's fabricated row ids against it, and then POSTed the record — HMAC
 * approve link and all — to the same forged origin. That is the grounding
 * guarantee, the moderation gate and the approve token, all lost to one
 * header. Netlify's edge happens to 404 a mismatched Host in production, but
 * the guarantee cannot rest on an undocumented platform behaviour the code
 * neither owns nor asserts, and it was fully live under `astro dev` and
 * `netlify dev`. See NAAM_ORIGIN below.
 */
import { getStore, type Store } from '@netlify/blobs'
import type { NaamRow } from '@/types/naam'

/* ────────────────────────────────────────────────────────────────────────────
   THE ORIGIN

   One constant, resolved once, never from the request. Order:
     NAAM_ORIGIN         explicit override — point it at localhost for `astro dev`
     URL                 Netlify's own primary site URL (platform-set)
     import.meta.env.SITE  astro.config.mjs `site`, baked in at build time
   ──────────────────────────────────────────────────────────────────────────── */

function toOrigin(value: string | undefined): string {
  if (!value) return ''
  try {
    return new URL(value).origin
  } catch {
    return ''
  }
}

export const NAAM_ORIGIN: string =
  toOrigin(process.env.NAAM_ORIGIN) ||
  toOrigin(process.env.URL) ||
  toOrigin(import.meta.env.SITE as string | undefined) ||
  ''

/* ────────────────────────────────────────────────────────────────────────────
   RESPONSES

   Nothing on this page returns a 5xx. DESIGN.md §4 rule 4: failure is honest,
   never blank — and the client's LOCAL branch is a signal, not an exception.
   ──────────────────────────────────────────────────────────────────────────── */

export function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json', 'Cache-Control': 'no-store' },
  })
}

/* ────────────────────────────────────────────────────────────────────────────
   RATE LIMITING

   Copied from netlify/functions/vibeset-demo.ts, including its honesty: this
   is per warm Lambda instance — best effort, not a wall. It stops a loop, not
   a determined actor.

   Keyed by endpoint AND ip, not by ip alone: submitting a suggestion and
   asking a question are different budgets, and one map keyed on ip only means
   a visitor who asked six questions cannot then send their picks.

   NO x-forwarded-for. It is client-supplied, so keying on it means the limit
   is bypassed by rotating a header — measured: same IP hit 429 on the sixth
   submit, a rotating X-Forwarded-For never did. Netlify sets
   x-nf-client-connection-ip itself and that is checked first; behind any other
   runtime the adapter's own client address is asked for, and when neither
   exists the key is a constant, so everyone shares one bucket. A shared bucket
   is a worse experience than a per-visitor one and a much better one than no
   limit at all.

   A per-ip limit is also no defence against a distributed flood of a route
   that costs real money, so `ceilinged()` puts a process-wide wall behind it.
   ──────────────────────────────────────────────────────────────────────────── */

const hits = new Map<string, { count: number; windowStart: number }>()
/** Keys are client-influenced, so the map is bounded and dropped wholesale. */
const HITS_MAX_KEYS = 5000

export function clientIp(request: Request, address?: () => string | undefined): string {
  const platform = request.headers.get('x-nf-client-connection-ip')
  if (platform) return platform
  try {
    const direct = address?.()
    if (direct) return direct
  } catch {
    /* the adapter does not expose a client address */
  }
  return 'unknown'
}

export function rateLimited(scope: string, ip: string, perMinute: number): boolean {
  const key = `${scope}:${ip}`
  const now = Date.now()
  const entry = hits.get(key)
  if (!entry || now - entry.windowStart > 60_000) {
    if (hits.size >= HITS_MAX_KEYS) hits.clear()
    hits.set(key, { count: 1, windowStart: now })
    return false
  }
  entry.count += 1
  return entry.count > perMinute
}

const ceilings = new Map<string, { count: number; windowStart: number }>()

/**
 * A process-wide budget, independent of who is asking. `/api/naam-chat` is an
 * unauthenticated call to a paid model; a distributed flood defeats a per-ip
 * limit by definition, and this is the wall it hits instead. Per warm instance,
 * like everything else here — it caps a runaway, not an adversary with a
 * botnet and patience.
 */
export function ceilinged(scope: string, perMinute: number): boolean {
  const now = Date.now()
  const entry = ceilings.get(scope)
  if (!entry || now - entry.windowStart > 60_000) {
    ceilings.set(scope, { count: 1, windowStart: now })
    return false
  }
  entry.count += 1
  return entry.count > perMinute
}

/* ────────────────────────────────────────────────────────────────────────────
   BLOBS
   ──────────────────────────────────────────────────────────────────────────── */

export const NAAM_STORE = 'naam-suggestions'
export const WALL_KEY = 'wall.json'

/** One approved suggestion, denormalized into wall.json. */
export interface NaamWallEntry {
  id: string
  from: string
  relation: string
  picks: Array<{ id: string; spelling: string }>
  /**
   * Names typed rather than picked. The no-JS path has no way to fill `picks`,
   * and a name someone loves is often not in a Vedic corpus anyway — which is
   * the wall's own point.
   */
  names: string
  reason: string
  at: string
}

/**
 * Ambient context first — see the header. Returns null rather than throwing:
 * every caller degrades to "the wall stays as it is", which is honest, and the
 * Netlify Forms email is the durable path for a submission either way.
 */
export function naamStore(): Store | null {
  try {
    return getStore(NAAM_STORE)
  } catch (err) {
    if ((err as Error)?.name !== 'MissingBlobsEnvironmentError') return null
    const siteID = process.env.NETLIFY_BLOBS_SITE_ID || process.env.BLOBS_SITE_ID || process.env.NETLIFY_SITE_ID || ''
    const token = process.env.NETLIFY_BLOBS_TOKEN || process.env.BLOBS_TOKEN || process.env.NETLIFY_AUTH_TOKEN || ''
    if (!siteID || !token) return null
    try {
      return getStore(NAAM_STORE, { siteID, token })
    } catch {
      return null
    }
  }
}

/* ────────────────────────────────────────────────────────────────────────────
   THE DATASET
   ──────────────────────────────────────────────────────────────────────────── */

const DATA_TIMEOUT_MS = 6000

/**
 * ONLY A NON-EMPTY RESULT IS CACHED. The first version of this wrote `[]` into
 * the cache on failure and then returned it forever, because `[]` is truthy —
 * so one transient CDN blip bricked the warm instance: every free-form ask
 * degraded to LOCAL for good, and, much worse, every legitimate submission was
 * rejected with `error: 'picks'` because `idsExist()` could no longer resolve a
 * real id. The moderation queue stopped receiving and the page still said
 * "sent". An empty dataset is never a valid answer here — the build asserts
 * 2,098 core rows — so emptiness means "try again", not "the document is
 * empty".
 *
 * `inFlight` keeps a cold-start burst to one fetch instead of one per request.
 */
let coreRows: NaamRow[] | null = null
let coreInFlight: Promise<NaamRow[]> | null = null
let restIds: Set<string> | null = null
let restInFlight: Promise<Set<string>> | null = null

async function fetchJson(path: string): Promise<unknown> {
  if (!NAAM_ORIGIN) throw new Error('no site origin configured')
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), DATA_TIMEOUT_MS)
  try {
    const res = await fetch(new URL(path, NAAM_ORIGIN).toString(), { signal: controller.signal })
    if (!res.ok) throw new Error(`${path} ${res.status}`)
    return await res.json()
  } finally {
    clearTimeout(timer)
  }
}

/** The 2,098 core rows, cached per warm instance once it has some. [] on failure. */
export async function loadCoreRows(): Promise<NaamRow[]> {
  if (coreRows && coreRows.length > 0) return coreRows
  if (!coreInFlight) {
    coreInFlight = fetchJson('/naam/names-core.json')
      .then((data) => {
        const rows = Array.isArray(data) ? (data as NaamRow[]) : []
        if (rows.length > 0) coreRows = rows
        return rows
      })
      .catch(() => [] as NaamRow[])
      .finally(() => {
        coreInFlight = null
      })
  }
  return coreInFlight
}

/**
 * Does this id name a real row? Checks the core rows first and only pulls the
 * other 4,617 when it has to, because that is the 2.5 MB half.
 */
export async function idsExist(ids: readonly string[]): Promise<Set<string>> {
  const found = new Set<string>()
  if (ids.length === 0) return found
  const core = await loadCoreRows()
  const coreIds = new Set(core.map((row) => row.id))
  const missing = ids.filter((id) => {
    if (coreIds.has(id)) {
      found.add(id)
      return false
    }
    return true
  })
  if (missing.length === 0) return found

  const rest = await loadRestIds()
  for (const id of missing) if (rest.has(id)) found.add(id)
  return found
}

/** Same rule as the core rows: an empty set is a failure, not an answer. */
async function loadRestIds(): Promise<Set<string>> {
  if (restIds && restIds.size > 0) return restIds
  if (!restInFlight) {
    restInFlight = fetchJson('/naam/names-rest.json')
      .then((data) => {
        const ids = new Set(Array.isArray(data) ? (data as NaamRow[]).map((row) => row.id) : [])
        if (ids.size > 0) restIds = ids
        return ids
      })
      .catch(() => new Set<string>())
      .finally(() => {
        restInFlight = null
      })
  }
  return restInFlight
}

/* ────────────────────────────────────────────────────────────────────────────
   INPUT HYGIENE

   Everything a stranger types passes through here before it is written down.
   Rendering is textContent / JSX interpolation everywhere, so this is defence
   in depth rather than the defence — but a control character in a blob is
   still a control character in an email.
   ──────────────────────────────────────────────────────────────────────────── */

/** Control characters out, runs of whitespace collapsed, trimmed, capped. */
export function tidy(value: unknown, max: number): string {
  return (
    String(value ?? '')
      // eslint-disable-next-line no-control-regex -- stripping them is the point
      .replace(/[\u0000-\u001f\u007f]/g, ' ')
      .replace(/\s+/g, ' ')
      .trim()
      .slice(0, max)
  )
}

/** Row ids are slugs minted by the dataset build: lowercase, digits, dashes. */
export function isRowId(value: unknown): value is string {
  return typeof value === 'string' && /^[a-z][a-z0-9-]{0,48}$/.test(value)
}
