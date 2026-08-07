/**
 * Request plumbing for the site's own endpoints.
 *
 * The JSON envelope, client-ip resolution and rate limiter here mirror the ones
 * in lib/naam/server.ts. That is a deliberate second copy, not an oversight:
 * /naam is a self-contained feature with its own owner, and coupling the
 * contact endpoint to its module would mean pulling in the Blobs client and the
 * dataset loader in order to send an email. If the two ever converge, this is
 * the copy to keep.
 *
 * The honesty of the original carries over unchanged: the rate limit is per
 * warm Lambda instance. It stops a loop, not a determined actor.
 */

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
