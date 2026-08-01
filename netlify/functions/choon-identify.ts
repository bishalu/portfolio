/**
 * Choon identify proxy — powers the stress test on /vibeset/choon.
 *
 * Why a proxy at all: the browser should never see the upstream URL, the call
 * needs a hard timeout and a per-IP limit, and every failure path has to return
 * something rather than blank the panel (docs/design/DESIGN.md §4).
 *
 * Why no credentials: the matcher itself (audiofp-api) sits behind IAM, but the
 * Choon demo frontend already runs inside GCP, authenticates to it with the
 * metadata server, and exposes the route publicly. So this forwards to that
 * gateway. The alternative — a service-account key in Netlify — is blocked
 * org-wide by constraints/iam.disableServiceAccountKeyCreation, and rightly so:
 * nothing in the fingerprinting repo uses a long-lived key either.
 *
 * Env:
 *   CHOON_API_BASE  the public Choon gateway. Not committed — this repo is
 *                   public. Unset means we fail closed to recorded results.
 *
 * POST { preset: 'clean'|'subway'|'nightcore'|'fried' }
 *   → { match, tier, confidence, classical, neural, ms, title, artist, source }
 */

const API_BASE = process.env.CHOON_API_BASE ?? ''

/** The demo track. FMA, Creative Commons — the catalog family the 66k benchmark used. */
const TRACK_ID = '93710'

/**
 * The four presets, expressed as parameters the matcher itself accepts. The
 * browser applies the same numbers to the same track, so what the model is
 * asked to identify is what you just heard — not a second rendering of it.
 */
const PRESETS: Record<string, Record<string, number>> = {
  clean: {},
  subway: { lowPassFreq: 2200, noise: 0.14, bitcrush: 12 },
  nightcore: { playbackRate: 1.26, noise: 0.02, lowPassFreq: 16000 },
  fried: { bitcrush: 6, lowPassFreq: 3400, noise: 0.05 },
}

/** Recorded from this same endpoint, so the replay path is real output, just older. */
const FALLBACK: Record<string, { tier: string; confidence: number; classical: number; neural: number; ms: number }> = {
  clean: { tier: 'classical', confidence: 0.909, classical: 0.295, neural: 0.0, ms: 13405 },
  subway: { tier: 'neural', confidence: 0.922, classical: 0.002, neural: 0.36, ms: 2682 },
  nightcore: { tier: 'neural', confidence: 0.984, classical: 0.008, neural: 0.93, ms: 2403 },
  fried: { tier: 'neural', confidence: 0.865, classical: 0.002, neural: 0.076, ms: 2073 },
}
const FALLBACK_TRACK = { title: 'Palm Tree', artist: 'The Chapin Sisters' }

// The matcher is an ML service: warm it answers in ~5s, cold it spends 30s+
// loading the model. Cap well under Netlify's ceiling and fall back instead.
const UPSTREAM_TIMEOUT_MS = 20_000
const RATE_LIMIT_PER_MIN = 8
const ipHits = new Map<string, { count: number; windowStart: number }>()

function rateLimited(ip: string): boolean {
  const now = Date.now()
  const e = ipHits.get(ip)
  if (!e || now - e.windowStart > 60_000) {
    ipHits.set(ip, { count: 1, windowStart: now })
    return false
  }
  e.count++
  return e.count > RATE_LIMIT_PER_MIN
}

const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json', 'Cache-Control': 'no-store' },
  })

const replay = (preset: string) =>
  json({ ...FALLBACK[preset], ...FALLBACK_TRACK, match: true, source: 'replay' })

export default async (req: Request) => {
  if (req.method !== 'POST') return json({ error: 'POST only' }, 405)

  let payload: { preset?: string }
  try {
    payload = await req.json()
  } catch {
    return json({ error: 'invalid JSON body' }, 400)
  }

  const preset = String(payload.preset ?? '')
  if (!(preset in PRESETS)) return json({ error: 'unknown preset' }, 400)

  const ip = req.headers.get('x-nf-client-connection-ip') || req.headers.get('x-forwarded-for') || 'unknown'
  if (rateLimited(ip) || !API_BASE) return replay(preset)

  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), UPSTREAM_TIMEOUT_MS)
  try {
    const res = await fetch(`${API_BASE}/api/simulate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ track_id: TRACK_ID, ...PRESETS[preset] }),
      signal: controller.signal,
    })
    if (!res.ok) throw new Error(`upstream ${res.status}`)
    const r = (await res.json()) as Record<string, unknown>
    if (!r.match_found) throw new Error('no match')

    return json({
      match: true,
      tier: String(r.tier ?? ''),
      confidence: Number(r.confidence ?? 0),
      classical: Number(r.classical_score ?? 0),
      neural: Number(r.neural_score ?? 0),
      ms: Math.round(Number(r.latency_ms ?? 0)),
      title: String(r.title ?? ''),
      artist: String(r.artist ?? ''),
      source: 'live',
    })
  } catch {
    return replay(preset)
  } finally {
    clearTimeout(timer)
  }
}

export const config = { path: '/api/choon-identify' }
