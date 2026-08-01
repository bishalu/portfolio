/**
 * Choon identify proxy — powers the stress test on /vibeset/choon.
 *
 * The matcher runs on Cloud Run behind IAM, so the browser can't call it and
 * the service URL never reaches the client. This mints a Google-signed OIDC
 * ID token for the service's audience and forwards a fixed, validated set of
 * degradation parameters.
 *
 * Every failure path returns the recorded fallback tagged source:'replay', so
 * the widget degrades honestly rather than blanking (docs/design/DESIGN.md §4).
 *
 * Env:
 *   CHOON_API_BASE  the Cloud Run service URL
 *   CHOON_SA_KEY    service-account JSON with roles/run.invoker on that service
 *
 * POST { preset: 'clean'|'subway'|'nightcore'|'fried' }
 *   → { match, tier, confidence, classical, neural, ms, title, artist, source }
 */

import { createSign } from 'node:crypto'

const API_BASE = process.env.CHOON_API_BASE ?? ''
const SA_KEY = process.env.CHOON_SA_KEY ?? ''

/** The demo track. FMA, Creative Commons — the same catalog family the 66k benchmark used. */
const TRACK_ID = '93710'

/**
 * The four presets, as parameters the matcher itself accepts. The browser
 * applies the same numbers to the same track so you can hear what the model is
 * being asked to identify — these are not two different degradations.
 */
const PRESETS: Record<string, Record<string, number>> = {
  clean: {},
  subway: { lowPassFreq: 2200, noise: 0.14, bitcrush: 12 },
  nightcore: { playbackRate: 1.26, noise: 0.02, lowPassFreq: 16000 },
  fried: { bitcrush: 6, lowPassFreq: 3400, noise: 0.05 },
}

/** Recorded from this same endpoint, so the replay path is real output. */
const FALLBACK: Record<string, { tier: string; confidence: number; classical: number; neural: number; ms: number }> = {
  clean: { tier: 'classical', confidence: 0.909, classical: 0.295, neural: 0.0, ms: 13405 },
  subway: { tier: 'neural', confidence: 0.922, classical: 0.002, neural: 0.36, ms: 2682 },
  nightcore: { tier: 'neural', confidence: 0.984, classical: 0.008, neural: 0.93, ms: 2403 },
  fried: { tier: 'neural', confidence: 0.865, classical: 0.002, neural: 0.076, ms: 2073 },
}
const FALLBACK_TRACK = { title: 'Palm Tree', artist: 'The Chapin Sisters' }

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

const b64url = (b: Buffer | string) =>
  Buffer.from(b).toString('base64').replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '')

// ID tokens last an hour; re-minting on every call would add a round trip.
let tokenCache: { token: string; exp: number } | null = null

/**
 * Exchange a service-account key for an OIDC ID token scoped to `audience`.
 * Hand-rolled with node:crypto so this stays dependency-free.
 */
async function idToken(audience: string): Promise<string> {
  if (tokenCache && Date.now() < tokenCache.exp) return tokenCache.token
  if (!SA_KEY) throw new Error('CHOON_SA_KEY is not set')

  const sa = JSON.parse(SA_KEY) as { client_email: string; private_key: string; token_uri?: string }
  const iat = Math.floor(Date.now() / 1000)
  const claims = {
    iss: sa.client_email,
    aud: sa.token_uri || 'https://oauth2.googleapis.com/token',
    iat,
    exp: iat + 3600,
    target_audience: audience,
  }
  const signingInput = `${b64url(JSON.stringify({ alg: 'RS256', typ: 'JWT' }))}.${b64url(JSON.stringify(claims))}`
  const signature = createSign('RSA-SHA256').update(signingInput).sign(sa.private_key)
  const assertion = `${signingInput}.${b64url(signature)}`

  const res = await fetch(claims.aud, {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: new URLSearchParams({
      grant_type: 'urn:ietf:params:oauth:grant-type:jwt-bearer',
      assertion,
    }),
  })
  if (!res.ok) throw new Error(`token exchange ${res.status}`)
  const body = (await res.json()) as { id_token?: string }
  if (!body.id_token) throw new Error('no id_token in response')

  tokenCache = { token: body.id_token, exp: Date.now() + 50 * 60_000 }
  return body.id_token
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
  if (rateLimited(ip)) return replay(preset)
  if (!API_BASE) return replay(preset)

  // The matcher is a cold-start-prone ML service; a warm call lands in ~2.5s
  // but a cold one can exceed 30s. Cap it and fall back rather than hang.
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), 20_000)
  try {
    const token = await idToken(API_BASE)
    const res = await fetch(`${API_BASE}/api/simulate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
      body: JSON.stringify({ track_id: TRACK_ID, ...PRESETS[preset] }),
      signal: controller.signal,
    })
    if (!res.ok) throw new Error(`upstream ${res.status}`)
    const r = (await res.json()) as Record<string, unknown>

    return json({
      match: Boolean(r.match_found),
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
