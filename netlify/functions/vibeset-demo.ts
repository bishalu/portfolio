/**
 * Vibeset demo proxy — powers the landing page's "Find the vibe" widget.
 *
 * Why a proxy: the Vibeset API's CORS allowlist doesn't include this site,
 * and the browser should never see upstream URLs. Only two read-only
 * endpoints are reachable through here, responses are slimmed (no embeddings),
 * and every failure path returns replay fixtures — the widget never blanks.
 *
 * POST { kind: 'autocomplete', query }               → { suggestions, source }
 * POST { kind: 'search', artist?, genres?, moods?,
 *        tempo_min?, tempo_max?, limit? }            → { tracks, response_time_ms, source }
 */
import { fixtureSuggestions, fixtureTracks, type SlimSuggestion, type SlimTrack } from './vibeset-fixtures'

const UPSTREAM = process.env.VIBESET_API_BASE || 'https://5vboufmeboomn64dgugikyqez40tunvp.lambda-url.us-east-2.on.aws'

const AUTOCOMPLETE_TIMEOUT_MS = 3500
const SEARCH_TIMEOUT_MS = 6500

// Small in-memory guards (per warm Lambda instance — best effort, not a wall)
const RATE_LIMIT_PER_MIN = 30
const ipHits = new Map<string, { count: number; windowStart: number }>()
const cache = new Map<string, { at: number; body: unknown }>()
const CACHE_TTL_MS = 5 * 60_000

function rateLimited(ip: string): boolean {
  const now = Date.now()
  const entry = ipHits.get(ip)
  if (!entry || now - entry.windowStart > 60_000) {
    ipHits.set(ip, { count: 1, windowStart: now })
    return false
  }
  entry.count++
  return entry.count > RATE_LIMIT_PER_MIN
}

async function upstream(path: string, body: unknown, timeoutMs: number): Promise<any> {
  const controller = new AbortController()
  const t = setTimeout(() => controller.abort(), timeoutMs)
  try {
    const res = await fetch(`${UPSTREAM}${path}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal: controller.signal,
    })
    if (!res.ok) throw new Error(`upstream ${res.status}`)
    return await res.json()
  } finally {
    clearTimeout(t)
  }
}

function slimSuggestions(raw: any): SlimSuggestion[] {
  const list = Array.isArray(raw?.suggestions) ? raw.suggestions : []
  return list.slice(0, 6).map((s: any) => ({
    display: String(s.display ?? ''),
    type: String(s.type ?? 'artist'),
    artist: s.data?.artist_0 ?? s.data?.artist ?? undefined,
    track: s.data?.track ?? undefined,
    genre: s.data?.genre_0 ?? undefined,
    tempo: typeof s.data?.tempo === 'number' ? Math.round(s.data.tempo) : undefined,
  }))
}

function slimTracks(raw: any): SlimTrack[] {
  const list = Array.isArray(raw?.tracks) ? raw.tracks : []
  return list.map((t: any) => ({
    track: String(t.track ?? t.song ?? ''),
    artist: String(t.artist_0 ?? t.artist ?? ''),
    genre: String(t.genre_0 ?? ''),
    mood: String(t.mood_0 ?? ''),
    energy: String(t.energy ?? '').toLowerCase(),
    tempo: typeof t.tempo === 'number' ? Math.round(t.tempo) : null,
    year: typeof t.release_year === 'number' ? t.release_year : null,
  }))
}

const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json', 'Cache-Control': 'no-store' },
  })

export default async (req: Request) => {
  if (req.method !== 'POST') return json({ error: 'POST only' }, 405)

  let payload: any
  try {
    payload = await req.json()
  } catch {
    return json({ error: 'invalid JSON body' }, 400)
  }

  const ip = req.headers.get('x-nf-client-connection-ip') || req.headers.get('x-forwarded-for') || 'unknown'

  if (payload?.kind === 'autocomplete') {
    const query = String(payload.query ?? '').slice(0, 64)
    if (query.trim().length < 2) return json({ suggestions: [], source: 'live' })

    if (rateLimited(ip)) return json({ suggestions: fixtureSuggestions(query), source: 'replay' })

    const cacheKey = `ac:${query.toLowerCase()}`
    const hit = cache.get(cacheKey)
    if (hit && Date.now() - hit.at < CACHE_TTL_MS) return json(hit.body)

    try {
      const raw = await upstream('/api/v3/autocomplete', { query }, AUTOCOMPLETE_TIMEOUT_MS)
      const body = { suggestions: slimSuggestions(raw), source: 'live' }
      cache.set(cacheKey, { at: Date.now(), body })
      return json(body)
    } catch {
      return json({ suggestions: fixtureSuggestions(query), source: 'replay' })
    }
  }

  if (payload?.kind === 'search') {
    const limit = Math.min(Math.max(Number(payload.limit) || 6, 1), 8)
    const request: Record<string, unknown> = {
      limit,
      columns: ['id', 'track', 'artist_0', 'genre_0', 'mood_0', 'energy', 'tempo', 'release_year'],
    }
    if (typeof payload.artist === 'string' && payload.artist.trim()) {
      request.artist_list = [payload.artist.trim().slice(0, 64)]
    }
    if (Array.isArray(payload.genres) && payload.genres.length) {
      request.genre_list = payload.genres.slice(0, 3).map((g: unknown) => String(g).slice(0, 40))
    }
    if (Array.isArray(payload.moods) && payload.moods.length) {
      request.mood_list = payload.moods.slice(0, 3).map((m: unknown) => String(m).slice(0, 40))
    }
    if (Number.isFinite(Number(payload.tempo_min))) request.tempo_min = Number(payload.tempo_min)
    if (Number.isFinite(Number(payload.tempo_max))) request.tempo_max = Number(payload.tempo_max)

    if (!request.artist_list && !request.genre_list && !request.mood_list && !request.tempo_min && !request.tempo_max) {
      return json({ error: 'give me something to search with' }, 400)
    }

    if (rateLimited(ip)) {
      return json({ tracks: fixtureTracks(limit), response_time_ms: null, search_method: 'replay', source: 'replay' })
    }

    const cacheKey = `s:${JSON.stringify(request)}`
    const hit = cache.get(cacheKey)
    if (hit && Date.now() - hit.at < CACHE_TTL_MS) return json(hit.body)

    try {
      const raw = await upstream('/api/v2/intelligent_search', request, SEARCH_TIMEOUT_MS)
      const tracks = slimTracks(raw)
      if (!tracks.length) {
        return json({ tracks: [], response_time_ms: raw?.response_time_ms ?? null, search_method: raw?.search_method ?? null, source: 'live' })
      }
      const body = {
        tracks,
        response_time_ms: typeof raw?.response_time_ms === 'number' ? Math.round(raw.response_time_ms) : null,
        search_method: raw?.search_method ?? null,
        source: 'live',
      }
      cache.set(cacheKey, { at: Date.now(), body })
      return json(body)
    } catch {
      return json({ tracks: fixtureTracks(limit), response_time_ms: null, search_method: 'replay', source: 'replay' })
    }
  }

  return json({ error: 'unknown kind' }, 400)
}

export const config = { path: '/api/vibeset-demo' }
