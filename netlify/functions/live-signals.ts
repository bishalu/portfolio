/**
 * Live signals — one small JSON for the landing page's ambient ticker.
 * Sources: the arena-leaderboard cron's Netlify Blobs state (already written
 * every 30 min) + the Vibeset catalog stats API. Every source is optional;
 * the strip renders build-time fallbacks for anything missing.
 */
import { getStore } from '@netlify/blobs'

type ArenaState = {
  current?: { model?: string; org?: string; score?: number }
  meta?: { scraped_at?: string }
}

const STATS_URL = process.env.VIBESET_STATS_URL || 'https://v7xjbxvuzoqscj7l4qgdlcubbi0bidtm.lambda-url.us-east-2.on.aws/stats'

let cached: { at: number; body: string } | null = null
const TTL_MS = 15 * 60_000

function arenaStore() {
  const siteID = process.env.BLOBS_SITE_ID || process.env.NETLIFY_SITE_ID || process.env.NETLIFY_BLOBS_SITE_ID
  const token = process.env.BLOBS_TOKEN || process.env.NETLIFY_AUTH_TOKEN || process.env.NETLIFY_BLOBS_TOKEN
  if (siteID && token) return getStore('arena-leaderboard', { siteID, token })
  return getStore('arena-leaderboard')
}

export default async () => {
  if (cached && Date.now() - cached.at < TTL_MS) {
    return new Response(cached.body, {
      headers: { 'Content-Type': 'application/json', 'Cache-Control': 'public, max-age=900' },
    })
  }

  const [arena, catalog] = await Promise.all([
    (async () => {
      try {
        const state = await arenaStore().get<ArenaState>('state.json', { type: 'json' })
        if (!state?.current?.model) return null
        return {
          model: state.current.model,
          org: state.current.org ?? null,
          score: state.current.score ?? null,
          checkedAt: state.meta?.scraped_at ?? null,
        }
      } catch {
        return null
      }
    })(),
    (async () => {
      try {
        const controller = new AbortController()
        const t = setTimeout(() => controller.abort(), 4000)
        const res = await fetch(STATS_URL, { signal: controller.signal })
        clearTimeout(t)
        if (!res.ok) return null
        const data = await res.json()
        if (typeof data?.songs !== 'number') return null
        return { songs: data.songs, artists: data.artists ?? null }
      } catch {
        return null
      }
    })(),
  ])

  const body = JSON.stringify({ arena, catalog })
  cached = { at: Date.now(), body }
  return new Response(body, {
    headers: { 'Content-Type': 'application/json', 'Cache-Control': 'public, max-age=900' },
  })
}

export const config = { path: '/api/live-signals' }
