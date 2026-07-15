/**
 * Live signals — one small JSON for the landing page's ambient ticker.
 * Source: the arena-leaderboard cron's Netlify Blobs state (written every
 * 30 min). Optional by design; the strip renders fallbacks if it's missing.
 */
import { getStore } from '@netlify/blobs'

type ArenaState = {
  current?: { model?: string; org?: string; score?: number }
  meta?: { scraped_at?: string }
}

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

  let arena = null
  try {
    const state = await arenaStore().get<ArenaState>('state.json', { type: 'json' })
    if (state?.current?.model) {
      arena = {
        model: state.current.model,
        org: state.current.org ?? null,
        score: state.current.score ?? null,
        checkedAt: state.meta?.scraped_at ?? null,
      }
    }
  } catch {
    arena = null
  }

  const body = JSON.stringify({ arena })
  cached = { at: Date.now(), body }
  return new Response(body, {
    headers: { 'Content-Type': 'application/json', 'Cache-Control': 'public, max-age=900' },
  })
}

export const config = { path: '/api/live-signals' }
