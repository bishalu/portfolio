/**
 * The wall, in one read (docs/design/DESIGN.md §4 rule 4, P8).
 *
 * ONE `get`, not `list()` + N `get`s. `list({ prefix })` returns keys and
 * etags only and carries no consistency option, so a list-then-get wall would
 * read stale on the exact request that follows an approval. naam-approve.ts
 * does the read-modify-write instead — it is single-actor and rare — and this
 * route is one strongly-consistent get of `wall.json`.
 *
 * `{ entries: [] }` on every failure, never a 500 and never a spinner. The
 * page renders the family's own seeded shortlist above whatever comes back, so
 * an empty or unreachable wall degrades to "nothing up yet", which is honest,
 * rather than to an error state, which would not be.
 */
import type { APIRoute } from 'astro'
import { json, naamStore, WALL_KEY, type NaamWallEntry } from '@/lib/naam/server'

export const prerender = false

const TTL_MS = 60_000

let cache: { at: number; body: { entries: NaamWallEntry[] } } | null = null

export const GET: APIRoute = async () => {
  if (cache && Date.now() - cache.at < TTL_MS) return json(cache.body)

  let entries: NaamWallEntry[] = []
  // Only a read that actually reached the store is worth caching. Caching the
  // empty fallback too meant one cold Blobs error blanked the wall for a full
  // minute for everybody — a failure wearing the "nothing up yet" state, which
  // is the opposite of §4 rule 4.
  let read = false
  try {
    const store = naamStore()
    if (store) {
      const wall = (await store.get(WALL_KEY, { type: 'json', consistency: 'strong' })) as {
        entries?: unknown
      } | null
      if (wall && Array.isArray(wall.entries)) entries = wall.entries as NaamWallEntry[]
      read = true
    }
  } catch {
    entries = []
    read = false
  }

  const body = { entries }
  if (read) cache = { at: Date.now(), body }
  return json(body)
}
