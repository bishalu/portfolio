/**
 * Moderation, in one POST (docs/design/DESIGN.md P9).
 *
 * WHY THIS IS NOT A GET. The obvious design is an approve link in the
 * notification email — `/api/naam-approve?id=…&key=…`. Gmail, Outlook and
 * every corporate mail gateway fetch every URL in every message to scan it.
 * A GET approve link is therefore fired the moment the notification lands,
 * which auto-approves every submission on a public page about a baby. So:
 * POST only, from a real button on /naam/approve, with a per-record HMAC so a
 * leaked link approves exactly one record and nothing else. Link prefetchers
 * do not POST.
 *
 * WHY the wall is one denormalized blob: `list({ prefix })` returns keys and
 * etags only and takes no consistency option, so a list-then-get wall has a
 * consistency window on the read path — the one path every visitor hits.
 * Approval is rare and single-actor, so it can afford the read-modify-write
 * that keeps the read to a single strongly-consistent get.
 */
import type { APIRoute } from 'astro'
import { createHmac, timingSafeEqual } from 'node:crypto'
import { clientIp, json, naamStore, rateLimited, WALL_KEY, type NaamWallEntry } from '@/lib/naam/server'

export const prerender = false

const RATE_LIMIT_PER_MIN = 20
const WALL_MAX = 200

function constantTimeEqual(a: string, b: string): boolean {
  const left = Buffer.from(a, 'utf8')
  const right = Buffer.from(b, 'utf8')
  if (left.length !== right.length) return false
  return timingSafeEqual(left, right)
}

export const POST: APIRoute = async (context) => {
  const { request } = context
  // `clientAddress` is a getter that throws on adapters that cannot supply it,
  // so it is passed lazily rather than destructured.
  if (
    rateLimited(
      'approve',
      clientIp(request, () => context.clientAddress),
      RATE_LIMIT_PER_MIN,
    )
  )
    return json({ ok: false, error: 'rate-limited' }, 429)

  let payload: unknown
  try {
    payload = await request.json()
  } catch {
    return json({ ok: false, error: 'invalid' }, 400)
  }

  const body = (payload ?? {}) as { id?: unknown; token?: unknown }
  const id = typeof body.id === 'string' ? body.id : ''
  const token = typeof body.token === 'string' ? body.token : ''
  if (!/^[a-z0-9]+-[a-f0-9]{8}$/.test(id) || !/^[a-f0-9]{64}$/.test(token)) {
    return json({ ok: false, error: 'invalid' }, 400)
  }

  const secret = process.env.NAAM_ADMIN_SECRET || import.meta.env.NAAM_ADMIN_SECRET
  if (!secret) return json({ ok: false, error: 'invalid' }, 403)

  const expected = createHmac('sha256', secret).update(id).digest('hex')
  if (!constantTimeEqual(token, expected)) return json({ ok: false, error: 'invalid' }, 403)

  const store = naamStore()
  if (!store) return json({ ok: false, error: 'error' }, 200)

  const key = `pending/${id}.json`
  let record: NaamWallEntry | null = null
  try {
    record = (await store.get(key, { type: 'json', consistency: 'strong' })) as NaamWallEntry | null
  } catch {
    record = null
  }

  if (!record) {
    // Already approved is not an error; it is the second click on the same link.
    try {
      const wall = ((await store.get(WALL_KEY, { type: 'json', consistency: 'strong' })) ?? { entries: [] }) as {
        entries?: NaamWallEntry[]
      }
      const already = Array.isArray(wall.entries) && wall.entries.some((entry) => entry.id === id)
      return json({ ok: already, error: already ? 'already' : 'missing' })
    } catch {
      return json({ ok: false, error: 'missing' })
    }
  }

  try {
    const wall = ((await store.get(WALL_KEY, { type: 'json', consistency: 'strong' })) ?? { entries: [] }) as {
      entries?: NaamWallEntry[]
    }
    const entries = Array.isArray(wall.entries) ? wall.entries.filter((entry) => entry.id !== id) : []
    entries.unshift(record)
    await store.setJSON(WALL_KEY, { entries: entries.slice(0, WALL_MAX) })
    await store.delete(key)
  } catch {
    return json({ ok: false, error: 'error' })
  }

  return json({ ok: true })
}
