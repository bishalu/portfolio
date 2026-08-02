/**
 * A suggestion, on its way to the wall (docs/design/DESIGN.md §4 rule 4, P9).
 *
 * WHY it returns { ok: true } even when Blobs is unavailable: the visible form
 * also POSTs to Netlify Forms, which emails us, and that is the durable path.
 * This endpoint is the moderation queue, not the delivery mechanism. Telling a
 * visitor their name failed to send when it did not fail would be a lie; so
 * the response carries `stored` and the client can be honest about which half
 * happened without inventing an error.
 *
 * WHY nothing here is trusted: the wall is public and it is about a baby. Hard
 * caps on every field, a closed relation list, control characters stripped,
 * anything containing a URL refused outright, ids checked against the
 * document's own rows, and per-IP rate limiting. Nothing published without a
 * human clicking approve — see naam-approve.ts, which is POST-only for a
 * reason.
 *
 * The record is written to `pending/<ts>-<rand>.json`: one key per submission,
 * never a read-modify-write, because this is the path a stranger can call and
 * two concurrent submissions must not be able to lose each other. The wall's
 * own read-modify-write happens on approval, which is single-actor and rare.
 *
 * KNOWN LIMIT, deliberately not fixed here: a pending record that is never
 * approved is never deleted. There is no reject path and no expiry, so the
 * store grows with every unapproved suggestion. Adding a sweep means deciding
 * to delete a stranger's suggestion on a timer while the only moderator might
 * simply be away for a fortnight, which is a product decision and not a fix.
 * The rate limit bounds the rate, not the total.
 *
 * `picks` OR `names` — one of the two is required, never neither. `names` is
 * the free-text field the visible form carries so the page still works with
 * JavaScript off: with JS off nothing can fill the hidden picks array, so a
 * no-JS submission used to arrive carrying no name at all, which is the one
 * thing this page exists to collect.
 */
import type { APIRoute } from 'astro'
import { randomBytes, createHmac } from 'node:crypto'
import { NAAM_COPY, NAAM_RELATIONS } from '@/lib/naam/copy'
import { clientIp, idsExist, isRowId, json, NAAM_ORIGIN, naamStore, rateLimited, tidy } from '@/lib/naam/server'

export const prerender = false

const RATE_LIMIT_PER_MIN = 6
const NAME_MAX = NAAM_COPY.limits.name
/** The form asks for 240; the endpoint accepts 400 so a paste is trimmed, not lost. */
const REASON_MAX = 400
/** The cap the tray, the form and this endpoint all agree on. Read, not retyped. */
const PICKS_MAX = NAAM_COPY.limits.picks
const NAMES_MAX = NAAM_COPY.limits.names
const SPELLING_MAX = 48

/** Anything else the visitor typed collapses to this. Stored value, not copy. */
const RELATION_FALLBACK = 'Other'
const RELATIONS = new Set<string>(NAAM_RELATIONS)

/** A suggestion is a sentence about a name. A link is something else. */
const LINKISH = /https?:|\bwww\.|:\/\//i

export const POST: APIRoute = async (context) => {
  const { request } = context

  let payload: unknown
  try {
    payload = await request.json()
  } catch {
    return json({ ok: false, stored: false, error: 'bad-request' }, 400)
  }

  // `clientAddress` is a getter that throws on adapters that cannot supply it,
  // so it is passed lazily rather than destructured.
  if (
    rateLimited(
      'submit',
      clientIp(request, () => context.clientAddress),
      RATE_LIMIT_PER_MIN,
    )
  ) {
    return json({ ok: false, stored: false, error: 'rate-limited' }, 429)
  }

  const body = (payload ?? {}) as {
    from?: unknown
    relation?: unknown
    picks?: unknown
    names?: unknown
    reason?: unknown
  }

  const from = tidy(body.from, NAME_MAX)
  if (!from) return json({ ok: false, stored: false, error: 'from' }, 400)

  const relationRaw = tidy(body.relation, 40)
  const relation = RELATIONS.has(relationRaw) ? relationRaw : RELATION_FALLBACK

  const reason = tidy(body.reason, REASON_MAX)
  const names = tidy(body.names, NAMES_MAX)
  if (LINKISH.test(from) || LINKISH.test(reason) || LINKISH.test(names)) {
    return json({ ok: false, stored: false, error: 'link' }, 400)
  }

  const submitted = Array.isArray(body.picks) ? body.picks : []
  const wanted: Array<{ id: string; spelling: string }> = []
  for (const entry of submitted.slice(0, PICKS_MAX)) {
    const pick = (entry ?? {}) as { id?: unknown; spelling?: unknown }
    if (!isRowId(pick.id)) continue
    if (wanted.some((p) => p.id === pick.id)) continue
    wanted.push({ id: pick.id, spelling: tidy(pick.spelling, SPELLING_MAX) })
  }

  // The ids are checked against the document's own rows, from a constant
  // origin — never `request.url`, or a forged Host decides what a real row is.
  const real = wanted.length > 0 ? await idsExist(wanted.map((p) => p.id)) : new Set<string>()
  const picks = wanted.filter((p) => real.has(p.id))

  // A suggestion is names, or typed names, or both. Never neither.
  if (picks.length === 0 && !names) return json({ ok: false, stored: false, error: 'picks' }, 400)

  const id = `${Date.now().toString(36)}-${randomBytes(4).toString('hex')}`
  const record = { id, from, relation, picks, names, reason, at: new Date().toISOString() }

  const store = naamStore()
  let stored = false
  if (store) {
    try {
      await store.setJSON(`pending/${id}.json`, record)
      stored = true
    } catch {
      stored = false
    }
  }

  // Best-effort notification with a one-click approve link. The link points at
  // the prerendered /naam/approve page, never at the endpoint — Gmail and
  // Outlook fire GET on every link they see, and they do not POST.
  //
  // Both the link and the request target are built from the constant origin.
  // When they were built from `request.url`, a forged Host sent the record's
  // HMAC approve token straight to the attacker, which is self-approval — the
  // exact thing the POST-only/HMAC design exists to prevent.
  //
  // WHY SLACK AND NOT NETLIFY FORMS. The first cut posted to a registered
  // Netlify Form, mirroring api/chat.ts's balgo-lead forward. On this site that
  // silently discards every notification: Netlify Forms intercepts POSTs at the
  // edge, but the adapter claims path '/*' with preferStatic, and preferStatic
  // only covers GET because Netlify's static handler does not answer POST. So
  // the POST lands in the SSR function, which renders a page and returns 200
  // while Forms never sees it. Verified against production on four paths — '/',
  // '/naam/', '/__forms.html', and a forced 200 rewrite — all forwarded, and the
  // site has zero form submissions on record since it launched. The forms are
  // still declared in public/__forms.html so they register; they just cannot
  // receive. SLACK_WEBHOOK_URL is already set in production and already carries
  // the arena watcher's alerts, so it is the one channel proven to arrive.
  //
  // To move this to email later, swap the fetch below for a provider call; the
  // record and the approve link are already assembled.
  if (stored && NAAM_ORIGIN) {
    const secret = process.env.NAAM_ADMIN_SECRET || import.meta.env.NAAM_ADMIN_SECRET
    const hook = process.env.SLACK_WEBHOOK_URL || import.meta.env.SLACK_WEBHOOK_URL
    if (secret && hook) {
      const token = createHmac('sha256', secret).update(id).digest('hex')
      const approve = `${NAAM_ORIGIN}/naam/approve?id=${encodeURIComponent(id)}&t=${token}`
      const chosen = [...picks.map((p) => p.spelling || p.id), names].filter(Boolean).join(', ')
      // Plain text, not Block Kit: a stranger's name and reason are the two
      // fields here, and mrkdwn would let them fake links and formatting.
      const lines = [
        `*A name suggestion* — ${from}${relation ? ` · ${relation}` : ''}`,
        chosen ? `> ${chosen}` : null,
        reason ? `> “${reason}”` : null,
        `Approve: ${approve}`,
      ].filter(Boolean)
      void fetch(hook, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: lines.join('\n'), mrkdwn: false }),
      }).catch(() => {})
    }
  }

  return json({ ok: true, stored })
}
