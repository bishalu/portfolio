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

  /**
   * JSON FROM THE APP, FORM-ENCODED FROM THE NO-JS FORM.
   *
   * This endpoint used to take JSON only, because the page without JavaScript
   * posted to Netlify Forms instead. Forms is gone — it was a second delivery
   * path that emailed us, it could not be exercised anywhere but production,
   * and it meant a submission had two ways to half-succeed. There is one path
   * now and it is the one that keeps the record.
   *
   * `picks` arrives as a JSON string in the encoded case, which is how the
   * form has always carried it.
   */
  const encoded = (request.headers.get('content-type') ?? '').includes('application/x-www-form-urlencoded')
  let payload: unknown
  try {
    if (encoded) {
      const form = new URLSearchParams(await request.text())
      let picks: unknown = []
      try {
        picks = JSON.parse(form.get('picks') || '[]')
      } catch {
        picks = []
      }
      payload = {
        from: form.get('from'),
        relation: form.get('relation'),
        reason: form.get('reason'),
        names: form.get('names'),
        picks,
        // The honeypot the form still carries. Netlify used to read it; now we
        // do. A filled one is a bot, and a bot gets the same page a person
        // gets so it learns nothing from the difference.
        trap: form.get('bot-field'),
      }
    } else {
      payload = await request.json()
    }
  } catch {
    return json({ ok: false, stored: false, error: 'bad-request' }, 400)
  }

  /** A browser posting a real form wants a page back, not JSON. */
  const done = (ok: boolean) =>
    encoded
      ? new Response(null, { status: 303, headers: { Location: ok ? '/thank-you' : '/naam?send=failed' } })
      : null

  // `clientAddress` is a getter that throws on adapters that cannot supply it,
  // so it is passed lazily rather than destructured.
  if (
    rateLimited(
      'submit',
      clientIp(request, () => context.clientAddress),
      RATE_LIMIT_PER_MIN,
    )
  ) {
    return done(false) ?? json({ ok: false, stored: false, error: 'rate-limited' }, 429)
  }

  const body = (payload ?? {}) as {
    from?: unknown
    relation?: unknown
    picks?: unknown
    names?: unknown
    reason?: unknown
    trap?: unknown
  }

  // Silently accepted, never stored. See the honeypot note above.
  if (typeof body.trap === 'string' && body.trap.trim() !== '') {
    return done(true) ?? json({ ok: true, stored: false }, 200)
  }

  const from = tidy(body.from, NAME_MAX)
  if (!from) return done(false) ?? json({ ok: false, stored: false, error: 'from' }, 400)

  const relationRaw = tidy(body.relation, 40)
  const relation = RELATIONS.has(relationRaw) ? relationRaw : RELATION_FALLBACK

  const reason = tidy(body.reason, REASON_MAX)
  const names = tidy(body.names, NAMES_MAX)
  if (LINKISH.test(from) || LINKISH.test(reason) || LINKISH.test(names)) {
    return done(false) ?? json({ ok: false, stored: false, error: 'link' }, 400)
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
  if (picks.length === 0 && !names) return done(false) ?? json({ ok: false, stored: false, error: 'picks' }, 400)

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
  // EMAIL, when a key exists. Set RESEND_API_KEY and every suggestion arrives
  // as mail instead; Slack stays as the fallback so there is never a silent
  // gap between the two. Nothing else has to change — the record and the signed
  // approve link are assembled either way. AWS SES was the preferred route
  // since the credentials are already here, but the IAM user is scoped to
  // Bedrock (ses:GetAccount is denied), so it would need a policy change first.
  if (stored && NAAM_ORIGIN) {
    const secret = process.env.NAAM_ADMIN_SECRET || import.meta.env.NAAM_ADMIN_SECRET
    const hook = process.env.SLACK_WEBHOOK_URL || import.meta.env.SLACK_WEBHOOK_URL
    const resendKey = process.env.RESEND_API_KEY || import.meta.env.RESEND_API_KEY
    const mailTo = process.env.NAAM_NOTIFY_EMAIL || import.meta.env.NAAM_NOTIFY_EMAIL
    if (secret && (hook || resendKey)) {
      const token = createHmac('sha256', secret).update(id).digest('hex')
      const approve = `${NAAM_ORIGIN}/naam/approve?id=${encodeURIComponent(id)}&t=${token}`
      const chosen = [...picks.map((p) => p.spelling || p.id), names].filter(Boolean).join(', ')
      const who = `${from}${relation ? ` · ${relation}` : ''}`

      if (resendKey && mailTo) {
        // Plain text, never HTML: `from` and `reason` are a stranger's words,
        // and an HTML mail body would let them inject markup into the inbox.
        // `from` defaults to Resend's shared sender, which delivers to the
        // account's own address without domain verification — enough to run,
        // and NAAM_MAIL_FROM overrides it once a domain is verified.
        const body = [
          `${who} suggested a name.`,
          chosen ? `\nName: ${chosen}` : '',
          reason ? `\nWhy: ${reason}` : '',
          `\n\nApprove it: ${approve}`,
          `\nNothing appears on the wall until you do.`,
        ].join('')
        void fetch('https://api.resend.com/emails', {
          method: 'POST',
          headers: { Authorization: `Bearer ${resendKey}`, 'Content-Type': 'application/json' },
          body: JSON.stringify({
            from: process.env.NAAM_MAIL_FROM || import.meta.env.NAAM_MAIL_FROM || 'onboarding@resend.dev',
            to: [mailTo],
            subject: `A name suggestion from ${from}`,
            text: body,
          }),
        }).catch(() => {})
      }

      if (hook) {
        // Plain text, not Block Kit: a stranger's name and reason are the two
        // fields here, and mrkdwn would let them fake links and formatting.
        const lines = [
          `*A name suggestion* — ${who}`,
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
  }

  return done(stored) ?? json({ ok: true, stored })
}
