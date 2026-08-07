/**
 * The contact endpoint — the one thing on this site that has to work.
 *
 * WHY IT EXISTS. The form used to post to Netlify Forms. Forms intercepts POSTs
 * at the edge, but this site is `output: 'server'` and the adapter claims '/*'
 * with preferStatic, and preferStatic only covers GET — Netlify's static
 * handler does not answer POST. So every submission landed in the SSR function
 * and Forms never saw one. The site has zero submissions on record since
 * launch. That is the entire reason this file exists.
 *
 * WORKS WITH JS OFF. A plain form POST gets a 303 to /thank-you; a fetch with
 * Accept: application/json gets JSON back. Same handler, same validation.
 *
 * NEVER 5xx AT THE VISITOR. A delivery problem is not something the person who
 * just typed a message can act on, so the confirmation is the same either way
 * and the failure is logged loudly server-side instead.
 *
 * REQUIRES RESEND_API_KEY and a notify address (CONTACT_NOTIFY_EMAIL, falling
 * back to NAAM_NOTIFY_EMAIL) to be set in the Netlify dashboard. Without them
 * sendMail returns { sent: false } and the message is lost — the console.error
 * below is the only trace. Check the function logs after any deploy that
 * touches env vars.
 */
export const prerender = false

import type { APIRoute } from 'astro'
import { clientIp, json, rateLimited, tidy } from '@/lib/http'
import { sendMail } from '@/lib/mail'

const DIRECT = 'bishal@vibeset.ai'

/** Generous enough for a real description, short enough to bound the body. */
const LIMITS = { name: 120, email: 160, link: 400, message: 4000 } as const

const looksLikeEmail = (v: string) => /^[^@\s]+@[^@\s.]+\.[^@\s]+$/.test(v)

export const POST: APIRoute = async ({ request, clientAddress }) => {
  const wantsJson = (request.headers.get('accept') || '').includes('application/json')

  // A form POST arrives urlencoded; the island posts JSON. Accept both rather
  // than making the no-JS path a second code path that can rot separately.
  let raw: Record<string, unknown> = {}
  try {
    const type = request.headers.get('content-type') || ''
    if (type.includes('application/json')) {
      raw = (await request.json()) as Record<string, unknown>
    } else {
      raw = Object.fromEntries(await request.formData())
    }
  } catch {
    raw = {}
  }

  // Honeypot. Named plausibly enough that a bot fills it and no human sees it;
  // a filled value is answered with the same success the visitor would get, so
  // a bot learns nothing from the response.
  if (tidy(raw.company, 80)) {
    return wantsJson ? json({ ok: true }) : Response.redirect(new URL('/thank-you', request.url), 303)
  }

  const name = tidy(raw.firstname ?? raw.name, LIMITS.name)
  const email = tidy(raw.email, LIMITS.email)
  const link = tidy(raw.link, LIMITS.link)
  const message = tidy(raw.message, LIMITS.message)

  const problems: string[] = []
  if (!name) problems.push('name')
  if (!email || !looksLikeEmail(email)) problems.push('email')
  if (!message) problems.push('message')
  if (problems.length) {
    return wantsJson
      ? json({ ok: false, error: 'incomplete', fields: problems }, 400)
      : Response.redirect(new URL('/#contact', request.url), 303)
  }

  const ip = clientIp(request, () => clientAddress)
  if (rateLimited('contact', ip, 5)) {
    return wantsJson
      ? json({ ok: false, error: 'rate-limited', direct: DIRECT }, 429)
      : Response.redirect(new URL('/thank-you', request.url), 303)
  }

  const result = await sendMail({
    subject: `bishal.ai — ${name}`,
    // Plain text: every field below is a stranger's words.
    text: [
      `${name} <${email}>`,
      link ? `\nLink: ${link}` : '',
      `\n\n${message}`,
      `\n\n— sent from the contact form at bishal.ai`,
    ].join(''),
    replyTo: email,
  })

  if (!result.sent) {
    // The visitor is told the same thing either way — they did nothing wrong and
    // cannot act on a delivery problem. It is logged loudly instead, because a
    // real person's message going nowhere is the worst failure this site has.
    console.error(`[contact] send failed (${result.reason}) — from ${email}: ${message.slice(0, 200)}`)
  }

  return wantsJson ? json({ ok: true }) : Response.redirect(new URL('/thank-you', request.url), 303)
}

/** A GET here is someone poking at the URL; send them to the form. */
export const GET: APIRoute = ({ request }) => Response.redirect(new URL('/#contact', request.url), 303)
