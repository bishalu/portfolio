/**
 * Outbound mail, and the one thing worth knowing about this site's forms.
 *
 * NOT NETLIFY FORMS. Forms intercepts POSTs at the edge, but this site is
 * `output: 'server'` and the adapter claims '/*' with preferStatic — and
 * preferStatic only covers GET, because Netlify's static handler does not
 * answer POST. So every form POST lands in the SSR function and Forms never
 * sees it. The contact form shipped that way since launch and has zero
 * submissions on record; the Balgo lead alert POSTed to the same dead path.
 * /naam hit this first and routed around it with Resend. This is that route,
 * lifted out so there is one of it.
 *
 * Plain text, never HTML. Every body assembled here contains a stranger's
 * words, and an HTML mail body lets them inject markup into the inbox.
 *
 * Fire-and-forget by default: a send that fails must never turn into a 5xx for
 * the visitor, who has done nothing wrong and cannot fix it. `await` the
 * returned promise only where the caller genuinely needs to report delivery.
 */

const env = (name: string): string | undefined =>
  process.env[name] || (import.meta.env as Record<string, string | undefined>)[name]

export interface MailOptions {
  subject: string
  text: string
  /** Where a human reply should go. Not the sender — that would fail SPF. */
  replyTo?: string
}

export interface MailResult {
  /** false means the message did not go out. Callers log it; nothing renders. */
  sent: boolean
  reason?: 'no-key' | 'no-recipient' | 'rejected'
}

export async function sendMail({ subject, text, replyTo }: MailOptions): Promise<MailResult> {
  const key = env('RESEND_API_KEY')
  if (!key) return { sent: false, reason: 'no-key' }

  // CONTACT_NOTIFY_EMAIL first so the professional inbox can be split from the
  // family one later without touching this file; NAAM_NOTIFY_EMAIL is the
  // address already configured, so nothing has to be set up twice today.
  const to = env('CONTACT_NOTIFY_EMAIL') || env('NAAM_NOTIFY_EMAIL')
  if (!to) return { sent: false, reason: 'no-recipient' }

  try {
    const res = await fetch('https://api.resend.com/emails', {
      method: 'POST',
      headers: { Authorization: `Bearer ${key}`, 'Content-Type': 'application/json' },
      body: JSON.stringify({
        // Defaults to Resend's shared sender, which delivers to the account's
        // own address without domain verification — enough to run today.
        from: env('CONTACT_MAIL_FROM') || env('NAAM_MAIL_FROM') || 'onboarding@resend.dev',
        to: [to],
        subject,
        text,
        ...(replyTo ? { reply_to: replyTo } : {}),
      }),
    })
    return res.ok ? { sent: true } : { sent: false, reason: 'rejected' }
  } catch {
    return { sent: false, reason: 'rejected' }
  }
}
