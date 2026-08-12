/**
 * Site verification runner — see .claude/skills/verify-site/SKILL.md
 *
 * Usage:  node scripts/verify/run.mjs <shots|a11y|console|widgets|responsive|all> [baseUrl]
 * Default baseUrl: http://localhost:4321 (`npm run dev`). Netlify is for
 * production deploys only — `netlify serve` keeps serving a stale unzipped
 * function bundle after a rebuild, so it silently verifies old code. Pass a
 * deploy preview URL explicitly when you want to check production.
 *
 * Output goes to scripts/verify/out/ (gitignored).
 */
import { chromium } from 'playwright'
import { AxeBuilder } from '@axe-core/playwright'
import { mkdirSync } from 'node:fs'

const [cmd = 'all', base = 'http://localhost:4321'] = process.argv.slice(2)
const OUT = new URL('./out/', import.meta.url).pathname
mkdirSync(OUT, { recursive: true })

/** Every page the site serves. Add a route here when you add one. */
const ROUTES = [
  '/',
  '/about',
  '/research',
  '/vibeset/curation',
  '/vibeset/cue',
  '/vibeset/choon',
  '/notes/choon',
  '/accessibility-statement',
  '/thank-you',
]

const VIEWPORTS = [
  { w: 390, h: 844, name: '390' },
  { w: 768, h: 1024, name: '768' },
  { w: 1280, h: 800, name: '1280' },
  { w: 1920, h: 1080, name: '1920' },
  { w: 1280, h: 800, name: '1280-rm', rm: true },
  { w: 390, h: 844, name: '390-rm', rm: true },
]

/**
 * Skip the intro overlay and walk the page so scroll reveals fire.
 *
 * The click exists only to dismiss LoadingScreen, and app-shell routes (/naam)
 * never render it — DefaultLayout gates it on `appShell`. On those pages (10,10)
 * is the site header's home link, so an unguarded click navigates away and every
 * assertion after it runs against '/'. Click only when the overlay is present;
 * it stays in the DOM as display:none after it finishes, so this stays true for
 * the second and later visits in a session.
 */
async function settle(page) {
  if (await page.locator('#loading-screen').count()) await page.mouse.click(10, 10)
  await page.waitForTimeout(1500)
  await page.evaluate(async () => {
    const step = window.innerHeight * 0.7
    for (let y = 0; y <= document.body.scrollHeight; y += step) {
      window.scrollTo(0, y)
      await new Promise((r) => setTimeout(r, 160))
    }
    window.scrollTo(0, 0)
  })
  await page.waitForTimeout(1200)
}

async function shots(browser) {
  for (const v of VIEWPORTS) {
    const ctx = await browser.newContext({
      viewport: { width: v.w, height: v.h },
      reducedMotion: v.rm ? 'reduce' : 'no-preference',
    })
    const page = await ctx.newPage()
    await page.goto(base + '/', { waitUntil: 'networkidle' })
    await settle(page)
    await page.screenshot({ path: `${OUT}/landing-${v.name}.png`, fullPage: true })
    await ctx.close()
    console.log('shot', v.name)
  }
}

async function consoleCheck(browser) {
  const ctx = await browser.newContext({ viewport: { width: 1280, height: 800 } })
  const page = await ctx.newPage()
  const problems = []
  /**
   * Messages the PAGE cannot cause and cannot fix.
   *
   * Chromium's GL layer emits `GL Driver Message (... Performance ...)` from the
   * driver itself. Under headless swiftshader — which is what this gate runs on
   * — every accelerated canvas produces a handful of `GPU stall due to
   * ReadPixels` lines while Playwright composites it. They are software-renderer
   * artefacts of the test rig, absent on real hardware, and they scale with how
   * many canvases the page has rather than with anything being wrong.
   *
   * Counting them meant the gate failed for drawing at all, which is the kind
   * of failure that teaches you to ignore the gate. The filter is deliberately
   * narrow: it matches the driver's own prefix and nothing else, so a genuine
   * console.warn from page code — a Pixi deprecation, say — still fails.
   */
  const RIG_NOISE = /GL Driver Message \(.*Performance/
  page.on(
    'console',
    (m) =>
      ['error', 'warning'].includes(m.type()) &&
      !RIG_NOISE.test(m.text()) &&
      problems.push(`[${m.type()}] ${m.text().slice(0, 200)}`),
  )
  page.on('pageerror', (e) => problems.push(`[pageerror] ${String(e).slice(0, 300)}`))
  for (const path of ROUTES) {
    await page.goto(base + path, { waitUntil: 'networkidle' })
    await settle(page)
    console.log('visited', path)
  }
  const vf = await page
    .goto(base + '/')
    .then(() => page.$$eval('.vf', (e) => e.length))
    .catch(() => 0)
  problems.forEach((p) => console.log(p))
  console.log(`console problems: ${problems.length}`)
  await ctx.close()
  return problems.length
}

async function a11y(browser) {
  const ctx = await browser.newContext({ viewport: { width: 1280, height: 800 } })
  const page = await ctx.newPage()
  let total = 0
  for (const path of ROUTES) {
    await page.goto(base + path, { waitUntil: 'networkidle' })
    await settle(page)
    const results = await new AxeBuilder({ page }).analyze()
    for (const v of results.violations) {
      console.log(`${path} [${v.impact}] ${v.id}: ${v.help} (${v.nodes.length} nodes)`)
      v.nodes.slice(0, 3).forEach((n) => console.log('   ', n.target.join(' ')))
    }
    total += results.violations.length
    console.log(`${path} — violations: ${results.violations.length}`)
  }
  return total
}

async function widgets(browser) {
  const ctx = await browser.newContext({ viewport: { width: 1280, height: 900 } })
  const page = await ctx.newPage()
  let problems = 0

  await page.goto(base + '/', { waitUntil: 'networkidle' })
  await settle(page)

  // ── VibeFinder: typeahead, search round trip, and which path answered ──
  await page.locator('.vf').scrollIntoViewIfNeeded()
  await page.fill('.vf-input', 'Bicep')
  await page.waitForTimeout(4200) // let a cold upstream answer or time out into fixtures
  const sug = await page.$$eval('.vf-sug', (e) => e.length)
  await page.click('.vf-go')
  await page.waitForTimeout(8000)
  const rows = await page.$$eval('.vf-track', (e) => e.length)
  const badge = (await page.textContent('.vf-badge').catch(() => '')) || ''
  const route = (await page.textContent('.vf-route').catch(() => '')) || '(none — replay has no route)'
  console.log(`vibefinder — suggestions: ${sug}, tracks: ${rows}, badge: ${badge.trim()}`)
  console.log(`vibefinder — route: ${route.replace(/\s+/g, ' ').trim()}`)
  if (!rows) {
    console.log('  FAIL: no tracks rendered — the never-blank guarantee is broken')
    problems += 1
  }
  await page.screenshot({ path: `${OUT}/widget-vibefinder.png`, clip: await page.locator('.vf').boundingBox() })

  // ── CueSync: switching offsets must change the fit, with zero network ──
  await page.locator('.cx').scrollIntoViewIfNeeded()
  await page.waitForTimeout(500)
  // The invariant is that the FIT is computed locally — no API call. Each
  // candidate's audio is a separate clip and is expected to load on demand,
  // so media requests don't count against it.
  // Count what a violation would actually look like: the fit reaching an API
  // or pulling a data file. The old rule counted every non-media request on the
  // page, which caught an <a href="/about"> prefetch of about-portrait.webp and
  // reported it as the fit calling out. That is a false positive, not a
  // loosened check — a prefetched image provably cannot be the fit computing,
  // and any real call would be to /api/ or a .json.
  let net = 0
  const other = []
  const countRequests = (r) => {
    const u = r.url()
    if (/\/api\/|\.json(\?|$)/i.test(u)) net += 1
    else if (!/\.(mp3|mp4|webm|wav|m4a|webp|png|jpe?g|svg|css|js|woff2?)(\?|$)/i.test(u)) other.push(u)
  }
  page.on('request', countRequests)
  const fits = []
  for (const n of [1, 2, 3]) {
    await page.click(`.cx-pick:nth-child(${n})`)
    await page.waitForTimeout(600)
    fits.push(((await page.textContent('.cx-score-val')) || '').trim())
  }
  page.off('request', countRequests)
  console.log(`cue — fits across candidates: ${fits.join(' / ')}, API/data requests: ${net}`)
  if (new Set(fits).size < 2) {
    console.log('  FAIL: the fit never changed — the demo has nothing to demonstrate')
    problems += 1
  }
  if (net > 0) {
    console.log(`  FAIL: ${net} API/data request(s) — the fit is supposed to need no call`)
    problems += 1
  }
  if (other.length) console.log(`  (note: ${other.length} other request(s) in the window, not attributable to the fit)`)
  await page.screenshot({ path: `${OUT}/widget-cue.png`, clip: await page.locator('.cx').boundingBox() })

  // ── Choon: lives on its product page now, and identifies for real ──
  await page.goto(base + '/vibeset/choon', { waitUntil: 'networkidle' })
  await settle(page)
  await page.locator('.ch').scrollIntoViewIfNeeded()
  const presets = await page.$$('.ch-presets button')
  if (presets.length >= 3) await presets[2].click()
  await page.waitForTimeout(400)
  await page.click('.ch-identify')
  await page.waitForTimeout(9000) // a cold matcher can take a while before falling back
  const meta = ((await page.textContent('.ch-result-meta').catch(() => '')) || '').trim()
  const src = ((await page.textContent('.ch-src').catch(() => '')) || '').trim()
  console.log(`choon — ${meta} [${src || 'no source badge'}]`)
  if (!meta) {
    console.log('  FAIL: no identification rendered — the fallback should have covered this')
    problems += 1
  }
  await page.screenshot({ path: `${OUT}/widget-choon.png`, clip: await page.locator('.ch').boundingBox() })

  // ── /naam: the whole page is the widget ──────────────────────────────────
  // THE AGENT LEADS THIS PAGE. Nothing renders behind the model's call — the
  // local matcher no longer answers first and is reachable only through an
  // opt-in button under a failure line (NaamApp.tsx) — so the assertion is no
  // longer "names arrived" unconditionally. It is: one of exactly two honest
  // outcomes arrived, and neither of them is blank or silently substituted.
  // Note this navigates away from '/', so it must stay last.
  await page.goto(base + '/naam', { waitUntil: 'networkidle' })

  // EVERY SELECTOR HERE IS SCOPED UNDER #naam-app. The no-JS fallback renders
  // twelve more .nm-card / .nm-pick nodes into the normal DOM and hides them
  // with CSS (src/pages/naam.astro) — an unscoped '.nm-card' therefore counts
  // twelve cards on a page that dealt none, and an unscoped Keep locator
  // resolves to a hidden, disabled fallback button that never accepts a click.
  //
  // It was scoped to .nm-stream until the hand moved onto the tray. Dealt cards
  // are no longer in the transcript — they lie on the plate beside the three
  // slots, because a hand of cards inside the stream pushed the model's own
  // reply off the top of the screen (see NaamApp.tsx, `hand`). #naam-app is the
  // scope that still excludes the fallback and now spans both.
  const stream = page.locator('#naam-app')

  // The composer ships disabled until names-core.json lands, so wait on the
  // control rather than on a guessed number of milliseconds.
  await page.locator('#nma-ask:not([disabled])').waitFor({ timeout: 20000 })

  // The app owns the viewport. If the document scrolls, the shell broke.
  const naamScrolls = await page.evaluate(() => document.documentElement.scrollHeight > window.innerHeight + 1)
  if (naamScrolls) {
    console.log('  FAIL: /naam scrolls the document — the app shell is not holding the viewport')
    problems += 1
  }

  // One of the page's own starter chips, and chosen for a reason: it parses to
  // a wish the document can actually answer, so with Bedrock up the model
  // reliably picks names and this drill exercises the happy path. An ask for a
  // quality the document does not have ("something calm") is a fine thing for a
  // visitor to type and a poor probe — the model correctly answers that it has
  // nothing, and the gate then only ever sees the empty-handed branch.
  await page.fill('#nma-ask', 'short, and easy to say abroad')
  await page.click('.nm-composer-go')

  // One of two things must arrive: the model's names, or the honest failure
  // block. /api/naam-chat still always answers 200 — `degraded` rather than an
  // error — but the client no longer turns that into a quiet local deal, so the
  // wait is on whichever outcome the environment produces. ~15s covers a
  // Bedrock cold start; the client's own ceiling is 12s.
  const dealtCard = stream.locator('.nm-card').first()
  const escapeBtn = stream.locator('[data-nm-escape]').first()
  await dealtCard
    .or(escapeBtn)
    .waitFor({ timeout: 15000 })
    .catch(() => {})

  // No names came back. Two ways that happens and they are told apart by
  // whether there is a failure line: Bedrock off/slow/broken says so and must
  // carry Try again, and a model that answered but named nothing must NOT —
  // there is nothing to redo. Both offer the opt-in escape. §4 rule 4 is
  // satisfied by never being blank and never slipping the matcher's list in as
  // though the model had produced it.
  const noNames = (await escapeBtn.count()) > 0
  if (noNames) {
    const retries = await stream.locator('[data-nm-retry]').count()
    const failureLine = await stream.locator('.nm-said--note').count()
    console.log(`naam — no names from the model; escape offered, retry=${retries}, failure line=${failureLine}`)
    if (failureLine > 0 && retries === 0) {
      console.log('  FAIL: a failure line with no Try again — the visitor is told no and left nowhere to go')
      problems += 1
    }
    if (failureLine === 0 && retries > 0) {
      console.log('  FAIL: Try again offered on a turn the model answered — there is nothing to try again')
      problems += 1
    }
    // Take the escape. It is the only route to the matcher now, so this is both
    // the check that it works and how the rest of the drill gets a card to Keep.
    await escapeBtn.click()
    await dealtCard.waitFor({ timeout: 5000 }).catch(() => {})
  }

  const cards = await stream.locator('.nm-card').count()
  const naamBadge = ((await page.textContent('.nm-badge--rail').catch(() => '')) || '').trim()
  console.log(`naam — cards dealt: ${cards}, badge: ${naamBadge || 'none'}`)
  if (cards === 0) {
    console.log('  FAIL: no names on screen — neither the model nor the opt-in escape produced any')
    problems += 1
  }
  // The rail badge is a claim, not a status light: empty until the model has
  // answered, and then live. `local` on the rail means the local-first render
  // is back and the page is showing the matcher's work as if it were the
  // agent's (DESIGN.md §4).
  if (naamBadge !== '' && naamBadge !== 'live') {
    console.log(`  FAIL: rail badge reads "${naamBadge}" — it may only be empty or live`)
    problems += 1
  }
  if (!noNames && naamBadge !== 'live') {
    console.log('  FAIL: the model answered and the rail never said live')
    problems += 1
  }

  // Keep one, and confirm it reaches the tray. The pick is state, not animation:
  // togglePick() runs on the click, so the slot is filled well before the arc
  // lands and this does not have to wait out FLIGHT_MS to be true.
  const keep = stream.locator('.nm-pick').first()
  if (await keep.count()) {
    await keep.click()
    await page.waitForTimeout(900)
    const filled = await page.locator('.nm-slot[data-filled]').count()
    console.log(`naam — slots filled after one keep: ${filled}`)
    if (filled !== 1) {
      console.log(`  FAIL: keeping a name filled ${filled} slots, expected exactly 1`)
      problems += 1
    }
  } else {
    console.log('  FAIL: no Keep control rendered')
    problems += 1
  }

  // responsive()'s overflow sweep skips descendants of clipped ancestors, and
  // the whole app is clipped — so the panes have to be measured by hand.
  const naamOverflow = await page.evaluate(() =>
    ['.nm-stream', '.nm-slots', '.nm-composer']
      .filter((sel) => {
        const el = document.querySelector(sel)
        return el && el.scrollWidth > el.clientWidth
      })
      .join(', '),
  )
  if (naamOverflow) {
    console.log(`  FAIL: overflows horizontally: ${naamOverflow}`)
    problems += 1
  }

  await page.screenshot({ path: `${OUT}/widget-naam.png` })

  await ctx.close()
  return problems
}

/**
 * Responsive audit — the automatic part of "does it work on smaller screens".
 *
 * Screenshots only help if someone looks at them. These are assertions:
 *
 *  - horizontal overflow: the page must never scroll sideways. This is the
 *    single most common responsive break and it is trivially detectable.
 *  - offenders: when it does overflow, name the elements sticking out, so the
 *    failure points at a selector instead of a screenshot.
 *  - tap targets: interactive elements must clear the 24px floor the tokens
 *    already declare (--target-size-min), per WCAG 2.2 AA 2.5.8.
 *
 * Widths straddle the site's own SCSS breakpoints (320/480/768/1024/1280) and
 * the nav breakpoint at 1000, because that is where layout actually changes.
 */
const RESP_WIDTHS = [320, 390, 414, 768, 1000, 1024, 1280, 1440]

async function responsive(browser) {
  let problems = 0
  for (const w of RESP_WIDTHS) {
    const ctx = await browser.newContext({ viewport: { width: w, height: 900 } })
    const page = await ctx.newPage()
    for (const path of ROUTES) {
      await page.goto(base + path, { waitUntil: 'networkidle' })
      await settle(page)

      const report = await page.evaluate((vw) => {
        const doc = document.documentElement
        const overflow = Math.max(doc.scrollWidth, document.body.scrollWidth) - vw
        const offenders = []
        if (overflow > 1) {
          for (const el of document.querySelectorAll('body *')) {
            const r = el.getBoundingClientRect()
            if (r.width === 0 || r.height === 0) continue
            // Ignore things that scroll on purpose, and anything an ancestor
            // clips — a decorative blob inside overflow:hidden sticks out of
            // the viewport box but contributes nothing to scrollWidth.
            const ox = getComputedStyle(el).overflowX
            if (ox === 'auto' || ox === 'scroll') continue
            let clipped = false
            for (let a = el.parentElement; a && a !== document.body; a = a.parentElement) {
              const s = getComputedStyle(a)
              if (s.overflow !== 'visible' || s.position === 'fixed') {
                clipped = true
                break
              }
            }
            if (clipped) continue
            if (r.right > vw + 1 || r.left < -1) {
              const sel =
                el.tagName.toLowerCase() +
                (el.className && typeof el.className === 'string'
                  ? '.' + el.className.trim().split(/\s+/).slice(0, 2).join('.')
                  : '')
              offenders.push(`${sel} [${Math.round(r.left)}→${Math.round(r.right)}]`)
            }
          }
        }
        const small = []
        for (const el of document.querySelectorAll('a, button, input, select, textarea, [role="slider"]')) {
          const r = el.getBoundingClientRect()
          if (r.width === 0 || r.height === 0) continue
          // Inline links inside a paragraph are exempt (WCAG 2.5.8 exception).
          if (el.tagName === 'A' && el.closest('p, li, figcaption')) continue
          if (r.height < 24 || r.width < 24) {
            const sel =
              el.tagName.toLowerCase() +
              (el.className && typeof el.className === 'string' ? '.' + el.className.trim().split(/\s+/)[0] : '')
            small.push(`${sel} ${Math.round(r.width)}x${Math.round(r.height)}`)
          }
        }
        return { overflow, offenders: [...new Set(offenders)].slice(0, 5), small: [...new Set(small)].slice(0, 5) }
      }, w)

      if (report.overflow > 1) {
        console.log(`  FAIL ${w}px ${path} — scrolls sideways by ${report.overflow}px`)
        report.offenders.forEach((o) => console.log(`        ${o}`))
        problems += 1
      }
      if (report.small.length) {
        console.log(`  FAIL ${w}px ${path} — tap targets under 24px: ${report.small.join(', ')}`)
        problems += 1
      }
    }
    console.log(`responsive ${w}px — checked ${ROUTES.length} routes`)
    await ctx.close()
  }
  return problems
}

const browser = await chromium.launch()
let failures = 0
if (cmd === 'shots' || cmd === 'all') await shots(browser)
if (cmd === 'console' || cmd === 'all') failures += await consoleCheck(browser)
if (cmd === 'a11y' || cmd === 'all') failures += await a11y(browser)
if (cmd === 'widgets' || cmd === 'all') failures += await widgets(browser)
if (cmd === 'responsive' || cmd === 'all') failures += await responsive(browser)
await browser.close()
console.log(failures ? `\nFAILURES: ${failures}` : '\nAll checks passed.')
process.exit(failures ? 1 : 0)
