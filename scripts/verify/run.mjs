/**
 * Site verification runner — see .claude/skills/verify-site/SKILL.md
 *
 * Usage:  node scripts/verify/run.mjs <shots|a11y|console|widgets|all> [baseUrl]
 * Default baseUrl: http://localhost:8888 (netlify dev) — pass another if using
 * `netlify serve` (8899) or a deploy preview URL.
 *
 * Output goes to scripts/verify/out/ (gitignored).
 */
import { chromium } from 'playwright'
import { AxeBuilder } from '@axe-core/playwright'
import { mkdirSync } from 'node:fs'

const [cmd = 'all', base = 'http://localhost:8888'] = process.argv.slice(2)
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

/** Skip the intro overlay and walk the page so scroll reveals fire. */
async function settle(page) {
  await page.mouse.click(10, 10)
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
  page.on('console', (m) => ['error', 'warning'].includes(m.type()) && problems.push(`[${m.type()}] ${m.text().slice(0, 200)}`))
  page.on('pageerror', (e) => problems.push(`[pageerror] ${String(e).slice(0, 300)}`))
  for (const path of ROUTES) {
    await page.goto(base + path, { waitUntil: 'networkidle' })
    await settle(page)
    console.log('visited', path)
  }
  const vf = await page.goto(base + '/').then(() => page.$$eval('.vf', (e) => e.length)).catch(() => 0)
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
  let net = 0
  const countRequests = (r) => {
    if (!/\.(mp3|mp4|webm|wav|m4a)(\?|$)/i.test(r.url())) net += 1
  }
  page.on('request', countRequests)
  const fits = []
  for (const n of [1, 2, 3]) {
    await page.click(`.cx-pick:nth-child(${n})`)
    await page.waitForTimeout(600)
    fits.push(((await page.textContent('.cx-score-val')) || '').trim())
  }
  page.off('request', countRequests)
  console.log(`cue — fits across candidates: ${fits.join(' / ')}, non-media requests: ${net}`)
  if (new Set(fits).size < 2) {
    console.log('  FAIL: the fit never changed — the demo has nothing to demonstrate')
    problems += 1
  }
  if (net > 0) {
    console.log(`  FAIL: ${net} non-media request(s) — the fit is supposed to need no call`)
    problems += 1
  }
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

  await ctx.close()
  return problems
}

const browser = await chromium.launch()
let failures = 0
if (cmd === 'shots' || cmd === 'all') await shots(browser)
if (cmd === 'console' || cmd === 'all') failures += await consoleCheck(browser)
if (cmd === 'a11y' || cmd === 'all') failures += await a11y(browser)
if (cmd === 'widgets' || cmd === 'all') failures += await widgets(browser)
await browser.close()
console.log(failures ? `\nFAILURES: ${failures}` : '\nAll checks passed.')
process.exit(failures ? 1 : 0)
