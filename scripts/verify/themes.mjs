/**
 * Both grounds, every route — axe plus a sideways-scroll check.
 *
 * `run.mjs` only ever saw one theme, which was fine when there was one. Now a
 * token can pass on night and fail on paper (and did: a filled CTA labelled
 * with --paper is 15.9:1 on night and 2.3:1 on paper), so every check has to
 * run twice or it is not a check.
 *
 * The settle is 900ms after the scroll sweep, not the 140ms `run.mjs` uses.
 * Reveals cap at 320ms and axe measures the *rendered* colour, so analysing
 * mid-animation reports blended text as a contrast failure. That produced six
 * phantom violations on the first pass; the elements were never wrong. Waiting
 * is the fix — lowering axe's threshold would have been turning the check off.
 *
 * Usage: node scripts/verify/themes.mjs [baseUrl]
 */
import { chromium } from 'playwright'
import { AxeBuilder } from '@axe-core/playwright'

const base = process.argv[2] || 'http://localhost:4321'
const ROUTES = [
  '/',
  '/about',
  '/research',
  '/vibeset/curation',
  '/vibeset/cue',
  '/notes/choon',
  '/accessibility-statement',
  '/thank-you',
]
const WIDTHS = [320, 390, 768, 1280, 1440]

const browser = await chromium.launch()
let failures = 0

for (const theme of ['dark', 'light']) {
  const ctx = await browser.newContext({ viewport: { width: 1280, height: 800 } })
  const page = await ctx.newPage()
  await page.addInitScript((t) => {
    try {
      t === 'light' ? localStorage.setItem('theme', 'light') : localStorage.removeItem('theme')
    } catch {}
  }, theme)

  let violations = 0
  for (const route of ROUTES) {
    await page.goto(base + route, { waitUntil: 'networkidle' })
    await page.mouse.click(10, 10)
    await page.waitForTimeout(1200)
    await page.evaluate(async () => {
      const step = window.innerHeight * 0.7
      for (let y = 0; y <= document.body.scrollHeight; y += step) {
        window.scrollTo(0, y)
        await new Promise((r) => setTimeout(r, 170))
      }
    })
    await page.waitForTimeout(900)

    const res = await new AxeBuilder({ page }).analyze()
    for (const v of res.violations) {
      console.log(`  ${theme} ${route} [${v.impact}] ${v.id} (${v.nodes.length})`)
      v.nodes.slice(0, 3).forEach((n) => console.log('       ', n.target.join(' ')))
    }
    violations += res.violations.length
  }

  // The theme must be resolved before paint, and dark must need no attribute.
  await page.goto(base + '/', { waitUntil: 'domcontentloaded' })
  const attr = await page.evaluate(() => document.documentElement.dataset.theme ?? null)
  const wanted = theme === 'light' ? 'light' : null
  const themeOk = attr === wanted
  if (!themeOk) console.log(`  ${theme}: FAIL theme attribute is ${attr}, expected ${wanted}`)

  let overflow = 0
  for (const w of WIDTHS) {
    await page.setViewportSize({ width: w, height: 900 })
    await page.goto(base + '/', { waitUntil: 'networkidle' })
    await page.mouse.click(10, 10)
    await page.waitForTimeout(900)
    const over = await page.evaluate((vw) => document.documentElement.scrollWidth - vw, w)
    if (over > 1) {
      console.log(`  ${theme}: FAIL ${w}px scrolls sideways by ${over}px`)
      overflow += 1
    }
  }

  console.log(`${theme}: ${violations} axe violations · theme attr ${themeOk ? 'ok' : 'WRONG'} · ${overflow} overflow`)
  failures += violations + overflow + (themeOk ? 0 : 1)
  await ctx.close()
}

await browser.close()
console.log(failures ? `\nFAILURES: ${failures}` : '\nBoth grounds clean.')
process.exit(failures ? 1 : 0)
