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
  for (const path of ['/', '/about', '/vibeset/curation', '/vibeset/cue', '/vibeset/choon', '/research']) {
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
  for (const path of ['/', '/about', '/vibeset/choon', '/research']) {
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
  await page.goto(base + '/', { waitUntil: 'networkidle' })
  await settle(page)

  // VibeFinder: typeahead + search round trip
  await page.locator('.vf').scrollIntoViewIfNeeded()
  await page.fill('.vf-input', 'Bicep')
  await page.waitForTimeout(4200) // allow a cold upstream to answer or time out into fixtures
  const sug = await page.$$eval('.vf-sug', (e) => e.length)
  await page.click('.vf-go')
  await page.waitForTimeout(8000)
  const rows = await page.$$eval('.vf-track', (e) => e.length)
  const badge = await page.textContent('.vf-badge').catch(() => 'none')
  console.log(`vibefinder — suggestions: ${sug}, tracks: ${rows}, badge: ${(badge || '').trim()}`)
  await page.screenshot({ path: `${OUT}/widget-vibefinder.png`, clip: await page.locator('.vf').boundingBox() })

  // Choon: preset + identify (canned)
  await page.locator('.ch').scrollIntoViewIfNeeded()
  await page.click('.ch-presets .ch-chip:nth-child(4)')
  await page.click('.ch-identify')
  await page.waitForTimeout(2200)
  const result = await page.textContent('.ch-result-meta').catch(() => 'none')
  console.log(`choon — result: ${(result || '').trim()}`)
  await page.screenshot({ path: `${OUT}/widget-choon.png`, clip: await page.locator('.ch').boundingBox() })
  await ctx.close()
}

const browser = await chromium.launch()
let failures = 0
if (cmd === 'shots' || cmd === 'all') await shots(browser)
if (cmd === 'console' || cmd === 'all') failures += await consoleCheck(browser)
if (cmd === 'a11y' || cmd === 'all') failures += await a11y(browser)
if (cmd === 'widgets' || cmd === 'all') await widgets(browser)
await browser.close()
console.log(failures ? `\nFAILURES: ${failures}` : '\nAll checks passed.')
process.exit(failures ? 1 : 0)
