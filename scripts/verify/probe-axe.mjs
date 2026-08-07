import { chromium } from 'playwright'
import AxeBuilder from '@axe-core/playwright'

const base = process.argv[2] || 'http://localhost:8888'
const route = process.argv[3] || '/naam'

const browser = await chromium.launch()
const ctx = await browser.newContext({ viewport: { width: 1280, height: 800 } })
const page = await ctx.newPage()
await page.goto(base + route, { waitUntil: 'networkidle' })
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

const results = await new AxeBuilder({ page }).analyze()
for (const v of results.violations) {
  console.log(`[${v.impact}] ${v.id}: ${v.help} (${v.nodes.length})`)
  v.nodes.slice(0, 3).forEach((n) => console.log('   ', n.target.join(' ')))
}
console.log(`${route} — violations: ${results.violations.length}`)
await browser.close()
