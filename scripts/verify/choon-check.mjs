/**
 * Drives the Choon stress test end to end: pick a preset, hit identify, and
 * report what came back and whether it was live or replay.
 *
 * Usage: node scripts/verify/choon-check.mjs [baseUrl]
 */
import { chromium } from 'playwright'

const base = process.argv[2] || 'http://localhost:8899'
const browser = await chromium.launch()
const page = await browser.newPage({ viewport: { width: 1280, height: 900 } })

const errors = []
page.on('pageerror', (e) => errors.push(String(e).slice(0, 140)))

await page.goto(`${base}/vibeset/choon`, { waitUntil: 'networkidle' })
await page.waitForTimeout(2000)
await page.evaluate(() => document.querySelector('.ch')?.scrollIntoView({ block: 'center' }))
await page.waitForTimeout(1000)

const presets = await page.$$('.ch-presets button')
if (presets.length >= 3) {
  await presets[2].click()
  await page.waitForTimeout(400)
}

await page.click('.ch-identify')
await page.waitForTimeout(6000)

const result = await page
  .$eval('.ch-result', (e) => e.innerText.replace(/\s+/g, ' ').trim())
  .catch(() => 'NO RESULT')
console.log('result:', result)
console.log('errors:', errors.length ? errors.join(' | ') : 'none')

const zone = await page.$('.ch-result-zone')
if (zone) await zone.screenshot({ path: 'scripts/verify/out/choon-live.png' })

await browser.close()
