import { chromium } from 'playwright'

const base = process.argv[2] || 'http://localhost:8888'
const route = process.argv[3] || '/about'
const out = process.argv[4] || 'scripts/verify/out/page.png'

const browser = await chromium.launch()
const page = await browser.newPage({
  viewport: { width: 1280, height: 900 },
  reducedMotion: 'reduce', // reveals land immediately; full-page capture is honest
})
await page.goto(base + route, { waitUntil: 'networkidle' })
await page.addStyleTag({ content: 'astro-dev-toolbar{display:none!important}' })
await page.waitForTimeout(600)
await page.screenshot({ path: out, fullPage: true })
console.log('shot', route, '→', out)
await browser.close()
