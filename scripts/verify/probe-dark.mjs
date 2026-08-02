import { chromium } from 'playwright'

const base = process.argv[2] || 'http://localhost:8888'
const browser = await chromium.launch()
const page = await browser.newPage({ viewport: { width: 1280, height: 900 } })
await page.goto(base + '/', { waitUntil: 'networkidle' })
// dismiss the intro overlay the same way the verify runner does
await page.mouse.click(640, 450).catch(() => {})
await page.waitForTimeout(1800)

const dark = await page.evaluate(() => {
  const lum = (r, g, b) => {
    const f = (v) => { v /= 255; return v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4 }
    return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b)
  }
  const out = []
  for (const el of document.querySelectorAll('body *')) {
    const cs = getComputedStyle(el)
    const m = cs.backgroundColor.match(/rgba?\(([\d.]+),\s*([\d.]+),\s*([\d.]+)(?:,\s*([\d.]+))?\)/)
    if (!m) continue
    const a = m[4] === undefined ? 1 : parseFloat(m[4])
    if (a < 0.25) continue
    const L = lum(+m[1], +m[2], +m[3])
    if (L > 0.25) continue
    const r = el.getBoundingClientRect()
    if (r.width < 8 || r.height < 8) continue
    out.push({
      tag: el.tagName.toLowerCase(),
      cls: (el.className && String(el.className).slice(0, 60)) || '',
      id: el.id || '',
      bg: cs.backgroundColor,
      L: +L.toFixed(3),
      box: `${Math.round(r.x)},${Math.round(r.y)} ${Math.round(r.width)}x${Math.round(r.height)}`,
      color: cs.color,
    })
  }
  return out
})
console.log(JSON.stringify(dark, null, 1))
await browser.close()
