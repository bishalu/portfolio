/**
 * SVG text contrast — the gap axe cannot see.
 *
 * axe-core does not run its colour-contrast rule on SVG <text>, so a chart can
 * ship white-on-bone through a green gate. This walks every <text> and <rect>
 * inside an inline SVG, resolves its computed fill against the nearest painted
 * ancestor background, and prints the ratio.
 */
import { chromium } from 'playwright'

const base = process.argv[2] || 'http://localhost:8888'
const route = process.argv[3] || '/notes/choon'

const browser = await chromium.launch()
const page = await browser.newPage({ viewport: { width: 1280, height: 900 }, reducedMotion: 'reduce' })
await page.goto(base + route, { waitUntil: 'networkidle' })
await page.waitForTimeout(500)

const rows = await page.evaluate(() => {
  const parse = (c) => {
    const m = c.match(/rgba?\(([\d.]+),\s*([\d.]+),\s*([\d.]+)(?:,\s*([\d.]+))?\)/)
    return m ? { r: +m[1], g: +m[2], b: +m[3], a: m[4] === undefined ? 1 : +m[4] } : null
  }
  const lum = ({ r, g, b }) => {
    const f = (v) => { v /= 255; return v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4 }
    return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b)
  }
  const ratio = (a, b) => {
    const [x, y] = [lum(a), lum(b)].sort((m, n) => n - m)
    return +((x + 0.05) / (y + 0.05)).toFixed(2)
  }
  const groundOf = (el) => {
    for (let n = el; n && n !== document.documentElement; n = n.parentElement) {
      const bg = parse(getComputedStyle(n).backgroundColor)
      if (bg && bg.a > 0.5) return bg
    }
    return parse(getComputedStyle(document.documentElement).backgroundColor) || { r: 255, g: 255, b: 255, a: 1 }
  }

  const out = []
  for (const el of document.querySelectorAll('svg text, svg tspan')) {
    const fill = parse(getComputedStyle(el).fill)
    if (!fill) continue
    const g = groundOf(el)
    const size = parseFloat(getComputedStyle(el).fontSize) || 12
    out.push({
      text: (el.textContent || '').trim().slice(0, 28),
      fill: getComputedStyle(el).fill,
      size: Math.round(size),
      ratio: ratio(fill, g),
      // WCAG large text = >=24px, or >=18.66px bold
      bar: size >= 24 ? 3 : 4.5,
    })
  }
  return out
})

let bad = 0
for (const r of rows) {
  const ok = r.ratio >= r.bar
  if (!ok) bad++
  console.log(`${ok ? '  ok ' : 'FAIL'}  ${String(r.ratio).padStart(6)} / ${r.bar}  ${String(r.size).padStart(3)}px  ${r.fill.padEnd(22)} "${r.text}"`)
}
console.log(`\n${rows.length} svg text nodes · ${bad} below their bar`)
await browser.close()
if (bad) process.exitCode = 1
