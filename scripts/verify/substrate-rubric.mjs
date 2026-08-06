/**
 * Graded rubric for the Substrate diagram.
 *
 * Machine checks only — the axes that need eyes are graded from the crops this
 * writes to scripts/verify/out/rubric-*.png. Every axis runs at 1280 and 390,
 * plus reduced-motion, keyboard-only and no-JS as their own targets.
 *
 * Usage: node scripts/verify/substrate-rubric.mjs [baseUrl]
 */
import { chromium } from 'playwright'
import { mkdirSync } from 'node:fs'

const base = process.argv[2] || 'http://localhost:4321'
const OUT = new URL('./out/', import.meta.url).pathname
mkdirSync(OUT, { recursive: true })

const rows = []
const grade = (axis, target, ok, detail) => {
  rows.push({ axis, target, ok, detail })
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${axis.padEnd(30)} ${target.padEnd(16)} ${detail}`)
}

const settle = async (p) => {
  await p.mouse.click(10, 10)
  await p.waitForTimeout(900)
  await p.evaluate(async () => {
    const s = window.innerHeight * 0.7
    for (let y = 0; y <= document.body.scrollHeight; y += s) {
      window.scrollTo(0, y)
      await new Promise((r) => setTimeout(r, 170))
    }
  })
  await p.locator('.sb').scrollIntoViewIfNeeded()
  await p.waitForTimeout(1100)
}

const browser = await chromium.launch()

for (const [w, h, target] of [
  [1280, 950, 'desktop'],
  [390, 844, 'phone'],
]) {
  const ctx = await browser.newContext({ viewport: { width: w, height: h } })
  const p = await ctx.newPage()
  await p.goto(base + '/', { waitUntil: 'networkidle' })
  await settle(p)

  const desktop = target === 'desktop'

  // ── composition: one spine ───────────────────────────────────────────────
  // This axis used to measure a shared LEFT edge. The section is deliberately
  // centred now — the diagram's own composition is symmetric and the header was
  // fighting it — so a left-edge check measures an intent that no longer exists.
  // Replaced rather than relaxed: it still fails if the blocks disagree, it just
  // asks about centres.
  //
  // Only rendered elements count. A display:none box reports left: 0, which made
  // the phone pass look like a 24px misalignment when everything was flush.
  const centres = await p.evaluate(() => {
    const out = {}
    for (const s of ['.sb-head', '.sb-stage', '.sb-lens', '.sb-compact']) {
      const r = document.querySelector(s)?.getBoundingClientRect()
      if (r && r.width > 0) out[s] = (r.left + r.right) / 2
    }
    // The drawing's ink, which is what the eye centres on — not its frame.
    const svg = document.querySelector('.sb-svg')
    if (svg && svg.getBoundingClientRect().width > 0) {
      let l = Infinity
      let r = -Infinity
      for (const el of svg.querySelectorAll('text, circle')) {
        const b = el.getBoundingClientRect()
        l = Math.min(l, b.left)
        r = Math.max(r, b.right)
      }
      out.ink = (l + r) / 2
    }
    return out
  })
  const cs = Object.values(centres)
  const spread = Math.max(...cs) - Math.min(...cs)
  grade(
    'composition: one spine',
    target,
    spread <= 8,
    `centres within ${spread.toFixed(1)}px across ${cs.length} blocks`,
  )

  // ── labels: nothing clipped by the viewBox, nothing overlapping ──────────
  if (desktop) {
    const label = await p.evaluate(() => {
      const svg = document.querySelector('.sb-svg')
      const box = svg.getBoundingClientRect()
      let minL = Infinity
      let maxR = -Infinity
      for (const t of svg.querySelectorAll('text')) {
        const r = t.getBoundingClientRect()
        minL = Math.min(minL, r.left)
        maxR = Math.max(maxR, r.right)
      }
      return { over: box.left - minL, under: maxR - box.right, w: box.width }
    })
    grade(
      'labels: inside the frame',
      target,
      label.over <= 1 && label.under <= 1,
      `left overflow ${label.over.toFixed(1)}px · right ${label.under.toFixed(1)}px`,
    )

    // Rows must not collide vertically.
    const collide = await p.evaluate(() => {
      const ys = [...document.querySelectorAll('.sb-cap')].map((g) => g.getBoundingClientRect())
      let worst = Infinity
      for (let i = 1; i < ys.length; i++) worst = Math.min(worst, ys[i].top - ys[i - 1].bottom)
      return worst
    })
    grade('labels: row separation', target, collide >= 4, `tightest gap ${collide.toFixed(1)}px`)

    // ── fan-out: the argument must be visible as edge counts ───────────────
    const fan = await p.evaluate(() => ({
      train: document.querySelectorAll('.sb-edge.is-train').length,
      retrieve: document.querySelectorAll('.sb-edge.is-retrieve').length,
      total: document.querySelectorAll('.sb-edge').length,
    }))
    grade(
      'fan-out: 1-to-1 vs 1-to-many',
      target,
      fan.train === 1 && fan.retrieve === 3,
      `train ${fan.train} edge · retrieve ${fan.retrieve} · ${fan.total} total`,
    )

    // ── lens: choosing a product isolates it ───────────────────────────────
    await p.click('.sb-lens-btn[data-lens="cue"]')
    await p.waitForTimeout(700)
    const lens = await p.evaluate(() => {
      const op = (s) => Number(getComputedStyle(document.querySelector(s)).opacity)
      const lit = [...document.querySelectorAll('.sb-edge.to-cue')].map((e) => Number(getComputedStyle(e).opacity))
      const dim = [...document.querySelectorAll('.sb-edge:not(.to-cue)')].map((e) =>
        Number(getComputedStyle(e).opacity),
      )
      return { minLit: Math.min(...lit), maxDim: Math.max(...dim), capLit: op('.sb-cap.has-cue .sb-cap-name') }
    })
    grade(
      'lens: isolates the choice',
      target,
      lens.minLit >= 0.9 && lens.maxDim <= 0.1,
      `lit ≥${lens.minLit.toFixed(2)} · dimmed ≤${lens.maxDim.toFixed(2)}`,
    )
    grade(
      'lens: lights what it reaches',
      target,
      lens.capLit >= 0.9,
      `connected capability at ${lens.capLit.toFixed(2)}`,
    )

    await p.screenshot({ path: `${OUT}/rubric-lens.png`, clip: await p.locator('.sb').boundingBox() })
    await p.click('.sb-lens-btn[data-lens="cue"]')
    await p.waitForTimeout(500)
  }

  // ── control: findable, and big enough to hit ─────────────────────────────
  const ctrl = await p.evaluate(() => {
    const bs = [...document.querySelectorAll('.sb-lens-btn')]
    const vis = bs.filter((b) => b.getBoundingClientRect().height > 0)
    return { n: vis.length, minH: Math.min(...vis.map((b) => b.getBoundingClientRect().height)) }
  })
  grade(
    'control: present and hittable',
    target,
    ctrl.n === 3 && ctrl.minH >= 24,
    `${ctrl.n} buttons · min height ${ctrl.minH.toFixed(0)}px`,
  )

  // ── phone: the compact reading carries the same data ─────────────────────
  if (!desktop) {
    const compact = await p.evaluate(() => {
      const rows = [...document.querySelectorAll('.sb-crow')]
      const train = rows.find((r) => r.classList.contains('is-train'))
      return {
        rows: rows.length,
        marks: rows[0]?.querySelectorAll('.sb-mark').length ?? 0,
        trainLeans: train?.querySelectorAll('.sb-mark.w-lean').length ?? -1,
      }
    })
    grade(
      'phone: same argument, no diagram',
      target,
      compact.rows === 6 && compact.marks === 3 && compact.trainLeans === 1,
      `${compact.rows} rows · ${compact.marks} marks · train leans ${compact.trainLeans}`,
    )
    await p.screenshot({ path: `${OUT}/rubric-phone.png`, clip: await p.locator('.sb').boundingBox() })
  } else {
    await p.screenshot({ path: `${OUT}/rubric-idle.png`, clip: await p.locator('.sb').boundingBox() })
  }

  // ── quality floor: no sideways scroll ────────────────────────────────────
  const overflow = await p.evaluate((vw) => Math.max(document.documentElement.scrollWidth, 0) - vw, w)
  grade('floor: no sideways scroll', target, overflow <= 1, `${overflow}px`)

  await ctx.close()
}

// ── keyboard-only ──────────────────────────────────────────────────────────
{
  const ctx = await browser.newContext({ viewport: { width: 1280, height: 950 } })
  const p = await ctx.newPage()
  await p.goto(base + '/', { waitUntil: 'networkidle' })
  await settle(p)
  const kb = await p.evaluate(() => {
    const g = document.querySelector('.sb-prod')
    g.focus()
    const focused = document.activeElement === g
    g.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }))
    return { focused, lens: document.querySelector('.sb').dataset.lens ?? null }
  })
  grade('keyboard: node reachable + Enter', 'keyboard', kb.focused && kb.lens === 'curation', `lens → ${kb.lens}`)

  const ring = await p.evaluate(() => {
    const g = document.querySelector('.sb-prod')
    g.focus()
    return getComputedStyle(g.querySelector('.sb-hit')).strokeWidth
  })
  grade('keyboard: visible focus', 'keyboard', parseFloat(ring) >= 1.5, `focus ring ${ring}`)
  await ctx.close()
}

// ── reduced motion: drawn, still ───────────────────────────────────────────
{
  const ctx = await browser.newContext({ viewport: { width: 1280, height: 950 }, reducedMotion: 'reduce' })
  const p = await ctx.newPage()
  await p.goto(base + '/', { waitUntil: 'networkidle' })
  await settle(p)
  const rmState = await p.evaluate(() => {
    const e = document.querySelector('.sb-edge')
    const cs = getComputedStyle(e)
    return { offset: cs.strokeDashoffset, anims: e.getAnimations().length }
  })
  grade(
    'reduced motion: fully drawn, still',
    'reduced-motion',
    rmState.offset === '0px' || rmState.offset === '0',
    `dashoffset ${rmState.offset} · ${rmState.anims} running`,
  )
  await ctx.close()
}

// ── no JS: the content is still there ──────────────────────────────────────
{
  const ctx = await browser.newContext({ viewport: { width: 1280, height: 950 }, javaScriptEnabled: false })
  const p = await ctx.newPage()
  await p.goto(base + '/', { waitUntil: 'domcontentloaded' })
  const nojs = await p.evaluate(() => ({
    edges: document.querySelectorAll('.sb-edge').length,
    caps: document.querySelectorAll('.sb-cap').length,
    table: document.querySelectorAll('.sb .sr-only tbody tr').length,
  }))
  grade(
    'no-JS: content present',
    'no-js',
    nojs.edges === 16 && nojs.caps === 6 && nojs.table === 6,
    `${nojs.edges} edges · ${nojs.caps} capabilities · ${nojs.table} table rows`,
  )
  await ctx.close()
}

await browser.close()
const failed = rows.filter((r) => !r.ok)
console.log(`\n${rows.length - failed.length}/${rows.length} pass`)
failed.forEach((f) => console.log(`  FAIL  ${f.axis} @ ${f.target} — ${f.detail}`))
process.exit(failed.length ? 1 : 0)
