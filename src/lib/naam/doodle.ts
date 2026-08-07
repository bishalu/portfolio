/**
 * The pointing doodle, drawn to fit the space it actually has.
 *
 * ─── WHY THIS IS GENERATED AND NOT A LITERAL PATH ──────────────────────────
 *
 * It used to be one fixed 90x214 path, absolutely positioned against the bottom
 * of the stream. Measured, that put it in the wrong place at every width:
 *
 *   1280x900   text ends 559, composer starts 732 — gap 173. Arrow top 502,
 *              so it began 57px INSIDE the paragraph it was pointing away from.
 *   1440x1000  gap 381. Arrow began 151px BELOW the text, floating in nothing.
 *   1024x768   gap 176. Arrow top 182 — most of it lay across the copy.
 *
 * The gap swings between 173 and 381px because the invitation is a fixed block
 * in a stream whose height is the viewport's. No fixed drawing satisfies "starts
 * at the text" and "reaches the box" at both ends of that range, and scaling one
 * to fit distorts it — at 381px a 214px drawing stretches 1.78x and the loops
 * become tall ovals, which is exactly the tell that a squiggle was drawn by
 * something other than a hand.
 *
 * So the curve is built to the measured span, and the TAIL takes up the slack:
 * past a certain gap the loops stop growing and only the run-out lengthens,
 * which is what a person does when they draw an arrow across a bigger space —
 * the same curl, a longer tail.
 *
 * Below that, the loops flatten rather than the arrow vanishing. Their ADVANCE
 * shrinks and their SWING does not, because the constraint is vertical only —
 * the gap is short, the column is not. A squeezed arrow therefore curls wider
 * and flatter, which is what a hand does with less room to come down in.
 *
 * Both halves of that were learned the hard way. Fixing the loops' size
 * returned null at two of the three test widths and the arrow simply did not
 * appear: full-size loops need ~121px and the box at 1280 is 151px, so no tail
 * was left. Then scaling them proportionally made them 17px across in a 90px
 * column, which reads as a wiggle rather than a curl.
 *
 * ─── THE LOOPS ARE A PROLATE TROCHOID ──────────────────────────────────────
 *
 *   y(t) = r·t − d·sin t        (advance down the page)
 *   x(t) = cx + d·cos t         (swing across it)
 *
 * With d > r the vertical speed r − d·cos t goes negative near the top of each
 * turn, so the curve doubles back and genuinely crosses itself. That crossing is
 * the whole difference between a loop and a wave, and it is not something you
 * can place reliably by hand with bezier control points — the first attempt at
 * that left a cusp where the curve met its own tail.
 */

/** Radius of the loop's advance at full size. 4π·r is the height they occupy. */
const LOOP_R = 9.6
/**
 * Loop width at full size, as a multiple of the advance. Must exceed 1 or the
 * curve waves instead of looping — the vertical speed r − d·cos t never goes
 * negative and so the curve never doubles back on itself.
 *
 * This sets the width the swing KEEPS. It is not a ratio maintained under
 * scaling: see loopD in buildDoodle, where the width holds while the advance
 * shrinks.
 */
const LOOP_ASPECT = 1.583
/**
 * The most of the available run the loops may occupy.
 *
 * Full-size loops need ~121px, and at 1280 the whole box is 151px — so the
 * first cut returned null at two of three test widths and the arrow simply did
 * not appear. The gesture has to survive a short gap, so the loops scale to the
 * budget and always leave a real tail behind them. Slightly over half: the tail
 * is what carries the eye to the box, and loops that eat it read as a doodle
 * that forgot what it was pointing at.
 */
const LOOP_SHARE = 0.55
/** Under this there is not enough run for two loops and a tail to read at all. */
const MIN_RUN = 74
/** Two turns. Three reads as a spring, one as a hesitation. */
const TURNS = 2
/** Drawing width, and the SVG's coordinate width. */
const W = 90
const CX = W / 2

/** Room above the first loop, so the curve does not touch the last line of type. */
const LEAD = 10
/** Room below the arrowhead, so the point does not sit on the composer's edge. */
const TAIL_PAD = 8
/** The arrowhead's own length. */
const BARB = 12

const MIN_W = 1.45
const MAX_W = 3.3

export interface Doodle {
  viewBox: string
  /** The ink: a filled outline whose width varies along the stroke. */
  outline: string
  /** The centreline, stroked wide in a mask to draw the ink on. */
  centre: string
  /** The two arrowhead strokes, drawn on separate beats after the shaft. */
  barbs: [string, string]
}

/**
 * Build the doodle for a box `height` px tall.
 *
 * Returns null when there is not enough room for the gesture to read — better
 * nothing than a compressed scribble.
 */
export function buildDoodle(height: number): Doodle | null {
  const run = height - LEAD - BARB - TAIL_PAD
  if (!Number.isFinite(run) || run < MIN_RUN) return null

  // Loops first, capped to their share of the run; the tail is the remainder.
  const loopSpan = Math.min(TURNS * 2 * Math.PI * LOOP_R, run * LOOP_SHARE)
  const loopR = loopSpan / (TURNS * 2 * Math.PI)
  /**
   * WIDTH DOES NOT SHRINK WITH HEIGHT. Scaling both together kept the loops'
   * proportions and made them 17px across in a 90px box — which reads as a
   * wiggle, not as a curl. The constraint is vertical only: the gap is short,
   * the column is not. So the advance shrinks and the swing stays, and a
   * squeezed arrow curls wider and flatter — which is what a hand does when it
   * has less room to come down in.
   */
  const loopD = Math.max(loopR * 1.25, LOOP_R * LOOP_ASPECT)
  const tail = run - loopSpan

  const pts: [number, number][] = []

  // The two loops.
  const STEPS = 130
  for (let i = 0; i <= STEPS; i++) {
    const t = (i / STEPS) * TURNS * 2 * Math.PI
    pts.push([CX + loopD * Math.cos(t), LEAD + loopR * t - loopD * Math.sin(t)])
  }

  /**
   * The run-out. It drifts rather than ruling straight: a dead-vertical segment
   * after two hand-drawn loops is the one place the whole stroke would give
   * itself away. The drift is a single slow sine over the tail's length, well
   * under a millimetre per centimetre — read as a wobble, not as a curve.
   */
  const [, loopEndY] = pts[pts.length - 1]
  const startX = pts[pts.length - 1][0]
  const TAIL_STEPS = Math.max(18, Math.round(tail / 6))
  for (let i = 1; i <= TAIL_STEPS; i++) {
    const k = i / TAIL_STEPS
    const drift = Math.sin(k * Math.PI * 1.15) * 3.4
    // Ease back to the centre so the arrowhead lands under the loops rather
    // than off to one side, which reads as the arrow pointing past the box.
    const x = startX + (CX - startX) * k + drift
    pts.push([x, loopEndY + tail * k])
  }

  const tipX = pts[pts.length - 1][0]
  const tipY = pts[pts.length - 1][1] + BARB
  pts.push([tipX, tipY])

  return {
    viewBox: `0 0 ${W} ${Math.ceil(height)}`,
    outline: taperedOutline(pts),
    centre: polyline(pts),
    barbs: [
      `M${(tipX - 6.4).toFixed(1)} ${(tipY - 10.2).toFixed(1)}L${tipX.toFixed(1)} ${tipY.toFixed(1)}`,
      `M${(tipX + 6.4).toFixed(1)} ${(tipY - 10.2).toFixed(1)}L${tipX.toFixed(1)} ${tipY.toFixed(1)}`,
    ],
  }
}

function polyline(pts: readonly [number, number][]): string {
  return (
    `M${pts[0][0].toFixed(1)} ${pts[0][1].toFixed(1)}` +
    pts
      .slice(1)
      .map((p) => `L${p[0].toFixed(1)} ${p[1].toFixed(1)}`)
      .join('')
  )
}

/**
 * The centreline offset either side by a width that follows the curve's own
 * curvature — ink pools where the pen slows, and on this path it slows in
 * exactly one place: the loops, where the wrist has to work around a corner.
 *
 * Only the START tapers. Tapering both ends is the reflex and it is wrong here:
 * this stroke does not lift at the end, it STOPS so two barbs can be added to
 * it, and thinning it to nothing there leaves the arrowhead on a hairline.
 */
function taperedOutline(pts: readonly [number, number][]): string {
  const n = pts.length
  const raw = pts.map((_, i) => {
    if (i === 0 || i === n - 1) return 0
    const [ax, ay] = pts[i - 1]
    const [bx, by] = pts[i]
    const [cx, cy] = pts[i + 1]
    const v1x = bx - ax
    const v1y = by - ay
    const v2x = cx - bx
    const v2y = cy - by
    const l1 = Math.hypot(v1x, v1y) || 1e-6
    const l2 = Math.hypot(v2x, v2y) || 1e-6
    const cross = (v1x * v2y - v1y * v2x) / (l1 * l2)
    const dot = (v1x * v2x + v1y * v2y) / (l1 * l2)
    return Math.abs(Math.atan2(cross, dot)) / ((l1 + l2) / 2)
  })

  // Raw discrete curvature is noisy and would read as a wobbly edge rather than
  // as ink, so it is smoothed over a small window.
  const smooth = raw.map((_, i) => {
    let sum = 0
    let count = 0
    for (let k = Math.max(0, i - 3); k <= Math.min(n - 1, i + 3); k++) {
      sum += raw[k]
      count++
    }
    return sum / count
  })
  const peak = Math.max(...smooth) || 1

  const widths = smooth.map((c, i) => {
    const pool = Math.min(1, c / (peak * 0.62))
    const w = MIN_W + (MAX_W - MIN_W) * pool
    return w * (0.4 + 0.6 * Math.min(1, i / (n - 1) / 0.035))
  })

  const side = (sign: number) =>
    pts.map((p, i) => {
      const a = pts[Math.max(0, i - 1)]
      const b = pts[Math.min(n - 1, i + 1)]
      const dx = b[0] - a[0]
      const dy = b[1] - a[1]
      const l = Math.hypot(dx, dy) || 1e-6
      const h = ((widths[i] / 2) * sign) / 1
      return [p[0] + (-dy / l) * h, p[1] + (dx / l) * h] as [number, number]
    })

  const forward = side(1)
  const back = side(-1).reverse()
  return `${polyline(forward)}${polyline(back).replace(/^M/, 'L')}Z`
}
