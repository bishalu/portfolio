/**
 * The valley — /naam's atmosphere layer, rendered in PixiJS.
 *
 * ─── WHAT THIS IS AND WHAT IT IS NOT ───────────────────────────────────────
 *
 * It is a DIORAMA, not a world you travel through. The page is one viewport
 * with no document scroll and a composer pinned to the bottom, so a traversable
 * scene was never on the table; what is achievable is a place that responds to
 * you. Everything readable — the invitation, the agent's replies, the names,
 * every control — stays in the DOM above this canvas. The canvas is aria-hidden
 * and never carries information, so axe still sees a complete page and the
 * names stay selectable.
 *
 * ─── WHY THE RANGE IS DRAWN IN FLAT LAYERS ─────────────────────────────────
 *
 * Aerial perspective is the Himalayan signature: from the valley the ridges
 * stack back in bands, each paler and bluer than the one in front, until the
 * furthest is barely separable from the sky. That is not a lighting effect to
 * be simulated — it is how the range reads, and it is the reason this is a 2D
 * composition rather than geometry. Parallax across the bands supplies all the
 * depth the page needs, at a fraction of the cost of a camera.
 *
 * ─── THE ROOM AND THE WINDOW ───────────────────────────────────────────────
 *
 * One canvas spans both columns, but the left column carries dark ink on paper
 * and would be unreadable over a dusk sky. So the scene is veiled back to the
 * page's own paper on the left and left open on the right: you are inside,
 * looking out. That is the ankhi jhyal — the Newar lattice window — used as a
 * structural idea rather than drawn as ornament.
 *
 * ─── DETERMINISM ───────────────────────────────────────────────────────────
 *
 * Every ridge, roof and lamp position comes from a seeded PRNG, never
 * Math.random, so the same page draws the same valley twice. The screenshot
 * harness depends on it, and so does anyone trying to tell a real visual
 * regression from noise.
 */
import { Container, Graphics, WebGLRenderer } from 'pixi.js'

/* ────────────────────────────────────────────────────────────────────────────
   PALETTE — sampled off the page's own tokens, so the canvas and the CSS
   ground are one material rather than two palettes meeting at a seam.
   ──────────────────────────────────────────────────────────────────────────── */

/** The page's own ground, --naam-sky. COOL, not cream — worth stating, because
 *  assuming it was warm is what produced the first version of this palette. */
const PAPER = 0xedeff2

/**
 * THE LAMP HAS TO STAY THE WARMEST THING ON THE PAGE.
 *
 * The first cut of this sky was #e6cba6 into #cfa583 — a terracotta sunset,
 * which is both the commonest machine-generated palette there is and wrong for
 * what it costs here: a warm sky puts a second warm mass on screen and the diyo
 * stops being the warm thing you are sitting beside. It becomes a small detail
 * inside a warm picture. The lamp is this page's whole centre of gravity, so
 * the sky has to be COLD and give the warmth back to it.
 *
 * Which is also what the place actually does. Once the sun is behind the range
 * the valley is already in blue shadow while the high snow still holds rose
 * light — so the one warm note in the landscape sits at the very BACK of the
 * frame, as far from the lamp as the composition allows, and everything between
 * them is cold. Near hills coldest, far peaks warmest: the inverse of the usual
 * haze rule, and the reason the effect is worth drawing rather than a gradient.
 *
 * High and washed, not dark and saturated. Thin air at altitude is luminous,
 * and this is a light page carrying dark ink.
 */
const SKY_TOP = 0xcbd4e3 //   thin cool blue, barely off the paper
const SKY_MID = 0xdad7de //   the wash where cold sky meets the lit band
const SKY_LOW = 0xf1cec2 //   the rose band standing above the range

/**
 * Front to back. Index 0 is nearest — coldest and darkest, the valley wall in
 * shadow. Index 3 is furthest: the snow, and the only thing still lit.
 *
 * `speed` is the parallax rate. Near layers travel further per unit of pointer
 * movement, which is the entire depth cue; the numbers are a geometric-ish
 * fall-off rather than a linear one because that is how distance actually
 * attenuates apparent motion.
 */
/**
 * DRAWN BACK TO FRONT, so index 0 is the FURTHEST — the snow — and index 3 is
 * the near valley wall. The array used to run the other way and the near ridges
 * were painted first, then buried under the lit peak; raising that peak's alpha
 * to make the alpenglow read turned the whole lower frame into one pink mass
 * with no range in it. Painter's order is not a detail here, it IS the depth.
 *
 * Every layer is now opaque. Distance is carried entirely by COLOUR approaching
 * the sky's — which is what aerial perspective actually is — and not by
 * transparency. Fading them as well applied the same cue twice and dissolved
 * the subject; it is also why the cold ridges vanished.
 */
const RIDGES = [
  // the snow, still lit — the one warm note, and the furthest thing in frame
  { color: 0xf3cabd, alpha: 1, y: 0.545, amp: 0.115, rough: 1.0, speed: 0.014 },
  { color: 0xcdc3cd, alpha: 1, y: 0.63, amp: 0.088, rough: 1.5, speed: 0.026 },
  { color: 0x9fa3b8, alpha: 1, y: 0.71, amp: 0.07, rough: 2.2, speed: 0.042 },
  // the near valley wall, deepest in shadow
  { color: 0x6e7288, alpha: 1, y: 0.795, amp: 0.052, rough: 3.1, speed: 0.062 },
] as const

/** Cap the device pixel ratio. Above 1.5 the cost is real and the gain on a
 *  soft-edged dusk scene is not visible. */
const MAX_DPR = 1.5

/* ────────────────────────────────────────────────────────────────────────────
   DETERMINISTIC NOISE
   ──────────────────────────────────────────────────────────────────────────── */

/** mulberry32 — small, fast, and seeded, which is the only property that
 *  matters here. */
function prng(seed: number): () => number {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

/** Value noise over a 1D lattice, smoothed. Ridges built from stacked octaves
 *  of this read as rock; a single sine reads as a wave, which is the tell that
 *  separates a mountain from a decoration. */
function ridgeNoise(seed: number) {
  const rand = prng(seed)
  const lattice = Array.from({ length: 256 }, () => rand())
  const at = (i: number) => lattice[((i % 256) + 256) % 256]
  return (x: number): number => {
    const i = Math.floor(x)
    const f = x - i
    // smoothstep, so the joins between lattice points are not visible creases
    const s = f * f * (3 - 2 * f)
    return at(i) * (1 - s) + at(i + 1) * s
  }
}

/* ────────────────────────────────────────────────────────────────────────────
   THE SCENE
   ──────────────────────────────────────────────────────────────────────────── */

export interface ValleyOptions {
  canvas: HTMLCanvasElement
  /** Fraction of the width that stays paper — the room you are standing in. */
  roomWidth?: number
  /** One frame, then stop. Set under prefers-reduced-motion. */
  still?: boolean
}

export interface ValleyHandle {
  /** Pointer position in 0..1, for parallax. Ignored when `still`. */
  look(x: number, y: number): void
  resize(): void
  destroy(): void
}

export async function createValley(options: ValleyOptions): Promise<ValleyHandle> {
  const { canvas, roomWidth = 0.44, still = false } = options

  const renderer = new WebGLRenderer()
  await renderer.init({
    canvas,
    antialias: false,
    resolution: Math.min(window.devicePixelRatio || 1, MAX_DPR),
    autoDensity: true,
    background: PAPER,
    width: canvas.clientWidth || 1,
    height: canvas.clientHeight || 1,
    // The page owns when to draw; a second ticker would fight the rAF loop below.
    powerPreference: 'low-power',
  })

  const stage = new Container()
  const sky = new Graphics()
  const ridgeLayer = new Container()
  const roofs = new Graphics()
  const veil = new Graphics()
  stage.addChild(sky, ridgeLayer, roofs, veil)

  const noise = RIDGES.map((_, i) => ridgeNoise(0x5eed + i * 977))
  const ridgeGraphics = RIDGES.map(() => new Graphics())
  for (const g of ridgeGraphics) ridgeLayer.addChild(g)

  let w = 0
  let h = 0

  /**
   * A vertical gradient in bands, because Pixi's own gradient fills have moved
   * between v8 minors and a sky is not worth a breaking change on a patch.
   *
   * 160 BANDS, NOT 48. At 48 the steps were plainly visible as stripes across
   * the sky — 900px of frame over 48 steps is a ~19px band, and a 2-3% colour
   * step at that width is exactly where banding becomes legible. The overdraw
   * is trivial (160 rects, once, on resize) and the artefact is gone.
   */
  function paintSky() {
    sky.clear()
    const bands = 160
    for (let i = 0; i < bands; i++) {
      const t = i / (bands - 1)
      // Two-stop ramp: cold above, the lit band low where the range stands.
      const c = t < 0.62 ? mix(SKY_TOP, SKY_MID, t / 0.62) : mix(SKY_MID, SKY_LOW, (t - 0.62) / 0.38)
      sky.rect(0, t * h - 1, w, h / bands + 2).fill({ color: c })
    }
  }

  function paintRidges() {
    RIDGES.forEach((spec, i) => {
      const g = ridgeGraphics[i]
      const n = noise[i]
      g.clear()
      const baseY = h * spec.y
      const step = Math.max(6, w / 90)
      g.moveTo(-40, h)
      for (let x = -40; x <= w + 40; x += step) {
        // Two octaves: the second at half amplitude and triple frequency is
        // what turns a rolling hill into something with crags on it.
        const u = (x / w) * spec.rough
        const e = n(u * 4) * 0.72 + n(u * 12) * 0.28
        g.lineTo(x, baseY - (e - 0.5) * h * spec.amp)
      }
      g.lineTo(w + 40, h)
      g.fill({ color: spec.color, alpha: spec.alpha })
    })
  }

  /**
   * The valley floor: a low town of tiered roofs, seen from across the fields.
   *
   * THE FIRST CUT DREW PINE TREES. Three faults compounded: the tiers were
   * ~34px on a 900px frame — the height of a building you are standing next to,
   * not a town on the far side of a valley — they were spaced three widths
   * apart so each stood alone against the sky, and the eaves flared WIDER as
   * they descended, which is a fir. A pagoda narrows as it rises and its eaves
   * are near-horizontal with only a slight lift at the ends.
   *
   * So: small (a tier is ~1.4% of frame height), dense and overlapping (a
   * skyline is a mass with a silhouette, not a row of objects), and each roof a
   * wide flat trapezoid over a shorter body. What survives at this size is the
   * STEPPED PROFILE against the sky, which is the whole of what makes a
   * Kathmandu skyline recognisable from a distance.
   */
  function paintRoofs() {
    roofs.clear()
    const rand = prng(0x1200f)
    const baseY = h * 0.9
    const unit = Math.max(5, h * 0.014)
    const ink = 0x413a4e

    let x = -30
    while (x < w + 40) {
      const tiers = 1 + Math.floor(rand() * 3)
      const width = unit * (2.6 + rand() * 2.4)
      const bodyH = unit * (1.1 + rand() * 1.5)
      const top = baseY - bodyH - unit * tiers * 0.8

      // Body first, so the eaves overhang it the way they actually do.
      roofs.rect(x + width * 0.2, top + unit * 0.5, width * 0.6, baseY - top).fill({ color: ink, alpha: 0.9 })

      for (let t = 0; t < tiers; t++) {
        // Narrower as it rises — the defining proportion.
        const shrink = 1 - t * 0.17
        const halfW = (width / 2) * shrink
        const y = top + t * unit * 0.8 + unit * 0.5
        const cx = x + width / 2
        roofs
          .moveTo(cx - halfW, y)
          .lineTo(cx - halfW * 0.55, y - unit * 0.52) // eaves lift slightly at the ends
          .lineTo(cx + halfW * 0.55, y - unit * 0.52)
          .lineTo(cx + halfW, y)
          .fill({ color: ink, alpha: 0.92 })
      }
      // Overlapping, not spaced: a town, not an exhibit of temples.
      x += width * (0.42 + rand() * 0.5)
    }

    roofs.rect(0, baseY, w, h - baseY).fill({ color: 0x353046, alpha: 0.85 })
  }

  /** THE ROOM. The left of the frame veils back to paper so dark ink stays
   *  readable over it; the right stays open. Banded for the same reason as the
   *  sky, and eased rather than linear so the transition has no visible edge —
   *  a straight ramp leaves a seam exactly where the eye is looking. */
  /**
   * THE ROOM. The left of the frame veils back to paper so dark ink stays
   * readable over it; the right stays open.
   *
   * OVERLAPPING BANDS CANNOT BE USED FOR A FADE. The first cut drew 40
   * translucent rects side by side with a +2px overlap — which is correct for
   * an OPAQUE ramp like the sky, and completely wrong here: every overlap
   * composites twice, so the seam is darker than either neighbour and the whole
   * veil rendered as vertical stripes across the canvas. It was the single
   * ugliest thing on the page.
   *
   * Fixed by making the bands butt rather than overlap, and by drawing them
   * back-to-front as a monotonically decreasing alpha so no pixel is painted by
   * two bands at once. 96 steps over a 30%-width feather is a sub-pixel step at
   * any viewport this page supports.
   */
  function paintVeil() {
    veil.clear()
    const start = w * (roomWidth - 0.08)
    const feather = w * 0.34
    const bands = 96
    const step = feather / bands

    // Solid paper up to where the feather begins.
    if (start > 0) veil.rect(0, 0, start, h).fill({ color: PAPER, alpha: 1 })

    for (let i = 0; i < bands; i++) {
      const t = i / (bands - 1)
      // smootherstep — flatter at both ends than smoothstep, so neither the
      // solid edge nor the open sky has a detectable start.
      const a = 1 - t * t * t * (t * (t * 6 - 15) + 10)
      veil.rect(start + i * step, 0, step + 0.5, h).fill({ color: PAPER, alpha: a })
    }
  }

  function layout() {
    w = canvas.clientWidth || 1
    h = canvas.clientHeight || 1
    renderer.resize(w, h)
    paintSky()
    paintRidges()
    paintRoofs()
    paintVeil()
  }

  /* — parallax ————————————————————————————————————————————————————— */

  let targetX = 0.5
  let targetY = 0.5
  let curX = 0.5
  let curY = 0.5

  let raf = 0
  let alive = true

  function frame() {
    if (!alive) return
    // Ease toward the pointer rather than tracking it. A layer that snaps to
    // the cursor reads as a mouse-follower; one that lags reads as distance.
    curX += (targetX - curX) * 0.045
    curY += (targetY - curY) * 0.045
    RIDGES.forEach((spec, i) => {
      const g = ridgeGraphics[i]
      g.x = (0.5 - curX) * w * spec.speed
      g.y = (0.5 - curY) * h * spec.speed * 0.45
    })
    roofs.x = (0.5 - curX) * w * 0.085
    renderer.render(stage)
    raf = requestAnimationFrame(frame)
  }

  layout()

  if (still) {
    renderer.render(stage)
  } else {
    raf = requestAnimationFrame(frame)
  }

  /** Stop when the tab is hidden. A scene animating into a background tab is
   *  battery spent on nobody. */
  const onVisibility = () => {
    if (document.hidden) {
      cancelAnimationFrame(raf)
      raf = 0
    } else if (!still && alive && raf === 0) {
      raf = requestAnimationFrame(frame)
    }
  }
  document.addEventListener('visibilitychange', onVisibility)

  return {
    look(x, y) {
      targetX = x
      targetY = y
    },
    resize: layout,
    destroy() {
      alive = false
      cancelAnimationFrame(raf)
      document.removeEventListener('visibilitychange', onVisibility)
      renderer.destroy()
    },
  }
}

/** Linear blend of two packed RGB values. */
function mix(a: number, b: number, t: number): number {
  const k = t < 0 ? 0 : t > 1 ? 1 : t
  const ar = (a >> 16) & 255
  const ag = (a >> 8) & 255
  const ab = a & 255
  const br = (b >> 16) & 255
  const bg = (b >> 8) & 255
  const bb = b & 255
  // Each channel truncated before it is shifted: a fractional red silently
  // becomes a fractional shift and the colour lands somewhere else entirely.
  const r = (ar + (br - ar) * k) | 0
  const g = (ag + (bg - ag) * k) | 0
  const b2 = (ab + (bb - ab) * k) | 0
  return (r << 16) | (g << 8) | b2
}
