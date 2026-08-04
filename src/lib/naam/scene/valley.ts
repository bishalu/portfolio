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
import { Container, FillGradient, Graphics, WebGLRenderer } from 'pixi.js'
import { lampSpots } from './lamps'

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
 *
 * FOUR STOPS, AND THE TOP ONE IS DARK ENOUGH TO BE A SKY. It began at #cbd4e3
 * against #edeff2 paper — a 4% difference — so the upper half of the frame read
 * as nothing and the scene looked like it only occupied the bottom of the
 * panel. A dusk sky is deepest at the zenith and lightens all the way down to
 * the horizon, and that vertical fall-off is most of what makes a sky read as
 * depth rather than as a flat backing colour.
 */
const SKY_ZENITH = 0x8d9dbd // deep blue overhead, where the night is arriving
const SKY_TOP = 0xb4c1d8 //   cooling down toward the range
const SKY_MID = 0xd8d3d8 //   the wash where cold sky meets the lit band
const SKY_LOW = 0xf1cec2 //   the rose band standing above the range

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
  { color: 0xf3cabd, alpha: 1, y: 0.545, amp: 0.115, rough: 1.0 },
  { color: 0xcdc3cd, alpha: 1, y: 0.63, amp: 0.088, rough: 1.5 },
  { color: 0x9fa3b8, alpha: 1, y: 0.71, amp: 0.07, rough: 2.2 },
  // the near valley wall, deepest in shadow
  { color: 0x6e7288, alpha: 1, y: 0.795, amp: 0.052, rough: 3.1 },
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
  /**
   * How many lamps are burning out there. One per name the family has chosen —
   * the wall of notes, moved into the world it was always describing.
   */
  setLamps(count: number): void
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
  const stupa = new Graphics()
  const flags = new Container()
  const birds = new Graphics()
  const veil = new Graphics()
  const lamps = new Graphics()
  /**
   * ADDITIVE, BECAUSE LIGHT ADDS. The lamps were three concentric discs of
   * warm colour at partial alpha, which is how you draw a DOT that happens to
   * be orange — over the town they read as pale stickers sitting on the roofs.
   * Real light does not occlude what is behind it, it sums with it: a lamp
   * brightens the wall it stands against, and two lamps close together are
   * brighter still where they overlap. `blendMode: 'add'` is that, and it is in
   * Pixi's base set rather than the advanced-blend-modes extension, so it costs
   * nothing to import.
   *
   * It works here specifically because the lamps sit low, over the town and the
   * valley floor, which are the darkest parts of the frame. Additive light on
   * the pale sky above would blow straight out to white — which is also why the
   * alphas below are lower than they were: summed light needs less of it.
   */
  lamps.blendMode = 'add'
  // Order is the composition: sky, range, town, the stupa standing above the
  // town, flags strung from it, birds in front of everything, then the veil
  // that turns the left half back into paper.
  // The static half — everything that only changes on resize — grouped so it
  // can be cached to a single texture.
  const still2 = new Container()
  still2.addChild(sky, ridgeLayer, roofs, stupa)
  // Lamps sit ABOVE the veil. They are the one thing in the scene that is
  // supposed to read through the paper — a light seen from inside a room is not
  // dimmed by the wall it is beyond, and on a phone the veil is heavy enough
  // over the town that lamps under it would simply not be visible.
  stage.addChild(still2, flags, birds, veil, lamps)

  const noise = RIDGES.map((_, i) => ridgeNoise(0x5eed + i * 977))
  const ridgeGraphics = RIDGES.map(() => new Graphics())
  for (const g of ridgeGraphics) ridgeLayer.addChild(g)

  let w = 0
  let h = 0

  /**
   * THE HORIZON, and it moves with the layout.
   *
   * At 0.9 of the frame the valley floor sits BELOW the keep-slots on a phone,
   * so the stupa was drawn straight through the third slot and its Devanagari
   * ३. Raising the horizon to 0.775 when stacked puts the entire town, the
   * stupa and the flags in the clear band between the lamp and the slots — the
   * scene stops overlapping the one row of controls a visitor has to press, and
   * gains a real foreground of quiet paper beneath it.
   */
  const horizon = () => h * (w < 600 ? 0.83 : 0.9)

  /**
   * ONE RECT, ONE GRADIENT — no bands at all.
   *
   * This was drawn as a stack of flat rects twice, first 48 of them and then
   * 160, on the belief that Pixi's gradient fills were too unstable across v8
   * minors to rely on. That was wrong on the facts and it cost two visible
   * defects: at 48 the sky was visibly striped, and the same technique applied
   * to the veil produced vertical lines down the whole canvas, because
   * overlapping TRANSLUCENT bands composite twice at every seam.
   *
   * FillGradient is the v8 API for exactly this and it interpolates on the GPU,
   * so there are no steps to be visible at any viewport size. The banding class
   * of bug is gone rather than pushed below the threshold where I happened to
   * stop noticing it.
   */
  let skyFill: FillGradient | null = null

  function paintSky() {
    sky.clear()
    skyFill?.destroy()
    skyFill = new FillGradient({
      type: 'linear',
      start: { x: 0, y: 0 },
      end: { x: 0, y: 1 },
      colorStops: [
        { offset: 0, color: SKY_ZENITH },
        { offset: 0.34, color: SKY_TOP },
        { offset: 0.66, color: SKY_MID },
        { offset: 1, color: SKY_LOW },
      ],
    })
    sky.rect(0, 0, w, h).fill(skyFill)
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

      /**
       * TERRACES, on the two near ridges only.
       *
       * The hills around the valley are cut in steps right to their tops, and
       * from across the valley that reads as a stack of horizontal contour
       * lines catching the last of the light. It is the single detail that
       * makes a Nepali hillside impossible to mistake for an alpine one, and
       * without it these ridges were just smooth mass.
       *
       * Near ridges only, because at distance the steps close up and vanish
       * into the haze — drawing them on the far peaks would flatten the depth
       * the whole composition is built on. They follow the ridge's own noise so
       * they sit ON the hill rather than being ruled across it.
       */
      if (i >= 2) {
        const rows = 9
        for (let r = 1; r <= rows; r++) {
          const drop = (r / rows) * h * 0.075
          g.moveTo(-40, h)
          let started = false
          for (let x = -40; x <= w + 40; x += step) {
            const u = (x / w) * spec.rough
            const e = n(u * 4) * 0.72 + n(u * 12) * 0.28
            // Each contour wanders a little on its own, the way a real terrace
            // follows the land rather than a survey line.
            const wobble = n(u * 7 + r * 3.1) * h * 0.006
            const y = baseY - (e - 0.5) * h * spec.amp + drop + wobble
            if (!started) {
              g.moveTo(x, y)
              started = true
            } else {
              g.lineTo(x, y)
            }
          }
          g.stroke({
            color: 0xffffff,
            width: Math.max(0.6, h * 0.0011),
            // Fading downward: the near edge of a terraced hill is in shadow.
            // Fainter: at 0.10 they read as survey contours drawn ON the hill
            // rather than as light catching the edge of a step.
            alpha: 0.055 * (1 - r / rows) + 0.018,
          })
        }
      }
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
  let floorFill: FillGradient | null = null

  function paintRoofs() {
    roofs.clear()
    const rand = prng(0x1200f)
    const baseY = horizon()
    const unit = Math.max(5, h * 0.014)
    // LIFTED OUT OF BLACK, THEN BACK UP OFF THE FLOOR. At 0x413a4e the town was
    // the heaviest mass on the page and fought the invitation from across the
    // fold; at 0x6f6b83 over 0.66 alpha it dissolved into the haze completely
    // and the valley floor read as empty wash. Both are the same mistake in
    // opposite directions — the town has to be the third-darkest thing in the
    // frame, below the near ridge and above the ground, so the stepped profile
    // reads without the mass shouting.
    const ink = 0x585470

    let x = -30
    while (x < w + 40) {
      const tiers = 1 + Math.floor(rand() * 3)
      const width = unit * (2.6 + rand() * 2.4)
      const bodyH = unit * (1.1 + rand() * 1.5)
      const top = baseY - bodyH - unit * tiers * 0.8

      // Body first, so the eaves overhang it the way they actually do.
      roofs.rect(x + width * 0.2, top + unit * 0.5, width * 0.6, baseY - top).fill({ color: ink, alpha: 0.8 })

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
          .fill({ color: ink, alpha: 0.86 })
      }
      // Overlapping, not spaced: a town, not an exhibit of temples.
      x += width * (0.42 + rand() * 0.5)
    }

    /**
     * THE VALLEY FLOOR FADES OUT rather than filling to the bottom edge.
     *
     * A flat rect from the horizon down was a grey slab under the composer on
     * desktop and, once the horizon rose to clear the slots on a phone, a slab
     * directly behind the three keep-slots and their Devanagari numerals. Both
     * are the same error: the ground was being drawn as an object with a bottom
     * edge, when what is actually down there is foreground haze that gives way
     * to the page. Fading it means the horizon can sit wherever the composition
     * wants without the controls ever landing on a hard mass.
     */
    // HELD, NOT DESTROYED INLINE. The docs say to destroy gradients when done —
    // and "done" is not the next line. Destroying it immediately after .fill()
    // threw `Cannot read properties of null (reading 'style')` at render time,
    // because the Graphics still holds the fill and reads it every frame.
    floorFill?.destroy()
    floorFill = new FillGradient({
      type: 'linear',
      start: { x: 0, y: baseY / h },
      end: { x: 0, y: 1 },
      colorStops: [
        { offset: 0, color: 'rgba(97, 93, 121, 0.78)' },
        { offset: 0.45, color: 'rgba(107, 103, 128, 0.34)' },
        { offset: 1, color: 'rgba(120, 117, 138, 0)' },
      ],
    })
    roofs.rect(0, baseY, w, h - baseY).fill(floorFill)
  }

  /* ──────────────────────────────────────────────────────────────────────────
     WHAT YOU ACTUALLY SEE FROM THE VALLEY

     A skyline of tiered roofs is true but it is not enough to place anyone.
     Three things are unmistakable to someone who has stood there, and all three
     survive being small and in silhouette, which is the test that matters at
     this scale:

       the stupa      a white hemisphere under a stepped golden spire. There is
                      no other building shaped like it anywhere, so it does the
                      work of naming the place on its own.
       prayer flags   strung down from the spire in the fixed order blue, white,
                      red, green, yellow — sky, air, fire, water, earth. Getting
                      that order wrong is the kind of mistake the family would
                      notice immediately.
       terraces       the hills around the valley are cut in steps to their tops.
                      From a distance they are horizontal contour lines, and
                      they are the reason a Nepali hillside cannot be mistaken
                      for an alpine one.

     Deliberately NOT drawn: the painted eyes. At this size they would be three
     dark smudges, and the Buddha's eyes rendered as smudges is worse than
     leaving them off. The silhouette already carries it.
     ────────────────────────────────────────────────────────────────────────── */

  /**
   * PIXI'S OWN PERFORMANCE GUIDANCE IS EXPLICIT: do not clear and rebuild a
   * Graphics every frame; geometry changes are the expensive part, while
   * transform, alpha and tint are cheap. The first cut of this did exactly the
   * wrong thing — 48 flag quads and 9 birds torn down and re-tessellated 60
   * times a second, on a page that also runs an LLM round trip.
   *
   * So every flag is built once as a single-rect Graphics and thereafter only
   * has its x/y written. The catenary and the travelling wave are the same
   * maths; they just drive a transform now instead of a rebuild.
   */
  interface Flag {
    g: Graphics
    u: number
  }
  interface FlagLine {
    flags: Flag[]
    x1: number
    y1: number
    x2: number
    y2: number
    sag: number
    phase: number
  }
  let flagLines: FlagLine[] = []

  /** Enough to read as a line of cloth, few enough to batch. */
  const FLAGS_PER_LINE = 16

  /** Sky, air, fire, water, earth — in that order, always. */
  const FLAG_COLORS = [0x3b6fb0, 0xf2f2ef, 0xc0392b, 0x3d8b56, 0xe0a52b] as const

  function paintStupa() {
    stupa.clear()
    const baseY = horizon()
    /**
     * Off to one side when stacked, and smaller. At 0.735 of 390px the dome ran
     * off the right edge; centred at 0.5 it sat directly behind the middle
     * keep-slot with the Devanagari २ printed across it. A monument competing
     * with the control a visitor has to press is worse than no monument.
     */
    const stacked = w < 600
    /**
     * OFF THE TRAY'S CENTRELINE. The right column runs from 0.44 to 1.0, so its
     * centre is 0.72 — and the diyo is centred in it. At 0.735 the stupa's dome
     * sat directly behind the flame: two lit objects overlapping, and the one
     * that matters (yours) reading as an ornament on the one that does not.
     * Pushed right, they are two things in one valley instead of one confusion.
     */
    const cx = stacked ? w * 0.7 : w * 0.87
    const s = Math.max(12, h * (stacked ? 0.026 : 0.052))
    const domeY = baseY - h * 0.055
    const white = 0xf4f1ef
    const gold = 0xc9922f
    // LIFTED OUT OF BLACK, THEN BACK UP OFF THE FLOOR. At 0x413a4e the town was
    // the heaviest mass on the page and fought the invitation from across the
    // fold; at 0x6f6b83 over 0.66 alpha it dissolved into the haze completely
    // and the valley floor read as empty wash. Both are the same mistake in
    // opposite directions — the town has to be the third-darkest thing in the
    // frame, below the near ridge and above the ground, so the stepped profile
    // reads without the mass shouting.
    const ink = 0x585470

    // Plinth: two square terraces the dome sits on.
    stupa.rect(cx - s * 1.5, domeY - 2, s * 3, s * 0.42).fill({ color: white, alpha: 0.94 })
    stupa.rect(cx - s * 1.2, domeY - s * 0.3, s * 2.4, s * 0.34).fill({ color: white, alpha: 0.96 })

    // The dome — a hemisphere, flatter than a half-circle, which is the real
    // proportion at Boudha and the thing that keeps it from reading as a ball.
    stupa.moveTo(cx - s, domeY - s * 0.28)
    for (let a = Math.PI; a <= Math.PI * 2 + 0.01; a += Math.PI / 24) {
      stupa.lineTo(cx + Math.cos(a) * s, domeY - s * 0.28 + Math.sin(a) * s * 0.78)
    }
    stupa.fill({ color: white, alpha: 0.97 })

    // Harmika: the square block above the dome that the eyes are painted on.
    const hY = domeY - s * 1.06
    stupa.rect(cx - s * 0.34, hY - s * 0.42, s * 0.68, s * 0.44).fill({ color: white, alpha: 0.97 })

    // Thirteen steps to enlightenment, tapering — the spire.
    const steps = 13
    for (let i = 0; i < steps; i++) {
      const p = i / steps
      const hw = s * 0.3 * (1 - p * 0.72)
      const y = hY - s * 0.42 - i * (s * 0.06)
      stupa.rect(cx - hw, y - s * 0.055, hw * 2, s * 0.05).fill({ color: gold, alpha: 0.9 })
    }

    // The gajur — the finial — and a mast for the flags to hang from.
    const tipY = hY - s * 0.42 - steps * (s * 0.06) - s * 0.12
    stupa.moveTo(cx, tipY - s * 0.2).lineTo(cx + s * 0.14, tipY).lineTo(cx - s * 0.14, tipY).fill({ color: gold })
    stupa.rect(cx - s * 0.02, tipY - s * 0.34, s * 0.04, s * 0.2).fill({ color: gold })

    // A second, smaller stupa further off, so it reads as a valley of them
    // rather than a single monument placed for effect.
    const fx = w * 0.955
    const fs = s * 0.3
    const fy = baseY - h * 0.036
    stupa.moveTo(fx - fs, fy)
    for (let a = Math.PI; a <= Math.PI * 2 + 0.01; a += Math.PI / 16) {
      stupa.lineTo(fx + Math.cos(a) * fs, fy + Math.sin(a) * fs * 0.78)
    }
    stupa.fill({ color: white, alpha: 0.6 })
    // A taller, thinner spire. At fs*0.42 with a stub finial it read as a
    // mushroom — the dome has to be small against its own spire, not the
    // other way round.
    stupa.rect(fx - fs * 0.12, fy - fs * 2.2, fs * 0.24, fs * 1.4).fill({ color: gold, alpha: 0.5 })

    void ink
    return { cx, tipY, s }
  }

  /**
   * Flags hang from the spire out to the ground, and along a couple of lines
   * between rooftops. Each line is its own Graphics so it can be redrawn per
   * frame without touching the rest of the scene.
   */
  function buildFlags(anchor: { cx: number; tipY: number; s: number }) {
    for (const line of flagLines) for (const f of line.flags) f.g.destroy()
    flagLines = []
    const baseY = horizon()
    const rand = prng(0xf1a65)

    const fw = Math.max(3, h * 0.011)
    const fh = fw * 1.25
    const add = (x1: number, y1: number, x2: number, y2: number, sag: number) => {
      const per: Flag[] = []
      for (let i = 0; i < FLAGS_PER_LINE; i++) {
        // One rect, drawn once. Well under the ~100-point threshold where Pixi
        // stops batching Graphics as efficiently as Sprites.
        const g = new Graphics().rect(-fw / 2, 0, fw * 0.86, fh).fill({
          color: FLAG_COLORS[i % 5],
          alpha: 0.82,
        })
        flags.addChild(g)
        per.push({ g, u: i / FLAGS_PER_LINE })
      }
      flagLines.push({ flags: per, x1, y1, x2, y2, sag, phase: rand() * 6.28 })
    }

    // From the spire, fanning down to either side — the shape everyone photographs.
    add(anchor.cx, anchor.tipY, anchor.cx - anchor.s * 3.4, baseY - h * 0.012, h * 0.05)
    add(anchor.cx, anchor.tipY, anchor.cx + anchor.s * 3.1, baseY - h * 0.02, h * 0.045)
    // And strung between rooftops, which is just as common and much quieter.
    add(w * 0.5, baseY - h * 0.045, w * 0.62, baseY - h * 0.03, h * 0.022)
  }

  /** A catenary, sampled, with a small quad per flag. The wind is a travelling
   *  wave rather than a uniform sway: a line where every flag moves together is
   *  a rendering, and one where the motion runs along it is cloth. */
  function animateFlags(time: number) {
    for (const line of flagLines) {
      const { flags: per, x1, y1, x2, y2, sag, phase } = line
      for (const f of per) {
        const u = f.u
        const droop = Math.sin(u * Math.PI) * sag
        // A travelling wave, not a uniform sway: a line where every flag moves
        // together is a rendering; one where the motion runs along it is cloth.
        const wave = Math.sin(time * 1.6 + u * 5 + phase) * (h * 0.004) * Math.sin(u * Math.PI)
        f.g.x = x1 + (x2 - x1) * u
        f.g.y = y1 + (y2 - y1) * u + droop + wave
      }
    }
  }

  /**
   * Pigeons over the stupa. There are always birds up there, and a scene with
   * nothing alive in it reads as a backdrop no matter how well it is drawn —
   * which is the whole reason the parallax was there before this replaced it.
   */
  const FLOCK = Array.from({ length: 9 }, (_, i) => {
    const rand = prng(0xb12d + i * 31)
    return { r: 0.06 + rand() * 0.1, sp: 0.16 + rand() * 0.2, ph: rand() * 6.28, yr: 0.4 + rand() * 0.5 }
  })

  const birdShapes = FLOCK.map(() => {
    const g = new Graphics()
    birds.addChild(g)
    return g
  })

  /** Rebuilt only when the viewport changes; per frame these move and scale. */
  function buildBirds() {
    const s2 = Math.max(2.5, h * 0.0045)
    for (const g of birdShapes) {
      g.clear()
      g.moveTo(-s2, s2 * 0.5)
        .lineTo(0, 0)
        .lineTo(s2, s2 * 0.5)
        .stroke({ color: 0x4a4658, width: Math.max(1.1, h * 0.0016), alpha: 0.72 })
    }
  }

  function animateBirds(time: number) {
    // No birds on a stacked layout: the only open sky is behind the invitation,
    // and nine drifting chevrons across someone's sentence is not atmosphere.
    const hidden = w < 600
    const cx = w * 0.78
    const cy = h * 0.34
    FLOCK.forEach((b, i) => {
      const g = birdShapes[i]
      g.visible = !hidden
      if (hidden) return
      const a = time * b.sp + b.ph
      g.x = cx + Math.cos(a) * w * b.r
      g.y = cy + Math.sin(a) * h * b.r * b.yr
      // The wingbeat is a vertical scale on a fixed chevron rather than a
      // redrawn one — the only thing that says "alive" at three pixels is that
      // it flexes, and scale is free where geometry is not.
      g.scale.y = Math.sin(time * 7 + b.ph) * 0.45 + 0.75
    })
  }

  let veilFill: FillGradient | null = null

  /**
   * THE ROOM. The left of the frame veils back to paper so dark ink stays
   * readable over it; the right stays open.
   *
   * THE VERTICAL LINES CAME FROM HERE, twice. Banding a TRANSLUCENT ramp cannot
   * work the way banding an opaque one does: each rect has to overlap its
   * neighbour or a sub-pixel seam shows the sky through — but every overlapped
   * strip then composites the paper twice and comes out darker. I removed the
   * overlap, saw a hairline, put half a pixel back, and the stripes returned.
   * A single rect with an alpha gradient has no seams to reconcile.
   */
  function paintVeil() {
    veil.clear()
    veilFill?.destroy()
    const p = `${(PAPER >> 16) & 255}, ${(PAPER >> 8) & 255}, ${PAPER & 255}`

    /**
     * ON A PHONE THE WINDOW IS BELOW, NOT BESIDE.
     *
     * The room-and-window idea is a LEFT/RIGHT split and it only exists while
     * the layout is two columns. At 390 the layout stacks, so a horizontal veil
     * put the sky directly behind the invitation: dark blue under dark ink,
     * birds drifting across "6,715 names, out of the Vedas and the Sutras", and
     * the stupa sliced in half by the right edge. It was the worst frame in the
     * pass, and it only appeared at the width most of the family will use.
     *
     * So the veil turns ninety degrees. Paper at the top where the conversation
     * is, opening to the valley at the bottom where the lamp and the slots are —
     * the same structure, rotated to match the layout.
     */
    const stacked = w < 600

    veilFill = stacked
      ? new FillGradient({
          type: 'linear',
          start: { x: 0, y: 0 },
          end: { x: 0, y: 1 },
          colorStops: [
            { offset: 0, color: `rgba(${p}, 1)` },
            { offset: 0.46, color: `rgba(${p}, 0.97)` },
            { offset: 0.62, color: `rgba(${p}, 0.86)` },
            // The keep-slots and their Devanagari numerals live in this band,
            // so the veil stays heavy enough here that they read as controls.
            { offset: 0.8, color: `rgba(${p}, 0.62)` },
            { offset: 1, color: `rgba(${p}, 0.4)` },
          ],
        })
      : new FillGradient({
          type: 'linear',
          start: { x: 0, y: 0 },
          end: { x: 1, y: 0 },
          colorStops: [
            { offset: 0, color: `rgba(${p}, 1)` },
            { offset: Math.max(0, roomWidth - 0.1), color: `rgba(${p}, 1)` },
            // Eased rather than linear, so neither the solid edge nor the open
            // sky has a point where the transition visibly begins.
            { offset: roomWidth + 0.06, color: `rgba(${p}, 0.72)` },
            { offset: roomWidth + 0.16, color: `rgba(${p}, 0.24)` },
            { offset: Math.min(1, roomWidth + 0.3), color: `rgba(${p}, 0)` },
          ],
        })
    veil.rect(0, 0, w, h).fill(veilFill)
  }

  /**
   * THE SCENE IS MOSTLY STATIC NOW, and that is worth cashing in. Once the
   * parallax went, the sky, the four ridges, the town and the stupa stopped
   * changing between frames — but they were still being re-submitted 60 times a
   * second. Pixi's guidance is to cache exactly this: a container of static
   * content rendered once to a texture and thereafter drawn as one quad.
   *
   * Recached on resize, which is the only time any of it changes.
   */
  function layout() {
    w = canvas.clientWidth || 1
    h = canvas.clientHeight || 1
    renderer.resize(w, h)
    still2.cacheAsTexture(false)
    paintSky()
    paintRidges()
    paintRoofs()
    buildFlags(paintStupa())
    buildBirds()
    paintVeil()
    // Cache after painting, or an empty texture is what gets stored.
    still2.cacheAsTexture(true)
  }

  let raf = 0
  let alive = true

  /**
   * THE VALLEY DOES NOT SLIDE.
   *
   * Every layer used to translate with the pointer, which is the standard
   * parallax move and wrong for this page. A landscape that scrolls sideways
   * under the cursor reads as a widget being manipulated; the entire argument
   * for the scene is that you are sitting in a room looking out of a window,
   * and a window does not pan. The mountains stay where they are.
   *
   * What moves is what actually moves in a valley at dusk — the flags on the
   * line and the birds over the stupa. That is a place being alive, which is
   * what the parallax was reaching for and never had.
   */
  /**
   * A LAMP IS A NAME SOMEBODY CHOSE. Warm, small, and the only warm thing in
   * the frame besides the diyo itself — which is the point: the lamp on the
   * left is yours, and these are everybody else's, lit across the valley.
   *
   * Three concentric discs rather than a filter: a bloom pass on nine points
   * costs a full-screen render target, and at this size the difference between
   * a real bloom and a stack of soft alphas is not visible.
   */
  let lampCount = 0

  function paintLamps(time: number) {
    lamps.clear()
    if (lampCount <= 0) return
    const spots = lampSpots(lampCount, w < 600)
    const r = Math.max(2.2, h * 0.0042)
    spots.forEach((spot, i) => {
      const x = spot.x * w
      const y = spot.y * h
      // Each flickers on its own slow beat, the way a butter lamp does. Never
      // in step: a row of lights pulsing together is a progress indicator.
      const flick = 0.84 + Math.sin(time * (1.1 + i * 0.17) + i * 2.3) * 0.16
      // Three falls of light rather than three discs — a wide dim halo, a
      // brighter core, and the flame itself. Alphas are well under what they
      // were: additive blending sums them, so the old values burned white.
      lamps.circle(x, y, r * 6).fill({ color: 0xff9b3d, alpha: 0.07 * flick })
      lamps.circle(x, y, r * 2.6).fill({ color: 0xffb85c, alpha: 0.16 * flick })
      lamps.circle(x, y, r * 1.05).fill({ color: 0xffdca8, alpha: 0.5 * flick })
    })
  }

  let clock = 0

  function frame() {
    if (!alive) return
    clock += 1 / 60
    animateFlags(clock)
    animateBirds(clock)
    paintLamps(clock)
    renderer.render(stage)
    raf = requestAnimationFrame(frame)
  }

  layout()

  if (still) {
    // One frame, with the flags hung and the birds placed — not an empty sky.
    animateFlags(0)
    animateBirds(0)
    paintLamps(0)
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
    setLamps(count) {
      lampCount = count
      if (still) {
        paintLamps(0)
        renderer.render(stage)
      }
    },
    resize: layout,
    destroy() {
      alive = false
      cancelAnimationFrame(raf)
      document.removeEventListener('visibilitychange', onVisibility)
      // Pixi's docs are explicit that gradients hold GPU resources and must be
      // destroyed; this island remounts on every astro navigation.
      skyFill?.destroy()
      veilFill?.destroy()
      floorFill?.destroy()
      renderer.destroy()
    },
  }
}
