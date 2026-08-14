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
import { Container, FillGradient, Graphics, Ticker, WebGLRenderer } from 'pixi.js'
import { lampSpots } from './lamps'
import { LANTERN_ASPECT, lanternField, lanternSpots } from './lanterns'
import { drive, type LivingLayer } from './living'

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
/**
 * Backing-store resolution cap.
 *
 * Was 1.5, which on a 1280x841 canvas gives a 1920x1262 buffer — every curve in
 * the scene resolved at three-quarters of the display's own grid, on top of
 * antialiasing being off. 2 is the standard ceiling: it covers every retina
 * class, and above it the memory cost doubles again for a difference nobody has
 * ever been able to point at.
 */
const MAX_DPR = 2

/** Blend two packed RGB colours. `t` 0 gives `a`, 1 gives `b`. */
function mixRgb(a: number, b: number, t: number): number {
  const k = Math.max(0, Math.min(1, t))
  const r = Math.round(((a >> 16) & 255) + (((b >> 16) & 255) - ((a >> 16) & 255)) * k)
  const g = Math.round(((a >> 8) & 255) + (((b >> 8) & 255) - ((a >> 8) & 255)) * k)
  const c = Math.round((a & 255) + ((b & 255) - (a & 255)) * k)
  return (r << 16) | (g << 8) | c
}

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
  /**
   * The chosen names, as support counts in the order the wall lists them.
   * Depth encodes support — see lanterns.ts.
   */
  setLanterns(counts: readonly number[]): void
  /** The note key per lantern, so simulation state survives a rebuild. */
  setLanternKeys(keys: readonly string[]): void
  /**
   * STOP THE SKY, OR START IT AGAIN — WCAG 2.2.2 (Pause, Stop, Hide), Level A.
   *
   * The scene animates for as long as the page is open, and `still` above is
   * set from prefers-reduced-motion — which is a SYSTEM setting, not a control
   * on the page. Someone who finds this sky distracting on this page, on this
   * visit, had no way to stop it without leaving to change an OS preference.
   * This is that way.
   *
   * Turning it on renders one more frame and then draws nothing: the picture
   * you get is the picture that was there, not a blank canvas — a stopped scene
   * is still a scene.
   */
  setStill(on: boolean): void
  resize(): void
  destroy(): void
}

export async function createValley(options: ValleyOptions): Promise<ValleyHandle> {
  const { canvas, roomWidth = 0.44, still = false } = options

  /**
   * `still` IS WHERE THIS STARTED; `stilled` IS WHERE IT IS NOW.
   *
   * They were one variable, which was fine while the only input was a media
   * query read once at construction. With a control on the page they are two
   * different questions — "was this built for a reduced-motion visitor" and
   * "is the sky stopped at this moment" — and every read AFTER construction
   * wants the second one.
   */
  let stilled = still

  /**
   * Boot timing, DEV ONLY.
   *
   * The scene's construction is the largest single main-thread task on the
   * page and the one that decides whether a weak phone feels stuck, so it is
   * worth being able to see its parts from outside. `import.meta.env.DEV` is
   * statically false in the build, so the whole thing is dropped from
   * production output rather than merely skipped.
   */
  /**
   * ── LET THE BROWSER BREATHE BETWEEN THE PHASES ──────────────────────────
   *
   * Building this scene is the largest single task on the page, and a task is
   * the unit the main thread cannot be interrupted inside: for as long as one
   * runs, a tap does nothing and the opening's own timers cannot fire.
   * Measured at 4x CPU on a 412px phone, the whole construction landed as ONE
   * 443ms task — init 262, layout 38, first render 134 — and the three
   * numbers add to 434, so that task is this function and nothing else.
   *
   * Yielding between the phases does not make any phase faster. It breaks one
   * unresponsive 443ms into three stretches the browser can schedule around,
   * and only the part of each over 50ms counts against blocking time. The
   * scene appears at the same moment; the page stops being deaf while it is
   * being built.
   *
   * `scheduler.yield()` where it exists, because it resumes at the front of
   * the queue rather than behind every other pending task — a plain
   * setTimeout(0) hands the scene to the back of the line and the backdrop
   * arrives visibly later. setTimeout is the fallback for everything else.
   */
  const breathe = async (): Promise<void> => {
    const s = (globalThis as { scheduler?: { yield?: () => Promise<void> } }).scheduler
    if (typeof s?.yield === 'function') await s.yield()
    else await new Promise((done) => setTimeout(done, 0))
    // The clock restarts on the far side. Without this the NEXT phase's mark
    // includes however long the browser took to come back, and layout read
    // 111ms on a phone where the work itself is nearer 38.
    __t = performance.now()
  }

  const boot = import.meta.env.DEV
    ? ((window as unknown as Record<string, Record<string, number>>).__vb ||= {})
    : null
  let __t = performance.now()
  const mark = (k: string) => {
    if (boot) boot[k] = performance.now() - __t
    __t = performance.now()
  }
  const renderer = new WebGLRenderer()
  await renderer.init({
    canvas,
    /**
     * ON, and it should never have been off.
     *
     * The scene is ridges, a dome, flag corners and lamp discs — diagonals and
     * curves in almost every shape it draws. Without multisampling every one of
     * those resolved to a hard stair-step, which is most of what "the graphics
     * look poor" was: the stupa's dome had visible stepping, and so did every
     * ridge line behind it.
     *
     * The cost is real but small here: the static half of the scene is cached
     * to a texture and only the flags, lanterns and lamps are redrawn, so
     * multisampling is paid on a handful of small shapes per frame rather than
     * on the whole frame.
     */
    antialias: true,
    resolution: Math.min(window.devicePixelRatio || 1, MAX_DPR),
    autoDensity: true,
    background: PAPER,
    width: canvas.clientWidth || 1,
    height: canvas.clientHeight || 1,
    // The page owns when to draw; a second ticker would fight the rAF loop below.
    powerPreference: 'low-power',
  })

  /**
   * ── PIXI'S OWN TICKER, THROTTLED ────────────────────────────────────────
   *
   * "The page owns when to draw; a second ticker would fight the rAF loop
   * below" is written above `powerPreference`, and it was half true: nothing
   * schedules a RENDER but `Ticker.system` runs anyway. Measured on a loaded
   * page it is started, with two listeners — Pixi's texture and renderable
   * garbage collectors — and it was calling them at the display's full rate,
   * uncapped, forever. It showed up in the profile as an unattributed
   * 0.6–1.0ms per frame on a throttled phone, more than everything this page
   * writes itself.
   *
   * Stopping it outright would take the texture GC with it, and this scene DOES
   * churn textures — every resize re-bakes the cached static layer. So it is
   * throttled instead: the collectors are housekeeping on a fixed set of
   * objects, and four passes a second is many more than they need.
   */
  Ticker.system.maxFPS = 4

  mark('init')
  const stage = new Container()
  const sky = new Graphics()
  const ridgeLayer = new Container()
  const grove = new Graphics()
  const roofs = new Graphics()
  const stupa = new Graphics()
  const flags = new Container()
  /**
   * CONTAINERS, NOT GRAPHICS. Both of these are only ever parents — nothing is
   * drawn into either — and calling addChild on a Graphics is deprecated in
   * Pixi v8 and slated for removal. It logged a deprecation on every load; the
   * console gate counted it, and the message was invisible because Pixi puts
   * the text in console.groupCollapsed and only the stack in console.warn.
   */
  /** The cords the flags hang from. Redrawn per frame — it is three polylines
   *  and it has to follow the same wave the flags ride. */
  const cords = new Graphics()
  const lanterns = new Container()
  const veil = new Graphics()
  const lamps = new Container()
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
  // Additive blending moved onto each lamp as it is built (see buildLamps).
  // A Container's blendMode is not the same thing as its children's: setting it
  // here promotes the container to its own render group, which is a heavier
  // object than nine circles need.
  // Order is the composition: sky, range, town, the stupa standing above the
  // town, flags strung from it, lanterns in front of everything, then the veil
  // that turns the left half back into paper.
  // The static half — everything that only changes on resize — grouped so it
  // can be cached to a single texture.
  const still2 = new Container()
  still2.addChild(sky, ridgeLayer, grove, roofs, stupa)
  // Lamps sit ABOVE the veil. They are the one thing in the scene that is
  // supposed to read through the paper — a light seen from inside a room is not
  // dimmed by the wall it is beyond, and on a phone the veil is heavy enough
  // over the town that lamps under it would simply not be visible.
  /**
   * LANTERNS ABOVE THE VEIL, with the lamps.
   *
   * The veil fades the left of the frame back to paper so dark ink stays
   * readable over it, and the lanterns sat underneath it — so the leftmost was
   * washed almost to nothing while its NAME, drawn in the DOM above
   * everything, stayed fully legible. A name floating over a lantern nobody
   * can see reads as a missing lantern, which is exactly how it was reported.
   *
   * They belong with the lamps, for the lamps' own reason: a light seen from
   * inside a room is not dimmed by the wall it is beyond.
   */
  stage.addChild(still2, cords, flags, veil, lanterns, lamps)

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
  /**
   * THE MIDDLE DISTANCE — a treeline between the last ridge and the town.
   *
   * The composition jumped straight from a smooth ridge to rooftops with
   * nothing in between, and that gap is what kept reading as flat: aerial
   * perspective needs something at every depth to step through, or the eye
   * reads two planes rather than a valley. It is the one thing that has held
   * the desktop composition at A− through four passes.
   *
   * Trees rather than more hills, because the thing missing is TEXTURE at a
   * middle scale — another smooth silhouette would just be a fifth ridge. From
   * across a valley a treeline is not individual trees, it is a ragged upper
   * edge on a soft mass, so that is what is drawn: one filled band whose top is
   * a run of overlapping arcs at slightly different heights. Individually they
   * are unreadable, which is correct — you are looking at woodland, not at a
   * tree.
   *
   * Sits between the third ridge and the town in both position and colour, so
   * it reads as the next step back rather than as an object in front.
   */
  function paintGrove() {
    grove.clear()
    const rand = prng(0x7ee5)
    const baseY = horizon()
    // Taller than the first cut. At 0.062 the band was there and doing nothing
    // — a step between two planes has to be big enough to BE a plane.
    const top = baseY - h * (w < 600 ? 0.062 : 0.09)
    const unit = Math.max(5, h * 0.014)

    grove.moveTo(-20, baseY + 4)
    let x = -20
    while (x < w + 30) {
      // Each clump a slightly different height, and the arc is wider than it is
      // tall — a canopy spreads, it does not spike. Spikes read as conifers on a
      // ridgeline, which is a different place entirely.
      const rise = unit * (0.7 + rand() * 1.5)
      const half = unit * (0.9 + rand() * 1.1)
      const cx = x + half
      grove.quadraticCurveTo(cx, top - rise, cx + half, top + unit * 0.25)
      x = cx + half * (0.7 + rand() * 0.4)
    }
    grove.lineTo(w + 30, baseY + 4)
    grove.lineTo(-20, baseY + 4)
    /* Pitched deliberately BETWEEN its neighbours: darker than the near ridge
       above it (0x6e7288) and lighter than the town below (0x585470), which is
       what makes it read as the next step back rather than as part of either.
       At 0x7c7f95 it was lighter than both and dissolved into the haze. */
    grove.fill({ color: 0x666a82, alpha: 0.88 })

    // A thin lighter edge where the last of the sky catches the canopy — the
    // same reason the terraces are drawn: an edge is what makes a mass read as
    // having a top rather than being a cut-out.
    grove.stroke({ color: 0xc2c5d6, width: Math.max(0.8, h * 0.0014), alpha: 0.42 })
  }

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
   * wrong thing — 48 flag quads and every lantern torn down and re-tessellated 60
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
  /**
   * Lungta order, and it is fixed: blue, white, red, green, yellow — sky, air,
   * fire, water, earth. It repeats in that sequence on every real line, so
   * getting it wrong is the kind of mistake a Nepali visitor sees instantly.
   *
   * Muted from the primaries they were. Cotton that has been on a rooftop
   * through a monsoon is not poster paint, and at full saturation these five
   * were the loudest thing in a frame whose whole palette is dusk.
   */
  const FLAG_COLORS = [0x4a6f9e, 0xe8e6df, 0xb05043, 0x5f8a63, 0xd0a24e] as const

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
    /**
     * THE DOME IS THE BIGGEST OBJECT IN THE FRAME AND IT WAS FLAT.
     *
     * A single white fill on a hemisphere gives a white circle: the shape says
     * dome and the shading says sticker. Whitewashed lime is bright but not
     * luminous — it takes a soft terminator and picks up the sky on the side
     * away from the light.
     *
     * A GRADIENT, not overlaid ellipses. The first attempt drew a light blob
     * and a dark blob on top of the fill, reasoning they would sit inside the
     * silhouette. They did not: both spilled past the curve and read as two
     * bubbles stuck to the dome, which was worse than the flat fill it
     * replaced. A gradient is bounded by the path it fills, which is the
     * property that was actually needed.
     */
    domeFill?.destroy()
    domeFill = new FillGradient({
      type: 'linear',
      start: { x: cx - s, y: domeY - s },
      end: { x: cx + s, y: domeY + s * 0.5 },
      textureSpace: 'global',
      colorStops: [
        { offset: 0, color: 0xfffdf8 },
        { offset: 0.52, color: white },
        { offset: 1, color: 0xc9ccdb },
      ],
    })
    stupa.fill(domeFill)

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
        const g = new Graphics()
        const base = FLAG_COLORS[i % 5]

        /**
         * CLOTH, NOT A SWATCH.
         *
         * These were flat rects, axis-aligned, with no string — five primaries
         * floating in a curve, which at 4x read as confetti rather than as
         * flags on a line. Three things fix that and none of them is expensive:
         *
         *   · the top edge is pinched to the cord and the bottom edge hangs
         *     free, so the shape is a trapezoid rather than a rectangle;
         *   · a darker band down the hoist, where cloth gathers at the
         *     stitching and light never reaches;
         *   · the free edge lifts slightly, drawn as a curve rather than a
         *     straight cut.
         *
         * Built once each; the wind below only ever transforms them.
         */
        g.moveTo(-fw * 0.43, 0)
          .lineTo(fw * 0.43, 0)
          .lineTo(fw * 0.43, fh * 0.92)
          .quadraticCurveTo(0, fh * 1.06, -fw * 0.43, fh * 0.9)
          .closePath()
          .fill({ color: base, alpha: 0.88 })

        // The hoist: the fold nearest the cord, always in its own shadow.
        g.rect(-fw * 0.43, 0, fw * 0.2, fh * 0.94).fill({ color: mixRgb(base, 0x2b2a33, 0.34), alpha: 0.5 })

        // And the top seam, which is what makes it look tied on rather than
        // stuck on.
        g.moveTo(-fw * 0.43, 0)
          .lineTo(fw * 0.43, 0)
          .stroke({ color: mixRgb(base, 0x2b2a33, 0.5), width: Math.max(0.6, fw * 0.09), alpha: 0.55 })

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
    /**
     * THE CORD WAS MISSING ENTIRELY.
     *
     * The flags rode a catenary and nothing was ever drawn along it, so they
     * hung on nothing — the single biggest reason the line read as confetti.
     * Worse, every flag was axis-aligned no matter how steeply the line fell,
     * which no hanging cloth does.
     *
     * The same sampled curve now draws the string AND rotates each flag to sit
     * square to it. `cords` is cleared and redrawn each frame, which is the one
     * per-frame geometry rebuild in this file that is justified: it is three
     * polylines of a dozen points, and it has to track the same travelling wave
     * the flags do or the flags come off the string.
     */
    cords.clear()

    for (const line of flagLines) {
      const { flags: per, x1, y1, x2, y2, sag, phase } = line

      /**
       * SAMPLED INTO SCALARS, NOT INTO OBJECTS.
       *
       * This returned a fresh `{x, y}` per call and is called 23 times for the
       * string plus twice per flag, on three lines, every draw — a few
       * thousand short-lived objects a second whose only purpose was to carry
       * two numbers a few lines. That is not a leak, it is GC pressure, and
       * the phones that feel this page as sticky are the ones where a
       * collection lands in the middle of a frame.
       */
      let ax = 0
      let ay = 0
      const at = (u: number) => {
        const swell = Math.sin(u * Math.PI)
        const wave = Math.sin(time * 1.6 + u * 5 + phase) * (h * 0.004) * swell
        ax = x1 + (x2 - x1) * u
        ay = y1 + (y2 - y1) * u + swell * sag + wave
      }

      // The string first, so the flags sit on it.
      const SAMPLES = 22
      at(0)
      cords.moveTo(ax, ay)
      for (let k = 1; k <= SAMPLES; k++) {
        at(k / SAMPLES)
        cords.lineTo(ax, ay)
      }
      // Thin and low-contrast: it is a string, and at 0.5 alpha in near-black
      // it drew as the most prominent line in the frame — heavier than the
      // ridges behind it and heavier than the flags it carries.
      cords.stroke({ color: 0x5c5870, width: Math.max(0.6, h * 0.0009), alpha: 0.3 })

      for (const f of per) {
        const u = f.u
        at(u)
        f.g.x = ax
        f.g.y = ay
        // Square to the cord: sampled either side rather than differentiated,
        // because the wave makes the analytic slope longer than the whole
        // function it corrects.
        at(Math.max(0, u - 0.02))
        const bx = ax
        const by = ay
        at(Math.min(1, u + 0.02))
        f.g.rotation = Math.atan2(ay - by, ax - bx)
      }
    }
  }

  /**
   * THE CHOSEN NAMES, FLOATING.
   *
   * This replaces a flock of nine pigeons. The birds were there to stop the
   * scene reading as a backdrop, and they did that — but they were decoration
   * occupying the only open sky, while the names the family had actually chosen
   * sat invisible along the rooftops with `opacity: 0` labels. The sky is worth
   * more than atmosphere. It is where the answer goes.
   *
   * Depth is support: see lanterns.ts, which also owns the placement so the DOM
   * labels land on their own lights. Each lantern is built ONCE and thereafter
   * only moved and dimmed — geometry is the expensive thing, and this file has
   * twice grown a per-frame rebuild by accident.
   */
  interface Lantern {
    g: Graphics
    /** Placement in 0..1, from the shared geometry. */
    x: number
    y: number
    scale: number
    glow: number
    /** Feeds lanternDrift, which owns the motion for canvas and label alike. */
    index: number
  }
  let lanternList: Lantern[] = []
  /** Support counts, newest set wins. Drives how many and how near. */
  let lanternCounts: readonly number[] = []
  /** Note keys, parallel to the counts, so bodies survive a rebuild. */
  let lanternKeys: readonly string[] = []

  function buildLanterns() {
    for (const l of lanternList) l.g.destroy()
    lanternList = []
    lanterns.removeChildren()
    if (lanternCounts.length === 0) return

    // 1100 is where the side-by-side layout begins; below it the page is one
    // column and there is no sky to hang anything in.
    const spots = lanternSpots(lanternCounts, w < 1100, h)
    // The field is seeded from the resting spots and then simulates. Both this
    // canvas and the DOM labels read it — see lanterns.ts.
    // Keys so a rebuild keeps each lantern where it is — see reset().
    lanternField().reset(spots, w / Math.max(1, h), lanternKeys)
    // One frame and stop means an entrance that has not run is one that never
    // will — see settle() in lanterns.ts.
    //
    // `stilled`, not `still`: this is reachable AFTER construction. Send a name
    // while the sky is stopped and setLanterns rebuilds the field, and on the
    // start-value this branch was skipped — so the new lantern was built at the
    // bottom of its rise, drawn once, and left there, with its name hanging in
    // the sky beneath the paper it is written on. A stopped scene still has to
    // be a finished one.
    if (stilled) lanternField().settle()

    /**
     * FAR FIRST. The lanterns are large enough now to overlap each other, and a
     * sky of them at different distances should occlude — but only in the right
     * order. Painting in list order put a far lantern over a near one wherever
     * two happened to cross.
     */
    const order = spots.map((_, i) => i).sort((a, z) => spots[a].depth - spots[z].depth)
    /**
     * DEPTH IS NOW DYNAMIC, so paint order has to be too. Sorting once at build
     * time ordered them by RESTING depth, but the drift moves each lantern
     * toward and away from the viewer independently — so one that had drifted
     * forward could still be painted behind one that had drifted back, which
     * is the single thing that would give the third dimension away.
     */
    lanterns.sortableChildren = true

    for (const i of order) {
      const spot = spots[i]
      const g = new Graphics()

      /**
       * ─── A LANTERN THE NAME IS WRITTEN ON ───────────────────────────────
       *
       * It was an 8px hexagon with the name floating well above it. What the
       * page is actually describing is a sky lantern with a name written on
       * its paper, so the paper has to be big enough to hold two lines of
       * type — see LANTERN_H in lanterns.ts.
       *
       * The silhouette is built from curves rather than the straight segments
       * the small version used: a dome that swells past its widest point and
       * draws back in to a narrow mouth, which is what gives paper its
       * inflated look. A polygon at this size reads as a shed.
       */
      const hh = spot.size * h
      const ww = hh * LANTERN_ASPECT
      const top = -hh * 0.5
      const bot = hh * 0.5
      const half = ww * 0.5

      const silhouette = (gfx: Graphics) =>
        gfx
          .moveTo(0, top)
          .bezierCurveTo(half * 0.86, top, half, top + hh * 0.3, half, top + hh * 0.46)
          .bezierCurveTo(half, bot - hh * 0.24, half * 0.62, bot - hh * 0.04, half * 0.34, bot)
          .lineTo(-half * 0.34, bot)
          .bezierCurveTo(-half * 0.62, bot - hh * 0.04, -half, bot - hh * 0.24, -half, top + hh * 0.46)
          .bezierCurveTo(-half, top + hh * 0.3, -half * 0.86, top, 0, top)
          .closePath()

      // The halo the paper throws, before the paper itself.
      g.ellipse(0, hh * 0.08, ww * 0.78, hh * 0.86).fill({
        color: 0xffb765,
        alpha: 0.05 + spot.glow * 0.07,
      })

      // The paper. Pale and warm, because the NAME has to be legible on it —
      // this is the one surface in the scene that carries type.
      silhouette(g).fill({ color: 0xfbeacd, alpha: 0.72 + spot.glow * 0.2 })

      // Lit from within and from BELOW, where the flame is: the base of a sky
      // lantern is always the brightest part of it.
      g.ellipse(0, bot - hh * 0.18, ww * 0.34, hh * 0.26).fill({
        color: 0xffd79a,
        alpha: 0.3 + spot.glow * 0.34,
      })

      // Two ribs. Paper stretched over a frame creases along it, and this is
      // the cheapest thing that says "paper" rather than "balloon".
      for (const k of [-0.34, 0.34]) {
        g.moveTo(half * k * 0.92, top + hh * 0.14)
          .bezierCurveTo(half * k, top + hh * 0.42, half * k, bot - hh * 0.3, half * k * 0.5, bot - hh * 0.02)
          .stroke({ color: 0xd8a463, width: Math.max(0.5, hh * 0.012), alpha: 0.3 })
      }

      // The edge, so the shape holds against a sky close to it in value.
      silhouette(g).stroke({
        color: 0xc98a4c,
        width: Math.max(0.6, hh * 0.016),
        alpha: 0.4 + spot.glow * 0.25,
      })

      // The mouth ring and the flame hanging in it.
      g.ellipse(0, bot, ww * 0.34, hh * 0.045).stroke({
        color: 0xb4703a,
        width: Math.max(0.5, hh * 0.014),
        alpha: 0.5,
      })
      g.ellipse(0, bot - hh * 0.05, ww * 0.07, hh * 0.06).fill({
        color: 0xfff1cd,
        alpha: 0.55 + spot.glow * 0.45,
      })

      lanterns.addChild(g)
      lanternList.push({
        g,
        x: spot.x,
        y: spot.y,
        scale: spot.scale,
        glow: spot.glow,
        index: i,
      })
    }
  }

  /** Wall-clock of the previous frame, for the simulation's timestep. */
  let lastLanternTime = 0

  function animateLanterns(time: number) {
    const f = lanternField()
    if (f.bodies.length !== lanternList.length) return

    // THE CANVAS OWNS THE CLOCK. One stepper, or the field advances twice per
    // frame and everything moves at double speed.
    const dt = lastLanternTime === 0 ? 1 / 60 : time - lastLanternTime
    lastLanternTime = time
    f.step(dt, time)

    lanternList.forEach((l, i) => {
      const body = f.bodies[i]
      l.g.x = body.x * w
      l.g.y = body.y * h
      // The third dimension: nearer is bigger, brighter, and in front.
      l.g.scale.set(body.scale / l.scale)
      l.g.alpha = body.alpha
      l.g.zIndex = body.scale
    })
  }

  let veilFill: FillGradient | null = null
  /** The dome's shading. Held so it can be released with the others. */
  let domeFill: FillGradient | null = null

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
     * lanterns drifting across "6,715 names, out of the Vedas and the Sutras", and
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
    paintGrove()
    paintRoofs()
    // One call, so a layer cannot be added to the render loop and forgotten
    // in the resize path — the failure that looks fine until you rotate a phone.
    living.build(w, h)
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
   * line and the lanterns over the stupa. That is a place being alive, which is
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

  /**
   * BUILT ONCE, THEN ONLY DIMMED — and I got this wrong here after getting it
   * right everywhere else. Pixi's guidance is that Graphics GEOMETRY is the
   * expensive thing while alpha and transform are cheap, which is why the flags
   * and the lanterns are built once and only moved. The lamps were written after
   * that and still cleared and re-tessellated all 27 circles every frame: 1,620
   * rebuilds a second, plus a fresh lampSpots() array each time, for an effect
   * whose only per-frame change is brightness.
   *
   * One Graphics per lamp, holding its three falls of light at their base
   * alphas. The flicker is `g.alpha`, which multiplies all three proportionally
   * — the same picture, with nothing rebuilt.
   */
  let lampShapes: Graphics[] = []

  function buildLamps() {
    for (const g of lampShapes) g.destroy()
    lampShapes = []
    if (lampCount <= 0) return
    const spots = lampSpots(lampCount, w < 600)
    const r = Math.max(2.2, h * 0.0042)
    for (const spot of spots) {
      const g = new Graphics()
      /**
       * A REAL FALLOFF, NOT THREE DISCS.
       *
       * Three concentric circles at three alphas produce three hard edges, and
       * at 4x they read exactly as what they are: a bullseye sticker with two
       * visible rings. Light does not have edges.
       *
       * Sampled falloff instead — a stack of rings whose alpha follows an
       * inverse-square-ish curve, which is what a small source actually does.
       * Twelve steps is where the banding stops being visible at 4x; the cost
       * is twelve circles built ONCE per lamp, never per frame.
       */
      const STEPS = 12
      for (let k = STEPS; k >= 1; k--) {
        const t = k / STEPS
        // Warmer toward the core, as a real flame is: the outer haze is orange
        // where the centre is nearly white.
        const warm = 1 - t
        const color = mixRgb(0xff8a2e, 0xfff0cc, warm * warm)
        g.circle(0, 0, r * 6.2 * t).fill({ color, alpha: 0.055 * (1 - t) ** 1.6 + 0.012 })
      }
      // The source itself, small and bright, sitting inside its own light.
      g.circle(0, 0, r * 0.85).fill({ color: 0xfff4d8, alpha: 0.65 })
      // Additive, per lamp. Summed light is why the alphas above are so low.
      g.blendMode = 'add'
      g.x = spot.x * w
      g.y = spot.y * h
      lamps.addChild(g)
      lampShapes.push(g)
    }
  }

  /**
   * ── EVENING FALLS WHILE YOU ARE THERE ──────────────────────────────────
   *
   * The town used to be fully lit in the first frame and stay that way, every
   * window breathing on its own sine. Beautiful, and a loop — which is the one
   * thing a place is not. Nothing in that town could ever be different from how
   * you found it.
   *
   * The windows come on one at a time now, over about a hundred seconds,
   * unevenly, and they never go off again. It is not an animation anybody is
   * meant to watch: it is slow enough that no single light is caught arriving,
   * and the only way to notice is to look up after a while and find there are
   * more of them than there were. That is the difference between scenery and a
   * place — a place is going on whether or not you are looking at it, and it
   * does not rewind.
   *
   * A third are lit from the start, because dusk is not darkness and a wholly
   * black town at the top of the sequence reads as broken rather than as early.
   *
   * `Math.sin` on the index rather than a random: deterministic, so a lamp does
   * not change character between renders, and irregular enough that no two
   * neighbours come on together.
   */
  const LIGHTING_WINDOW = 100
  let lampsFrom = 0

  function litBy(i: number, count: number): number {
    // Deterministic 0..1 per lamp, then squared so most of them arrive late —
    // a village fills in slowly and then all at once, never at a steady rate.
    const r = Math.abs(Math.sin(i * 12.9898 + 78.233) * 43758.5453) % 1
    if (i % 3 === 0) return 0
    return r * r * LIGHTING_WINDOW * (0.4 + (i / Math.max(1, count)) * 0.6)
  }

  function animateLamps(time: number) {
    if (lampsFrom === 0) lampsFrom = time
    const since = still ? LIGHTING_WINDOW * 2 : time - lampsFrom
    lampShapes.forEach((g, i) => {
      const at = litBy(i, lampShapes.length)
      // 1.6s to come up — a window being lit, not a light switch.
      const up = Math.max(0, Math.min(1, (since - at) / 1.6))
      if (up <= 0) {
        g.alpha = 0
        return
      }
      // Each on its own slow beat, never in step: a row of lights pulsing
      // together is a progress indicator, not a village at dusk.
      const breath = 0.84 + Math.sin(time * (1.1 + i * 0.17) + i * 2.3) * 0.16
      g.alpha = breath * (up * up * (3 - 2 * up))
    })
  }

  /**
   * THE THREE MOVING LAYERS, DECLARED AGAINST THE CONTRACT.
   *
   * `LivingLayer` (./living.ts) exists because prose did not hold: the rule
   * "build once, animate transform and alpha only" was written above each of
   * these and I still wrote a per-frame rebuild into the lamps. Naming them as
   * the interface means the compiler checks the shape, `drive()` guarantees
   * build and animate stay in step, and a fourth layer has one obvious way in.
   */
  const flagLayer: LivingLayer = {
    build: () => buildFlags(paintStupa()),
    animate: animateFlags,
    destroy: () => {
      for (const line of flagLines) for (const f of line.flags) f.g.destroy()
      flagLines = []
    },
  }

  const cordLayer: LivingLayer = {
    // Nothing to build: the cord is a per-frame polyline (see animateFlags).
    build: () => {},
    animate: () => {},
    destroy: () => cords.destroy(),
  }

  const lanternLayer: LivingLayer = {
    // build() takes the viewport and is called on mount and resize, which is
    // exactly when a lantern's pixel size changes. The COUNTS arrive separately
    // via setLanterns, so both paths funnel into the same builder.
    build: buildLanterns,
    animate: animateLanterns,
    destroy: () => {
      for (const l of lanternList) l.g.destroy()
      lanternList = []
    },
  }

  const lampLayer: LivingLayer = {
    build: buildLamps,
    animate: animateLamps,
    destroy: () => {
      for (const g of lampShapes) g.destroy()
      lampShapes = []
    },
  }

  const living = drive([flagLayer, cordLayer, lanternLayer, lampLayer])

  let clock = 0

  /**
   * ── THIRTY, NOT SIXTY ────────────────────────────────────────────────────
   *
   * This is a backdrop. Nothing in it is a response to a touch: the flags
   * sway, the lanterns drift on a wander a few percent of the screen wide,
   * and the lamps breathe. All of it is slow enough that the second half of
   * every pair of frames drew a picture indistinguishable from the first.
   *
   * Measured on a 412px phone at 4x CPU — roughly a mid-range Android against
   * this host — the loop cost 4.0ms per frame, of which 2.9ms was
   * renderer.render and 1.1ms the layer animation. At 60fps that is a quarter
   * of the frame budget spent on scenery, every frame, forever; the page's own
   * work then has to fit in what is left, which is what makes a phone feel
   * sticky rather than slow.
   *
   * Halving the rate halves both halves of that, and it is the ONLY change
   * here that costs nothing visually — the alternatives (fewer lanterns,
   * coarser gradients, a lower backing-store resolution) all take something
   * off the screen.
   *
   * The tolerance matters: a bare `< FRAME_MS` test against 16.67ms frames
   * lands within a rounding error of the threshold and drops to 20fps
   * whenever the browser's timestamps jitter the wrong way. Four milliseconds
   * of slack is far below one frame and immune to that.
   */
  const FRAME_MS = 1000 / 30
  let lastDraw = 0

  function frame() {
    if (!alive) return
    /**
     * ── THE GUARD HAS TO BE HERE, NOT AT THE CALL SITE ──────────────────────
     *
     * cancelAnimationFrame alone loses a race and it is not a subtle one: the
     * re-arm below happens BEFORE any work, so a callback the browser has
     * already dispatched re-schedules the loop microseconds after you cancel
     * it. Measured on the first attempt at this control — rAF held at ~40 a
     * second with the button reading "still".
     *
     * Same shape as `alive` above, for the same reason: the only place a loop
     * can be reliably stopped is inside the loop.
     */
    if (stilled) {
      raf = 0
      return
    }
    /**
     * WALL CLOCK, NOT A FRAME COUNT. This was `clock += 1/60`, which runs slow
     * whenever frames drop and — the reason it had to change — is unshareable:
     * the DOM lantern labels compute the same drift from the same time and
     * cannot see this loop's private counter. performance.now() is a base both
     * sides already have.
     */
    raf = requestAnimationFrame(frame)
    const now = performance.now()
    if (now - lastDraw < FRAME_MS - 4) return
    lastDraw = now
    clock = now / 1000
    living.animate(clock)
    renderer.render(stage)
  }

  mark('setup')
  await breathe()
  layout()
  mark('layout')
  await breathe()

  // One frame either way, with the flags hung and the lanterns placed — not an
  // empty sky. Under reduced motion it is also the only frame.
  living.animate(0)
  renderer.render(stage)
  mark('firstRender')
  if (!still) raf = requestAnimationFrame(frame)

  /** Stop when the tab is hidden. A scene animating into a background tab is
   *  battery spent on nobody. */
  const onVisibility = () => {
    if (document.hidden) {
      cancelAnimationFrame(raf)
      raf = 0
    } else if (!stilled && alive && raf === 0) {
      raf = requestAnimationFrame(frame)
    }
  }
  document.addEventListener('visibilitychange', onVisibility)

  return {
    setLamps(count) {
      if (count === lampCount) return
      lampCount = count
      buildLamps()
      if (stilled) {
        animateLamps(0)
        renderer.render(stage)
      }
    },
    setLanternKeys(keys) {
      lanternKeys = [...keys]
    },
    setLanterns(counts) {
      // Same length AND same values, or a re-render on every parent update
      // would rebuild the whole field and reset every lantern's drift.
      if (
        counts.length === lanternCounts.length &&
        counts.every((c, i) => c === lanternCounts[i])
      ) {
        return
      }
      lanternCounts = [...counts]
      buildLanterns()
      if (stilled) {
        animateLanterns(0)
        renderer.render(stage)
      }
    },
    setStill(on) {
      if (on === stilled) return
      stilled = on
      if (on) {
        // Belt and braces: the guard inside frame() is what actually stops the
        // loop, but a pending callback would otherwise draw one more frame
        // after the button says it has stopped.
        cancelAnimationFrame(raf)
        raf = 0
        /**
         * SETTLE, NOT FREEZE. A lantern caught halfway through its rise is a
         * lantern with its name hanging in the sky beneath it — see settle()
         * in lanterns.ts, which exists for exactly this. Stopping the sky
         * should give you the composition, not the frame you happened to
         * press on.
         */
        lanternField().settle()
        // One last frame, so the settle is on screen. After this nothing draws.
        living.animate(clock)
        renderer.render(stage)
      } else if (alive && raf === 0 && !document.hidden) {
        raf = requestAnimationFrame(frame)
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
      domeFill?.destroy()
      living.destroy()
      renderer.destroy()
    },
  }
}
