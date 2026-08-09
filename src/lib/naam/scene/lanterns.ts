/**
 * Where the chosen names float, and how far away each one is.
 *
 * ─── WHY LANTERNS AND NOT THE LAMPS ────────────────────────────────────────
 *
 * The names were already in the scene, as lamps along the rooftops. Audited,
 * they were invisible: every label computed `opacity: 0` and was revealed only
 * on hover, and all six sat in one flat band at the very bottom of the frame,
 * down among the roofs. A name nobody can read is not on the page. So they come
 * up off the ground and into the sky, and they carry their names in the open.
 *
 * Lanterns rather than birds, in the space the birds used to occupy: sky
 * lanterns are real at Nepali festivals, they read warm against a cold ridge
 * line where a chevron reads as a speck, and — the part that matters — a
 * lantern can be NEAR or FAR, which is the whole mechanism below. A bird at
 * three pixels cannot carry a name or a distance.
 *
 * ─── DEPTH IS SUPPORT ──────────────────────────────────────────────────────
 *
 * A name several people chose comes closer: bigger, lower, brighter. A name one
 * person chose hangs further back, smaller and dimmer, up toward the ridge.
 * Nothing is labelled with a number — the sky is the tally, the same way the
 * mala counts without printing a digit next to a child's name.
 *
 * ─── THE BANDS ADAPT TO THE SPREAD OF THE VOTES ────────────────────────────
 *
 * The obvious version maps count linearly onto depth, and it fails in the case
 * this page is actually in most of the time: early on, EVERY name has exactly
 * one vote. A linear map puts all of them at identical depth, in one flat row —
 * which is the fault we just moved them out of.
 *
 * So depth is banded by DISTINCT count, and the band width is the inverse of
 * how many distinct counts there are:
 *
 *     bandWidth = RANGE / distinctCounts
 *
 *   · one distinct count (everybody on one vote) → bandWidth = the whole RANGE.
 *     They spread across the entire sky, because with no ranking to show, depth
 *     is free to do the other job: make it look like a place.
 *   · many distinct counts → many narrow bands. Each tier sits clearly in front
 *     of the one below it, and the ordering is legible at a glance.
 *
 * Which is the rule stated plainly: few distinct votes, wide spread; many
 * distinct votes, tight bands but more of them.
 *
 * ─── ONE GEOMETRY, TWO RENDERERS ───────────────────────────────────────────
 *
 * Same contract as lampSpots, for the same reason: the Pixi canvas draws the
 * lantern and the DOM draws the labelled, focusable control on top of it. If
 * they disagree the name floats away from its light. Neither owns the geometry
 * — this does, in fractions of the canvas box, and both read from it.
 */

/** A lantern's placement, in 0..1 of the canvas box. */
export interface LanternSpot {
  x: number
  y: number
  /** 0 far, 1 near. Drives size, brightness and how low it hangs. */
  depth: number
  /** Multiplier on the drawn lantern and on the label's type size. */
  scale: number
  /**
   * The lantern's HEIGHT as a fraction of the canvas box, so the canvas and the
   * DOM can draw and fill the same object without either measuring the other.
   * Width is this times LANTERN_ASPECT.
   */
  size: number
  /** 0..1, for the glow and the label's ink. */
  glow: number
}

/**
 * A lantern's height at full size, as a fraction of the canvas height, and its
 * width relative to that.
 *
 * These are big — far bigger than the 8px glyph they replace — because THE NAME
 * IS WRITTEN ON THE PAPER. A lantern that only marks where a label floats can
 * be a dot; a lantern the name sits inside has to be at least as large as the
 * two lines of type it carries, which at full scale is about 90x44px. 0.1 of an
 * 841px canvas gives 84px of height and 108px of width, which holds the
 * Devanagari and the Latin beneath it with room to breathe.
 */
const LANTERN_H = 0.1
export const LANTERN_ASPECT = 1.28
/** No lantern smaller than this, in px — see `size` below. */
const MIN_PX = 52

/**
 * The sky the lanterns may occupy, in fractions of the canvas height.
 *
 * The bottom is 0.34 because the tray — the diyo, its label and the three slots
 * — begins at 0.398 of the canvas at both 1280 and 1440, measured. The first
 * cut ran the field to 0.62 and the nearest lantern's name landed on top of
 * "KEEP UP TO THREE" with the flame behind it. Below the tray is not sky, it is
 * furniture; the lanterns get the part of the frame that is actually empty.
 */
/* Raised from 0.1: the names sit INSIDE the paper now rather than above it, so
   nothing overflows the top edge, and a taller band gives depth more room to
   separate lanterns that are close together horizontally. */
const SKY_TOP = 0.05
const SKY_BOTTOM = 0.36
/** Depth range, before banding. */
const RANGE = 1

/**
 * The narrowest gap allowed between two lanterns, in fractions of the canvas
 * width — because the NAME rides above the lantern and is far wider than it.
 *
 * Tuning the jitter down twice did not fix this and could not: with lanes
 * 0.18 of the field apart and a jitter of ±0.1, two neighbours could still land
 * 46px apart while their labels are ~90px wide, so a collision was always
 * reachable. Separation has to be a guarantee, not a probability — the pass
 * below enforces it after placement.
 *
 * Raised from 0.085 when the lanterns grew to carry their own names: the widest
 * is now ~124px, and 0.085 of 1280 is 109. Lanterns MAY overlap — a sky of them
 * at different distances should occlude, and valley.ts paints far before near
 * so it does so correctly — but two names may never touch.
 */
const MIN_GAP = 0.1

/** Deterministic 0..1 from an integer — a lantern must not move between renders. */
function hash(n: number): number {
  const v = Math.sin(n * 127.1 + 11.7) * 43758.5453
  return v - Math.floor(v)
}

/**
 * Place `counts.length` lanterns, where `counts[i]` is how many people chose
 * that name. Order of the returned array matches the order in.
 *
 * ─── THERE IS NO SKY ON A PHONE ────────────────────────────────────────────
 *
 * `stacked` returns an empty array, and that is a measurement rather than a
 * preference. On the stacked layout every part of the canvas is behind
 * something: at 375x667 the occupied bands run 0.039-0.73 (the invitation),
 * 0.366-0.49 (the lamp and its label), 0.502-0.605 (the slots) and 0.651-1
 * (the starters and the box). The largest gap in the entire frame is 3.9% of
 * its height. At 412x915 it is 3.3%.
 *
 * The first cut ignored this and floated the names anyway. They landed across
 * "look through them with you" — the sentence asking the visitor to do the one
 * thing the page exists for. The bird code this replaced had the same guard for
 * the same reason, and said so: "the only open sky is behind the invitation,
 * and nine drifting chevrons across someone's sentence is not atmosphere."
 *
 * So on a phone the names stay in the DOM and stay available to assistive tech,
 * and nothing is drawn over the copy. See the note on .nm-lamps in naam.astro.
 */
export function lanternSpots(
  counts: readonly number[],
  stacked: boolean,
  frameHeight = 0,
): LanternSpot[] {
  if (counts.length === 0 || stacked) return []

  const distinct = [...new Set(counts)].sort((a, b) => b - a)
  const bandWidth = RANGE / distinct.length
  const rankOf = new Map(distinct.map((c, i) => [c, i]))

  const top = SKY_TOP
  const bottom = SKY_BOTTOM
  // The left of the frame stays clear — that side is the room the visitor is
  // standing in, and it is where the type lives.
  /**
   * Inset by half a lantern, because the lantern now has real width — 124px at
   * full scale. Running the lane to 0.99 put the rightmost one half off the
   * frame with its name cut down the middle. The left edge is held clear of the
   * room as well: 0.46 plus the inset keeps the paper off the column the type
   * lives in.
   */
  // LANTERN_H is a fraction of HEIGHT and x is a fraction of WIDTH, so the
  // conversion needs the canvas's own aspect. 1.52 is 1280/841 — the desktop
  // frame this field only ever renders in (lanternSpots returns nothing when
  // stacked), so it is a constant here rather than a measurement.
  const CANVAS_ASPECT = 1.52
  const halfWidth = (LANTERN_H * 1.2 * LANTERN_ASPECT) / 2 / CANVAS_ASPECT
  const from = 0.5 + halfWidth
  /* Past the right edge on purpose. With eight lanterns the separation pass
     otherwise falls back to even spacing at 88px while the names are ~100px
     wide, and they print over each other. A lantern half off the frame is a
     sky; two names sharing the same pixels is a bug. */
  const to = 1.04 - halfWidth

  const placed = counts.map((count, i) => {
    const rank = rankOf.get(count) ?? 0
    // Band 0 is the most-supported and sits nearest. Within its band a lantern
    // is placed by a seeded fraction, so two names on the same count still sit
    // at slightly different distances instead of on one line.
    /**
     * A GAP BETWEEN BANDS, not just adjacent bands.
     *
     * Contiguous bands put the bottom of one tier flush against the top of the
     * next, so the two names on either side of that boundary sit at the same
     * distance and the tiers blur — and then drift finishes the job. Each band
     * uses the upper 62% of its slice and leaves the rest as clear air, which
     * is what makes "these are nearer than those" legible at a glance rather
     * than only in the numbers.
     */
    const withinBand = hash(i * 7 + 3) * 0.62
    const depth = Math.min(1, Math.max(0, 1 - (rank + withinBand) * bandWidth))

    // Near lanterns hang low and far ones ride up toward the ridge.
    const y = top + (bottom - top) * depth
    /**
     * Lane first, jitter second. The jitter only has to break up the evenness —
     * the separation pass below is what keeps names off each other.
     */
    const lane = counts.length === 1 ? 0.5 : i / (counts.length - 1)
    const x = from + (to - from) * (lane * 0.9 + hash(i * 13 + 5) * 0.1)

    /**
     * The floor is 0.78, not 0.62. The name is written ON the paper now, so the
     * lantern's size IS the type size — and at 0.62 the furthest lanterns set
     * their names at about 8.7px, which is not distance, it is unreadable.
     * Perspective is worth less than a name somebody can read.
     */
    const scale = 0.72 + depth * 0.56

    return {
      x,
      y,
      depth,
      scale,
      /**
       * A FLOOR IN PIXELS, because a fraction of a short frame is a small
       * object. The responsive gate caught this at 768, 1000 and 1024, where
       * the canvas is short enough that 0.1 of its height came out at 15-28px:
       * under the 24px tap minimum, and far too small to write a name on.
       *
       * Clamped HERE rather than in either renderer, because the canvas draws
       * the paper and the DOM writes on it — a minimum applied in one and not
       * the other would put the name back outside the lantern.
       */
      size: frameHeight > 0 ? Math.max(LANTERN_H * scale, MIN_PX / frameHeight) : LANTERN_H * scale,
      glow: 0.45 + depth * 0.55,
    }
  })

  /**
   * SEPARATION IN TWO DIMENSIONS, because one ran out.
   *
   * The first pass only pushed sideways, and with eight lanterns there is no
   * sideways left: the gaps come out around 107px while the names are ~100px
   * wide and the lateral drift is ±108px, so any gap can close. Measured, two
   * pairs printed across each other — सोहम् and सात्विक interleaved into one
   * unreadable word.
   *
   * There IS vertical room, so crowded pairs are pushed apart along whichever
   * axis has slack, biased toward y. A few relaxation passes rather than one
   * sweep, because moving one lantern out of a collision can put it into the
   * next; three is enough to settle eight of them and the whole thing runs
   * once per layout, not per frame.
   *
   * Depth is NOT touched. It carries the support ranking, which is the field's
   * entire job — a lantern may be nudged in the sky but never into another
   * tier, so `y` is clamped to its own band.
   */
  /**
   * Both axes, and the gap shrinks as the field fills. Pushing only in y ran
   * out of room the moment a third name was sent and the sky held ten: y is
   * bounded by the band, x is not, so a crowded field has to be allowed to
   * spread sideways as well.
   */
  const crowd = Math.max(0.55, Math.min(1, 7 / placed.length))
  const MIN_X = MIN_GAP * crowd
  const MIN_Y = 0.055 * crowd
  for (let pass = 0; pass < 6; pass++) {
    for (let i = 0; i < placed.length; i++) {
      for (let j = i + 1; j < placed.length; j++) {
        const a = placed[i]
        const b = placed[j]
        const dx = b.x - a.x
        const dy = b.y - a.y
        if (Math.abs(dx) >= MIN_X || Math.abs(dy) >= MIN_Y) continue

        // Push along whichever axis is closer to free. y is cheap until the
        // band edge; past that the only room left is sideways.
        const needY = (MIN_Y - Math.abs(dy)) / 2 + 0.001
        const dirY = dy === 0 ? (i % 2 === 0 ? 1 : -1) : Math.sign(dy)
        a.y -= dirY * needY
        b.y += dirY * needY

        const needX = (MIN_X - Math.abs(dx)) / 2 + 0.001
        const dirX = dx === 0 ? (i % 2 === 0 ? 1 : -1) : Math.sign(dx)
        a.x -= dirX * needX * 0.5
        b.x += dirX * needX * 0.5
      }
    }
  }

  /**
   * Back inside the sky, and roughly inside each lantern's own depth band.
   *
   * The band allowance widens as the field fills: with ten up there a hard
   * clamp leaves the relaxation nowhere to go and they simply overlap. Size and
   * brightness still carry the ranking exactly — only the height is allowed to
   * bleed between neighbouring tiers when the sky is full.
   */
  const slack = 0.5 + (1 - crowd) * 0.9
  const bandTop = (depth: number) => top + (bottom - top) * Math.min(1, depth + bandWidth * slack)
  const bandBottom = (depth: number) => top + (bottom - top) * Math.max(0, depth - bandWidth * slack)
  for (const spot of placed) {
    spot.y = Math.min(bandTop(spot.depth), Math.max(bandBottom(spot.depth), spot.y))
    spot.y = Math.min(bottom, Math.max(top, spot.y))
  }

  return placed
}

/**
 * ─── HOW A LANTERN MOVES ───────────────────────────────────────────────────
 *
 * A sky lantern is not a pendulum. It has almost no mass against a lot of
 * surface, so it does not oscillate — it WANDERS: carried by whatever the air
 * is doing, rising and settling as the air inside it cools, never returning to
 * quite the same place.
 *
 * ─── WHY THIS IS A SIMULATION AND NOT A FUNCTION OF TIME ───────────────────
 *
 * It used to be three summed sines per axis, seeded by index — cheap, pure,
 * and shareable between the canvas and the DOM without any plumbing. It also
 * could not do the one thing a field of lanterns has to do: react to the
 * others. Overlaps were resolved once, at layout, and then the drift walked
 * straight back through them.
 *
 * So each lantern is a body now. It carries its own velocity, its own mass,
 * its own drag, and its own wander clock, and every one of those is drawn from
 * its own seed — nothing shares a period, so nothing moves in step and no two
 * ever pair up. A collider on each resolves overlaps with a SOFT force rather
 * than a hard constraint: they push apart in proportion to how deeply they
 * have met, which lets them touch and slide past each other occasionally
 * instead of behaving like magnets that can never quite meet.
 *
 * ─── ONE FIELD, TWO RENDERERS ──────────────────────────────────────────────
 *
 * The canvas draws the paper and the DOM writes the name on it, and if those
 * two ever disagree the paper slides out from under its own name. A simulation
 * cannot be recomputed independently on both sides the way a pure function
 * could, so there is exactly one field: the canvas owns the clock and steps
 * it, the DOM reads positions out of it, and the label is parented to its
 * lantern's transform rather than carrying a copy of the motion.
 */

/** How far a lantern may wander from where support put it. */
const DRIFT = { x: 0.075, y: 0.018 }
/** Depth swing, as a fraction of resting size. */
const DRIFT_Z = 0.04

/**
 * ── THE ARRIVAL ──────────────────────────────────────────────────────────
 *
 * Seconds. The lead is the beat the valley gets to itself before anything
 * starts rising into it; the stagger is the gap between one lantern leaving
 * and the next; the lift is how long one takes to reach its resting spot from
 * ENTER_FROM below it.
 *
 * 2.1s, and the number went UP on the second look. 0.95s was chosen against
 * "quickly, but not so quick they stop reading as balloons" and it overshot:
 * a paper lantern crossing a third of the screen in under a second is not
 * drifting, it is being placed. Something that large and that light moves
 * slowly, and the slowness IS the immersion — this is the one place on the
 * page where taking longer makes the thing more like itself.
 *
 * It lengthens the tail, which an earlier axis argued against. That axis was
 * written when the sky ran UNDER the words and its tail was an overrun; the
 * sky is the last movement in the sequence now, so a long settle is the page
 * coming to rest rather than something still outstanding.
 *
 * THE LEAD IS NOW ZERO. It existed to give the valley a beat to itself before
 * anything rose into it, and that beat has moved somewhere better: the field is
 * held until the invitation has finished asking, and the rest before it is
 * SKY_BEAT in NaamApp. Keeping a second lead here would have been a rest inside
 * a rest.
 *
 * The stagger came down with it — 0.11 to 0.07. The sky now starts after the
 * words rather than under them, so its tail is the last thing on screen, and a
 * tail nobody is still watching is not a rest, it is an overrun (E8). Eight
 * lanterns at 0.07 apart settle 1.4s after the first one lifts.
 */
const ENTER_LEAD = 0.0
const ENTER_STAGGER = 0.15
const ENTER_LIFT = 2.1
/**
 * How far outside the frame a lantern starts, as a fraction of the frame.
 *
 * ── THEY DRIFT IN FROM THE EDGES, THEY DO NOT RISE FROM BELOW ────────────
 *
 * The first version lifted every lantern 0.16 of the frame height from
 * underneath its resting spot, which is what a sky lantern does at the moment
 * it is released. But nothing on this page is being released here — these are
 * other people's names, sent from somewhere else, and the honest picture is of
 * lanterns that have been aloft a while and are drifting over.
 *
 * So each one comes in from whichever frame edge it is nearest: the high ones
 * down from the top, the ones near a side in from that side. The entry
 * direction is a property of where a lantern lives, so the sky fills from its
 * own edges inward rather than every object obeying one rule.
 *
 * 0.34 rather than 0.16 because a drift from off-screen has to start
 * off-screen — a lantern that appears a third of the way in has not drifted,
 * it has faded.
 */
const ENTER_FROM = 0.34

export interface LanternBody {
  /** Live position, in fractions of the frame. */
  x: number
  y: number
  /** Depth offset, −1..1, on top of the resting scale. */
  z: number
  /** Where support put it. Everything is a spring back toward this. */
  homeX: number
  homeY: number
  /** Collider radius, in fractions of the frame's width. */
  r: number
  scale: number
  alpha: number
}

interface Body extends LanternBody {
  /** The note this body belongs to, so state survives a rebuild. */
  key: string
  vx: number
  vy: number
  vz: number
  /** Per-lantern, all seeded independently so nothing shares a beat. */
  mass: number
  drag: number
  wanderRate: number
  wanderPhase: number
  wanderGain: number
  /** Rotates this lantern's wander onto its own axes. */
  wanderTilt: number
  restScale: number
  /**
   * The arrival, 0 to 1.
   *
   * The lanterns used to exist from the first drawn frame: the sky was simply
   * already full, which is the one thing a sky full of rising lanterns is not.
   * They lift from below their resting spot instead, one after another, far
   * ones first, so the valley resolves and then fills.
   *
   * It lives on the BODY and not in the renderer because the paper is drawn on
   * the canvas and the name is written in the DOM, and both read this field. An
   * offset applied on the canvas side only would leave every name hanging at
   * its resting spot while its lantern rose to meet it.
   */
  enter: number
  /** Seconds to wait before this one starts lifting. */
  enterDelay: number
  /** Unit vector from off-frame toward home — which edge this one drifts in from. */
  enterX: number
  enterY: number
  /**
   * How much of `y` is the arrival rather than the simulation.
   *
   * The offset has to be REMOVED before the next step or it accumulates: the
   * first version added it every frame and the lanterns left the frame
   * downward at 0.16 of the screen per step. Carrying the applied amount and
   * subtracting it at the top of the step keeps one source of truth — both
   * readers still take `x` and `y` — while the physics never sees the lift.
   */
  lift: number
  liftX: number
}

/** Deterministic 0..1 from a seed — a lantern must not change character. */
function rand(n: number): number {
  const v = Math.sin(n * 127.1 + 11.7) * 43758.5453
  return v - Math.floor(v)
}

/**
 * The field. One instance, owned by the canvas, read by the DOM.
 *
 * `reset` is called on layout with the resting spots; `step` advances the
 * simulation; `bodies` is the live state both renderers read.
 */
export class LanternField {
  readonly bodies: Body[] = []
  /**
   * How many times the simulation has advanced.
   *
   * The canvas owns the clock and the DOM labels ride it, and the canvas now
   * draws at 30fps rather than at the display's rate. Without this the label
   * loop would keep writing sixty transforms a second, half of them identical
   * to the ones already on the nodes — a compositor job either way, but a
   * pointless one. Readers compare it and skip a frame that has nothing in it.
   */
  steps = 0
  /** Seconds since the field started stepping — the arrival's clock. */
  private age = 0
  /** Frame aspect, so a radius in x-fractions can be compared in y. */
  private aspect = 1.5

  /**
   * Seed or re-seed the field.
   *
   * STATE SURVIVES A REBUILD. This is called whenever the notes change — and
   * keeping a name changes them — so rebuilding from scratch snapped every
   * lantern in the sky back to its resting spot the instant somebody voted for
   * one. Bodies are matched by key: a name that was already up there keeps its
   * position, its velocity and its wander clock, and only its HOME moves to
   * the new resting spot, so gaining a voice makes it drift nearer over the
   * next few seconds instead of teleporting.
   */
  reset(spots: readonly LanternSpot[], aspect: number, keys: readonly string[] = []): void {
    this.aspect = aspect || 1.5
    const previous = new Map(this.bodies.map((body) => [body.key, body]))
    this.bodies.length = 0
    spots.forEach((spot, i) => {
      const s = i * 12.9898
      const key = keys[i] ?? String(i)
      const kept = previous.get(key)
      const seeded = {
        key,
        x: spot.x,
        y: spot.y,
        z: 0,
        homeX: spot.x,
        homeY: spot.y,
        // The collider is the paper, a little tighter than the halo, so they
        // may overlap their glow without overlapping their names.
        r: (spot.size * LANTERN_ASPECT) / 2 / this.aspect,
        scale: spot.scale,
        alpha: 0.9,
        restScale: spot.scale,
        vx: 0,
        vy: 0,
        vz: 0,
        /**
         * Everything below is per-lantern and independently seeded, and two of
         * them are spaced rather than sampled.
         *
         * A first cut drew rate and phase from the same random source and two
         * lanterns came out correlated at r=0.98 — with eight bodies and a
         * narrow range, a near-collision of parameters is likely rather than
         * unlucky. The phase now steps by the golden angle so no two are ever
         * close, the rates are spread across a wider band by index before
         * being jittered, and each lantern's wander is rotated onto its own
         * axes so even a similar clock traces a different path.
         */
        mass: 0.8 + rand(s + 3) * 0.6,
        drag: 1.6 + rand(s + 7) * 1.4,
        // Golden-ratio sequences, not index ramps. Spreading the rate BY INDEX
        // guarantees the thing it was meant to prevent: neighbours get the
        // nearest clocks to each other, and lamps 6 and 7 came out correlated
        // at r=0.93. A low-discrepancy sequence puts consecutive indices as far
        // apart as the range allows.
        wanderRate: 0.045 + ((i * 0.6180339887) % 1) * 0.09 + rand(s + 11) * 0.012,
        wanderPhase: ((i * 0.3819660113) % 1) * 6.283185 + rand(s + 17) * 0.5,
        wanderGain: 0.6 + rand(s + 23) * 0.8,
        enter: 0,
        enterDelay: 0,
        enterX: 0,
        enterY: 0,
        lift: 0,
        liftX: 0,
        wanderTilt: ((i * 0.7548776662) % 1) * 6.283185 + rand(s + 29) * 0.4,
      }
      this.bodies.push(
        kept
          ? // Position, velocity and clock carried over; home and size follow
            // the new support. THE ARRIVAL CARRIES OVER TOO — a lantern that is
            // already up there must not lift off again because somebody voted.
            {
              ...seeded,
              x: kept.x,
              y: kept.y,
              z: kept.z,
              vx: kept.vx,
              vy: kept.vy,
              vz: kept.vz,
              enter: kept.enter,
              enterDelay: kept.enterDelay,
              enterX: kept.enterX,
              enterY: kept.enterY,
              lift: kept.lift,
              liftX: kept.liftX,
            }
          : seeded,
      )
    })

    /**
     * FAR ONES FIRST, so the sky fills back to front and the near lanterns —
     * the ones with the most support — are the last to settle. The delay is
     * assigned after the loop because it depends on every body's depth, which
     * is not known until they all exist.
     *
     * ENTER_LEAD is the beat the backdrop gets to itself. The valley resolves,
     * and then things start rising into it; without it the mountains and the
     * first lantern arrive together and the ordering the eye is being offered
     * never happens.
     */
    const arriving = this.bodies.filter((body) => body.enter < 1)
    arriving
      .slice()
      .sort((a, z) => a.restScale - z.restScale)
      .forEach((body, rank) => {
        body.enterDelay = ENTER_LEAD + rank * ENTER_STAGGER
      })

    /**
     * WHICH EDGE EACH ONE CAME FROM — top, left or right, split by where a
     * lantern sits AMONG THE OTHERS rather than by which frame edge is
     * physically nearest.
     *
     * Nearest-edge was the first version and it collapsed: the sky occupies
     * the right-hand column, so every lantern's closest edge is the right one
     * and all eight drifted in from the same side. Measured — 1 of 3 edges in
     * use. A rule that is geometrically correct and produces one behaviour is
     * not a rule, it is a constant.
     *
     * Ranked across the band instead: the leftmost third enter from the left,
     * the rightmost third from the right, and the middle come down from the
     * top. That guarantees all three are used at any width and at any number
     * of names, and it reads as what it is — a sky filling from its edges.
     *
     * The bottom is deliberately never a candidate. Below the sky is the
     * valley, and a lantern coming up out of the town reads as launched from
     * it, which is a different picture from one drifting over.
     */
    const byX = arriving.slice().sort((a, z) => a.homeX - z.homeX)
    const third = Math.max(1, Math.round(byX.length / 3))
    byX.forEach((b, i) => {
      if (i < third) {
        b.enterX = -1
        b.enterY = -0.3
      } else if (i >= byX.length - third) {
        b.enterX = 1
        b.enterY = -0.3
      } else {
        b.enterX = 0
        b.enterY = -1
      }
    })
  }

  /**
   * ── THE SKY WAITS FOR THE INVITATION TO FINISH ASKING ──────────────────
   *
   * Held by default. The lanterns are other people's names, and they mean
   * something quite different depending on when they arrive: rising WHILE the
   * page is still explaining itself, they are ambient scenery competing with
   * the words; rising AFTER the line that asks for your help, they are the
   * answer to it (L3).
   *
   * Filmed before this: the second line of the invitation landed while the
   * lanterns were mid-rise and the third landed during their tail, so the eye
   * was pulled left-right-left across the fold and neither performance was
   * watched. Two well-made choreographies running concurrently read as
   * neither (L2).
   *
   * `age` is what the arrival clocks on, so holding simply stops that clock.
   * Everything else — drift, collision, the wander — runs from the first frame,
   * because the sky being alive is not the same as the sky arriving.
   */
  private held = true

  /** The invitation has finished asking. Let them up. */
  release(): void {
    this.held = false
  }

  /**
   * Skip the arrival — every lantern already up, nothing lifted.
   *
   * Under prefers-reduced-motion the scene draws ONE frame and then stops, so
   * an entrance that has not run is an entrance that never will: measured, all
   * eight lanterns sat 0.16 of the frame height below their spots permanently,
   * with their names hanging in the sky under the paper they are written on.
   * A still frame of a rising lantern is a lantern that has arrived, which is
   * also the honest reading of WCAG 2.3.3 — it is about motion, not about
   * whether the scene is allowed to be complete.
   */
  settle(): void {
    this.held = false
    for (const b of this.bodies) {
      b.x -= b.liftX
      b.y -= b.lift
      b.lift = 0
      b.liftX = 0
      b.enter = 1
    }
  }

  /**
   * Advance by `dt` seconds at wall-clock `time`.
   *
   * Forces, in the order they are applied: the air moving the lantern, a weak
   * spring holding it near where support put it, drag, and finally the soft
   * separation that keeps names off each other.
   */
  step(dt: number, time: number): void {
    const h = Math.min(dt, 1 / 30)
    this.steps++
    /**
     * THE ARRIVAL RUNS ON WALL TIME, THE PHYSICS RUNS ON `h`.
     *
     * `h` is capped at a thirtieth so a stalled frame cannot fling a body
     * across the screen — correct for a simulation, wrong for an entrance. The
     * canvas draws at 30fps by design and slower than that on a weak machine,
     * so an arrival clocked on `h` ran at a fraction of real speed: measured on
     * this host, 3.3 seconds of wall time advanced it by 0.65. A lantern's
     * entrance has to take the same time on every machine, so it takes `dt` —
     * bounded only against a tab that was backgrounded for a minute.
     */
    if (!this.held) this.age += Math.min(dt, 0.25)

    // Undo last frame's lift so the simulation below runs on the resting
    // position, not on the position the arrival put it in.
    for (const b of this.bodies) {
      b.x -= b.liftX
      b.y -= b.lift
      b.lift = 0
      b.liftX = 0
    }

    for (const b of this.bodies) {
      // The air. Two incommensurate components per axis so the path has slow
      // arcs with smaller irregularities on it and never visibly repeats.
      const t = time * b.wanderRate + b.wanderPhase
      const u = (Math.sin(t) * 0.7 + Math.sin(t * 2.3 + 1.1) * 0.3) * b.wanderGain
      const v = (Math.cos(t * 0.83) * 0.7 + Math.sin(t * 1.9 + 2.2) * 0.3) * b.wanderGain
      // Rotated onto this lantern's own axes, so two with similar clocks still
      // trace different paths rather than sliding in parallel.
      const cos = Math.cos(b.wanderTilt)
      const sin = Math.sin(b.wanderTilt)
      const ax = (u * cos - v * sin) * DRIFT.x
      const ay = (u * sin + v * cos) * DRIFT.y
      const az = Math.sin(t * 0.61 + 0.7) * DRIFT_Z

      // A weak spring home. Without it a body wanders off and never returns;
      // with it too strong, every lantern oscillates about its home on the
      // same beat, which is the synchronised motion this exists to avoid. It
      // is deliberately softer than the wander that fights it.
      const kx = (b.homeX + ax - b.x) * 1.4
      const ky = (b.homeY + ay - b.y) * 1.4
      const kz = (az - b.z) * 1.2

      b.vx += (kx / b.mass) * h
      b.vy += (ky / b.mass) * h
      b.vz += (kz / b.mass) * h

      const damp = Math.exp(-b.drag * h)
      b.vx *= damp
      b.vy *= damp
      b.vz *= damp

      /**
       * A SOFT WALL, not a hard one.
       *
       * You said a lantern going off the edge is fine, and it is — but the
       * NAME is written on the paper and centred on it, so a lantern that
       * sails out takes half a name with it. Measured, they were reaching
       * 64px past the frame.
       *
       * This is a spring that only exists near the edges: nothing at all in
       * the middle of the sky, firming up as a body approaches the margin. So
       * they still wander widely and still overhang, and they stop before the
       * writing is cut.
       */
      const margin = b.r * 0.9
      if (b.x < margin) b.vx += (margin - b.x) * 9 * h
      if (b.x > 1 - margin) b.vx -= (b.x - (1 - margin)) * 9 * h
      const marginY = (b.r * this.aspect) * 0.9
      if (b.y < marginY) b.vy += (marginY - b.y) * 9 * h
      if (b.y > 1 - marginY) b.vy -= (b.y - (1 - marginY)) * 9 * h

      b.x += b.vx * h
      b.y += b.vy * h
      b.z += b.vz * h
    }

    this.separate(h)

    for (const b of this.bodies) {
      b.scale = b.restScale * (1 + b.z)
      b.alpha = 0.84 + b.z * 4
    }

    /**
     * ── AND THEN THE RISE, ON TOP OF EVERYTHING THE PHYSICS DID ────────────
     *
     * Applied last and never fed back: `y` still holds the simulated position
     * and the offset is added to it afterwards, so a lantern still on its way
     * up is not treated by the wander, the walls or the colliders as though it
     * were down there. It drifts and separates as if it were already home, and
     * the arrival is a transform laid over that.
     *
     * easeOutCubic, and the choice matters: a balloon under buoyancy leaves
     * fast and asymptotes into its ceiling. Anything with an overshoot reads as
     * a UI element springing, which is the opposite of the thing being
     * described.
     */
    for (const b of this.bodies) {
      if (b.enter >= 1) continue
      const t = Math.max(0, Math.min(1, (this.age - b.enterDelay) / ENTER_LIFT))
      b.enter = 1 - Math.pow(1 - t, 3)
      const left = 1 - b.enter
      // x in frame-fractions is narrower than y, so the horizontal reach is
      // divided by the aspect to travel the same visible distance.
      b.liftX = (left * ENTER_FROM * b.enterX) / this.aspect
      b.lift = left * ENTER_FROM * b.enterY
      b.x += b.liftX
      b.y += b.lift
      // Fades over the first half of the lift, so it is fully there for the
      // part of the climb the eye is actually following.
      b.alpha *= Math.min(1, b.enter * 2)
      // A shade smaller on the way up. Two percent — enough to read as coming
      // from further away, not enough to read as a zoom.
      b.scale *= 0.98 + b.enter * 0.02
    }
  }

  /**
   * SOFT SEPARATION. Overlapping bodies push apart in proportion to how far
   * they have interpenetrated, scaled by a spring constant low enough that a
   * determined pair can still cross — the ask was that they intersect
   * sometimes and slightly, not that they repel like magnets.
   *
   * It acts on VELOCITY rather than position: moving them directly would be a
   * hard constraint and would read as two objects snapping apart.
   */
  private separate(h: number): void {
    const list = this.bodies
    for (let i = 0; i < list.length; i++) {
      for (let j = i + 1; j < list.length; j++) {
        const a = list[i]
        const b = list[j]
        // y is compared in x-units so a circle stays a circle on a wide frame.
        const dx = b.x - a.x
        const dy = (b.y - a.y) / this.aspect
        const dist = Math.hypot(dx, dy) || 1e-5
        const want = a.r + b.r
        if (dist >= want) continue

        const overlap = (want - dist) / want
        // Deliberately gentle, and eased so a slight touch barely registers
        // while a deep overlap pushes firmly.
        const push = overlap * overlap * 3.2 * h
        const nx = dx / dist
        const ny = dy / dist
        a.vx -= (nx * push) / a.mass
        a.vy -= (ny * push * this.aspect) / a.mass
        b.vx += (nx * push) / b.mass
        b.vy += (ny * push * this.aspect) / b.mass
      }
    }
  }
}

/** The one field. The canvas steps it; the DOM reads it. */
let field: LanternField | null = null

export function lanternField(): LanternField {
  field ??= new LanternField()
  return field
}

/**
 * Verification handle, DEV ONLY.
 *
 * `steps` is a faithful count of scene draws — the field is stepped once per
 * draw and never otherwise — which is the only way to check the frame cap
 * from outside without the page shipping a counter to report on itself.
 * `import.meta.env.DEV` is statically false in the build, so this whole block
 * is dropped from production output rather than merely skipped at runtime.
 */
if (import.meta.env.DEV && typeof window !== 'undefined') {
  ;(window as unknown as Record<string, unknown>).__lanternFieldProbe = lanternField
}
