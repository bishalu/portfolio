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
  const MIN_X = MIN_GAP
  const MIN_Y = 0.055
  for (let pass = 0; pass < 3; pass++) {
    for (let i = 0; i < placed.length; i++) {
      for (let j = i + 1; j < placed.length; j++) {
        const a = placed[i]
        const b = placed[j]
        const dx = b.x - a.x
        const dy = b.y - a.y
        if (Math.abs(dx) >= MIN_X || Math.abs(dy) >= MIN_Y) continue
        // Push along y first — it is the axis with room — and only fall back
        // to x when they are almost exactly stacked.
        const needY = (MIN_Y - Math.abs(dy)) / 2 + 0.001
        const dirY = dy === 0 ? (i % 2 === 0 ? 1 : -1) : Math.sign(dy)
        a.y -= dirY * needY
        b.y += dirY * needY
      }
    }
  }

  /** Back inside the sky, and inside each lantern's own depth band. */
  const bandTop = (depth: number) => top + (bottom - top) * Math.min(1, depth + bandWidth * 0.5)
  const bandBottom = (depth: number) => top + (bottom - top) * Math.max(0, depth - bandWidth * 0.5)
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
 * quite the same place. The first version bobbed on a single sine and read
 * exactly like what it was, a thing on a spring.
 *
 * Three summed sines at incommensurate frequencies per axis is the cheapest
 * honest wander. The lowest has a period near 57s and carries most of the
 * amplitude; the two above it are progressively faster and smaller, so the
 * path has large slow arcs with small irregularities on top and no repeat a
 * viewer could ever notice. That is a balloon's pace: metres per minute, not
 * per second.
 *
 * IT IS A PURE FUNCTION OF (index, time) AND THAT IS THE POINT. The canvas
 * draws the paper and the DOM writes the name on it, and if those two ever
 * disagree the paper slides out from under its own name. Publishing per-frame
 * offsets from the renderer into React would mean a callback, a ref and state
 * churn sixty times a second; a pure function needs none of it and cannot
 * desync, because there is nothing to keep in sync.
 *
 * THE THIRD DIMENSION IS REAL. `scale` is depth: a lantern drifting toward the
 * viewer grows and brightens, one drifting away shrinks and dims, and because
 * the DOM label takes the same scale the name recedes with its own paper.
 */

/**
 * How far a lantern may wander, as fractions of the frame and of its size.
 *
 * ─── WHY Y AND Z ARE SO MUCH SMALLER THAN X ────────────────────────────────
 *
 * The first cut used 0.055 and 0.17 and the whole point of the field stopped
 * working: measured with two tiers of support, a one-vote lantern read 113px
 * wide against a two-vote one at 93px. The drift range (±17% scale, ±46px of
 * height) was simply larger than the gap support opens between the tiers, so
 * the ranking the sky exists to show was being scrambled by its own weather.
 *
 * Height and depth CARRY THE MEANING here, so they get a small budget; sideways
 * carries nothing, so it keeps a generous one. The result still reads as free
 * drift — 108px of lateral wander over minutes is a lot of travel — while a
 * name two people chose stays visibly nearer than a name one person chose.
 */
const DRIFT = { x: 0.075, y: 0.018, z: 0.04 }

/**
 * Three octaves of sine, roughly ±1.
 *
 * The frequencies are the pace, and the first cut had them three times too
 * fast: measured, the labels moved at 15.6 px/s, which is a twitch. A sky
 * lantern crosses a frame in minutes. At 0.038 the slowest component has a
 * period near three minutes and carries most of the amplitude, which puts the
 * peak speed around 6 px/s — slow enough that you notice a lantern has moved
 * rather than watching it move.
 */
function wander(time: number, seed: number): number {
  return (
    Math.sin(time * 0.038 + seed) * 0.6 +
    Math.sin(time * 0.066 + seed * 1.7) * 0.28 +
    Math.sin(time * 0.105 + seed * 2.9) * 0.12
  )
}

export interface LanternDrift {
  dx: number
  dy: number
  /** Depth, as a multiplier on the resting size. */
  scale: number
  alpha: number
}

/**
 * Where lantern `index` has drifted to at `time` seconds, in px against a
 * frame of `width` x `height`.
 *
 * Lanterns are allowed to leave the frame. The range below is wide enough that
 * the outermost ones sometimes will, and that is correct — a sky with a hard
 * invisible wall at its edges is a diorama, not a sky.
 */
export function lanternDrift(
  index: number,
  time: number,
  width: number,
  height: number,
): LanternDrift {
  const seed = index * 12.9898
  const z = wander(time, seed + 23)
  return {
    dx: wander(time, seed) * width * DRIFT.x,
    // Slightly biased upward: they are buoyant, and they are cooling.
    dy: (wander(time, seed + 11) - 0.16) * height * DRIFT.y,
    scale: 1 + z * DRIFT.z,
    alpha: 0.84 + z * 0.16,
  }
}
