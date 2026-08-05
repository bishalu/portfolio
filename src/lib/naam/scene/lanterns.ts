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
  /** 0..1, for the glow and the label's ink. */
  glow: number
}

/**
 * The sky the lanterns may occupy, in fractions of the canvas height.
 *
 * The bottom is 0.34 because the tray — the diyo, its label and the three slots
 * — begins at 0.398 of the canvas at both 1280 and 1440, measured. The first
 * cut ran the field to 0.62 and the nearest lantern's name landed on top of
 * "KEEP UP TO THREE" with the flame behind it. Below the tray is not sky, it is
 * furniture; the lanterns get the part of the frame that is actually empty.
 */
const SKY_TOP = 0.1
const SKY_BOTTOM = 0.34
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
 */
const MIN_GAP = 0.085

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
export function lanternSpots(counts: readonly number[], stacked: boolean): LanternSpot[] {
  if (counts.length === 0 || stacked) return []

  const distinct = [...new Set(counts)].sort((a, b) => b - a)
  const bandWidth = RANGE / distinct.length
  const rankOf = new Map(distinct.map((c, i) => [c, i]))

  const top = SKY_TOP
  const bottom = SKY_BOTTOM
  // The left of the frame stays clear — that side is the room the visitor is
  // standing in, and it is where the type lives.
  const from = 0.52
  const to = 0.97

  const placed = counts.map((count, i) => {
    const rank = rankOf.get(count) ?? 0
    // Band 0 is the most-supported and sits nearest. Within its band a lantern
    // is placed by a seeded fraction, so two names on the same count still sit
    // at slightly different distances instead of on one line.
    const withinBand = hash(i * 7 + 3)
    const depth = Math.min(1, Math.max(0, 1 - (rank + withinBand) * bandWidth))

    // Near lanterns hang low and far ones ride up toward the ridge.
    const y = top + (bottom - top) * depth
    /**
     * Lane first, jitter second. The jitter only has to break up the evenness —
     * the separation pass below is what keeps names off each other.
     */
    const lane = counts.length === 1 ? 0.5 : i / (counts.length - 1)
    const x = from + (to - from) * (lane * 0.9 + hash(i * 13 + 5) * 0.1)

    return {
      x,
      y,
      depth,
      // Near is bigger, but never so small that the name under it stops being
      // readable — legibility outranks the perspective.
      scale: 0.62 + depth * 0.58,
      glow: 0.45 + depth * 0.55,
    }
  })

  /**
   * SEPARATION, ENFORCED. Walk left to right and push any lantern that has
   * crowded its neighbour out to the minimum gap. Deterministic, so a lantern
   * still keeps its place between renders.
   *
   * If there are too many lanterns for the field to hold them all at MIN_GAP,
   * spacing evenly is the honest fallback — every gap equal and slightly too
   * small beats a few correct gaps and one collision.
   */
  const span = to - from
  const order = placed.map((_, i) => i).sort((a, b) => placed[a].x - placed[b].x)
  if (placed.length > 1 && (placed.length - 1) * MIN_GAP > span) {
    order.forEach((idx, seat) => {
      placed[idx].x = from + (span * seat) / (placed.length - 1)
    })
  } else {
    for (let seat = 1; seat < order.length; seat++) {
      const prev = placed[order[seat - 1]]
      const here = placed[order[seat]]
      if (here.x - prev.x < MIN_GAP) here.x = prev.x + MIN_GAP
    }
    // Pushing rightward can walk the last one off the edge; slide the whole
    // row back rather than piling them against the frame.
    const overflow = placed[order[order.length - 1]].x - to
    if (overflow > 0) for (const spot of placed) spot.x -= overflow
  }

  return placed
}
