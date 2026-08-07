/**
 * Where the lamps sit in the valley.
 *
 * TWO THINGS HAVE TO AGREE ON THIS and they are rendered by completely
 * different machinery: the Pixi canvas draws the glow, and the DOM draws an
 * invisible button on top of it carrying the name, the aria-label and the focus
 * ring. If they disagree by even a few percent the visitor hovers a lamp and
 * nothing happens, or a tooltip appears over empty sky. So neither owns the
 * geometry — this does, in fractions of the canvas, and both read from it.
 *
 * The DOM is deliberately the half that takes input. A canvas can be made
 * clickable but it cannot be made focusable, labelled, or readable by a screen
 * reader without rebuilding all of that by hand, and the a11y gate on this page
 * is zero violations across ten routes. Canvas draws; DOM is the control.
 */

/** A lamp's position, in 0..1 of the canvas box. */
export interface LampSpot {
  x: number
  y: number
}

/**
 * Lamps are strung along the near edge of the town, not scattered.
 *
 * A random field of points reads as fireflies. What is actually down there at
 * dusk is lamps in windows and on ledges — so they follow the line of the
 * rooftops, at slightly different heights, the way a row of houses on a slope
 * does. The wander is seeded off the index rather than random, so a name keeps
 * its lamp between renders: a light that jumps to a different roof when
 * somebody else's name arrives is not a place.
 */
export function lampSpots(count: number, stacked: boolean): LampSpot[] {
  if (count <= 0) return []
  const spots: LampSpot[] = []

  // The band the town occupies, matching the scene's own horizon.
  const baseY = stacked ? 0.845 : 0.905
  // Left edge stays clear of the room on desktop, and of nothing on a phone.
  const from = stacked ? 0.08 : 0.5
  const to = stacked ? 0.94 : 0.99

  for (let i = 0; i < count; i++) {
    // Spread across the band, with the ends inset so no lamp lands on the frame.
    const t = count === 1 ? 0.5 : i / (count - 1)
    const x = from + (to - from) * t
    // A deterministic stagger — hashed off the index, so lamp 3 is always on
    // the same ledge no matter how many neighbours it has.
    const wobble = Math.sin(i * 12.9898) * 43758.5453
    const rise = (wobble - Math.floor(wobble)) * 0.05
    spots.push({ x, y: baseY - rise })
  }
  return spots
}
