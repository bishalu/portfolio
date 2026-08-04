/**
 * The shelf (docs/design/DESIGN.md §1 P1 exception, P2, §2, P10).
 *
 * WHERE OTHER PEOPLE'S CHOICES ACCUMULATE — the six the family keeps coming
 * back to, every suggestion Bishal has approved, and the visitor's own picks as
 * they make them. It lives in the right-hand column beside the three slots,
 * because in the conversation it was a thing that had already happened: two
 * turns later it had scrolled away and was never seen again. Here it is the
 * surface being added to.
 *
 * WHY LEAVES AND NOT POST-IT NOTES. The notes were well made — two shadows, an
 * uneven cut edge, a hashed tilt — and they were furniture from the wrong
 * world. A post-it is an office object; these names come out of a Sanskrit and
 * Pali source that was copied for a thousand years in POTHI form: long narrow
 * leaves, a double red rule down the binding margin, a hole where the cord
 * passes through. That is the shape this list has actually had, so it is the
 * shape it has here. It is also not the tourist shorthand — no prayer flags,
 * no mandala, no om — which is the rule the whole page is built on.
 *
 * ONE LEAF PER NAME, AND THE REPETITION IS THE POINT. This used to render one
 * note per (sender × pick), so a name two relatives both loved sat on the wall
 * twice and read as two different names. Grouping by id turns that duplication
 * into the single most useful thing the shelf knows: which names are gathering
 * support. Keeping a name the family already keeps adds a bead to their leaf
 * rather than writing the name again — which is exactly what agreeing means.
 *
 * THE TALLY IS BEADS, and that is a decision among three:
 *
 *   SIZE — the name set larger the more people chose it — is the word-cloud
 *   answer, and it makes the names with the LEAST support the hardest to read.
 *   That is backwards on a page asking people to consider them.
 *
 *   A DIGIT reads as a score. This is a family choosing their son's name, not a
 *   leaderboard, and "4" beside one name invites the reader to see the other as
 *   losing.
 *
 *   BEADS are already this page's counting object — the thread down the
 *   conversation, the three slots — so one bead per person needs no legend and
 *   counts at a glance. Past five they stop being countable, so the row gives
 *   way to a numeral rather than growing forever.
 *
 * Support also deepens the INK, because a name written over and over is a
 * darker line on a page and this material can carry that honestly. It is never
 * the only signal: the beads count it and an .sr-only line says it in words, so
 * nothing depends on seeing ink density or counting dots.
 *
 * ACCESSIBILITY (P10). A real <ul>/<li>, not div soup; role="list" is
 * restorative because `list-style: none` strips list semantics in Safari.
 * Nothing inside is focusable and nothing should be — a leaf is a record, not a
 * control, and eleven fake buttons that do nothing is a worse keyboard
 * experience than a list. Ink #33291c on #e3d3a8 is 9.4:1, and the lightest
 * leaf — a name nobody has chosen yet — still clears AA at 4.9:1, which is the
 * one that matters most, since it is the one asking to be noticed.
 *
 * Every visible string reaches the DOM through JSX interpolation. `from` and
 * `relation` are a stranger's typed text and are never assembled into markup.
 */
import { useEffect, useRef, useState, type CSSProperties } from 'react'
import { NAAM_COPY } from '@/lib/naam/copy'
import { lampSpots } from '@/lib/naam/scene/lamps'

const C = NAAM_COPY

export interface WallNote {
  /** Stable across renders. One entry per NAME, not per supporter. */
  key: string
  /** Empty when a suggestion names a row this build cannot resolve. */
  deva: string
  latin: string
  /**
   * HOW MANY PEOPLE HAVE CHOSEN IT. The wall used to render one leaf per
   * (sender × pick), so a name two relatives both loved appeared twice and
   * looked like two different names. Grouped, the repetition becomes the most
   * useful thing on the shelf: which names are gathering support.
   */
  count: number
  /** Who sent it, when it was one person. Dropped once several have. */
  who?: string
  /** Kept by the visitor in this session — their own contribution to the shelf. */
  mine?: boolean
}

export interface NaamWallProps {
  notes: readonly WallNote[]
}

export default function NaamWall({ notes }: NaamWallProps) {
  const shelfRef = useRef<HTMLUListElement>(null)

  /**
   * The SAME geometry the canvas uses to draw the glow. Read once per render
   * from the shared module rather than duplicated here — a label two percent
   * away from its lamp is a hover target over empty sky.
   */
  const [stacked, setStacked] = useState(false)
  useEffect(() => {
    if (typeof matchMedia !== 'function') return
    const q = matchMedia('(max-width: 599px)')
    const read = () => setStacked(q.matches)
    read()
    q.addEventListener('change', read)
    return () => q.removeEventListener('change', read)
  }, [])
  const spots = lampSpots(notes.length, stacked)

  /**
   * PARALLAX, which is the cheapest real depth cue there is.
   *
   * A stack of leaves that never moves is a table with a texture on it. What
   * makes a rendered scene read as a SPACE rather than a picture of one is that
   * things at different depths displace by different amounts when the viewpoint
   * shifts — so the shelf publishes the pointer's offset from its own centre as
   * two numbers, and each leaf multiplies them by its own depth. A leaf several
   * people have chosen floats higher, so it moves more.
   *
   * The listener is on the WINDOW, not on the shelf. Parallax that only responds
   * while the cursor is over the thing is a hover effect; a scene responds to
   * where you are even when you are looking somewhere else, and that difference
   * is most of the effect.
   *
   * One rAF at a time, passive, and it writes two custom properties — no layout
   * is read per move except a cached rect, and nothing here can schedule a
   * second frame's work from inside the first.
   */
  useEffect(() => {
    const el = shelfRef.current
    if (!el || typeof matchMedia !== 'function') return

    /**
     * TWO GATES, AND BOTH HAVE TO BE IN JAVASCRIPT.
     *
     * A `@media (prefers-reduced-motion)` block cannot stop a pointer listener
     * or a transform written from script — CSS can only decline to animate what
     * CSS owns. So the preference is read here, and it is WATCHED: somebody who
     * turns the setting on mid-session is asking for the motion to stop now, not
     * on their next visit.
     *
     * The second gate is the pointer itself. On a touch screen there is no
     * hovering cursor to parallax against, so the listener would be a wasted
     * subscription that can never fire meaningfully — and `deviceorientation` is
     * NOT the fallback: it needs an explicit permission prompt on iOS and
     * gyro-driven parallax is a considerably worse nausea trigger than the
     * pointer kind. Touch gets the static dealt arrangement, which is the honest
     * still frame of this effect.
     */
    const calm = matchMedia('(prefers-reduced-motion: reduce)')
    const fine = matchMedia('(hover: hover) and (pointer: fine)')

    let frame = 0
    let rect = el.getBoundingClientRect()
    let bound = false

    const remeasure = () => {
      rect = el.getBoundingClientRect()
    }
    const rest = () => {
      el.style.setProperty('--px', '0')
      el.style.setProperty('--py', '0')
    }
    const onMove = (event: PointerEvent) => {
      if (frame) return
      frame = requestAnimationFrame(() => {
        frame = 0
        if (rect.width === 0) return
        // −1…1 either side of the shelf's centre, clamped so a pointer at the
        // far edge of a wide screen does not send the leaves off their stack.
        const x = Math.max(-1, Math.min(1, (event.clientX - (rect.left + rect.width / 2)) / (rect.width * 0.9)))
        const y = Math.max(-1, Math.min(1, (event.clientY - (rect.top + rect.height / 2)) / (rect.height * 2.2)))
        // NEVER ROUNDED. The whole travel is a dozen pixels, so integers would
        // step visibly; sub-pixel transforms composite on the GPU anyway.
        el.style.setProperty('--px', x.toFixed(3))
        el.style.setProperty('--py', y.toFixed(3))
      })
    }

    const bind = () => {
      if (bound) return
      bound = true
      // The listener is on the WINDOW: a scene answers where you are, not
      // whether you are touching it. Passive, and it only ever reads two
      // coordinates — the rect is cached, so no move can force a reflow.
      window.addEventListener('pointermove', onMove, { passive: true })
      window.addEventListener('scroll', remeasure, { passive: true })
      window.addEventListener('resize', remeasure)
      document.addEventListener('pointerleave', rest)
    }
    const unbind = () => {
      if (!bound) return
      bound = false
      window.removeEventListener('pointermove', onMove)
      window.removeEventListener('scroll', remeasure)
      window.removeEventListener('resize', remeasure)
      document.removeEventListener('pointerleave', rest)
      if (frame) cancelAnimationFrame(frame)
      frame = 0
      rest()
    }

    const decide = () => {
      if (calm.matches || !fine.matches) unbind()
      else bind()
    }

    decide()
    calm.addEventListener('change', decide)
    fine.addEventListener('change', decide)
    return () => {
      calm.removeEventListener('change', decide)
      fine.removeEventListener('change', decide)
      unbind()
    }
  }, [])

  return (
    <>
      <p className="label-mono label-mono--sm nm-quiet-label" id="nm-shelf-label">
        {C.app.familyLead}
      </p>
      {/*
        THE SHELF BECAME LAMPS IN THE VALLEY.

        This was a scrolling stack of pothi leaves, and it was the single
        heaviest thing on the right of the page — six names at full contrast,
        each with a tally, sitting on top of the scene rather than in it. Two
        objects competing for the same column.

        What the list actually says is "these names are burning somewhere in the
        family". So it says it in the world: one lamp per name, out across the
        town at dusk, and the same warm colour as the diyo because they are the
        same gesture. The count is now light — a name several people chose burns
        brighter, which needs no legend and no digit beside a child's name.

        THE CANVAS DRAWS THE GLOW; THIS DRAWS THE CONTROL. Canvas cannot be
        focused, labelled, or read by a screen reader, and the gate on this page
        is zero axe violations. So each lamp gets a real <button> positioned
        over it from the SAME geometry (lampSpots), carrying the name for
        hover, focus and assistive tech. Nothing here depends on seeing light.
      */}
      {/* eslint-disable-next-line jsx-a11y/no-redundant-roles */}
      <ul className="nm-lamps" role="list" aria-labelledby="nm-shelf-label" ref={shelfRef}>
        {notes.map((note, i) => {
          const spot = spots[i]
          if (!spot) return null
          return (
            <li
              className="nm-lamp"
              key={note.key}
              data-mine={note.mine ? 'true' : undefined}
              style={
                {
                  left: `${spot.x * 100}%`,
                  top: `${spot.y * 100}%`,
                  // Support drives BRIGHTNESS, capped at five. Size would make
                  // the least-supported names hardest to see, which is backwards
                  // on a page asking people to consider them.
                  '--support': Math.min(note.count, 5),
                  '--i': i,
                } as CSSProperties
              }
            >
              <button type="button" className="nm-lamp-hit">
                <span className="nm-lamp-name">
                  {note.deva && (
                    <span className="nm-lamp-deva" lang="sa-Deva">
                      {note.deva}
                    </span>
                  )}
                  <span className="nm-lamp-latin">{note.latin}</span>
                </span>
                <span className="sr-only">
                  {C.wall.support(note.count)}
                  {note.who && note.count === 1 ? ` — ${note.who}` : ''}
                </span>
              </button>
            </li>
          )
        })}
      </ul>
    </>
  )
}
