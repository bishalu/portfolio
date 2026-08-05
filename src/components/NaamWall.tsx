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
import { type CSSProperties, useEffect, useLayoutEffect, useRef, useState } from 'react'
import { NAAM_COPY } from '@/lib/naam/copy'
import { LANTERN_ASPECT, lanternDrift, lanternSpots } from '@/lib/naam/scene/lanterns'

/** Kept in CSS custom properties so the stylesheet can size the paper. */
const ASPECT = LANTERN_ASPECT

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
  /**
   * ─── THE TWO BOXES ARE NOT THE SAME BOX ────────────────────────────────
   *
   * lanternSpots returns fractions of the CANVAS, which is the full width of
   * the shell. This list is not: it sits in the right-hand panel, which starts
   * where the room ends. Measured at 1280, the canvas is 0..1280 and this list
   * is 563..1280 — so a spot at x=0.6 meant 768px to the canvas and 993px here,
   * and every name floated ~225px right of the lantern it belongs to.
   *
   * This was true of the lamps before the lanterns, and invisible then only
   * because their labels were hidden until hover: you could not see that the
   * tooltip was over the wrong roof.
   *
   * Measured rather than assumed. The offset is a consequence of the valley's
   * `roomWidth`, and hard-coding 0.44 here would make this silently wrong the
   * day that option changes.
   */
  const [frame, setFrame] = useState({ dx: 0, dy: 0, cw: 1, ch: 0, pw: 1 })
  useLayoutEffect(() => {
    const read = () => {
      const panel = shelfRef.current
      // `.nm-valley` IS the canvas — not a wrapper around one. Querying
      // '.nm-valley canvas' matched nothing, read() silently took the fallback
      // branch every time, and the conversion below produced exactly the
      // unconverted percentages it exists to replace. The fix looked correct
      // and changed nothing, twice.
      const canvas = document.querySelector<HTMLCanvasElement>('canvas.nm-valley')
      if (!panel) return
      const pb = panel.getBoundingClientRect()
      // No canvas is the reduced/blocked case: fall back to treating this
      // panel as the frame, which keeps the names spread over their own box
      // rather than collapsing them into a corner.
      const cb = canvas ? canvas.getBoundingClientRect() : pb
      setFrame({
        dx: cb.left - pb.left,
        dy: cb.top - pb.top,
        cw: cb.width || 1,
        ch: cb.height || 0,
        pw: pb.width || 1,
      })
    }
    read()

    /**
     * AND READ AGAIN WHEN THE CANVAS ARRIVES.
     *
     * The valley is a dynamic import that mounts ~260ms after this list, so the
     * first read finds no canvas and falls back to this panel — which produces
     * exactly the un-converted percentages the fallback exists to replace. The
     * bug therefore survived the fix and looked identical to it: names still
     * ~225px right of their lanterns, with a correct-looking mapping that had
     * never been given the canvas.
     *
     * The panel's own size does not change when the canvas appears, so the
     * ResizeObserver below cannot catch it. This watches for the element.
     */
    /**
     * The canvas element exists from the first render; what arrives ~260ms
     * later is its SIZE, once the dynamic import runs and the renderer sets it.
     * So this watches the element's box rather than the DOM for a new node.
     */
    const canvas = document.querySelector<HTMLCanvasElement>('canvas.nm-valley')
    const canvasWatcher =
      typeof ResizeObserver === 'function' && canvas ? new ResizeObserver(read) : null
    if (canvas) canvasWatcher?.observe(canvas)

    const observer = typeof ResizeObserver === 'function' ? new ResizeObserver(read) : null
    if (shelfRef.current) observer?.observe(shelfRef.current)
    window.addEventListener('resize', read)
    return () => {
      canvasWatcher?.disconnect()
      observer?.disconnect()
      window.removeEventListener('resize', read)
    }
  }, [])

  /**
   * THE NAME RIDES ITS OWN BALLOON.
   *
   * The canvas drifts each lantern; the label used to sit perfectly still at
   * the resting position, so the paper slid out from under the name written on
   * it — the one thing a lantern carrying a name must never do.
   *
   * Same pure function, same clock, applied as a transform. Deliberately NOT
   * React state: this runs every frame, and re-rendering six list items sixty
   * times a second to move them turns a compositor job into a layout job. The
   * style is written straight onto the node.
   *
   * `transform` is free here — .nm-lamp does its centring with the `translate`
   * property, which is a separate one and is left alone.
   */
  useEffect(() => {
    const list = shelfRef.current
    if (!list || stacked || frame.ch === 0) return undefined
    // Reduced motion gets the resting composition, drawn once and left still.
    if (typeof matchMedia === 'function' && matchMedia('(prefers-reduced-motion: reduce)').matches) {
      return undefined
    }

    let raf = 0
    const tick = () => {
      const time = performance.now() / 1000
      list.querySelectorAll<HTMLElement>('.nm-lamp').forEach((el, i) => {
        const drift = lanternDrift(i, time, frame.cw, frame.ch)
        el.style.transform = `translate(${drift.dx.toFixed(2)}px, ${drift.dy.toFixed(2)}px) scale(${drift.scale.toFixed(3)})`
        // The label stacks by the same depth its paper does, so a name that has
        // drifted forward is not overlapped by one that has drifted back.
        el.style.zIndex = String(Math.round(drift.scale * 1000))
      })
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [stacked, frame.cw, frame.ch, notes.length])

  /**
   * Canvas fractions to px offsets inside this panel.
   *
   * Both axes, and in pixels rather than percentages. The first version
   * converted only x and left y and the lantern's size as percentages of the
   * PANEL, on the assumption that the panel and the canvas share a height. They
   * do at 1280 — which is where it was checked — and they do not at 1024 and
   * below, so the pixel floor on lantern size silently did nothing at exactly
   * the widths that needed it and the responsive gate kept reporting 19x15
   * tap targets.
   */
  const toPanelX = (x: number) => frame.dx + x * frame.cw
  const toPanelY = (y: number) => frame.dy + y * frame.ch

  /**
   * The SAME geometry the canvas draws the lanterns from, so a name always sits
   * on its own light. Depth carries support — see lanterns.ts.
   */
  const spots = lanternSpots(
    notes.map((note) => note.count),
    stacked,
    // The canvas's own height, measured. See `frame`.
    frame.ch,
  )


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
        family". So it says it in the world — and then, audited, it said it too
        quietly to count: every label computed opacity 0 and appeared only on
        hover, and all of them sat in one flat band down among the rooftops. Six
        names nobody could read. Whether a name is on the page is not a question
        of whether a 44px square exists at the right coordinates.

        They are LANTERNS now, up in the sky the birds used to have, with their
        names legible without hovering anything. Support is DISTANCE: a name
        several people chose floats nearer — larger, lower, brighter — and one
        person's choice hangs further back toward the ridge. No legend, no digit
        beside a child's name.

        THE CANVAS DRAWS THE LIGHT; THIS DRAWS THE CONTROL. Canvas cannot be
        focused, labelled, or read by a screen reader, and the gate on this page
        is zero axe violations. So each lantern gets a real <button> positioned
        from the SAME geometry (lanternSpots), carrying the name for hover,
        focus and assistive tech. Nothing here depends on seeing light.
      */}
      {/* eslint-disable-next-line jsx-a11y/no-redundant-roles */}
      <ul
        className="nm-lamps"
        role="list"
        aria-labelledby="nm-shelf-label"
        ref={shelfRef}
        style={{ '--lantern-aspect': ASPECT } as CSSProperties}
      >
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
                  left: `${toPanelX(spot.x).toFixed(1)}px`,
                  top: `${toPanelY(spot.y).toFixed(1)}px`,
                  // Support now drives DEPTH, and depth drives size — but the
                  // label's own scale is floored well above nothing, because a
                  // name too small to read is not a quieter answer, it is a
                  // missing one. See lanterns.ts.
                  '--support': Math.min(note.count, 5),
                  '--lantern-scale': spot.scale.toFixed(3),
                  '--lantern-glow': spot.glow.toFixed(3),
                  /* The lantern's own box, from the SAME number the canvas
                     draws it with — so the name is laid inside the paper rather
                     than parked above a glyph.

                     Height only. The panel and the canvas share a height
                     exactly, so a percentage resolves correctly; width then
                     comes from aspect-ratio in CSS rather than from px maths
                     that would need the canvas width and get it wrong. */
                  '--lantern-h': `${(spot.size * frame.ch).toFixed(1)}px`,
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
