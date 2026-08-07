/**
 * Mounting the mala: a second, small Pixi surface over the conversation rail.
 *
 * WHY A SEPARATE SURFACE FROM THE VALLEY. They have nothing in common. The
 * valley is a full-bleed diorama behind everything, painted once and cached;
 * this is a 60px-wide strip that has to stay pinned to lines of text as they
 * arrive. Sharing would mean one canvas spanning both, and then every bead
 * position would need converting out of a coordinate space that includes the
 * entire page — the exact class of drift that made lampSpots a shared module.
 * Two surfaces, each in the box it belongs to, cannot desync.
 *
 * WHY 2D AND NOT A SECOND WEBGL CONTEXT. It was WebGL first, and the console
 * gate caught the cost: four `GPU stall due to ReadPixels` driver messages that
 * disappeared when this module's import was blocked. Browsers cap simultaneous
 * WebGL contexts, the valley already holds one, and this draws a polyline and
 * forty circles. See mala.ts. It also means the rope needs no async import and
 * no renderer init, so it can fail in exactly one way: getContext returns null,
 * and the CSS cord stays.
 *
 * THE CANVAS LIVES INSIDE THE SCROLLER. Verified before building: the rail is a
 * descendant of .nm-stream, so a canvas placed there scrolls with the text for
 * free. Had it been outside, every scroll frame would have needed a re-read of
 * the bead positions.
 */
import type { Mala } from './mala'

export interface MalaHandle {
  /** Re-read the DOM bead positions and rebuild the rope. */
  relayout(): void
  /** Swing it — called when a turn arrives. */
  nudge(strength?: number): void
  destroy(): void
}

export interface MountMalaOptions {
  /** The scrolling conversation container the canvas is placed inside. */
  host: HTMLElement
  /** One frame then stop, under prefers-reduced-motion. */
  still?: boolean
}

export async function mountMala(options: MountMalaOptions): Promise<MalaHandle | null> {
  const { host, still = false } = options
  const { Mala: MalaClass } = await import('./mala')

  const canvas = document.createElement('canvas')
  canvas.setAttribute('aria-hidden', 'true')
  // Height is set per layout, in px, from the CONTENT height — `100%` inside a
  // scroll port resolves to the PORT, which would squash a content-sized
  // backing store back into the visible strip and undo the fix below.
  // Width and height are both set per layout, in px: the canvas covers only the
  // rail the rope hangs in, not the whole column.
  canvas.style.cssText = 'position:absolute;left:0;top:0;pointer-events:none;z-index:0'
  host.style.position = host.style.position || 'relative'
  host.prepend(canvas)

  /**
   * Bound to a non-null const AFTER the guard, and not used directly.
   *
   * `relayout` and `frame` below are hoisted function declarations, so
   * TypeScript will not carry a narrowing into them — it cannot prove they run
   * after the check. Reading the raw result inside them is `possibly null` and
   * was three real type errors.
   */
  const maybeCtx = canvas.getContext('2d')
  if (!maybeCtx) {
    // No 2D context at all is vanishingly rare, but the caller only hides the
    // CSS cord on a non-null return, so bailing here leaves the old rail intact
    // rather than an empty gutter.
    canvas.remove()
    return null
  }
  const ctx: CanvasRenderingContext2D = maybeCtx

  const mala: Mala = new MalaClass()

  let alive = true
  let raf = 0
  let pointer: { x: number; y: number } | null = null

  /**
   * Read where the DOM has put the marker beads, in CONTENT coordinates.
   *
   * `host` is a scroll port, so a bead's viewport rect is relative to what is
   * currently shown. The canvas spans the scrolled CONTENT, so the scroll
   * offset has to be added back or every bead below the fold lands in the
   * wrong place.
   */
  function anchors(): { x: number; y: number }[] {
    const box = host.getBoundingClientRect()
    const top = host.scrollTop
    const left = host.scrollLeft
    // VISIBLE beads only. A bead inside a display:none row measures 0x0, and
    // the rope would happily hang itself from the top-left corner of the page
    // to reach it — which is exactly what happened when a phone-only row was
    // added to this list.
    return [...host.querySelectorAll<HTMLElement>('.nm-bead')]
      .map((el) => el.getBoundingClientRect())
      .filter((r) => r.height > 0)
      .map((r) => ({
        x: r.left - box.left + left + r.width / 2,
        y: r.top - box.top + top + r.height / 2,
      }))
  }

  function relayout() {
    /**
     * ONLY AS WIDE AS THE ROPE.
     *
     * The canvas spanned the whole conversation column — 704px at 1440 — while
     * the mala hangs in a ~60px rail at its left edge. Every frame cleared and
     * composited a surface where more than four fifths of it was empty, and
     * the fix that made the rope reach the bottom of the transcript multiplied
     * that waste by the length of the conversation.
     *
     * Sized from the beads themselves, plus the bow and a bead's radius, so it
     * covers the rope and nothing else. On a phone that is roughly a tenth of
     * the fill it was doing.
     */
    const marker = host.querySelector<HTMLElement>('.nm-bead')
    const markerRadius = marker ? marker.getBoundingClientRect().width / 2 : 8
    const box = host.getBoundingClientRect()
    const rightmost = [...host.querySelectorAll<HTMLElement>('.nm-bead')]
      .map((el) => el.getBoundingClientRect())
      .filter((r) => r.height > 0)
      .reduce((max, r) => Math.max(max, r.right - box.left + host.scrollLeft), 0)
    const w = Math.max(1, Math.min(host.clientWidth, Math.ceil(rightmost + markerRadius + 24)))
    /**
     * THE CONTENT'S HEIGHT, NOT THE PORT'S.
     *
     * This read clientHeight, so the canvas was only ever as tall as the
     * visible strip while the beads run the length of the whole transcript.
     * Past the first screenful the cord simply stopped: measured on a 412px
     * phone mid-conversation, a stub of rope at the top and two beads hanging
     * with nothing between them. The mala looked broken because most of it was
     * being drawn outside the surface.
     */
    /**
     * ...AND MEASURED FROM THE TEXT, NOT FROM THE SCROLLER.
     *
     * An absolutely positioned child still counts toward its scroll
     * container's overflow, so sizing this canvas from `host.scrollHeight`
     * made the canvas part of what it was measuring. The height could then
     * only ratchet upward: one long relayout stretched it, and every later
     * one measured the canvas rather than the text and kept the figure.
     *
     * Measured under prefers-reduced-motion at 1440: 6006px of scroll range
     * over 454px of conversation. On a 375px phone the milder version of the
     * same fault parked the last line of chat 143px above the cards — the
     * stream was pinned to its end correctly and there was simply nothing
     * down there.
     *
     * Collapsing the canvas before the read is the obvious fix and it does
     * not work here: `relayout` runs from a ResizeObserver, where the write
     * reads back as `0px` while the box measures its old height. So the
     * content is measured directly — the bottom of the last row, plus the
     * padding under it — which needs no flush and cannot include the canvas.
     */
    const style = getComputedStyle(host)
    const padBottom = parseFloat(style.paddingBottom) || 0
    const content = [...host.children]
      .filter((el): el is HTMLElement => el instanceof HTMLElement && el !== canvas)
      .reduce((max, el) => Math.max(max, el.offsetTop + el.offsetHeight), 0)
    const h = Math.max(1, Math.ceil(content + padBottom), host.clientHeight)
    // Backing store at device resolution, drawing coordinates in CSS pixels —
    // capped at 2 because a 3x buffer on this strip buys nothing visible and
    // costs real memory on the phones that report it.
    const dpr = Math.min(window.devicePixelRatio || 1, 2)
    canvas.width = Math.round(w * dpr)
    canvas.height = Math.round(h * dpr)
    canvas.style.width = `${w}px`
    canvas.style.height = `${h}px`
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    mala.build({ anchors: anchors(), markerRadius })
    if (still) {
      // Settle it in one go rather than animating: reduced motion should get a
      // hanging rope, not a rope caught mid-swing.
      for (let i = 0; i < 60; i++) mala.animate(ctx, null)
    }
  }

  /**
   * IT STOPS WHEN IT HAS STOPPED.
   *
   * Verlet settles: within a few seconds of a nudge every node is moving less
   * than a hundredth of a pixel per frame and the drawing is identical from
   * one frame to the next. This kept redrawing it anyway, for as long as the
   * page was open. Idle frames now end the loop, and anything that can move
   * the rope again — a pointer near it, a new turn, a resize — restarts it.
   *
   * A few frames of grace after it goes quiet, because a single slow frame can
   * make a moving rope look momentarily still.
   */
  /** Half a device pixel: below this the next frame is the frame on screen. */
  const STILL = 0.25
  let quiet = 0

  function frame() {
    if (!alive) return
    mala.animate(ctx, pointer)
    quiet = mala.drift < STILL && !pointer ? quiet + 1 : 0
    if (quiet > 12) {
      raf = 0
      return
    }
    raf = requestAnimationFrame(frame)
  }

  /** Wake the loop if it has settled. Cheap when it is already running. */
  function wake() {
    quiet = 0
    if (alive && !still && raf === 0) raf = requestAnimationFrame(frame)
  }

  const onPointer = (event: PointerEvent) => {
    wake()
    const box = host.getBoundingClientRect()
    const x = event.clientX - box.left
    const y = event.clientY - box.top
    // Only track while near the rail; a rope reacting to a cursor on the far
    // side of the column is a gimmick rather than a physical object.
    pointer = x < 90 && x > -40 && y > -40 && y < box.height + 40 ? { x, y } : null
  }
  const onLeave = () => {
    pointer = null
  }

  const observer =
    typeof ResizeObserver === 'function'
      ? new ResizeObserver(() => {
          relayout()
          wake()
        })
      : null
  observer?.observe(host)
  if (!still) {
    window.addEventListener('pointermove', onPointer, { passive: true })
    window.addEventListener('pointerleave', onLeave, { passive: true })
  }

  const onVisibility = () => {
    if (document.hidden) {
      cancelAnimationFrame(raf)
      raf = 0
    } else if (!still && alive && raf === 0) {
      raf = requestAnimationFrame(frame)
    }
  }
  document.addEventListener('visibilitychange', onVisibility)

  relayout()
  if (!still) raf = requestAnimationFrame(frame)

  return {
    relayout: () => {
      relayout()
      wake()
    },
    nudge: (strength) => {
      mala.nudge(strength)
      wake()
    },
    destroy() {
      alive = false
      cancelAnimationFrame(raf)
      observer?.disconnect()
      window.removeEventListener('pointermove', onPointer)
      window.removeEventListener('pointerleave', onLeave)
      document.removeEventListener('visibilitychange', onVisibility)
      mala.destroy()
      canvas.remove()
    },
  }
}
