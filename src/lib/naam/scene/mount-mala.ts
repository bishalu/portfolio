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
  canvas.style.cssText =
    'position:absolute;left:0;top:0;width:100%;height:100%;pointer-events:none;z-index:0'
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

  /** Read where the DOM has put the marker beads, in host coordinates. */
  function anchors(): { x: number; y: number }[] {
    const box = host.getBoundingClientRect()
    // VISIBLE beads only. A bead inside a display:none row measures 0x0, and
    // the rope would happily hang itself from the top-left corner of the page
    // to reach it — which is exactly what happened when a phone-only row was
    // added to this list.
    return [...host.querySelectorAll<HTMLElement>('.nm-bead')]
      .map((el) => el.getBoundingClientRect())
      .filter((r) => r.height > 0)
      .map((r) => ({ x: r.left - box.left + r.width / 2, y: r.top - box.top + r.height / 2 }))
  }

  function relayout() {
    const w = Math.max(1, host.clientWidth)
    const h = Math.max(1, host.clientHeight)
    // Backing store at device resolution, drawing coordinates in CSS pixels —
    // capped at 2 because a 3x buffer on this strip buys nothing visible and
    // costs real memory on the phones that report it.
    const dpr = Math.min(window.devicePixelRatio || 1, 2)
    canvas.width = Math.round(w * dpr)
    canvas.height = Math.round(h * dpr)
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    const found = anchors()
    const marker = host.querySelector<HTMLElement>('.nm-bead')
    mala.build({
      anchors: found,
      markerRadius: marker ? marker.getBoundingClientRect().width / 2 : 8,
    })
    if (still) {
      // Settle it in one go rather than animating: reduced motion should get a
      // hanging rope, not a rope caught mid-swing.
      for (let i = 0; i < 60; i++) mala.animate(ctx, null)
    }
  }

  function frame() {
    if (!alive) return
    mala.animate(ctx, pointer)
    raf = requestAnimationFrame(frame)
  }

  const onPointer = (event: PointerEvent) => {
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

  const observer = typeof ResizeObserver === 'function' ? new ResizeObserver(relayout) : null
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
    relayout,
    nudge: (strength) => mala.nudge(strength),
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
