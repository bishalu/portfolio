/**
 * The contract every moving layer in the valley obeys.
 *
 * ─── WHY THIS IS A TYPE AND NOT A COMMENT ──────────────────────────────────
 *
 * PixiJS is explicit that Graphics GEOMETRY is the expensive thing, while alpha
 * and transform are cheap. So the flags, the birds and the lamps are each built
 * once and thereafter only moved or dimmed. valley.ts said so, in prose, above
 * each of them.
 *
 * I then wrote the lamps and had them `clear()` and re-tessellate 27 circles
 * every frame — 1,620 rebuilds a second — in the same file where I had already
 * fixed exactly that twice. The comments were correct, present, and useless: by
 * the time you are 800 lines into a file, the rule you are about to break is
 * off screen.
 *
 * A comment is advice you can walk past. This is a shape you have to fill in:
 * `build` is the only place geometry may be created, `animate` receives a clock
 * and nothing else, and there is nowhere in the interface to put a per-frame
 * rebuild. Adding a fourth moving layer — a river, smoke off a chimney,
 * whatever the scene grows next — means implementing this, and the wrong
 * pattern no longer has a home to be written into.
 *
 * ─── THE RULE, STATED ONCE ─────────────────────────────────────────────────
 *
 *   build(w, h)     Called on mount and on resize, and at NO other time.
 *                   Everything derived from viewport size belongs here:
 *                   positions, radii, path sampling, the geometry itself.
 *
 *   animate(time)   Called every frame. May set x, y, scale, rotation, alpha,
 *                   visible, tint. May NOT call clear(), draw a shape, allocate
 *                   an array, or read layout. If a frame needs new geometry,
 *                   the layer is modelled wrong.
 *
 *   destroy()       Free the Graphics. Pixi does not do it for you, and this
 *                   island remounts on every astro navigation.
 */
export interface LivingLayer {
  /** Build geometry. Mount and resize only. */
  build(width: number, height: number): void
  /** Per frame. Transform, alpha and visibility only — never geometry. */
  animate(time: number): void
  /** Release GPU resources. */
  destroy(): void
}

/**
 * Drive every living layer from one clock.
 *
 * Not a convenience: it is the second half of the contract. With each layer
 * calling its own animate from the render loop, a new one can quietly be added
 * to the loop WITHOUT being added to the resize path — which is the other bug
 * this shape prevents, and the one that would show up as a scene that looks
 * right until somebody turns their phone sideways.
 */
export function drive(layers: readonly LivingLayer[]) {
  return {
    build(width: number, height: number) {
      for (const layer of layers) layer.build(width, height)
    },
    animate(time: number) {
      for (const layer of layers) layer.animate(time)
    },
    destroy() {
      for (const layer of layers) layer.destroy()
    },
  }
}
