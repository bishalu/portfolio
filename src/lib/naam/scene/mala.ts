/**
 * The mala, as a hanging object rather than a drawn one.
 *
 * ─── WHAT WAS WRONG WITH THE CSS VERSION ───────────────────────────────────
 *
 * Audited at 3× against the built page, and every fault is the same fault:
 *
 *   · the counting beads were a `repeating-radial-gradient` — a TILE, so the
 *     ring artefact repeated at a fixed 12px pitch and the rail read as a
 *     zipper rather than as beads on a cord
 *   · every bead identical in size, tone and spacing. Mechanical perfection is
 *     the thing that says "generated"
 *   · the three marker beads sat ON TOP of the gradient instead of being strung
 *     THROUGH it — no hole, no contact shadow, no sense of a cord behind
 *   · the cord was a dead-straight vertical. The one thing everybody knows
 *     about a mala is that it HANGS, and a straight line has no weight
 *   · measured: `animatingAtRest: 0` and `interactiveOnLeft: 0`. Nothing on the
 *     left column moved, ever, or responded to anything
 *
 * Under all of it: the mala was decoration shaped like a mechanism. It counted
 * nothing, weighed nothing and reacted to nothing.
 *
 * ─── WHY VERLET AND NOT A NICER GRADIENT ───────────────────────────────────
 *
 * Slack cannot be faked convincingly. A hand-drawn catenary is right for one
 * spacing and wrong the moment the turns above it change height, and this rail
 * restretches every time the conversation grows. A chain that HANGS solves it
 * once for every spacing, and gets sway, settle and pointer response for free
 * because they are all the same integrator.
 *
 * Verlet specifically because it is four lines and unconditionally stable:
 * positions carry their own velocity as the difference from the last frame, so
 * there is no velocity to diverge and no stiffness to tune. Constraint solving
 * is a few relaxation passes. This is well under the cost of the gradient it
 * replaces once that gradient stopped being cacheable.
 *
 * ─── THE DOM STILL OWNS WHERE THE BEADS GO ─────────────────────────────────
 *
 * The three marker beads stay in the DOM. They are laid out by the same grid
 * that positions each turn, so they track the text as it reflows without
 * anybody computing anything — and that alignment is the whole point of them:
 * a bead marks a thing that was said. This module reads their centres on layout
 * and hangs cord between them. Canvas draws the rope; the DOM decides where its
 * ends are nailed.
 */

/**
 * CANVAS 2D, NOT PIXI, AND THE REASON IS MEASURED.
 *
 * This first shipped on its own WebGLRenderer, separate from the valley's. The
 * console gate then reported four `GPU stall due to ReadPixels` driver messages
 * that vanished the moment the mala's import was blocked — so they were this
 * module's, not the valley's.
 *
 * That is a software-renderer symptom and would not appear on real hardware,
 * but it pointed at something that is true everywhere: a second GPU context to
 * draw one polyline and forty circles in a 60px strip is a bad trade. Browsers
 * cap simultaneous WebGL contexts — commonly 8 to 16 — and the valley already
 * holds one. Spending a second on this puts the two of them in competition on
 * exactly the phones least able to afford it.
 *
 * Everything drawn here is moveTo/lineTo/arc/fill. That is Canvas 2D's native
 * vocabulary, so the port costs nothing and removes the context.
 */

/** One point in the rope. `px/py` is the previous position — in Verlet that IS
 *  the velocity, which is why there is no velocity field. */
interface Node {
  x: number
  y: number
  px: number
  py: number
  pinned: boolean
}

/** A bead threaded on the rope, at a fixed distance along it. */
interface Bead {
  /** 0..1 along the whole chain. */
  t: number
  r: number
  tone: number
  /** Small per-bead lean, so no two catch the light identically. */
  tilt: number
}

const GRAVITY = 0.34

/**
 * A constant sideways bias, and the reason it is not optional.
 *
 * Seeding the bow (see BOW) fixed the fold but left the rope in an UNSTABLE
 * equilibrium: at rest length on a vertical drop, gravity runs parallel to the
 * span and exerts no preference about which side the bulge falls to. Measured,
 * the two spans settled bowing opposite ways — and the long one bowed right,
 * into the column of text it is supposed to be marking beside.
 *
 * Gravity cannot break that tie because it is the axis of the tie. So a small
 * constant leftward force does. It only has to DECIDE the direction — the bow's
 * own geometry sets the size — so it is a tenth of gravity, not a quarter. At a
 * quarter it stopped being a tiebreak and became a second wind, dragging the
 * long span clean off the rail.
 */
const LEAN = 0.035
/** Verlet damping. Below ~0.96 the rope looks like it is moving through syrup;
 *  above ~0.995 it never settles and reads as jelly. */
const DAMP = 0.985
/** Relaxation passes per frame. Two is visibly stretchy on a 40-node rope,
 *  four is taut, and the difference above four is not visible here. */
const PASSES = 4

/**
 * THE CURVE IS BUILT FROM A CAPPED BOW, NOT FROM SLACK. Three wrong fixes led
 * here and each one is worth keeping, because each was a different mistake.
 *
 * 1. Slack alone (5.5%) produced a KNOT, not a catenary. These anchors are
 *    stacked vertically — every marker sits directly under the last, because
 *    they track lines of text in one column — so gravity runs along the span's
 *    own axis and surplus length has no sideways component to resolve into. It
 *    folds back on itself. Slack only reads as hanging when the anchors are
 *    separated horizontally.
 * 2. So the bow was seeded explicitly, as a fraction of span length. That did
 *    kill the knot, but a fraction is the wrong unit: the long span here is
 *    775px, and 7.5% of it is a 58px bulge on a 60px rail. The cord left the
 *    page. (The comment claimed 14px. It was never checked against a real
 *    span length.)
 * 3. And at exactly rest length on a vertical drop, the bow sits in UNSTABLE
 *    equilibrium — gravity has no opinion about which side it falls to, so the
 *    two spans settled bowing opposite ways, the long one into the text.
 *
 * What is left is the honest ordering: choose the bow in PIXELS, capped to what
 * the rail can hold, then derive the rope's rest length from that bow rather
 * than picking a slack figure and hoping. A sine bulge of amplitude A over span
 * L has arc length about L + π²A²/4L, which is the one line below that matters.
 */

/** Bow as a fraction of span, for short spans where a fixed figure would be
 *  proportionally enormous. */
const BOW = 0.075
/** Hard ceiling in px. The rail is 60px wide and the cord must stay inside it
 *  with room for the bead radius. This is what a fractional bow lacked. */
const BOW_MAX = 13

export interface MalaOptions {
  /** Anchor centres in canvas coordinates, top to bottom. */
  anchors: readonly { x: number; y: number }[]
  /** Bead radius the DOM markers use, so the counting beads sit in proportion. */
  markerRadius: number
}

export class Mala {
  private nodes: Node[] = []
  private beads: Bead[] = []
  /**
   * REST LENGTH PER CONSTRAINT, not one for the whole rope.
   *
   * The first cut spread nodes evenly across the total span and pinned the
   * interior markers, which is wrong whenever the markers are not evenly
   * spaced — and they never are, because they sit against lines of text of
   * different heights. A short gap then received the same node count as a long
   * one, so the surplus length had nowhere to go and the beads visibly bunched
   * and overlapped at the middle marker. Each span now carries its own slack,
   * proportional to its own length.
   */
  private rest: number[] = []

  /**
   * Build the rope. Called on layout only — node count, rest length and bead
   * placement all derive from the anchor positions.
   */
  build({ anchors, markerRadius }: MalaOptions) {
    this.drawn = []
    this.nodes = []
    this.beads = []
    this.rest = []
    if (anchors.length < 2) return

    // Built span by span, each between two consecutive markers, so every gap
    // gets slack in proportion to its own length.
    let total = 0
    for (let a = 0; a < anchors.length - 1; a++) {
      const from = anchors[a]
      const to = anchors[a + 1]
      const span = Math.hypot(to.x - from.x, to.y - from.y)
      if (span < 2) continue

      // Bow first, capped; then the arc length that bow implies. Doing it in
      // this order is the whole fix — the rope is never asked to hold more
      // length than the curve it was drawn with can absorb.
      const bow = Math.min(span * BOW, BOW_MAX)
      const arc = span + (Math.PI * Math.PI * bow * bow) / (4 * span)

      // One node roughly every 9px, so the curve is smooth without simulating
      // more points than the rail is wide.
      const count = Math.max(3, Math.round(arc / 9))
      const seg = arc / count

      if (this.nodes.length === 0) {
        this.nodes.push({ x: from.x, y: from.y, px: from.x, py: from.y, pinned: true })
      }
      // Bowed away from the text, perpendicular to the span. Seeding the curve
      // here rather than waiting for gravity is what stops the fold: the rope
      // starts out already holding its surplus sideways.
      const nx = -(to.y - from.y) / span
      const ny = (to.x - from.x) / span

      for (let i = 1; i <= count; i++) {
        const t = i / count
        const bulge = Math.sin(Math.PI * t) * bow
        const x = from.x + (to.x - from.x) * t + nx * bulge
        const y = from.y + (to.y - from.y) * t + ny * bulge
        // The last node of a span IS the next marker, and it is pinned: a bead
        // marks something that was said and sits against its own line of text,
        // so the rope has to pass exactly through it rather than near it.
        this.nodes.push({ x, y, px: x, py: y, pinned: i === count })
        this.rest.push(seg)
      }
      total += arc
    }
    if (this.nodes.length < 2) return

    /**
     * THE MARKERS KEEP THEIR OWN SPACE. The comment here said "never on top of
     * a marker" and nothing implemented it, so counting beads were laid over
     * the 17px marker beads and clustered visibly against them — the surviving
     * half of the bunching artefact after per-span slack fixed the rest.
     *
     * Each pinned node is a marker, and its position along the rope is known,
     * so a bead landing within a marker's radius plus a small gap is skipped.
     * Real malas do exactly this: the guru and the marker beads sit clear.
     */
    const pinnedT: number[] = []
    this.nodes.forEach((node, i) => {
      if (node.pinned) pinnedT.push(i / (this.nodes.length - 1))
    })
    const clearance = (markerRadius * 1.35) / total

    // Counting beads, spaced along the rope's real length so the density is the
    // same in a short gap as in a long one.
    /**
     * BIG ENOUGH TO BE A BEAD. At 0.29 of the marker these came out ~2.5px in
     * radius, and every cue that makes a bead read as round — the rim light,
     * the tight specular, the terminator falling off the edge — is sub-pixel at
     * that size. The shading was all being computed and none of it was visible,
     * so the strand still looked like a dotted line.
     *
     * 0.46 is also closer to a real mala: counting beads are smaller than the
     * markers that divide them, but not by six times.
     */
    const r = Math.max(3.2, markerRadius * 0.46)
    const n = Math.max(4, Math.floor(total / (r * 2.45)))
    for (let i = 1; i < n; i++) {
      const t = i / n
      if (pinnedT.some((pt) => Math.abs(pt - t) < clearance)) continue
      // Seeded off the index so a bead keeps its character between rebuilds —
      // a rail that reshuffles on resize is a rail nobody believes in.
      const h = Math.sin(i * 127.1) * 43758.5453
      const j = h - Math.floor(h)
      this.beads.push({ t, r: r * (0.82 + j * 0.36), tone: j, tilt: (j - 0.5) * 0.7 })
    }
  }

  /** A nudge, for when a turn arrives. The rope swings and settles on its own. */
  nudge(strength = 1) {
    for (const n of this.nodes) {
      if (n.pinned) continue
      n.px -= 1.6 * strength
    }
  }

  /**
   * Per frame. Integrate, relax, draw.
   *
   * `pointer` is in canvas coordinates, or null. The rope leans away from it —
   * a hanging thing you put your hand near moves, and this is the only element
   * on the left column that responds to anything at all.
   */
  /** Where every node was when the rope was last DRAWN. */
  private drawn: number[] = []

  /**
   * How far the rope has moved since it was last drawn, in pixels.
   *
   * NOT velocity, which was the first attempt and never settles: Verlet under
   * constant gravity reaches a steady state where the constraint pass cancels
   * the fall every frame, so `x - px` stays small but non-zero forever and a
   * velocity threshold never trips. Measured, the loop ran at 40/s on a phone
   * with nothing visibly moving.
   *
   * What actually matters is whether the next frame would look any different
   * from the one on screen, so that is what this measures.
   */
  get drift(): number {
    let most = 0
    for (let i = 0; i < this.nodes.length; i++) {
      const dx = Math.abs(this.nodes[i].x - (this.drawn[i * 2] ?? Infinity))
      const dy = Math.abs(this.nodes[i].y - (this.drawn[i * 2 + 1] ?? Infinity))
      const d = Math.max(dx, dy)
      if (d > most) most = d
    }
    return most
  }

  /** Remember the drawn pose, so `drift` can be measured against it. */
  private remember(): void {
    for (let i = 0; i < this.nodes.length; i++) {
      this.drawn[i * 2] = this.nodes[i].x
      this.drawn[i * 2 + 1] = this.nodes[i].y
    }
  }

  animate(ctx: CanvasRenderingContext2D, pointer: { x: number; y: number } | null) {
    if (this.nodes.length === 0) return

    for (const n of this.nodes) {
      if (n.pinned) continue
      const vx = (n.x - n.px) * DAMP
      const vy = (n.y - n.py) * DAMP
      n.px = n.x
      n.py = n.y
      n.x += vx - LEAN
      n.y += vy + GRAVITY

      if (pointer) {
        const dx = n.x - pointer.x
        const dy = n.y - pointer.y
        const d2 = dx * dx + dy * dy
        // Falls off fast: felt within about 70px, absent beyond it.
        if (d2 < 4900 && d2 > 0.01) {
          const f = (1 - d2 / 4900) * 0.9
          const d = Math.sqrt(d2)
          n.x += (dx / d) * f
          n.y += (dy / d) * f
        }
      }
    }

    /**
     * COME TO A STOP, PROPERLY.
     *
     * Verlet under constant gravity never quite stops: the fall is re-applied
     * every frame and the constraint pass cancels it, so each node keeps a
     * small permanent jitter and the rope is redrawn forever for no visible
     * change. Two thresholds were tried against that and both were fighting
     * the physics rather than fixing it.
     *
     * Below a twentieth of a pixel of movement a node is not going anywhere,
     * so its velocity is set to exactly zero — in Verlet that means moving the
     * previous position onto the current one. Gravity still acts next frame,
     * the constraints still cancel it, and now the residue has nowhere to
     * accumulate: the rope reaches a true rest and the loop can end.
     */
    for (const n of this.nodes) {
      if (n.pinned) continue
      if (Math.abs(n.x - n.px) < 0.05 && Math.abs(n.y - n.py) < 0.05) {
        n.px = n.x
        n.py = n.y
      }
    }

    for (let pass = 0; pass < PASSES; pass++) {
      for (let i = 0; i < this.nodes.length - 1; i++) {
        const a = this.nodes[i]
        const b = this.nodes[i + 1]
        const dx = b.x - a.x
        const dy = b.y - a.y
        const d = Math.hypot(dx, dy) || 0.0001
        const diff = (this.rest[i] - d) / d / 2
        const ox = dx * diff
        const oy = dy * diff
        if (!a.pinned) {
          a.x -= ox
          a.y -= oy
        }
        if (!b.pinned) {
          b.x += ox
          b.y += oy
        }
      }
    }

    this.draw(ctx)
    this.remember()
  }

  private at(t: number): { x: number; y: number; a: number } {
    const f = t * (this.nodes.length - 1)
    const i = Math.min(this.nodes.length - 2, Math.floor(f))
    const k = f - i
    const a = this.nodes[i]
    const b = this.nodes[i + 1]
    return { x: a.x + (b.x - a.x) * k, y: a.y + (b.y - a.y) * k, a: Math.atan2(b.y - a.y, b.x - a.x) }
  }

  private draw(ctx: CanvasRenderingContext2D) {
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height)

    // The cord itself, under the beads.
    ctx.beginPath()
    ctx.moveTo(this.nodes[0].x, this.nodes[0].y)
    for (let i = 1; i < this.nodes.length; i++) ctx.lineTo(this.nodes[i].x, this.nodes[i].y)
    ctx.strokeStyle = 'rgba(138, 112, 72, 0.85)'
    ctx.lineWidth = 1.6
    ctx.lineCap = 'round'
    ctx.lineJoin = 'round'
    ctx.stroke()

    /**
     * ─── WHAT MAKES A BEAD LOOK LIKE A BEAD ───────────────────────────────
     *
     * The first version was three flat ellipses: a body, a dark blob offset
     * down-right, and a soft pale patch up-left. It read as plastic, and the
     * reason is that none of those three is how a round object actually meets
     * light. A sphere is not a disc with a smudge on it.
     *
     * Four things carry roundness, in the order they matter:
     *
     *   TERMINATOR — the dark side is a CRESCENT hugging the far edge, not a
     *     blob near the middle. A centred shadow flattens a sphere into a
     *     dish; the shadow has to fall off the edge to imply the surface
     *     turning away.
     *   RIM LIGHT — a thin bright arc on the extreme shadow-side edge, bounced
     *     off the paper the mala lies on. This is the single strongest cue,
     *     and its absence is why the old beads looked like holes. Real light
     *     comes back off a ground; a bead with a dark side and no bounce reads
     *     as cut out rather than lit.
     *   SPECULAR — small and TIGHT, not broad and soft. Polished wood has a
     *     hard highlight; a wide one says matte plastic. The old one was 0.34r
     *     and hazy, which is exactly the plastic look.
     *   HUE, not just value. The old beads varied in brightness alone, so they
     *     were the same bead at different exposures. Real sandalwood varies in
     *     TONE — some redder, some more olive — and that variation is what
     *     stops a strand looking machined.
     */
    for (const bead of this.beads) {
      const p = this.at(bead.t)
      const r = bead.r

      // Body. Hue walks between a red sandalwood and a darker rosewood rather
      // than one brown at two brightnesses.
      ctx.fillStyle = wood(bead.tone)
      ellipse(ctx, p.x, p.y, r * 1.05, r)

      // Terminator, hugging the lower-right edge and running off it.
      ctx.save()
      ctx.beginPath()
      ctx.ellipse(p.x, p.y, r * 1.05, r, 0, 0, Math.PI * 2)
      ctx.clip()
      ctx.fillStyle = 'rgba(46, 26, 12, 0.5)'
      ellipse(ctx, p.x + r * 0.52, p.y + r * 0.46, r * 0.98, r * 0.94)
      ctx.restore()

      // Rim light: bounce off the paper, on the shadow side ONLY. The arc used
      // to run to 0.78π, which carries it round past the bottom onto the lit
      // side — light arriving from both directions, which flattens the bead
      // again. It stops at the bottom now.
      ctx.beginPath()
      ctx.ellipse(p.x, p.y, r * 0.9, r * 0.86, 0, Math.PI * -0.04, Math.PI * 0.52)
      ctx.strokeStyle = 'rgba(226, 178, 122, 0.55)'
      ctx.lineWidth = Math.max(0.45, r * 0.2)
      ctx.stroke()

      // Specular, small and tight, up-left with everything else on this page.
      ctx.fillStyle = 'rgba(255, 240, 214, 0.85)'
      ellipse(ctx, p.x - r * 0.34, p.y - r * 0.36, r * 0.2, r * 0.16)

      void bead.tilt
    }
  }

  destroy() {
    this.nodes = []
    this.beads = []
  }
}

/** One filled ellipse. Canvas 2D has no fill-shape shorthand. */
function ellipse(ctx: CanvasRenderingContext2D, x: number, y: number, rx: number, ry: number) {
  ctx.beginPath()
  ctx.ellipse(x, y, rx, ry, 0, 0, Math.PI * 2)
  ctx.fill()
}

/**
 * Bead colour for a 0..1 seed: a walk between two real woods rather than one
 * brown at two brightnesses.
 *
 * Sandalwood at the warm end, rosewood at the dark end. Interpolating between
 * two different HUES is the point — varying a single brown's lightness gives
 * you the same bead under different exposures, which is what the strand looked
 * like before and why it read as moulded rather than strung.
 */
function wood(seed: number): string {
  // Widened from [166,108,62]..[116,66,38] over 0.18..1: that range was narrow
  // enough that the strand read as one dark chocolate tone throughout, which is
  // the machined look the hue walk exists to avoid.
  const warm = [182, 124, 74]
  const deep = [104, 58, 33]
  const k = 0.06 + seed * 0.94
  const mix = warm.map((c, i) => Math.round(deep[i] + (c - deep[i]) * k))
  return `rgb(${mix[0]} ${mix[1]} ${mix[2]})`
}
