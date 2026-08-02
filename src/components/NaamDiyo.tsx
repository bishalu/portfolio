/**
 * The lamp (docs/design/DESIGN.md §1 P1 exception, P7 · docs/design/MOTION.md §5).
 *
 * WHY THIS EXISTS AT ALL. The verdict on the first build of /naam was that it
 * "still feels like a plain old webpage", and the diagnosis was not tween
 * quality — the card flight was already 430ms with an apex lift, a hitstop and
 * a spring recoil. What was missing is that NOTHING WAS CONTINUOUS. The screen
 * was completely inert between clicks, and a game never is. This is the fix:
 * one element that burns whether or not anyone touches the page.
 *
 * WHY A DIYO. The page has to feel familiar to a Nepali or Buddhist visitor
 * without ever saying so, which rules out the tourist shorthand — prayer
 * flags, mandalas, Everest, an om. A lamp is what is actually lit at a naming,
 * it is a light source rather than an ornament (so it can do real work — see
 * the ground below), and it is a living thing rather than a shape, which is
 * exactly what a mala of beads could not be.
 *
 * WHY IT IS NOT A MASCOT. It has no face and no personality beyond a flame's.
 * A character would be a second signature competing with the mala thread, and
 * would need authored assets. This is ~40 lines of SVG and two keyframes.
 *
 * THE FLICKER IS THE WHOLE POINT, so it must never look like a loop. Two
 * animations run at deliberately incommensurate periods — 2.3s and 3.7s, whose
 * ratio is irrational enough that the combined pattern does not visibly repeat
 * inside a session. A single sine wave reads as a CSS animation; two out of
 * phase read as fire.
 *
 * IT LIGHTS THE PAGE. `--nm-flame` is published on the root element every
 * frame-ish and the ground's radial pool in naam.astro rides it at a fifth of
 * the amplitude. So the paper itself breathes with the flame rather than being
 * a painted rectangle. That is the answer to "what do games do to make the
 * ground come to life" — they light it, they do not colour it.
 *
 * REDUCED MOTION: the flame is drawn, lit, and perfectly still. Not hidden —
 * the lamp is still on, it simply stops moving (WCAG 2.3.3; an unmoving lit
 * lamp is the honest still frame of a burning one).
 */
import { useEffect, useRef } from 'react'

export type DiyoState = 'idle' | 'thinking' | 'flare'

interface NaamDiyoProps {
  /** 'thinking' while the model is out; 'flare' pulses once when a name lands. */
  state: DiyoState
  /** Cleared by the parent after the flare has played. */
  onFlareEnd?: () => void
}

/** How long a flare reads as a reaction rather than as a mode change. */
const FLARE_MS = 180

export default function NaamDiyo({ state, onFlareEnd }: NaamDiyoProps) {
  const flareRef = useRef<number | null>(null)

  useEffect(() => {
    if (state !== 'flare' || !onFlareEnd) return
    flareRef.current = window.setTimeout(onFlareEnd, FLARE_MS)
    return () => {
      if (flareRef.current !== null) window.clearTimeout(flareRef.current)
    }
  }, [state, onFlareEnd])

  return (
    <div className="nm-diyo" data-state={state} aria-hidden="true">
      {/* 62×79, not 44×56. At the smaller size the flame rendered about 15px
          and a ±1.4° sway is sub-pixel there — it animated and looked frozen,
          which is the one failure this component exists to avoid. Scale is not
          decoration here; below a certain size a flicker is not perceptible at
          all. */}
      <svg viewBox="0 0 44 56" width="62" height="79" focusable="false">
        {/* The glow is painted first so the flame sits inside its own light. */}
        <defs>
          <radialGradient id="nm-diyo-glow" cx="50%" cy="34%" r="52%">
            <stop offset="0%" stopColor="var(--nm-keep)" stopOpacity="0.55" />
            <stop offset="100%" stopColor="var(--nm-keep)" stopOpacity="0" />
          </radialGradient>
          <linearGradient id="nm-diyo-flame" x1="0" y1="1" x2="0" y2="0">
            {/* --nm-ember, not --nm-accent. The latter does not exist; see the
                token's own comment in naam.astro. */}
            <stop offset="0%" stopColor="var(--nm-ember)" />
            <stop offset="55%" stopColor="var(--nm-keep)" />
            <stop offset="100%" stopColor="#FFF3D0" />
          </linearGradient>
        </defs>

        <circle className="nm-diyo-glow" cx="22" cy="19" r="20" fill="url(#nm-diyo-glow)" />

        {/* Flame: two nested teardrops. The outer one carries the slow sway,
            the inner the fast guttering, so neither period is legible. */}
        <g className="nm-diyo-sway">
          <path
            className="nm-diyo-outer"
            d="M22 4c4.6 5.9 7.4 10.2 7.4 14.6 0 4.6-3.3 8-7.4 8s-7.4-3.4-7.4-8C14.6 14.2 17.4 9.9 22 4z"
            fill="url(#nm-diyo-flame)"
          />
          <path
            className="nm-diyo-inner"
            d="M22 12c2.2 3.3 3.4 5.4 3.4 7.6 0 2.4-1.5 4-3.4 4s-3.4-1.6-3.4-4c0-2.2 1.2-4.3 3.4-7.6z"
            fill="#FFF6DE"
          />
        </g>

        {/* The wick. Without it the flame hovers eight pixels above the bowl
            and the whole thing reads as a teardrop icon rather than as
            something burning — the single most important line in the drawing,
            and it is two points long. */}
        <path
          className="nm-diyo-wick"
          d="M22 25.5V33"
          stroke="#6B5836"
          strokeWidth="1.6"
          strokeLinecap="round"
          fill="none"
        />

        {/* The oil it floats on: a thin ellipse just inside the rim. Cheap, and
            it is what turns a half-circle into a vessel with something in it. */}
        <ellipse className="nm-diyo-oil" cx="22" cy="34" rx="14.5" ry="2.4" fill="var(--nm-keep-fill)" />

        {/* The lamp. A pinched clay saucer — the shape, not a picture of one.
            The rim flares slightly wider than the bowl so the silhouette has a
            lip to catch the light, which is what makes it read as fired clay
            rather than as a semicircle. */}
        <path
          className="nm-diyo-body"
          d="M7.5 34h29c0 7.4-6.1 12.6-14.5 12.6S7.5 41.4 7.5 34z"
          fill="var(--nm-surface-2)"
          stroke="var(--nm-hair-strong)"
          strokeWidth="1"
        />
        <path
          className="nm-diyo-lip"
          d="M5.5 33.6c0-1 1-1.6 2.4-1.6h28.2c1.4 0 2.4.6 2.4 1.6 0 1-1 1.7-2.4 1.7H7.9c-1.4 0-2.4-.7-2.4-1.7z"
          fill="var(--nm-surface)"
          stroke="var(--nm-hair-strong)"
          strokeWidth="1"
        />
      </svg>
    </div>
  )
}
