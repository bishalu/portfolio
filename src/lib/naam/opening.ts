/**
 * The opening, as a sequence rather than three CSS delays.
 *
 * ─── WHY THIS IS NOT CSS ANIMATION-DELAY ───────────────────────────────────
 *
 * It was: `.nm-shell[data-first] .nm-turn:nth-of-type(n) .nm-said` with delays
 * of 0, 120 and 360ms. Two things were wrong with that, and only one of them is
 * about timing.
 *
 * The timing one: 820ms for the whole opening is a cascade, not someone
 * speaking. These three blocks are a real rhetorical sequence — greet, then
 * establish what the document is, then ask the question — so staggering them
 * encodes something true rather than decorating. That sets the beat at SPEECH
 * rhythm, about a second and a quarter between thoughts, not the ~300ms a HUD
 * uses. A person pauses between sentences; a menu does not.
 *
 * The structural one, which is the reason this file exists: `data-first` is
 * removed the moment somebody submits, so every rule above vanished with it. A
 * visitor who typed during the opening did not see it speed up — they saw lines
 * two and three POP into place at full opacity, which is exactly the choppiness
 * the whole pass is meant to remove. CSS cannot express "finish early but still
 * finish"; it can only be present or absent.
 *
 * ─── WHY A HOOK AND NOT MORE STATE IN NaamApp ──────────────────────────────
 *
 * NaamApp already carries eighteen useState, eleven refs and four effects. This
 * needs three more pieces of state — which stage we are at, whether it has
 * finished, and whether it was rushed — and adding them there is how a 2,000
 * line component becomes a 2,400 line one. The arrow (which may not appear
 * until the opening is done) and the turns both read from one place instead.
 */
import { useCallback, useEffect, useRef, useState } from 'react'

/** How many blocks the opening has. Three: greet, establish, ask. */
export const OPENING_STEPS = 3

/**
 * The gap between blocks, in ms.
 *
 * 1150 is a considered choice rather than a measured optimum, and it is the
 * number most likely to be argued with, so: it is roughly the pause a person
 * leaves between two sentences when they are being careful about what they say.
 * Below about 700 the three blocks read as one animation with steps in it;
 * above about 1500 the reader has finished and is waiting, which is the failure
 * the previous 820ms total was over-correcting for.
 */
const STEP_MS = 1150

/**
 * The first block does not wait. Someone who has just clicked a link is not
 * ready to be paused at, and a page whose first line is late reads as slow
 * rather than as deliberate.
 */
const FIRST_MS = 220

/**
 * RUSHED, NOT SKIPPED. When somebody types before the opening has finished they
 * have told you they are ready — but cutting instantly to the end throws away
 * the thing they were half-way through watching, and a block that appears with
 * no transition is the pop this whole pass exists to remove. So the remaining
 * blocks arrive fast and in order, 130ms apart, which still reads as a sequence
 * at a speed that does not hold anyone up.
 */
const RUSH_MS = 130

export interface Opening {
  /** How many blocks have arrived, 0..OPENING_STEPS. */
  stage: number
  /** True once every block is in. The arrow waits on this. */
  done: boolean
  /** True when the visitor interrupted — the blocks are arriving fast. */
  rushed: boolean
  /** Called on submit. Idempotent, and a no-op once the opening has finished. */
  rush(): void
}

/**
 * `enabled` is false when there is nothing to stage — a returning visitor with
 * a conversation already in progress should not watch an invitation arrive.
 * In that case every block is in from the first frame.
 */
export function useOpening(enabled: boolean): Opening {
  const [stage, setStage] = useState(() => (enabled ? 0 : OPENING_STEPS))
  const [rushed, setRushed] = useState(false)
  const timers = useRef<number[]>([])
  const rushedRef = useRef(false)

  const clear = useCallback(() => {
    for (const id of timers.current) window.clearTimeout(id)
    timers.current = []
  }, [])

  /** Schedule the blocks that have not arrived yet, at `gap` apart. */
  const schedule = useCallback(
    (from: number, gap: number, lead: number) => {
      clear()
      for (let i = from; i < OPENING_STEPS; i++) {
        const at = lead + (i - from) * gap
        timers.current.push(window.setTimeout(() => setStage(i + 1), at))
      }
    },
    [clear],
  )

  useEffect(() => {
    if (!enabled) {
      setStage(OPENING_STEPS)
      return
    }
    schedule(0, STEP_MS, FIRST_MS)
    return clear
    // schedule/clear are stable; enabled is the only real input.
  }, [enabled, schedule, clear])

  const rush = useCallback(() => {
    if (rushedRef.current) return
    rushedRef.current = true
    setRushed(true)
    // Re-schedule from wherever it got to, rather than jumping to the end.
    setStage((current) => {
      if (current >= OPENING_STEPS) return current
      schedule(current, RUSH_MS, 0)
      return current
    })
  }, [schedule])

  /**
   * REDUCED MOTION SKIPS THE SEQUENCE ENTIRELY, and that is not the same as
   * running it faster. Someone who has asked for less motion has not asked to
   * be shown the same choreography at speed — they have asked for the content.
   * Read once on mount and watched, because turning the setting on mid-session
   * is a request for the motion to stop now.
   */
  useEffect(() => {
    if (typeof matchMedia !== 'function') return
    const calm = matchMedia('(prefers-reduced-motion: reduce)')
    const settle = () => {
      if (!calm.matches) return
      clear()
      setStage(OPENING_STEPS)
    }
    settle()
    calm.addEventListener('change', settle)
    return () => calm.removeEventListener('change', settle)
  }, [clear])

  return { stage, done: stage >= OPENING_STEPS, rushed, rush }
}
