/**
 * The note wall (docs/design/DESIGN.md §1 P1 exception, P2, §2, P10).
 *
 * WHY IT IS NOT A ROW OF CHIPS ANY MORE. This is the one place on /naam where
 * other people's choices accumulate — the six names the family keeps coming
 * back to, plus every suggestion Bishal has approved onto the wall — and as a
 * flat pill row it read as filter chips: a control surface, something you were
 * expected to click. Nothing here is clickable. It is a record, so it should
 * look like one, and the object that says "several people put something up
 * here" without a word of explanation is a wall of pinned notes.
 *
 * WHAT MAKES PAPER READ AS PAPER, and every one of these is load-bearing:
 *
 *   TWO SHADOWS, NEVER ONE. A single blurred shadow is the Material-card tell.
 *   Real paper has a hard contact shadow where it touches the wall plus a wide
 *   faint lift, so it is `0 1px 1px` AND `0 8px 14px -6px`.
 *
 *   AN UNEVEN, SQUARE-ISH RADIUS (1px 3px 2px 4px). A 999px pill or a uniform
 *   8px is UI chrome. Four different sub-pixel radii read as a cut edge.
 *
 *   A DETERMINISTIC TILT OF ±3°, hashed off the name by the caller so the wall
 *   never reshuffles on a re-render — a random angle re-rolls on every render
 *   and reads as a glitch. Past ~4° it stops reading as "someone pinned this in
 *   a hurry" and starts reading as "someone applied a rotate".
 *
 *   UN-TILT TO 0° AND LIFT ON HOVER. Straightening a note is a stronger
 *   physical cue than any shadow change: it reads as picked up.
 *
 *   OVERLAP. A perfectly spaced row is a leaderboard. An overlapping cluster
 *   with ascending z-index is a wall, and that is the one this page wants.
 *
 * WHY THE TILT IS ON A WRAPPER. `transform` is a single property and two
 * things want it — the tilt, which is permanent and per-note, and the lift,
 * which is transient and on hover. Sharing one element means the lift has to
 * restate the tilt, and anything added later that animates layout (motion's
 * layoutId is the plan's next step) would clobber both. So .nm-note-pin owns
 * rotation and nothing else, and .nm-note owns translation and nothing else.
 *
 * THE SIGNATURE IS PART OF THE NOTE, not a caption under it: smaller, bottom
 * right, the way a person actually signs something they pinned up. Only
 * approved suggestions carry one — the family's own six do not need attributing
 * — so its presence is what visibly separates "ours, to start" from "somebody
 * else already chose this", with no label saying so.
 *
 * ACCESSIBILITY (P10). It is a real <ul>/<li>, not div soup, and role="list" is
 * restorative rather than redundant because `list-style: none` strips list
 * semantics in Safari. Nothing inside is focusable and nothing should be: a
 * note is a record, not a control, and a wall of eleven fake buttons that do
 * nothing is a worse keyboard experience than a list. Ink #2b2a26 on #f6e27a is
 * 11.0:1; the signature's mix is 5.6:1.
 *
 * Every visible string reaches the DOM through JSX interpolation. `from` and
 * `relation` are a stranger's typed text and are never assembled into markup.
 */
import type { CSSProperties } from 'react'
import { NAAM_COPY } from '@/lib/naam/copy'

const C = NAAM_COPY

export interface WallNote {
  /** Stable across renders — it is what the tilt was hashed from. */
  key: string
  /** Empty when a suggestion names a row this build cannot resolve. */
  deva: string
  latin: string
  /** Absent on the family's own seeds; a signature on everything else. */
  who?: string
  /** Degrees, ±3, computed by the caller from `key`. */
  tilt: number
}

export interface NaamWallProps {
  notes: readonly WallNote[]
}

export default function NaamWall({ notes }: NaamWallProps) {
  return (
    <>
      <p className="label-mono label-mono--sm nm-quiet-label">{C.app.familyLead}</p>
      {/* eslint-disable-next-line jsx-a11y/no-redundant-roles -- restorative:
          `list-style: none` strips list semantics in Safari/VoiceOver, and every
          list on this page is unstyled. */}
      <ul className="nm-wall" role="list">
        {notes.map((note, i) => (
          // --z ascends so a later note sits on top of the one it overlaps,
          // which is what turns a spaced row into a pile.
          <li className="nm-note-pin" key={note.key} style={{ '--tilt': `${note.tilt}deg`, '--z': i } as CSSProperties}>
            <div className="nm-note">
              {note.deva && (
                <span className="nm-note-deva" lang="sa-Deva">
                  {note.deva}
                </span>
              )}
              <span className="nm-note-latin">{note.latin}</span>
              {note.who && <span className="nm-note-who">{note.who}</span>}
            </div>
          </li>
        ))}
      </ul>
    </>
  )
}
