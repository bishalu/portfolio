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
import type { CSSProperties } from 'react'
import { NAAM_COPY } from '@/lib/naam/copy'

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
  return (
    <>
      <p className="label-mono label-mono--sm nm-quiet-label" id="nm-shelf-label">
        {C.app.familyLead}
      </p>
      {/* THE SHELF SCROLLS, SO IT MUST BE REACHABLE. axe's
          scrollable-region-focusable is a serious violation and it caught this
          the moment the shelf gained a max-height: a region a mouse can scroll
          and a keyboard cannot is content only some people can read. tabIndex
          makes it focusable and aria-labelledby borrows the heading already
          above it, so the label is not written twice and cannot drift.
          role="list" is restorative — `list-style: none` strips list semantics
          in Safari/VoiceOver, and every list on this page is unstyled. */}
      {/* eslint-disable-next-line jsx-a11y/no-redundant-roles, jsx-a11y/no-noninteractive-tabindex */}
      <ul className="nm-shelf" role="list" tabIndex={0} aria-labelledby="nm-shelf-label">
        {notes.map((note) => (
          <li
            className="nm-leaf"
            key={note.key}
            data-mine={note.mine ? 'true' : undefined}
            /* Support drives the ink, capped at five so a runaway favourite
               cannot black the leaf out and make its own name unreadable. */
            style={{ '--support': Math.min(note.count, 5) } as CSSProperties}
          >
            <span className="nm-leaf-name">
              {note.deva && (
                <span className="nm-leaf-deva" lang="sa-Deva">
                  {note.deva}
                </span>
              )}
              <span className="nm-leaf-latin">{note.latin}</span>
            </span>

            {/*
              THE TALLY IS BEADS, and that is a choice among three.

              SIZE — a name set larger the more people chose it — is the
              word-cloud answer, and it makes the names with least support the
              hardest to read, which is backwards on a page asking people to
              consider them.

              A DIGIT reads as a score. This is a family choosing a child's
              name, not a leaderboard, and "4" beside a name invites the reader
              to treat the other one as losing.

              BEADS are already this page's counting object — the thread down
              the conversation, the three slots — so one bead per person reads
              instantly, needs no legend, and stays warm. Past five they stop
              being countable at a glance, so the fifth carries a numeral
              instead of the row growing forever.

              The beads are aria-hidden and the count is stated in words, so
              nothing here depends on counting dots or seeing ink density.
            */}
            <span className="nm-leaf-tally" aria-hidden="true">
              {note.count > 5 ? (
                <span className="nm-leaf-many label-mono label-mono--sm">{note.count}</span>
              ) : (
                Array.from({ length: note.count }, (_, i) => <span className="nm-leaf-bead" key={i} />)
              )}
            </span>
            <span className="sr-only">{C.wall.support(note.count)}</span>

            {note.who && note.count === 1 && <span className="nm-leaf-who">{note.who}</span>}
          </li>
        ))}
      </ul>
    </>
  )
}
