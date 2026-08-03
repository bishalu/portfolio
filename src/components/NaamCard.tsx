/**
 * One name, on its sound rail (docs/design/DESIGN.md §1, P4, P5, §4).
 *
 * WHY this shape: the site's thesis is an instrument reading a signal, and a
 * name is a signal — a sound you will say ten thousand times. The document
 * this page is built from says so itself: its legend defines `f?` as
 * "grammatically feminine ending (say it aloud to judge)" and `!` as "harder
 * consonant cluster". Judging by ear is the source's own instruction, so every
 * name here sits on a 2px --grad-signal rule with one tick per syllable — the
 * site's stations-on-a-path motif (DESIGN.md §0), applied to a word. A tick
 * whose syllable opens on a consonant cluster is marked, on the rows the
 * document flagged. For a V name the FIRST tick carries the व/ब control,
 * because the first syllable is exactly where the swap happens: one control,
 * one visual, one joke, doing three jobs.
 *
 * WHY provenance is type, not a badge: DESIGN.md §4 allows three badge words
 * and forbids a fourth, and P5 already assigns mono the job of being the voice
 * of data. So everything the document said is set in --font-mono at
 * --paper-soft; everything we worked out is set in --font-body. Nothing
 * interpretive is ever in mono, nothing verbatim is ever in prose.
 *
 * THE MEANING LINE IS ALWAYS MONO, on every card. An earlier cut switched face
 * per row on `glossIsVerbatim` — mono when the tidy changed nothing, prose when
 * it did. The reasoning was right and the result was not: the tidy is
 * conservative by design, so the two cases interleave unpredictably down a
 * three-column grid, and a typeface that changes card-to-card for a reason the
 * reader cannot decode reads as a bug, not as provenance.
 *
 * So the line is drawn where it is actually true. A meaning ALWAYS originates
 * with the document — our tidy expands `wh` to `white` and drops `, Sch`, it
 * never writes a new sentence. `"'lord of the V', name of a man (= veda-dhara)"`
 * is not something this site said, and neither is its tidied form. Our own
 * voice on this page is the match reasons and the assistant's framing, and both
 * are set in prose. Verbatim-vs-tidied is then stated in WORDS inside the
 * disclosure (C.glossVerbatim / C.glossTidied) rather than encoded in a face:
 * if a distinction matters, say it.
 *
 * Mono here is sentence case, not .label-mono. §3's two-sizes rule governs the
 * *labeling* system — eyebrows, chips, status. A 60-character quotation is data
 * being shown, the same license §3 gives a stat's value ("a stat's value is not
 * a label — it is the proof"), and uppercasing it would make it unreadable.
 *
 * WHY the disclosure is not the card: the plan's sketch puts the pick button
 * next to the name and the document's words behind "▸ from the document". A
 * <button> inside a <summary> is an axe `nested-interactive` violation, and
 * axe 0 is a gate (P10) — so the card is an <article> and the <details> sits
 * inside it, holding the verbatim block, exactly as the sketch draws it.
 *
 * WHY the buttons can arrive disabled: this component server-renders the
 * shortlist on a prerendered page (the no-JS content guarantee) and also
 * renders inside two hydrated islands. With no handlers passed it emits the
 * same markup with `disabled` plus data hooks, and the page's one enhancement
 * script enables and wires it after boot. A disabled control is honest and
 * unfocusable; a dead enabled one is neither. Sizes are identical either way,
 * so nothing shifts (CLS floor is 0).
 */
import type { MouseEvent } from 'react'
import { NAAM_COPY } from '@/lib/naam/copy'
import { naamPreferredDevanagari, naamPreferredForm, type NaamRow } from '@/types/naam'

const C = NAAM_COPY.card

export interface NaamCardProps {
  row: NaamRow
  /** The page-wide व/ब display preference. See src/lib/naam/tray.ts. */
  preferB: boolean
  picked?: boolean
  /** Absent = server-rendered; the page script wires it after boot. */
  onSwap?: () => void
  onPick?: () => void
  /** The tray is at its cap and this name is not in it. */
  trayFull?: boolean
  /** false on the wall, which displays names rather than collecting them. */
  pickable?: boolean
  /**
   * A name the family carries that the document does not. Renders the name,
   * the rail and one mono line — no meaning, no etymology, no source badge,
   * because there is nothing to cite.
   */
  undocumented?: boolean
}

/**
 * Digraphs are one sound, not a cluster: bh, ch, dh, gh, jh, kh, ph, sh, th.
 * Everything left in the onset after they are folded away is a real stack.
 */
function opensOnCluster(syllable: string): boolean {
  const onset = /^[^aeiou]*/.exec(syllable.toLowerCase())?.[0] ?? ''
  return onset.replace(/[bcdgjkpst]h/g, 'x').length > 1
}

export default function NaamCard({
  row,
  preferB,
  picked = false,
  onSwap,
  onPick,
  trayFull = false,
  pickable = true,
  undocumented = false,
}: NaamCardProps) {
  const primary = naamPreferredForm(row, preferB)
  const alternate = row.bVariant ? (preferB ? row.latin : row.bVariant) : null
  // The script follows the spelling. Showing वस्तु over "Bastu" told the one
  // reader who can read it that the swap was not happening.
  const devanagari = naamPreferredDevanagari(row, preferB)
  const marked = row.badges.hardCluster

  /**
   * THE WHOLE CARD KEEPS THE NAME. Asking someone to find a small pill in the
   * corner of a card is three steps — notice it, aim at it, work out that it
   * and not the name above it is the thing that chooses — for one idea.
   *
   * One region opts out, because it does a different job and is one tap of its
   * own: the व/ब pill, which changes how a name is SPELLED rather than
   * choosing it. Everything else on the card is Keep.
   *
   * The real <button> is still rendered and still does the work, so the
   * accessibility tree has exactly one properly-named control per card and the
   * keyboard path is unchanged — this only widens where a pointer may land.
   * The button's own click is left to bubble: forwarding it here as well would
   * keep the name and immediately un-keep it.
   */
  const keepFromCard = (event: MouseEvent<HTMLElement>) => {
    if (!onPick || (trayFull && !picked)) return
    const target = event.target as HTMLElement
    if (target.closest('.nm-swap') || target.closest('.nm-pick')) return
    onPick()
  }

  return (
    /* Both rules are asking for the keyboard path, and the keyboard path is the
       real <button> rendered inside this article — properly labelled, in the
       tab order, doing exactly this. Adding onKeyDown here would give a
       screen-reader user a second, unlabelled way to keep the same name, and
       making the <article> a button would nest the disclosure inside it, which
       axe rejects outright (nested-interactive). This widens where a POINTER
       may land and changes nothing else. */
    // eslint-disable-next-line jsx-a11y/click-events-have-key-events, jsx-a11y/no-noninteractive-element-interactions
    <article className="nm-card" data-nm-card={row.id} data-picked={picked ? 'true' : undefined} onClick={keepFromCard}>
      <div className="nm-card-name">
        <span className="nm-deva" data-nm-deva lang="sa-Deva">
          {devanagari}
        </span>
        <p className="nm-latin">
          <span className="nm-latin-main">{primary}</span>
          {alternate && <span className="nm-latin-alt label-mono">{alternate}</span>}
        </p>
      </div>

      {/* role="presentation" because this is an instrument, not a list. Without
          it a screen reader announces "list, 3 items" on all 77 prerendered
          cards; the syllables themselves stay in the reading order, which is
          the part worth hearing. */}
      <div className="nm-rail" data-swap={row.bVariant ? 'true' : undefined}>
        <ol className="nm-ticks" role="presentation">
          {row.syllableSplit.map((syllable, i) => (
            <li
              key={`${row.id}-${i}`}
              className="nm-tick"
              data-cluster={marked && opensOnCluster(syllable) ? 'true' : undefined}
            >
              {i === 0 && row.bVariant && (
                <span className="nm-swap-slot">
                  <button
                    type="button"
                    className="nm-swap label-mono label-mono--sm"
                    aria-pressed={preferB}
                    disabled={!onSwap}
                    data-nm-swap={onSwap ? undefined : ''}
                    onClick={onSwap}
                  >
                    <span aria-hidden="true">{C.swapGlyph}</span>
                    <span className="sr-only">
                      {row.latin} · {C.swapAria}
                    </span>
                  </button>
                </span>
              )}
              <span className="nm-tick-dot" aria-hidden="true"></span>
              <span className="nm-tick-syllable label-mono label-mono--sm">{syllable}</span>
            </li>
          ))}
        </ol>
      </div>

      {undocumented ? (
        <p className="nm-undocumented label-mono label-mono--sm">{C.notInDocument}</p>
      ) : (
        <p className="nm-gloss">{row.gloss}</p>
      )}

      {/* THE MATCHER NO LONGER EXPLAINS ITSELF ON THE CARD. "Matched on ·
          attested name · evocative meaning" ran to three or four lines in a
          column this narrow, and it spent them on the matcher's bookkeeping in
          the space the MEANING wanted — which is the one thing a person reading
          a name actually needs. It was also saying it twice: the agent gives a
          real reason, in prose, in its own voice, in the column beside the card.
          `reasons` still exists and still orders the pool; it stopped being
          furniture. */}
      {/* The name rides in an .sr-only span rather than in aria-label so the
          visible word stays the start of the accessible name (WCAG 2.5.3).
          Without it a screen reader meets 72 buttons called "Pick" and 72
          disclosures called "From the document", with no way to tell which
          name any of them belongs to — the <article> carries no name either. */}
      <div className="nm-card-foot">
        {pickable && (
          <button
            type="button"
            className="nm-pick label-mono label-mono--sm"
            aria-pressed={picked}
            disabled={!onPick || (trayFull && !picked)}
            data-nm-pick={onPick ? undefined : row.id}
            onClick={onPick}
          >
            <span data-nm-pick-word>{picked ? C.picked : C.pick}</span>
            <span className="sr-only"> {primary}</span>
          </button>
        )}

        {/* THE "FROM THE DOCUMENT" DISCLOSURE IS GONE. It carried the verbatim
            source line, the corpus labels and a page number under every card,
            and it was the most machinery-looking thing on a page about naming a
            child — a footnote apparatus on a card someone is choosing between.

            The provenance is not weakened by removing it: the names, the
            spellings, the Devanagari and the meanings still come from the
            document and nowhere else, and the matcher still cannot surface a
            name that is not in it. What has gone is the CITATION, not the
            sourcing. The meaning on the card is the document's own line. */}
      </div>
    </article>
  )
}
