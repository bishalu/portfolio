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
import { Fragment } from 'react'
import { NAAM_COPY } from '@/lib/naam/copy'
import { naamPreferredDevanagari, naamPreferredForm, type NaamRow } from '@/types/naam'

const C = NAAM_COPY.card

export interface NaamCardProps {
  row: NaamRow
  /** The page-wide व/ब display preference. See src/lib/naam/tray.ts. */
  preferB: boolean
  picked?: boolean
  /** Absent = server-rendered; the page script wires it after boot. */
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
   * The name, cut where the syllables cut it — SLICED FROM THE SPELLING ON THE
   * CARD, not assembled from the split.
   *
   * `syllableSplit` is lowercase, so printing it directly turned Varisha into
   * "va·ri·sha": the break was right and the name was wrong, on a page whose
   * whole subject is how a name is written. Slicing the displayed spelling at
   * the split's own offsets keeps the capital and works for the B-form too,
   * which differs from the V-form only in its first letter.
   *
   * If the two ever disagree in length the name is printed whole, without
   * breaks. A missing hyphenation is a small loss; a mangled name is not.
   */
  const broken = (() => {
    const parts = row.syllableSplit
    const total = parts.reduce((n, part) => n + part.length, 0)
    if (total !== primary.length) return [{ text: primary, cluster: false }]
    let at = 0
    return parts.map((part) => {
      const text = primary.slice(at, at + part.length)
      at += part.length
      return { text, cluster: marked && opensOnCluster(part) }
    })
  })()

  return (
    /* NO HANDLER ON THE ARTICLE ANY MORE. It used to carry an onClick so a
       pointer could land anywhere on the card while the keyboard used the
       button inside — two paths to one action, and an eslint suppression to
       allow it. The button is now stretched over the whole card, so the
       pointer and the keyboard reach the same control and the article is back
       to being what it says it is. */
    <article className="nm-card" data-nm-card={row.id} data-picked={picked ? 'true' : undefined}>
      {/* ── ONE LINE FOR THE NAME, ONE FOR THE SOUND ─────────────────────────
          The card carried the Devanagari on its own line, the Latin under it,
          and then a whole separate 32px instrument — a rule with a dot and an
          uppercase label per syllable — to say how the name breaks. Three rows
          to print two facts, on the object there are nine of.

          The break is now IN the name: "Bar·dhi". It is the same information,
          it is read where the reader is already looking, and the interpunct is
          the typographic convention for exactly this. The cluster mark rides
          the separator that opens it, so the document's own `!` survives as a
          coloured dot rather than as a coloured row. */}
      <span className="nm-deva" data-nm-deva lang="sa-Deva">
        {devanagari}
      </span>
      <p className="nm-latin">
        <span className="nm-latin-main">
          {broken.map((part, i) => (
            <Fragment key={`${row.id}-${i}`}>
              {i > 0 && (
                <span className="nm-syl" aria-hidden="true" data-cluster={part.cluster ? 'true' : undefined}>
                  ·
                </span>
              )}
              {part.text}
            </Fragment>
          ))}
        </span>
        <span className="nm-latin-alt label-mono">{alternate ?? '\u00a0'}</span>
      </p>

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
      {/* ── THE CONTROL IS THE CARD ────────────────────────────────────────
          KEEP was a 59x44 pill on its own row — 44px of the card's height, and
          the widest thing on it, to say the one thing the whole card is for.

          The button is still a real button and still the keyboard path, but it
          is stretched over the entire card and its visible mark is a small ring
          in the corner. The tap target went UP (the whole card, not a pill in
          the bottom-left), the height came down by a row, and what is left is a
          mark that says "choosable" without spending a line saying it.

          `inset: 0` means the article no longer needs its own pointer handler:
          one control, one hit area, the same for a mouse and a keyboard.

          The name still rides in an .sr-only span rather than in aria-label so
          the visible word starts the accessible name (WCAG 2.5.3) — without it
          a screen reader meets 72 buttons called "Keep" with no way to tell
          which name any of them belongs to. */}
      {pickable && (
        <button
          type="button"
          className="nm-pick"
          aria-pressed={picked}
          disabled={!onPick || (trayFull && !picked)}
          data-nm-pick={onPick ? undefined : row.id}
          onClick={onPick}
        >
          {/* THE VERB IS ON THE PAGE. This was an empty ring and the word lived
              only in the .sr-only span below it, so no sighted visitor ever saw
              the name of the page's primary action — a card-shrinking pass took
              the label off the one control that needed it most. Same corner,
              same footprint, now carrying its own word. */}
          <span className="nm-pick-mark" aria-hidden="true">{picked ? C.picked : C.pick}</span>
          <span className="sr-only">
            {picked ? C.picked : C.pick} {primary}
          </span>
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
    </article>
  )
}
