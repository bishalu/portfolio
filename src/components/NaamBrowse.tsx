import { useEffect, useMemo, useRef, useState, useSyncExternalStore } from 'react'
import NaamCard from './NaamCard'
import { NAAM_COPY } from '@/lib/naam/copy'
import { search } from '@/lib/naam/match'
import {
  getDefaultPreferB,
  getEmptyPicks,
  getPicks,
  getPreferB,
  hasAllRows,
  hydrate,
  loadAllRows,
  loadCoreRows,
  onIdle,
  subscribe,
  togglePick,
  toggleSwap,
  PICK_MAX,
} from '@/lib/naam/tray'
import { naamThemeBit, type NaamLetter, type NaamRow, type NaamSource, type NaamTheme } from '@/types/naam'

/**
 * The whole collection, filterable (docs/design/DESIGN.md §2, P8, P10).
 *
 * WHY it is capped: there are 2,098 rows in the first payload and 6,715 in
 * total, and a page that renders them all is a page nobody can use — axe has to
 * walk it, the verify harness scrolls it in 0.7-viewport steps, and
 * virtualisation fights both. So: 50 rows at a time behind an explicit "show 50
 * more", the other 4,617 rows behind an explicit "load the rest", and the list
 * fetched in requestIdleCallback rather than on the critical path. The parsed
 * array is cached in module scope by src/lib/naam/tray.ts, so a
 * back-navigation costs nothing.
 *
 * WHY it owns no scoring: ordering and search are src/lib/naam/match.ts's job
 * (`search()` buckets exact > prefix > substring > meaning). This component
 * filters and paginates and nothing else, so the browse list and the guided
 * list can never disagree about what a row is.
 *
 * Honesty: this is LOCAL by construction — the filtering happens in the
 * visitor's browser over the document's own fields. The page's one .nm-badge
 * lives in NaamGuide, which is the surface that can be either of the two
 * words; here the same fact is a mono caption, because §4 says a badge states
 * what the visitor is looking at, not what the system can do elsewhere.
 *
 * Teardown: one AbortController per mount, aborted on `astro:before-swap`.
 */

const C = NAAM_COPY
const PAGE = 50

type FlatOption = { value: unknown; label: string; note?: string }

/** Option labels live in the guided questions; read them, never retype them. */
function optionsFor(key: string): readonly FlatOption[] {
  const question = C.guide.questions.find((q) => q.key === key)
  return (question?.options ?? []) as readonly FlatOption[]
}

const LETTERS = optionsFor('letters')
const SYLLABLES = optionsFor('syllables')
const SOURCES = optionsFor('sources')
const THEMES = optionsFor('themes')

function toggle<T>(list: readonly T[], value: T): T[] {
  return list.includes(value) ? list.filter((v) => v !== value) : [...list, value]
}

export default function NaamBrowse() {
  const [mounted, setMounted] = useState(false)
  const [rows, setRows] = useState<NaamRow[] | null>(null)
  const [dataFailed, setDataFailed] = useState(false)
  const [loadingRest, setLoadingRest] = useState(false)
  const [query, setQuery] = useState('')
  const [letters, setLetters] = useState<readonly NaamLetter[]>([])
  const [syllables, setSyllables] = useState<readonly number[]>([])
  const [sources, setSources] = useState<readonly NaamSource[]>([])
  const [attested, setAttested] = useState(false)
  const [evocative, setEvocative] = useState(false)
  const [theme, setTheme] = useState('')
  const [shown, setShown] = useState(PAGE)
  const mountRef = useRef<AbortController | null>(null)

  const picks = useSyncExternalStore(subscribe, getPicks, getEmptyPicks)
  const preferB = useSyncExternalStore(subscribe, getPreferB, getDefaultPreferB)
  const pickedIds = useMemo(() => new Set(picks.map((p) => p.id)), [picks])

  useEffect(() => {
    const ac = new AbortController()
    mountRef.current = ac
    setMounted(true)
    hydrate()
    const cancel = onIdle(() => {
      loadCoreRows()
        .then((loaded) => {
          if (!ac.signal.aborted) setRows(loaded)
        })
        .catch(() => {
          if (!ac.signal.aborted) setDataFailed(true)
        })
    })
    document.addEventListener('astro:before-swap', () => ac.abort(), { signal: ac.signal })
    // See NaamGuide: React's cleanup runs at astro:after-swap, so the idle
    // callback is also cancelled on abort rather than only on unmount.
    ac.signal.addEventListener('abort', cancel)
    return () => {
      cancel()
      ac.abort()
    }
  }, [])

  useEffect(() => {
    setShown(PAGE)
  }, [query, letters, syllables, sources, attested, evocative, theme, rows])

  const results = useMemo(() => {
    const base = rows ?? []
    const kept = base.filter((row) => {
      if (letters.length > 0 && !letters.includes(row.letter)) return false
      if (syllables.length > 0 && !syllables.includes(row.syllables)) return false
      if (sources.length > 0 && !row.sources.some((s) => sources.includes(s))) return false
      if (attested && !row.badges.attested) return false
      if (evocative && !row.badges.evocative) return false
      if (theme && (row.themeMask & naamThemeBit(theme as NaamTheme)) === 0) return false
      return true
    })
    return search(kept, query, Math.max(kept.length, 1))
  }, [rows, query, letters, syllables, sources, attested, evocative, theme])

  const loadRest = () => {
    const ac = mountRef.current
    if (!ac || loadingRest || hasAllRows()) return
    setLoadingRest(true)
    loadAllRows()
      .then((all) => {
        if (!ac.signal.aborted) setRows(all)
      })
      .catch(() => {
        if (!ac.signal.aborted) setDataFailed(true)
      })
      .finally(() => {
        if (!ac.signal.aborted) setLoadingRest(false)
      })
  }

  const clear = () => {
    setQuery('')
    setLetters([])
    setSyllables([])
    setSources([])
    setAttested(false)
    setEvocative(false)
    setTheme('')
  }

  const ready = rows !== null
  const visible = results.slice(0, shown)
  const countLabel = ready ? C.browse.count(visible.length, results.length) : ''

  /**
   * The count is a live region, and it recomputed on every keystroke — typing
   * "lotus" queued five announcements over a screen reader. The visible number
   * still updates instantly; only the announcement waits for a pause.
   */
  const [announced, setAnnounced] = useState('')
  useEffect(() => {
    const handle = window.setTimeout(() => setAnnounced(countLabel), 400)
    return () => window.clearTimeout(handle)
  }, [countLabel])

  return (
    <div className="nm-browse">
      <div className="nm-filters">
        <div className="nm-filter-row">
          <label className="label-mono nm-filter-label" htmlFor="nm-search">
            {C.browse.searchLabel}
          </label>
          <input
            id="nm-search"
            className="nm-search"
            type="search"
            value={query}
            placeholder={C.browse.searchPlaceholder}
            autoComplete="off"
            maxLength={64}
            disabled={!ready}
            onChange={(e) => setQuery(e.target.value)}
          />
        </div>

        <fieldset className="nm-filter-group">
          <legend className="label-mono label-mono--sm">{C.browse.filterLetter}</legend>
          {LETTERS.map((option) => (
            <button
              key={String(option.value)}
              type="button"
              className="nm-chip label-mono label-mono--sm"
              aria-pressed={letters.includes(option.value as NaamLetter)}
              disabled={!ready}
              onClick={() => setLetters((list) => toggle(list, option.value as NaamLetter))}
            >
              {option.label}
            </button>
          ))}
        </fieldset>

        <fieldset className="nm-filter-group">
          <legend className="label-mono label-mono--sm">{C.browse.filterSyllables}</legend>
          {SYLLABLES.map((option) => (
            <button
              key={String(option.value)}
              type="button"
              className="nm-chip label-mono label-mono--sm"
              aria-pressed={syllables.includes(option.value as number)}
              disabled={!ready}
              onClick={() => setSyllables((list) => toggle(list, option.value as number))}
            >
              {String(option.value)}
            </button>
          ))}
        </fieldset>

        <fieldset className="nm-filter-group">
          <legend className="label-mono label-mono--sm">{C.browse.filterSource}</legend>
          {SOURCES.map((option) => (
            <button
              key={String(option.value)}
              type="button"
              className="nm-chip label-mono label-mono--sm"
              aria-pressed={sources.includes(option.value as NaamSource)}
              disabled={!ready}
              onClick={() => setSources((list) => toggle(list, option.value as NaamSource))}
            >
              {option.label}
            </button>
          ))}
        </fieldset>

        <fieldset className="nm-filter-group">
          <legend className="label-mono label-mono--sm">{C.browse.filterKind}</legend>
          <button
            type="button"
            className="nm-chip label-mono label-mono--sm"
            aria-pressed={attested}
            disabled={!ready}
            onClick={() => setAttested((v) => !v)}
          >
            {C.card.attested}
          </button>
          <button
            type="button"
            className="nm-chip label-mono label-mono--sm"
            aria-pressed={evocative}
            disabled={!ready}
            onClick={() => setEvocative((v) => !v)}
          >
            {C.card.evocative}
          </button>
        </fieldset>

        <div className="nm-filter-row">
          <label className="label-mono nm-filter-label" htmlFor="nm-theme">
            {C.browse.filterTheme}
          </label>
          <select
            id="nm-theme"
            className="nm-select"
            value={theme}
            disabled={!ready}
            onChange={(e) => setTheme(e.target.value)}
          >
            <option value="">{C.guide.skip}</option>
            {THEMES.map((option) => (
              <option key={String(option.value)} value={String(option.value)}>
                {option.label}
              </option>
            ))}
          </select>
          <button type="button" className="nm-quiet" disabled={!ready} onClick={clear}>
            {C.browse.clear}
          </button>
        </div>
      </div>

      {/* The visible count is instant; the announcement is the debounced one,
          carried in the same always-present region so nothing is created with
          its content already in it. */}
      <p className="nm-browse-status label-mono" aria-live="polite">
        <span aria-hidden="true">{countLabel}</span>
        <span className="sr-only">{loadingRest ? C.browse.loading : announced}</span>
      </p>
      <p className="nm-browse-note">{C.badge.localCaption}</p>

      {mounted && !ready && !dataFailed && (
        <div className="nm-loading" aria-hidden="true">
          <div className="pulse-line on-ink"></div>
          <p className="pulse-caption">{C.results.loading}</p>
        </div>
      )}
      {dataFailed && <p className="nm-note">{C.failure.dataDown}</p>}

      {ready && results.length === 0 && <p className="nm-note">{C.browse.empty}</p>}

      {visible.length > 0 && (
        <div className="nm-grid">
          {visible.map((row) => (
            <NaamCard
              key={row.id}
              row={row}
              preferB={preferB}
              picked={pickedIds.has(row.id)}
              trayFull={picks.length >= PICK_MAX && !pickedIds.has(row.id)}
              onSwap={toggleSwap}
              onPick={() => togglePick(row)}
            />
          ))}
        </div>
      )}

      <div className="nm-browse-foot">
        {results.length > visible.length && (
          <button type="button" className="nm-next label-mono label-mono--sm" onClick={() => setShown((n) => n + PAGE)}>
            {C.results.more}
          </button>
        )}
        {!hasAllRows() && (
          <button type="button" className="nm-quiet" disabled={!ready || loadingRest} onClick={loadRest}>
            {C.browse.loadRest}
          </button>
        )}
        {loadingRest && (
          <div className="nm-loading" aria-hidden="true">
            <div className="pulse-line on-ink"></div>
            <p className="pulse-caption">{C.browse.loading}</p>
          </div>
        )}
      </div>
    </div>
  )
}
