import { useCallback, useEffect, useMemo, useRef, useState, useSyncExternalStore } from 'react'
import NaamCard from './NaamCard'
import { NAAM_COPY } from '@/lib/naam/copy'
import {
  normalizePrefs,
  parseFreeText,
  pool,
  rankRelaxed,
  scoreName,
  type NaamMatch,
  type Prefs,
} from '@/lib/naam/match'
import {
  getDefaultPreferB,
  getEmptyPicks,
  getPicks,
  getPreferB,
  hydrate,
  loadCoreRows,
  onIdle,
  removePick,
  subscribe,
  togglePick,
  toggleSwap,
  PICK_MAX,
} from '@/lib/naam/tray'
import type { NaamRow } from '@/types/naam'

/**
 * The listening instrument (docs/design/DESIGN.md §4, P8, P9, P11).
 *
 * Two ways in, one engine. Six questions or one sentence both end at
 * src/lib/naam/match.ts, which scores the document's shortlist in the browser
 * and hands back matches with reasons it computed from the document's own
 * fields. That is the whole reason this surface may wear a LOCAL badge (§4):
 * real algorithm, real data, running in your browser.
 *
 * THE MODEL NEVER NAMES A NAME. On the free-form path the matcher also builds
 * a pool of ids; /api/naam-chat hands that pool to Bedrock and drops anything
 * that comes back from outside it. Cards are rendered from the local dataset,
 * never from model text, so a hallucinated name is structurally impossible —
 * and when the model is off, slow or broken the badge simply reads LOCAL and
 * the matcher's own eight names stand. Degradation is LOCAL, never REPLAY:
 * §4 rule 2 says REPLAY means recorded, and a canned reply was never produced
 * by the real system.
 *
 * NOT A CHAT WIDGET (P11). No bubbles, no avatar, no assistant framing. A
 * family asked a question and something answered in one paragraph.
 *
 * Every visible string comes from src/lib/naam/copy.ts and every reason string
 * from match.ts. Nothing here writes prose.
 *
 * Teardown: one AbortController per mount, aborted on `astro:before-swap` and
 * used as the signal for every listener and every fetch, plus a per-request
 * controller so a slow model times out without tearing the island down.
 * Hero.astro re-registers window listeners on every view transition and never
 * removes them; do not copy that.
 *
 * EVERY CONTROL IN HERE GATES ON `notReady`, WHICH IS TRUE ON THE SERVER. This
 * component server-renders inside a prerendered page, and the first cut gated
 * on a `loading` flag that was itself false during SSR (`mounted` starts
 * false) — so the built HTML shipped five enabled buttons and an enabled text
 * input that did nothing at all until hydration, on a page whose whole point is
 * that the no-JS half is real. NaamBrowse gates on `rows !== null`, which is
 * false on the server and stays false until the dataset lands, and that is the
 * correct test. Same component family, same rule.
 *
 * LIVE REGIONS ARE CREATED EMPTY AND FILLED LATER, never created with their
 * content: `.nm-status` is always in the tree and its text changes. A polite
 * region inserted in the same mutation as its text is announced unreliably
 * across NVDA/JAWS/VoiceOver, and on this page it is the only feedback for an
 * operation with a 12s budget. The .pulse-line blocks are therefore decorative
 * (aria-hidden) and the announcement rides in the persistent region.
 */

const C = NAAM_COPY
const QUESTIONS = C.guide.questions
const RESULT_MAX = 8
const POOL_SIZE = 40
const CHAT_TIMEOUT_MS = 12_000

type Phase = 'wizard' | 'results'
type Source = 'local' | 'live'

/**
 * The two badge words this page is allowed (§4). Lowercase in the DOM,
 * uppercased by .label-mono — the verify harness reads textContent.
 */
const BADGE: Record<Source, string> = {
  live: C.badge.live.toLowerCase(),
  local: C.badge.local.toLowerCase(),
}

/** An option, once the question union has been flattened for rendering. */
type FlatOption = { value: unknown; label: string; note?: string }

export default function NaamGuide() {
  const [mounted, setMounted] = useState(false)
  const [rows, setRows] = useState<NaamRow[] | null>(null)
  const [dataFailed, setDataFailed] = useState(false)
  const [step, setStep] = useState(0)
  const [answers, setAnswers] = useState<Partial<Prefs>>({})
  const [phase, setPhase] = useState<Phase>('wizard')
  const [matches, setMatches] = useState<readonly NaamMatch[]>([])
  const [heading, setHeading] = useState<string>(C.results.heading)
  const [emptyNote, setEmptyNote] = useState<string>(C.results.empty)
  const [source, setSource] = useState<Source>('local')
  const [note, setNote] = useState('')
  const [ask, setAsk] = useState('')
  const [asking, setAsking] = useState(false)
  const [reply, setReply] = useState('')
  const mountRef = useRef<AbortController | null>(null)
  const stepRef = useRef<HTMLDivElement | null>(null)
  const resultsRef = useRef<HTMLElement | null>(null)
  const firstPhase = useRef(true)

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
    // React's cleanup runs at astro:after-swap, one tick too late: an idle
    // callback firing in that window would start a fetch on an already-aborted
    // signal. Harmless — every consumer re-checks — but asymmetric with the
    // page script, which cancels on abort.
    ac.signal.addEventListener('abort', cancel)
    return () => {
      cancel()
      ac.abort()
    }
  }, [])

  /**
   * A phase change unmounts the block the focused control was in — "Show me
   * names", "Change an answer" and "Start over" all do it — and focus fell back
   * to <body>, so the next Tab restarted at the site nav (WCAG 2.4.3). Move it
   * to whatever replaced it. Not on first render: this island is below the
   * fold and stealing focus on load would scroll the page out from under the
   * visitor.
   */
  useEffect(() => {
    if (firstPhase.current) {
      firstPhase.current = false
      return
    }
    const target = phase === 'results' ? resultsRef.current : stepRef.current
    target?.focus()
  }, [phase])

  /* — the six questions ————————————————————————————————————————————— */

  const question = QUESTIONS[step]
  const options = question.options as readonly FlatOption[]
  const total = QUESTIONS.length

  const chosen = useCallback(
    (key: string): unknown[] => {
      const value = (answers as Record<string, unknown>)[key]
      if (Array.isArray(value)) return value
      return value === undefined ? [] : [value]
    },
    [answers],
  )

  const choose = (key: string, value: unknown, multiple: boolean, max?: number) => {
    setAnswers((prev) => {
      const next = { ...prev } as Record<string, unknown>
      if (!multiple) {
        next[key] = value
        return next as Partial<Prefs>
      }
      const list = Array.isArray(next[key]) ? [...(next[key] as unknown[])] : []
      const at = list.indexOf(value)
      if (at >= 0) list.splice(at, 1)
      else if (max === undefined || list.length < max) list.push(value)
      next[key] = list
      return next as Partial<Prefs>
    })
  }

  const advance = () => {
    if (step + 1 < total) {
      setStep(step + 1)
      return
    }
    if (!rows) return
    // Never an empty result: letters, syllables and easySay are hard filters
    // and a reasonable set of six answers can intersect to nothing.
    const { matches: found, relaxed } = rankRelaxed(rows, normalizePrefs(answers), RESULT_MAX)
    setMatches(found)
    setHeading(C.results.heading)
    setEmptyNote(C.results.empty)
    setSource('local')
    setReply('')
    setNote(relaxed ? C.results.relaxed : C.badge.localCaption)
    setPhase('results')
  }

  const restart = () => {
    setPhase('wizard')
    setStep(0)
    setAnswers({})
    setMatches([])
    setReply('')
    setNote('')
  }

  /* — one sentence ——————————————————————————————————————————————————— */

  const runAsk = async () => {
    const text = ask.trim()
    const dataset = rows
    if (!text || asking || !dataset) return

    const mount = mountRef.current?.signal
    const parsed = parseFreeText(text, dataset)
    const named = parsed.compare.length > 0 ? parsed.compare : parsed.lookups
    // Same rule as the guided path, and it matters more here: this list is
    // also the LOCAL fallback when the model is off, slow or broken, and the
    // degradation contract is "never an error state and never an empty one".
    const { matches: ranked, relaxed } = rankRelaxed(dataset, parsed.prefs, RESULT_MAX)
    const local = mergeNamed(named, ranked, parsed.prefs)

    setAsking(true)
    setReply('')
    setNote('')
    setSource('local')
    setMatches(local)
    setEmptyNote(C.results.emptyAsk)
    setHeading(
      parsed.compare.length > 0
        ? C.results.compareHeading
        : parsed.lookups.length > 0
          ? C.results.lookupHeading
          : C.results.heading,
    )
    setPhase('results')

    const poolIds = pool(dataset, parsed.prefs, POOL_SIZE).map((row) => row.id)
    const request = new AbortController()
    const onMountAbort = () => request.abort()
    mount?.addEventListener('abort', onMountAbort, { once: true })
    const timer = setTimeout(() => request.abort(), CHAT_TIMEOUT_MS)

    let outcome: 'live' | 'degraded' | 'unreachable' = 'unreachable'
    let reason: unknown
    try {
      const res = await fetch('/api/naam-chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ask: text, poolIds }),
        signal: request.signal,
      })
      const data: unknown = await res.json()
      const body = (data ?? {}) as { degraded?: boolean; reason?: unknown; reply?: unknown; pickIds?: unknown }
      reason = body.reason
      if (body.degraded !== true && typeof body.reply === 'string' && body.reply.trim().length > 0) {
        const byId = new Map(dataset.map((row) => [row.id, row]))
        const picked = (Array.isArray(body.pickIds) ? body.pickIds : [])
          .map((id) => (typeof id === 'string' ? byId.get(id) : undefined))
          .filter((row): row is NaamRow => Boolean(row))
        if (!mount?.aborted) {
          setReply(body.reply.trim())
          setSource('live')
          setNote(C.badge.liveCaption)
          if (picked.length > 0) setMatches(withReasons(picked, parsed.prefs))
        }
        outcome = 'live'
      } else {
        outcome = 'degraded'
      }
    } catch {
      /* aborted, offline, or no endpoint at all — the matcher already answered */
    } finally {
      clearTimeout(timer)
      mount?.removeEventListener('abort', onMountAbort)
    }

    if (mount?.aborted) return
    if (outcome !== 'live') setNote(outcome === 'degraded' ? failureNote(reason) : C.failure.modelDown)
    else if (relaxed) setNote(`${C.badge.liveCaption} ${C.results.relaxed}`)
    setAsking(false)
  }

  /* — render ————————————————————————————————————————————————————————— */

  /**
   * No dataset, no answers — and true on the server, where `rows` is null and
   * always will be. This is the flag every control gates on. See the header.
   */
  const notReady = rows === null
  const loading = mounted && rows === null && !dataFailed
  const busy = loading || asking
  const busyCaption = asking ? C.results.loadingAsk : C.results.loading
  const trayLabel = picks.length > 0 ? C.tray.count(picks.length) : C.tray.empty

  return (
    <div className="nm-guide">
      <div className="nm-panel">
        {phase === 'wizard' ? (
          <div className="nm-step" data-active="true" data-step={step} tabIndex={-1} ref={stepRef}>
            <p className="label-mono label-mono--sm nm-step-count">{C.guide.stepLabel(step + 1, total)}</p>
            <h2 className="nm-panel-title nm-step-label">{question.label}</h2>
            <p className="nm-step-helper">{question.helper}</p>

            <div className="nm-opts">
              {options.map((option) => {
                const picked = chosen(question.key).includes(option.value)
                const capped =
                  question.multiple &&
                  question.max !== undefined &&
                  !picked &&
                  chosen(question.key).length >= question.max
                return (
                  <button
                    key={String(option.value)}
                    type="button"
                    className="nm-opt"
                    aria-pressed={picked}
                    disabled={notReady || capped}
                    onClick={() => choose(question.key, option.value, question.multiple, question.max)}
                  >
                    <span className="nm-opt-label">{option.label}</span>
                    {option.note && <span className="nm-opt-note">{option.note}</span>}
                  </button>
                )
              })}
            </div>

            <div className="nm-step-nav">
              {step > 0 && (
                <button type="button" className="nm-quiet" onClick={() => setStep(step - 1)}>
                  {C.guide.back}
                </button>
              )}
              <button type="button" className="nm-quiet" onClick={advance} disabled={notReady}>
                {C.guide.skip}
              </button>
              <button type="button" className="nm-next label-mono label-mono--sm" onClick={advance} disabled={notReady}>
                {step + 1 < total ? C.guide.next : C.guide.finish}
              </button>
            </div>
          </div>
        ) : (
          <div className="nm-step-nav">
            <button type="button" className="nm-quiet" onClick={() => setPhase('wizard')}>
              {C.guide.change}
            </button>
            <button type="button" className="nm-quiet" onClick={restart}>
              {C.guide.restart}
            </button>
          </div>
        )}

        {loading && (
          <div className="nm-loading" aria-hidden="true">
            <div className="pulse-line on-ink"></div>
            <p className="pulse-caption">{C.results.loading}</p>
          </div>
        )}
        {dataFailed && <p className="nm-note">{C.failure.dataDown}</p>}
      </div>

      <div className="nm-panel nm-ask" id="nm-ask">
        <label className="label-mono nm-ask-label" htmlFor="nm-ask-input">
          {C.ask.label}
        </label>
        <p className="nm-ask-hint">{C.ask.hint}</p>
        <div className="nm-ask-row">
          <input
            id="nm-ask-input"
            className="nm-ask-input"
            type="text"
            value={ask}
            placeholder={C.ask.placeholder}
            autoComplete="off"
            maxLength={400}
            disabled={notReady}
            onChange={(e) => setAsk(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') {
                e.preventDefault()
                void runAsk()
              }
            }}
          />
          <button
            type="button"
            className="btn-pill nm-ask-go"
            onClick={() => void runAsk()}
            disabled={notReady || asking || ask.trim().length === 0}
          >
            {C.ask.send}
          </button>
        </div>
        <ul className="nm-ask-examples">
          {C.ask.examples.map((example) => (
            <li key={example}>
              <button type="button" className="nm-quiet" disabled={notReady} onClick={() => setAsk(example)}>
                {example}
              </button>
            </li>
          ))}
        </ul>
      </div>

      {/* The one persistent live region on this island. It carries the LIVE /
          LOCAL caption, the honest failure lines, and — via .sr-only, so the
          visible caption below is not printed twice — the fact that something
          is happening. */}
      <p className="nm-status" aria-live="polite">
        {busy ? <span className="sr-only">{busyCaption}</span> : note}
      </p>

      {asking && (
        <div className="nm-loading" aria-hidden="true">
          <div className="pulse-line on-ink"></div>
          <p className="pulse-caption">{C.results.loadingAsk}</p>
        </div>
      )}

      <div className="nm-ask-log" role="log">
        {reply && <p className="nm-reply">{reply}</p>}
      </div>

      {phase === 'results' && (
        <section className="nm-results" aria-labelledby="nm-results-head" tabIndex={-1} ref={resultsRef}>
          <div className="nm-results-head">
            <h2 id="nm-results-head" className="nm-panel-title">
              {heading}
            </h2>
            <span className="nm-badge label-mono" data-source={source}>
              {BADGE[source]}
            </span>
          </div>

          {matches.length === 0 ? (
            <p className="nm-note">{emptyNote}</p>
          ) : (
            <div className="nm-grid">
              {matches.map((match) => (
                <NaamCard
                  key={match.row.id}
                  row={match.row}
                  preferB={preferB}
                  reasons={match.reasons}
                  picked={pickedIds.has(match.row.id)}
                  trayFull={picks.length >= PICK_MAX && !pickedIds.has(match.row.id)}
                  onSwap={toggleSwap}
                  onPick={() => togglePick(match.row)}
                />
              ))}
            </div>
          )}
        </section>
      )}

      {/* One sub-scale for every panel heading on this page (.nm-panel-title).
          There were three: the section triple at 52px, two panel headings at
          24.8px, and this one at 12px in uppercase mono — a heading rendered as
          a label. The heading levels are left alone; the scale is not. */}
      <section className="nm-tray" aria-labelledby="nm-tray-head">
        <div className="nm-tray-head">
          <h2 id="nm-tray-head" className="nm-panel-title">
            {C.tray.heading}
          </h2>
          <span className="label-mono label-mono--sm nm-tray-count">{trayLabel}</span>
        </div>
        {picks.length > 0 && (
          <>
            <ul className="nm-tray-list">
              {picks.map((pick) => (
                <li key={pick.id}>
                  <button type="button" className="nm-tray-item" onClick={() => removePick(pick.id)}>
                    <span className="nm-tray-name">{pick.spelling}</span>
                    <span className="label-mono label-mono--sm">{C.card.unpick}</span>
                  </button>
                </li>
              ))}
            </ul>
            <p className="nm-tray-foot">
              <a className="nm-quiet-link" href="#nm-form">
                {C.tray.send}
              </a>
              <span className="label-mono label-mono--sm nm-tray-note">
                {picks.length >= PICK_MAX ? C.tray.full(PICK_MAX) : C.tray.persisted}
              </span>
            </p>
          </>
        )}
      </section>
    </div>
  )
}

/* ────────────────────────────────────────────────────────────────────────────
   helpers — every reason string still comes out of the matcher
   ──────────────────────────────────────────────────────────────────────────── */

function withReasons(rows: readonly NaamRow[], prefs: Prefs): NaamMatch[] {
  return rows.map((row) => {
    const { score, reasons } = scoreName(row, prefs)
    return { row, score: Number.isFinite(score) ? score : 0, reasons }
  })
}

/** A name the visitor asked about outranks anything the wish alone found. */
function mergeNamed(named: readonly NaamRow[], ranked: readonly NaamMatch[], prefs: Prefs): NaamMatch[] {
  if (named.length === 0) return [...ranked]
  const seen = new Set(named.map((row) => row.id))
  return [...withReasons(named, prefs), ...ranked.filter((m) => !seen.has(m.row.id))].slice(0, RESULT_MAX)
}

function failureNote(reason: unknown): string {
  if (reason === 'timeout') return C.failure.modelSlow
  if (reason === 'unset') return C.failure.modelOff
  return C.failure.modelDown
}
