import { useEffect, useRef, useState } from 'react'

/**
 * VibeFinder — the Curation panel's live demo (docs/design/DESIGN.md §6).
 * Type an artist (real typeahead against the production catalog), add mood /
 * BPM chips, get tracks back with an honest latency badge: LIVE when the
 * production API answered, REPLAY when fixtures did. Never renders blank.
 */

type Suggestion = { display: string; type: string; artist?: string; track?: string; genre?: string; tempo?: number }
type Track = { track: string; artist: string; genre: string; mood: string; energy: string; tempo: number | null; year: number | null }
type SearchMeta = { ms: number | null; source: 'live' | 'replay' }

const MOODS = ['Atmospheric', 'Euphoric', 'Energetic', 'Introspective']
const TEMPOS: Array<{ label: string; min: number; max: number }> = [
  { label: '118–124', min: 118, max: 124 },
  { label: '124–128', min: 124, max: 128 },
  { label: '128–134', min: 128, max: 134 },
]

export default function VibeFinder() {
  const [query, setQuery] = useState('')
  const [suggestions, setSuggestions] = useState<Suggestion[]>([])
  const [sugOpen, setSugOpen] = useState(false)
  const [sugActive, setSugActive] = useState(-1)
  const [genres, setGenres] = useState<string[]>([])
  const [moods, setMoods] = useState<string[]>([])
  const [tempo, setTempo] = useState<number>(-1) // index into TEMPOS
  const [tracks, setTracks] = useState<Track[] | null>(null)
  const [meta, setMeta] = useState<SearchMeta | null>(null)
  const [status, setStatus] = useState<'idle' | 'loading' | 'done'>('idle')
  const debounceRef = useRef<ReturnType<typeof setTimeout>>(null)
  const boxRef = useRef<HTMLDivElement>(null)

  // Typeahead — debounced against the production autocomplete index
  useEffect(() => {
    if (debounceRef.current) clearTimeout(debounceRef.current)
    const q = query.trim()
    if (q.length < 2) {
      setSuggestions([])
      setSugOpen(false)
      return
    }
    debounceRef.current = setTimeout(async () => {
      try {
        const res = await fetch('/api/vibeset-demo', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ kind: 'autocomplete', query: q }),
        })
        const data = await res.json()
        setSuggestions(data.suggestions ?? [])
        setSugOpen((data.suggestions ?? []).length > 0)
        setSugActive(-1)
      } catch {
        setSuggestions([])
        setSugOpen(false)
      }
    }, 200)
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current)
    }
  }, [query])

  useEffect(() => {
    const close = (e: MouseEvent) => {
      if (boxRef.current && !boxRef.current.contains(e.target as Node)) setSugOpen(false)
    }
    document.addEventListener('mousedown', close)
    return () => document.removeEventListener('mousedown', close)
  }, [])

  const pickSuggestion = (s: Suggestion) => {
    if (s.type === 'genre' && s.genre) {
      setGenres((g) => (g.includes(s.genre!) ? g : [...g, s.genre!]))
      setQuery('')
    } else {
      setQuery(s.artist ?? s.display)
    }
    setSugOpen(false)
  }

  const toggle = (list: string[], set: (v: string[]) => void, item: string) =>
    set(list.includes(item) ? list.filter((x) => x !== item) : [...list, item])

  const canSearch = query.trim().length > 0 || genres.length > 0 || moods.length > 0 || tempo >= 0

  const search = async (override?: { genres?: string[]; tempo?: number }) => {
    const g = override?.genres ?? genres
    const t = override?.tempo ?? tempo
    if (!override && (!canSearch || status === 'loading')) return
    if (override && status === 'loading') return
    setStatus('loading')
    setSugOpen(false)
    try {
      const body: Record<string, unknown> = { kind: 'search', limit: 6 }
      if (!override && query.trim()) body.artist = query.trim()
      if (g.length) body.genres = g
      if (moods.length) body.moods = moods.map((m) => m.toLowerCase())
      if (t >= 0) {
        body.tempo_min = TEMPOS[t].min
        body.tempo_max = TEMPOS[t].max
      }
      const res = await fetch('/api/vibeset-demo', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      const data = await res.json()
      setTracks(data.tracks ?? [])
      setMeta({ ms: data.response_time_ms ?? null, source: data.source === 'live' ? 'live' : 'replay' })
    } catch {
      // the proxy already fixtures every failure; this is belt-and-braces
      setTracks([])
      setMeta({ ms: null, source: 'replay' })
    }
    setStatus('done')
  }

  const onKeyDown = (e: React.KeyboardEvent) => {
    if (sugOpen && suggestions.length) {
      if (e.key === 'ArrowDown') {
        e.preventDefault()
        setSugActive((i) => (i + 1) % suggestions.length)
        return
      }
      if (e.key === 'ArrowUp') {
        e.preventDefault()
        setSugActive((i) => (i <= 0 ? suggestions.length - 1 : i - 1))
        return
      }
      if (e.key === 'Escape') {
        setSugOpen(false)
        return
      }
      if (e.key === 'Enter' && sugActive >= 0) {
        e.preventDefault()
        pickSuggestion(suggestions[sugActive])
        return
      }
    }
    if (e.key === 'Enter') {
      e.preventDefault()
      search()
    }
  }

  const chip = (active: boolean) =>
    `vf-chip btn ${active ? 'vf-chip-on' : ''}`

  // One-tap example — the first success should cost one click (DESIGN.md §6)
  const runExample = () => {
    setGenres(['melodic techno'])
    setTempo(1)
    void search({ genres: ['melodic techno'], tempo: 1 })
  }

  return (
    <div className="vf" ref={boxRef}>
      <div className="vf-input-row">
        <div className="vf-input-wrap">
          <input
            type="text"
            className="vf-input"
            placeholder="Type an artist — try “Fred” or “Bicep”"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={onKeyDown}
            role="combobox"
            aria-expanded={sugOpen}
            aria-controls="vf-suggestions"
            aria-label="Search the catalog by artist"
            autoComplete="off"
          />
          {sugOpen && (
            <ul className="vf-suggestions" id="vf-suggestions" role="listbox">
              {suggestions.map((s, i) => (
                <li key={`${s.display}-${i}`} role="option" aria-selected={i === sugActive}>
                  <button
                    type="button"
                    className={`vf-sug ${i === sugActive ? 'vf-sug-active' : ''}`}
                    onMouseDown={(e) => {
                      e.preventDefault()
                      pickSuggestion(s)
                    }}
                  >
                    <span className="vf-sug-name">{s.display}</span>
                    <span className="vf-sug-meta label-mono">
                      {s.type}
                      {s.tempo ? ` · ${s.tempo} bpm` : ''}
                    </span>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>
        <button type="button" className="vf-go btn" onClick={() => search()} disabled={!canSearch || status === 'loading'}>
          Find tracks
        </button>
      </div>

      <div className="vf-chips" aria-label="Mood filters">
        {MOODS.map((m) => (
          <button key={m} type="button" className={chip(moods.includes(m))} onClick={() => toggle(moods, setMoods, m)} aria-pressed={moods.includes(m)}>
            {m}
          </button>
        ))}
        <span className="vf-chip-divider" aria-hidden="true"></span>
        {TEMPOS.map((t, i) => (
          <button key={t.label} type="button" className={chip(tempo === i)} onClick={() => setTempo(tempo === i ? -1 : i)} aria-pressed={tempo === i}>
            {t.label} bpm
          </button>
        ))}
        {genres.map((g) => (
          <button key={g} type="button" className="vf-chip vf-chip-on btn" onClick={() => toggle(genres, setGenres, g)} aria-pressed="true">
            {g} ✕
          </button>
        ))}
      </div>

      <div className="vf-results" aria-live="polite">
        {status === 'idle' && (
          <button type="button" className="vf-example btn" onClick={runExample}>
            ▶ try it — melodic techno at 124–128 bpm
          </button>
        )}

        {status === 'loading' && (
          <div className="vf-loading">
            <div className="pulse-line" role="status" aria-label="Searching the catalog"></div>
            <div className="pulse-caption">Searching the catalog…</div>
          </div>
        )}

        {status === 'done' && meta && (
          <div className="vf-meta">
            {meta.source === 'live' ? (
              <span className="vf-badge vf-badge-live label-mono">
                <span className="live-dot"></span>
                {meta.ms != null ? `${meta.ms} ms · ` : ''}live
              </span>
            ) : (
              <span className="vf-badge vf-badge-replay label-mono">replay — the API was napping; these are cached results</span>
            )}
          </div>
        )}

        {status === 'done' && tracks && tracks.length === 0 && (
          <p className="vf-empty">Nothing matched that combination — loosen a filter and try again.</p>
        )}

        {status === 'done' && tracks && tracks.length > 0 && (
          <ol className="vf-tracks">
            {tracks.map((t, i) => (
              <li key={`${t.track}-${i}`} className="vf-track">
                <div className="vf-track-main">
                  <span className="vf-track-name">{t.track}</span>
                  <span className="vf-track-artist">{t.artist}</span>
                </div>
                <div className="vf-track-meta label-mono">
                  {[t.genre, t.mood].filter(Boolean).join(' · ')}
                  {t.tempo ? ` · ${t.tempo} bpm` : ''}
                </div>
              </li>
            ))}
          </ol>
        )}
      </div>
    </div>
  )
}
