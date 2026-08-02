import { useEffect, useMemo, useRef, useState } from 'react'

/**
 * CueSync — the Cue panel's demo (docs/design/DESIGN.md §4).
 *
 * Every number here came out of Cue's production analyzer, run offline:
 * cut times from the shot detector, the beat grid from madmom, and the ranked
 * offsets from beat_sync.score_offsets. Only the *choice* of offset is replayed
 * — picking one recomputes cut→beat coverage here, in the browser, using the
 * same tolerance and hit floors as apps/web/src/lib/sync-fit.ts. That is why
 * the badge says LOCAL and not LIVE: real algorithm, real data, no network.
 *
 * Deliberately not a waveform. A waveform is the reflexive choice for anything
 * audio and it shows loudness, which is not what is in dispute. What is in
 * dispute is whether the cuts land on the beat — so the picture is two lanes
 * on one time axis, and a connector wherever they coincide.
 */

type Song = {
  /* No src: the track is shipped as one pre-cut segment per candidate. */
  duration: number
  tempo: number
  beats: number[]
  beat_weights: number[]
  downbeats: number[]
  downbeat_weights: number[]
  phrases: number[]
}
type Video = { src: string; duration: number; fps: number; cut_times: number[]; cut_strength: number }
type Alignment = { offset_seconds: number; correlation: number; label: string }
/** A shown candidate, with the track already cut to start at its offset. */
type Pick = Alignment & { audio: string }
type Fixture = {
  analyzer: { sync_profile_version: number; grid_provider: string; confidence: string; tempo: number }
  video: Video
  song: Song
  alignments: Alignment[]
  picks: Pick[]
}

/* Ported verbatim from apps/web/src/lib/sync-fit.ts — change these only when
   the analyzer's own constants change. */
const ON_BEAT_TOLERANCE_SECONDS = 0.1
const HIT_FLOOR = { beat: 0.3, downbeat: 0.6, phrase: 0.85 } as const

function nearestDistance(sorted: number[], t: number): number {
  if (!sorted.length) return Infinity
  let lo = 0
  let hi = sorted.length - 1
  if (t <= sorted[0]) return sorted[0] - t
  if (t >= sorted[hi]) return t - sorted[hi]
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1
    if (sorted[mid] > t) hi = mid
    else lo = mid
  }
  return Math.min(Math.abs(sorted[lo] - t), Math.abs(sorted[hi] - t))
}

function nearestIndex(sorted: number[], t: number): number {
  const n = sorted.length
  if (!n) return -1
  if (t <= sorted[0]) return 0
  if (t >= sorted[n - 1]) return n - 1
  let lo = 0
  let hi = n - 1
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1
    if (sorted[mid] > t) hi = mid
    else lo = mid
  }
  return Math.abs(sorted[lo] - t) <= Math.abs(sorted[hi] - t) ? lo : hi
}

/** How many cuts land on a beat at this offset, and how strongly each reads. */
function cutsOnBeat(song: Song, video: Video, offset: number) {
  const cuts = video.cut_times
  const strength: number[] = []
  const hits: boolean[] = []
  let count = 0
  for (let i = 0; i < cuts.length; i += 1) {
    const songTime = cuts[i] + offset
    const bi = nearestIndex(song.beats, songTime)
    if (bi < 0 || Math.abs(song.beats[bi] - songTime) > ON_BEAT_TOLERANCE_SECONDS) {
      strength.push(0)
      hits.push(false)
      continue
    }
    // The biggest structure a cut lands on sets the floor; per-beat salience can raise it.
    const floor =
      nearestDistance(song.phrases, songTime) <= ON_BEAT_TOLERANCE_SECONDS
        ? HIT_FLOOR.phrase
        : nearestDistance(song.downbeats, songTime) <= ON_BEAT_TOLERANCE_SECONDS
          ? HIT_FLOOR.downbeat
          : HIT_FLOOR.beat
    strength.push(Math.max(floor, Math.min(1, Math.max(0, song.beat_weights[bi] ?? 0))))
    hits.push(true)
    count += 1
  }
  return { strength, hits, count, total: cuts.length }
}

/** Resolve once the element knows its duration, so a seek can actually land. */
function ensureMetadata(el: HTMLMediaElement): Promise<void> {
  if (el.readyState >= 1) return Promise.resolve()
  return new Promise((resolve) => {
    const done = () => {
      el.removeEventListener('loadedmetadata', done)
      resolve()
    }
    el.addEventListener('loadedmetadata', done)
    el.load()
  })
}

function withTimeout<T>(p: Promise<T>, ms: number): Promise<T> {
  return Promise.race([p, new Promise<T>((_, rej) => setTimeout(() => rej(new Error('timeout')), ms))])
}

const fmt = (s: number) => {
  const m = Math.floor(s / 60)
  const r = Math.floor(s % 60)
  return `${m}:${String(r).padStart(2, '0')}`
}

export default function CueSync({ data }: { data: Fixture }) {
  const { song, video, alignments, analyzer, picks } = data

  const [pick, setPick] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [head, setHead] = useState(0)
  const videoRef = useRef<HTMLVideoElement>(null)
  const audioRef = useRef<HTMLAudioElement>(null)
  const rafRef = useRef<number>(0)

  const offset = picks[pick]?.offset_seconds ?? 0
  const fit = useMemo(() => cutsOnBeat(song, video, offset), [song, video, offset])

  // Beats inside the clip's window, in clip time.
  const grid = useMemo(() => {
    const out: Array<{ t: number; down: boolean }> = []
    const downSet = new Set(song.downbeats.map((d) => Math.round(d * 1000)))
    for (const b of song.beats) {
      const t = b - offset
      if (t < -0.05 || t > video.duration + 0.05) continue
      out.push({ t, down: downSet.has(Math.round(b * 1000)) })
    }
    return out
  }, [song, offset, video.duration])

  const stop = () => {
    cancelAnimationFrame(rafRef.current)
    videoRef.current?.pause()
    audioRef.current?.pause()
    setPlaying(false)
  }

  // Reset to the top whenever the offset changes, so each comparison starts level.
  useEffect(() => {
    stop()
    setHead(0)
    if (videoRef.current) videoRef.current.currentTime = 0
  }, [pick])

  useEffect(() => () => cancelAnimationFrame(rafRef.current), [])

  const toggle = async () => {
    const v = videoRef.current
    const a = audioRef.current
    if (!v || !a) return
    if (playing) {
      stop()
      return
    }
    v.currentTime = 0
    a.currentTime = 0
    try {
      // No seeking: each candidate ships its own segment of the track, already
      // cut to start at that offset. Seeking a long file needs HTTP Range, and
      // when the host doesn't serve ranges the element reports seekable.end = 0
      // and silently plays from the top — which would make the offset a lie.
      // Three short files remove the dependency and cost a quarter of the bytes.
      await withTimeout(ensureMetadata(a), 4000)
      await Promise.all([v.play(), a.play()])
    } catch {
      // Autoplay policy, a decode failure, or a stall: stay still rather than
      // half-playing out of sync.
      stop()
      return
    }
    setPlaying(true)
    const tick = () => {
      const t = v.currentTime
      setHead(t)
      if (t >= video.duration - 0.05) {
        stop()
        setHead(0)
        return
      }
      rafRef.current = requestAnimationFrame(tick)
    }
    rafRef.current = requestAnimationFrame(tick)
  }

  const pct = (t: number) => `${(t / video.duration) * 100}%`

  return (
    <div className="cx">
      <p className="cx-head label-mono">
        <span className="cx-badge">local</span>
        {/* One expression, not several: adjacent JSX text nodes get collapsed by
            the HTML minifier and then fail hydration. */}
        <span className="cx-prov">
          {`${analyzer.grid_provider} grid · ${analyzer.tempo} bpm · v${analyzer.sync_profile_version}`}
        </span>
      </p>

      <div className="cx-stage">
        <video
          ref={videoRef}
          className="cx-video"
          src={video.src}
          muted
          playsInline
          preload="metadata"
          aria-label="Fifteen-second clip with seven hard cuts"
        />
        {/* Keyed on the pick so React swaps the element rather than mutating
            src on a playing node. */}
        <audio key={picks[pick]?.audio} ref={audioRef} src={picks[pick]?.audio} preload="metadata" />
      </div>

      {/* The lanes are a picture of the numbers stated in cx-count below, which
          is the accessible version of the same fact. */}
      <div className="cx-lanes" aria-hidden="true">
        <div className="cx-lane cx-lane-beats">
          {grid.map((g, i) => (
            <span
              key={i}
              className={`cx-beat ${g.down ? 'cx-beat-down' : ''}`}
              style={{ left: pct(g.t) }}
            />
          ))}
        </div>

        <div className="cx-lane cx-lane-cuts">
          {video.cut_times.map((t, i) => (
            <span
              key={i}
              className={`cx-cut ${fit.hits[i] ? 'cx-cut-on' : ''}`}
              style={{ left: pct(t), ['--s' as string]: fit.strength[i] || 0 }}
            />
          ))}
          <span className="cx-head-line" style={{ left: pct(head) }} />
        </div>
      </div>

      {/* The headline is the fit, because the fit is what actually moves. This
          track is metronomic, so cut→beat coverage is identical at every
          beat-snapped offset (verified: XXX.X.. at all six). What the ranker is
          choosing between is not "more cuts on the beat" — it is where in the
          song those cuts mean something. Leading with a coverage count that
          never changes would be inventing a signal the choice wasn't made on. */}
      <p className="cx-score">
        <span className="cx-score-val">{picks[pick]?.correlation.toFixed(2)}</span>
        <span className="cx-score-meta">
          <span className="cx-score-label label-mono">fit</span>
          <span className="cx-score-where">
            {picks[pick]?.label ? `opens on the ${picks[pick].label}` : 'mid-song'}
          </span>
        </span>
      </p>

      <p className="cx-count label-mono">
        {`${fit.count} of ${fit.total} cuts on a beat`}
      </p>

      <div className="cx-controls">
        <button type="button" className="cx-play" onClick={toggle} aria-label={playing ? 'Stop' : 'Play the clip against the track'}>
          {playing ? '■ Stop' : '▶ Play'}
        </button>

        <div className="cx-picks" role="group" aria-label="Candidate offsets from the ranker">
          {picks.map((p, i) => (
            <button
              key={p.offset_seconds}
              type="button"
              className={`cx-pick ${i === pick ? 'cx-pick-on' : ''}`}
              aria-pressed={i === pick}
              onClick={() => setPick(i)}
            >
              {`${fmt(p.offset_seconds)}${p.label ? ` · ${p.label}` : ''}`}
            </button>
          ))}
        </div>
      </div>

      <p className="cx-note">
        {
          'Three of six offsets the ranker picked. The cuts land on a beat wherever you start. What changes is which beats, and what the song is doing there.'
        }
      </p>
    </div>
  )
}
