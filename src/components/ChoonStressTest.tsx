import { useEffect, useRef, useState } from 'react'

/**
 * ChoonStressTest — the Choon panel's demo (docs/design/DESIGN.md §6).
 * Entirely client-side: mangle a bundled clip with WebAudio DSP (ported from
 * the real Choon stress-test workbench), watch the spectrum react, then
 * "identify" it. Results are canned per clip+preset and labeled ILLUSTRATIVE —
 * they mirror how the real system behaves (clean audio → classical landmark
 * tier, mangled audio → neural embedding tier), without exposing the private
 * GCP backend. The real thing runs at choon.vibeset.ai.
 */

type PresetKey = 'clean' | 'subway' | 'nightcore' | 'fried'

const CLIPS = [
  { id: 'a', name: 'Neon Night', url: '/choon/clip-a.mp3' },
  { id: 'b', name: 'Late Cut', url: '/choon/clip-b.mp3' },
]

const PRESETS: Record<PresetKey, { label: string; rate: number; drive: number; bits: number; lowpass: number; noise: number }> = {
  clean: { label: 'Clean', rate: 1, drive: 0, bits: 16, lowpass: 20000, noise: 0 },
  subway: { label: 'Subway', rate: 1, drive: 10, bits: 12, lowpass: 2200, noise: 0.14 },
  nightcore: { label: 'Nightcore', rate: 1.26, drive: 0, bits: 16, lowpass: 16000, noise: 0.02 },
  fried: { label: 'Deep fried', rate: 1, drive: 340, bits: 6, lowpass: 3400, noise: 0.05 },
}

// Mirrors the real tiered matcher: clean/mild → classical landmarks (fast),
// heavy distortion → neural embeddings (slower, robust).
const RESULTS: Record<PresetKey, { tier: 'classical' | 'neural'; confidence: number; ms: number }> = {
  clean: { tier: 'classical', confidence: 0.99, ms: 212 },
  subway: { tier: 'classical', confidence: 0.94, ms: 384 },
  nightcore: { tier: 'neural', confidence: 0.91, ms: 1240 },
  fried: { tier: 'neural', confidence: 0.87, ms: 1418 },
}

function distortionCurve(amount: number): Float32Array {
  const n = 1024
  const curve = new Float32Array(n)
  for (let i = 0; i < n; i++) {
    const x = (i * 2) / n - 1
    curve[i] = amount === 0 ? x : ((3 + amount) * x * 20 * (Math.PI / 180)) / (Math.PI + amount * Math.abs(x))
  }
  return curve
}

function bitcrushCurve(bits: number): Float32Array {
  const n = 1024
  const steps = Math.pow(2, bits)
  const curve = new Float32Array(n)
  for (let i = 0; i < n; i++) {
    const x = (i * 2) / n - 1
    curve[i] = bits >= 16 ? x : Math.round(x * steps) / steps
  }
  return curve
}

export default function ChoonStressTest() {
  const [clipIdx, setClipIdx] = useState(0)
  const [preset, setPreset] = useState<PresetKey>('clean')
  const [playing, setPlaying] = useState(false)
  const [identifying, setIdentifying] = useState(false)
  const [result, setResult] = useState<(typeof RESULTS)[PresetKey] | null>(null)
  const [loadingClip, setLoadingClip] = useState(false)

  const ctxRef = useRef<AudioContext | null>(null)
  const buffersRef = useRef<Map<string, AudioBuffer>>(new Map())
  const nodesRef = useRef<{
    source?: AudioBufferSourceNode
    noise?: AudioBufferSourceNode
    shaper?: WaveShaperNode
    crusher?: WaveShaperNode
    filter?: BiquadFilterNode
    noiseGain?: GainNode
    gain?: GainNode
    analyser?: AnalyserNode
  }>({})
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const rafRef = useRef(0)
  const rootRef = useRef<HTMLDivElement>(null)

  const stop = () => {
    const n = nodesRef.current
    try {
      n.source?.stop()
      n.noise?.stop()
    } catch {}
    nodesRef.current = {}
    cancelAnimationFrame(rafRef.current)
    rafRef.current = 0
    setPlaying(false)
  }

  // Pause when scrolled away, on tab hide, and on page swap
  useEffect(() => {
    const el = rootRef.current
    if (!el) return
    const io = new IntersectionObserver((entries) => {
      if (!entries[0].isIntersecting) stop()
    })
    io.observe(el)
    const onHide = () => document.hidden && stop()
    document.addEventListener('visibilitychange', onHide)
    document.addEventListener('astro:before-swap', stop)
    return () => {
      io.disconnect()
      document.removeEventListener('visibilitychange', onHide)
      document.removeEventListener('astro:before-swap', stop)
      stop()
      ctxRef.current?.close().catch(() => {})
    }
  }, [])

  const drawSpectrum = () => {
    const analyser = nodesRef.current.analyser
    const canvas = canvasRef.current
    if (!analyser || !canvas) return
    const ctx2d = canvas.getContext('2d')
    if (!ctx2d) return

    const rm = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    const data = new Uint8Array(analyser.frequencyBinCount)
    const dpr = Math.min(window.devicePixelRatio || 1, 2)
    const w = canvas.clientWidth
    const h = canvas.clientHeight
    canvas.width = w * dpr
    canvas.height = h * dpr
    ctx2d.setTransform(dpr, 0, 0, dpr, 0, 0)

    let lastDraw = 0
    const frame = (now: number) => {
      rafRef.current = requestAnimationFrame(frame)
      if (rm && now - lastDraw < 250) return // still functional, just calm
      lastDraw = now
      analyser.getByteFrequencyData(data)
      ctx2d.clearRect(0, 0, w, h)
      const bars = 48
      const step = Math.floor(data.length / bars)
      const bw = w / bars
      for (let i = 0; i < bars; i++) {
        const v = data[i * step] / 255
        const bh = Math.max(2, v * h * 0.92)
        const hue = i / bars
        // alpenglow: crimson → marigold across the spectrum
        const r = Math.round(214 + (239 - 214) * hue)
        const g = Math.round(69 + (163 - 69) * hue)
        const b = Math.round(83 + (59 - 83) * hue)
        ctx2d.fillStyle = `rgba(${r},${g},${b},${0.35 + v * 0.65})`
        ctx2d.fillRect(i * bw + 1, h - bh, bw - 2, bh)
      }
    }
    rafRef.current = requestAnimationFrame(frame)
  }

  const play = async () => {
    if (playing) {
      stop()
      return
    }
    setResult(null)

    if (!ctxRef.current) {
      ctxRef.current = new AudioContext()
    }
    const ctx = ctxRef.current
    await ctx.resume()

    const clip = CLIPS[clipIdx]
    let buffer = buffersRef.current.get(clip.id)
    if (!buffer) {
      setLoadingClip(true)
      try {
        const res = await fetch(clip.url)
        const arr = await res.arrayBuffer()
        buffer = await ctx.decodeAudioData(arr)
        buffersRef.current.set(clip.id, buffer)
      } catch {
        setLoadingClip(false)
        return
      }
      setLoadingClip(false)
    }

    const p = PRESETS[preset]

    const source = ctx.createBufferSource()
    source.buffer = buffer
    source.loop = true
    source.playbackRate.value = p.rate

    const shaper = ctx.createWaveShaper()
    shaper.curve = distortionCurve(p.drive)
    shaper.oversample = '2x'

    const crusher = ctx.createWaveShaper()
    crusher.curve = bitcrushCurve(p.bits)

    const filter = ctx.createBiquadFilter()
    filter.type = 'lowpass'
    filter.frequency.value = p.lowpass

    const gain = ctx.createGain()
    gain.gain.value = 0.55

    const analyser = ctx.createAnalyser()
    analyser.fftSize = 256
    analyser.smoothingTimeConstant = 0.75

    source.connect(shaper).connect(crusher).connect(filter).connect(gain)

    let noise: AudioBufferSourceNode | undefined
    let noiseGain: GainNode | undefined
    if (p.noise > 0) {
      const nb = ctx.createBuffer(1, ctx.sampleRate, ctx.sampleRate)
      const nd = nb.getChannelData(0)
      for (let i = 0; i < nd.length; i++) nd[i] = Math.random() * 2 - 1
      noise = ctx.createBufferSource()
      noise.buffer = nb
      noise.loop = true
      noiseGain = ctx.createGain()
      noiseGain.gain.value = p.noise
      noise.connect(noiseGain).connect(gain)
      noise.start()
    }

    gain.connect(analyser).connect(ctx.destination)
    source.start()

    nodesRef.current = { source, noise, shaper, crusher, filter, noiseGain, gain, analyser }
    setPlaying(true)
    drawSpectrum()
  }

  // Live-retune the chain when the preset changes mid-play
  useEffect(() => {
    const n = nodesRef.current
    if (!playing || !n.source) return
    const p = PRESETS[preset]
    n.source.playbackRate.value = p.rate
    if (n.shaper) n.shaper.curve = distortionCurve(p.drive)
    if (n.crusher) n.crusher.curve = bitcrushCurve(p.bits)
    if (n.filter) n.filter.frequency.value = p.lowpass
    if (n.noiseGain) n.noiseGain.gain.value = p.noise
    else if (p.noise > 0) {
      // preset needs noise but chain has none — rebuild
      stop()
      void play()
    }
    setResult(null)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [preset])

  const identify = () => {
    if (identifying) return
    setResult(null)
    setIdentifying(true)
    const rm = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    setTimeout(
      () => {
        setResult(RESULTS[preset])
        setIdentifying(false)
      },
      rm ? 150 : 1400,
    )
  }

  const clip = CLIPS[clipIdx]

  return (
    <div className="ch" ref={rootRef}>
      <div className="ch-row">
        <div className="ch-clips" role="group" aria-label="Pick a clip">
          {CLIPS.map((c, i) => (
            <button
              key={c.id}
              type="button"
              className={`ch-chip btn ${i === clipIdx ? 'ch-chip-on' : ''}`}
              aria-pressed={i === clipIdx}
              onClick={() => {
                stop()
                setClipIdx(i)
                setResult(null)
              }}
            >
              {c.name}
            </button>
          ))}
        </div>
        <button type="button" className="ch-play btn" onClick={play} aria-label={playing ? `Pause ${clip.name}` : `Play ${clip.name}`}>
          {loadingClip ? '…' : playing ? '❚❚' : '▶'}
        </button>
      </div>

      <div className="ch-presets" role="group" aria-label="Mangle the audio">
        {(Object.keys(PRESETS) as PresetKey[]).map((k) => (
          <button
            key={k}
            type="button"
            className={`ch-chip btn ${preset === k ? 'ch-chip-on' : ''}`}
            aria-pressed={preset === k}
            onClick={() => setPreset(k)}
          >
            {PRESETS[k].label}
          </button>
        ))}
      </div>

      <div className="ch-scope-wrap">
        <canvas ref={canvasRef} className="ch-scope" aria-hidden="true"></canvas>
        {!playing && <p className="ch-scope-hint label-mono">▶ play the clip, mangle it, then identify it</p>}
      </div>

      <div className="ch-id-row">
        <button type="button" className="ch-identify btn" onClick={identify} disabled={identifying}>
          Identify this
        </button>
        <span className="ch-note label-mono">illustrative — the real matcher runs on GCP</span>
      </div>

      <div className="ch-result-zone" aria-live="polite">
        {identifying && (
          <div className="ch-loading">
            <div className="pulse-line" role="status" aria-label="Listening"></div>
            <div className="pulse-caption">Listening…</div>
          </div>
        )}
        {result && !identifying && (
          <div className="ch-result">
            <div className="ch-result-top">
              <span className="ch-check" aria-hidden="true">
                ✓
              </span>
              <span className="ch-result-name">
                {clip.name} <span className="ch-result-sub">— Vibeset demo catalog</span>
              </span>
            </div>
            <div className="ch-result-meta label-mono">
              tier: {result.tier} · confidence {result.confidence.toFixed(2)} · {result.ms} ms
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
