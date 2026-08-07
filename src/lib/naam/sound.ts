/**
 * The three sounds this page is allowed (docs/design/DESIGN.md §4, P7 ·
 * docs/design/MOTION.md §5).
 *
 * WHY SOUND AT ALL. It is the highest game-feel-per-kilobyte thing available
 * to a web page and almost nothing outside a game ships it. Every cue here is
 * SYNTHESISED — three oscillators and a noise buffer — so the whole feature is
 * about 0 kB of network. There are no audio assets to fetch, decode or cache,
 * and `howler` (7 kB) would only be wrapping what these forty lines already do.
 *
 * WHY THESE THREE SOUNDS AND NOT THE OBVIOUS ONES. The page is a room with a
 * lamp in it and a conversation about a child's name. So the cues are the
 * MATERIALS ACTUALLY IN THAT ROOM, one each, and never a UI vocabulary:
 *
 *   deal      PAPER  a card set down — a filtered noise tick, 18ms
 *   land      CLAY   a bead put on a tray — a low earthen tock, 90ms
 *   complete  METAL  a small bowl struck once, and the only time metal plays
 *
 * `complete` is the one that could most easily have been generic. The default
 * answer is a two-note rising chime, which is the success jingle every app on
 * a phone already owns, and next to a diyo it would sound like a notification.
 * A struck bowl is what is actually in that room, and — the part that matters
 * acoustically — its partials are INHARMONIC. The stack below is roughly
 * 1 : 2.71 : 5.42, which is a bowl rather than a musical interval, so the ear
 * files it as an object being struck instead of as a tune being played. Two
 * fundamentals six cents apart give it the slow beat a real bowl has.
 *
 * It also runs ~700ms rather than the 200 the other two get. A bowl cut off at
 * 200ms is a click; the decay IS the sound. It stays quiet instead: everything
 * here peaks at −18 dBFS, which is well under speech.
 *
 * DEFAULT OFF, and the toggle is the unlock rather than a preference buried in
 * a menu. Browsers require a gesture before an AudioContext may make noise, so
 * the switch that turns sound on is also the gesture that permits it — one act,
 * not two. Nothing can play before it is pressed, which is also the honest
 * behaviour: a page that makes noise at a visitor who did not ask is a page
 * they close.
 *
 * REDUCED MOTION SILENCES IT TOO. `prefers-reduced-motion` is not only about
 * transforms — it is set by people who find incidental stimulus costly, and an
 * unrequested sound is exactly that.
 */

export type NaamCue = 'deal' | 'land' | 'complete'

const KEY = 'naam.sound.v1'
/** −18 dBFS. Under speech, over the room. */
const PEAK = 0.126

let ctx: AudioContext | null = null
let noise: AudioBuffer | null = null
let on = false
const listeners = new Set<() => void>()

function announce() {
  for (const fn of listeners) fn()
}

function reduced(): boolean {
  return typeof matchMedia === 'function' && matchMedia('(prefers-reduced-motion: reduce)').matches
}

/**
 * Read once on first subscribe. Wrapped because Safari in private browsing
 * throws on localStorage rather than returning null, and a page that will not
 * render because it could not read a sound preference is a bad trade.
 */
export function hydrateSound(): void {
  try {
    on = window.localStorage.getItem(KEY) === 'on'
  } catch {
    on = false
  }
  announce()
}

export function soundOn(): boolean {
  return on
}

/** useSyncExternalStore needs a stable server value, and silence is the truth. */
export function soundOff(): boolean {
  return false
}

export function subscribeSound(fn: () => void): () => void {
  listeners.add(fn)
  return () => listeners.delete(fn)
}

/**
 * Called from the toggle's click handler, which is the gesture the autoplay
 * policy is waiting for — so the context is built here and nowhere else.
 */
export function setSound(next: boolean): void {
  on = next
  try {
    window.localStorage.setItem(KEY, next ? 'on' : 'off')
  } catch {
    /* a preference that cannot be stored still works for this visit */
  }
  if (next) void ensure()
  announce()
}

async function ensure(): Promise<AudioContext | null> {
  if (typeof window === 'undefined') return null
  const Ctor = window.AudioContext ?? (window as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext
  if (!Ctor) return null
  if (!ctx) {
    ctx = new Ctor()
    // One second of white noise, made once and shared. Both the paper tick and
    // the clay tock read it at different offsets, so two buffers would be two
    // copies of the same thing.
    const frames = Math.floor(ctx.sampleRate)
    noise = ctx.createBuffer(1, frames, ctx.sampleRate)
    const data = noise.getChannelData(0)
    for (let i = 0; i < frames; i++) data[i] = Math.random() * 2 - 1
  }
  if (ctx.state === 'suspended') await ctx.resume()
  return ctx
}

/** A noise burst through a bandpass: the sound of a surface, not a tone. */
function grain(at: number, freq: number, q: number, peak: number, ms: number) {
  if (!ctx || !noise) return
  const src = ctx.createBufferSource()
  src.buffer = noise
  // A different offset every time, so ten cards in a row are not one sample
  // played ten times — which is the thing that makes game audio sound cheap.
  src.loopStart = 0
  const band = ctx.createBiquadFilter()
  band.type = 'bandpass'
  band.frequency.value = freq
  band.Q.value = q
  const gain = ctx.createGain()
  gain.gain.setValueAtTime(0.0001, at)
  gain.gain.exponentialRampToValueAtTime(peak, at + 0.001)
  gain.gain.exponentialRampToValueAtTime(0.0001, at + ms / 1000)
  src.connect(band).connect(gain).connect(ctx.destination)
  src.start(at, Math.random() * 0.5)
  src.stop(at + ms / 1000 + 0.02)
}

function partial(at: number, freq: number, peak: number, ms: number, glideTo?: number) {
  if (!ctx) return
  const osc = ctx.createOscillator()
  osc.type = 'sine'
  osc.frequency.setValueAtTime(freq, at)
  if (glideTo) osc.frequency.exponentialRampToValueAtTime(glideTo, at + ms / 1000)
  const gain = ctx.createGain()
  gain.gain.setValueAtTime(0.0001, at)
  gain.gain.exponentialRampToValueAtTime(peak, at + 0.004)
  gain.gain.exponentialRampToValueAtTime(0.0001, at + ms / 1000)
  osc.connect(gain).connect(ctx.destination)
  osc.start(at)
  osc.stop(at + ms / 1000 + 0.02)
}

export function playCue(cue: NaamCue): void {
  if (!on || reduced()) return
  void ensure().then((audio) => {
    if (!audio) return
    const t = audio.currentTime + 0.001
    switch (cue) {
      // PAPER. High, dry, and over before it is noticed — this one fires three
      // times in a row when a hand is dealt, so anything with a tail would
      // smear into a rustle.
      case 'deal':
        grain(t, 2100, 1.1, PEAK * 0.5, 18)
        break

      // CLAY. The pitch drops as it sounds, which is what makes a struck solid
      // read as heavy rather than as a beep; the noise transient on top is the
      // contact itself.
      case 'land':
        partial(t, 196, PEAK * 0.8, 90, 118)
        grain(t, 700, 0.8, PEAK * 0.35, 12)
        break

      // METAL, once. 1 : 2.71 : 5.42 is a bowl's partial stack, not a chord —
      // and the second fundamental six cents sharp is what gives it the slow
      // beat you hear in a real one.
      case 'complete':
        partial(t, 660, PEAK * 0.6, 720)
        partial(t, 664, PEAK * 0.35, 700)
        partial(t + 0.004, 1789, PEAK * 0.22, 500)
        partial(t + 0.008, 3577, PEAK * 0.09, 320)
        grain(t, 3200, 1.4, PEAK * 0.3, 14)
        break
    }
  })
}
