/**
 * Say it instead of typing it.
 *
 * WHY THIS EXISTS: much of this family is over sixty and on a phone, and
 * "names that sound soft when you say them out loud" is a hard sentence to
 * thumb into a keyboard in a second language.
 *
 * WHY IT NEVER SUBMITS: measured against the pool builder, retrieval shrugs off
 * transcription noise on ordinary words — 38 to 40 of 40 pool overlap for
 * "agressive", "sutrus", "abrod", "nems" — and collapses when the mis-heard
 * word is the one carrying the meaning: 2/40 for "vise" where "wise" was meant,
 * 4/40 for "vater", 7/40 for "lite". That is exactly the v/w/b substitution a
 * Nepali speaker's accent produces, so it is the LIKELY error, not the rare
 * one.
 *
 * And none of the ten noisy queries returned nothing. A mis-hear is therefore
 * silent: the page answers confidently with names for a question nobody asked.
 * The only honest mitigation is to put the transcript in front of the person
 * before it is acted on — so this fills the composer and stops, and the send
 * button stays theirs to press.
 *
 * NO POLYFILL, NO UPSELL. Where the browser does not have it the hook reports
 * `available: false` and the page renders nothing at all, rather than a control
 * that explains itself or a link to a better browser.
 */
import { useCallback, useEffect, useRef, useState } from 'react'

/** The two names the same API ships under. */
interface SpeechCtor {
  new (): SpeechRecognitionLike
}
interface SpeechRecognitionLike {
  lang: string
  continuous: boolean
  interimResults: boolean
  start(): void
  stop(): void
  abort(): void
  onresult: ((event: SpeechResultEventLike) => void) | null
  onend: (() => void) | null
  onerror: (() => void) | null
}
interface SpeechResultEventLike {
  resultIndex: number
  results: ArrayLike<ArrayLike<{ transcript: string }> & { isFinal: boolean }>
}

function ctor(): SpeechCtor | null {
  if (typeof window === 'undefined') return null
  const w = window as unknown as { SpeechRecognition?: SpeechCtor; webkitSpeechRecognition?: SpeechCtor }
  return w.SpeechRecognition ?? w.webkitSpeechRecognition ?? null
}

export interface Speech {
  /** The browser has the API. False during SSR and on anything without it. */
  available: boolean
  listening: boolean
  toggle(): void
}

/**
 * `onText` receives the running transcript, interim included, so the box fills
 * as the person speaks rather than after they stop. Interim results are what
 * make it feel like dictation instead of a form submission.
 */
export function useSpeech(onText: (text: string) => void): Speech {
  const [available, setAvailable] = useState(false)
  const [listening, setListening] = useState(false)
  const recRef = useRef<SpeechRecognitionLike | null>(null)
  /** Kept in a ref so restarting never rebinds a stale handler. */
  const sink = useRef(onText)
  sink.current = onText

  // Detected at mount: this component server-renders and there is no window then.
  useEffect(() => {
    setAvailable(ctor() !== null)
  }, [])

  useEffect(
    () => () => {
      recRef.current?.abort()
      recRef.current = null
    },
    [],
  )

  const toggle = useCallback(() => {
    if (recRef.current) {
      recRef.current.stop()
      return
    }
    const Ctor = ctor()
    if (!Ctor) return
    const rec = new Ctor()
    // Indian English is the closest widely-shipped model to the accents this
    // page is actually spoken in, and it is markedly better on Sanskrit-derived
    // names than en-US. It is a better guess, not a correct one.
    rec.lang = 'en-IN'
    rec.continuous = false
    rec.interimResults = true

    let said = ''
    rec.onresult = (event) => {
      let text = ''
      for (let i = 0; i < event.results.length; i += 1) {
        text += event.results[i][0]?.transcript ?? ''
      }
      said = text.trim()
      if (said) sink.current(said)
    }
    const finish = () => {
      recRef.current = null
      setListening(false)
    }
    rec.onend = finish
    // Silent on error, deliberately. A permission refusal or a network blip is
    // not something to explain to somebody who was trying to name a baby; the
    // keyboard is still right there and still works.
    rec.onerror = finish

    recRef.current = rec
    setListening(true)
    try {
      rec.start()
    } catch {
      finish()
    }
  }, [])

  return { available, listening, toggle }
}
