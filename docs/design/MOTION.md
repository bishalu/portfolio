# Signal motion — the animation spec

> Companion to [DESIGN.md](./DESIGN.md). Implemented in `src/styles/motion.css` and
> `src/scripts/reveal.ts`. Cited by `tokens.css`, `motion.css`, `reveal.ts` and the
> component headers.

---

## 0. The idea

Motion on this site is **signal propagation**, not decoration. The vocabulary is named
`sig-*` because it descends from the same metaphor as everything else: Bishal's research is
electrical signalling between neurons, so content *arrives* the way a pulse arrives, and the
loading indicator is literally described in the source as *"a line carries a traveling
action-potential spike."*

Two consequences that shape every rule below:

1. **Motion carries meaning or it doesn't ship.** There is no motion here whose only job is
   to look nice. If a proposed animation can't answer "what is it telling the visitor?",
   it's decoration — cut it.
2. **The vocabulary is small and closed.** Six verbs total. Adding a seventh requires
   displacing one (§5). A large motion vocabulary is indistinguishable from no vocabulary.

---

## 1. Physics

```css
--ease-out:    cubic-bezier(0.22, 1, 0.36, 1);     /* default — expo-out */
--ease-spring: cubic-bezier(0.34, 1.56, 0.64, 1);  /* rare, playful. Currently unused. */
--ease-exit:   cubic-bezier(0.45, 0, 0.7, 0.2);    /* leaving */

--dur-fast: 120ms;   /* micro-feedback */
--dur-base: 200ms;   /* state changes, pops */
--dur-slow: 400ms;   /* entrances */
```

Scripted (WAAPI) motion in `Hero.astro` and `LoadingScreen.astro` uses
`EXPO_OUT = cubic-bezier(0.16, 1, 0.3, 1)` for larger, more theatrical moves, and
`EASE_INOUT = cubic-bezier(0.65, 0, 0.35, 1)` for symmetric ones.

**Hard rule: compositor properties only** — `transform`, `opacity`, `clip-path`. Never
animate `height`, `width`, `top`, `margin`, or anything else that triggers layout. This is
what keeps CLS at 0 and TBT under 200ms.

---

## 2. The vocabulary

### Entrances — "the pulse arrives"

| Verb | Motion | Use |
|---|---|---|
| `sig-arrive` | `translateY(12px)` + fade | **Default.** Content settles up into place. |
| `sig-settle` | fade only | Heavy compositions where a rise would feel like a shove. |
| `sig-pop` | `scale(0.985) → 1` + fade | Overlays, menus, result cards. |

Markup is declarative:

```html
<div data-reveal>…</div>              <!-- sig-arrive -->
<div data-reveal="settle">…</div>
<div data-reveal="pop">…</div>
<div data-reveal-stagger>…</div>      <!-- children delayed in reading order -->
```

`src/scripts/reveal.ts` observes at `threshold: 0.3, rootMargin: '0px 0px -5% 0px'`, adds
`.is-revealed` once per element (tracked in a `WeakSet`), and unobserves. Elements already
in the viewport on load reveal immediately, with no observer flash.

**Stagger: 40ms per item, capped at 320ms.** The cap is the important half — without it,
a long list turns into a slow wipe.

### Drawing — "a measured line writes itself"

```html
<polyline data-draw pathLength="1" … />   <!-- inside a [data-reveal] ancestor -->
```

`sig-draw` animates `stroke-dashoffset` 1 → 0 over 900ms. `pathLength="1"` is required
so the dash maths is independent of the path's real geometry.

Borrowed from the Choon monitor, where the same treatment marks a plot as a *reading*
rather than a picture. Reserve it for charts whose data is a sequence — a line that draws
itself says "this was measured over something". A bar chart doesn't qualify; bars are
categories, not a traversal.

It fires on scroll via the existing reveal observer, never on load, and the global
reduced-motion block collapses the duration so the line is simply present. Fully drawn is
the correct still state.

### Loading — "the pulse"

One loader, everywhere. **No spinners, no skeletons** (DESIGN.md P8).

```html
<div class="pulse-line" role="status" aria-label="Searching the catalog"></div>
<p class="pulse-caption">searching…</p>
```

A 2px hairline rule carrying a `--grad-signal` spike that travels
`translateX(-20%) → 520%` over 1.1s. The caption breathes at 1.8s. Under reduced motion the
spike becomes a static full-width gradient bar — still legible as "something is happening."

### Ambient — "alive"

Rationed to the budget in DESIGN.md P7.

| Verb | Where | Notes |
|---|---|---|
| `sig-live-breathe` | `.live-dot` | opacity 0.45↔1 @ 1.8s. Declared in-source as the only infinite ambient animation permitted outside the hero. |
| `sig-eq` | `.sig-eq` | Three bars, `scaleY(0.26 → 1)`, periods 0.9s / 0.7s / 1.1s, delays 0 / 0.18s / 0.36s. **Rests at `scaleY(0.35)`** so reduced motion freezes it into a legible idle equalizer rather than a flat line. |
| aurora drift | `Atmosphere.astro` | 110s and 140s, `translate3d` + `scale` only, disabled below 768px. |
| oscilloscope | `Hero.astro` | `requestAnimationFrame` canvas. Paused by IntersectionObserver and `visibilitychange`; never starts under reduced motion (draws one static frame instead). |

**`live-dot` vs `sig-eq`:** both mean "this is alive", in two idioms. A dot suits a service
("the server is up"); bars suit an audio system ("something is running, and here is its
magnitude"). They are a **swap**, not an addition — using `sig-eq` somewhere means
`.live-dot` leaves that spot. `.live-dot` remains in the hero and the live-signals strip.

### Scripted moves

Two theatrical moments are performed by WAAPI rather than CSS. They are part of the
vocabulary and should be named as such when reused:

- **`sig-scene`** — the Balgo console bloom. Enter `scale(0.97) → 1` @ 340ms `EXPO_OUT`;
  exit `scale(1) → 1.06` @ 300ms `--ease-exit`. Also the phone sheet's
  `translateY(100%) → 0`.
- **The handoff** — `LoadingScreen` dispatches `bishal:hero-go` and the hero entrance begins
  *underneath* the overlay fade, so the drawn trace hands off into the oscilloscope instead
  of cutting to it. Do not "simplify" this into a sequential animation; the overlap is the
  whole effect.

### Micro-interactions

Three moves. There are no others.

```css
hover  → transform: translateY(-1px)
press  → transform: scale(0.96);  transition-duration: 80ms
focus  → outline: 2px solid var(--crimson); outline-offset: 2px
```

**Focus is never merged with hover.** A `hocus`-style combined variant (as used in two of the
Vibeset product codebases) is wrong here: focus needs a *stronger, different* treatment than
hover, not the same one.

---

## 3. Reduced motion

**One global block, at the end of `motion.css`. Do not add per-component blocks.** The file
says so, and drift starts the moment a second block exists.

The contract is that the page must be **complete, functional, and fully drawn — just still**:

- All durations → `0.01ms`, iteration count → 1
- `[data-reveal] { opacity: 1; animation: none }` — content is never hidden
- `.pulse-line::after` → static full-width gradient bar
- Hover/press transforms → none
- The oscilloscope draws a single static frame rather than not rendering

Components opt into *stillness*, never into *absence*. Anything that vanishes under reduced
motion is a bug.

**Colour and tint feedback are not motion** and are deliberately left intact — hover
background changes, focus rings, and state colours all still work. This is correct and
intentional; don't "fix" it.

`html.no-js` gets parallel treatment: `[data-reveal] { opacity: 1 }` so crawlers and no-JS
visitors see a complete page. `reveal.ts` removes `no-js` from `<html>` on boot.

---

## 4. Performance rules

- Compositor properties only (§1).
- Every `requestAnimationFrame` loop must pause on `IntersectionObserver` exit **and** on
  `visibilitychange`. A canvas animating in a background tab is a bug.
- Cap devicePixelRatio at 2 for canvas work.
- Motion must not cause layout shift. CLS floor is **0**, not "good".
- No animation library. The site ships none by design — the vocabulary is small enough that
  CSS keyframes plus WAAPI cover it, and a library would cost more than it saves.

---

## 5. Extending the vocabulary

Before adding a verb, answer all four:

1. **What does it tell the visitor** that no existing verb tells them?
2. **Is it an entrance, a loading state, an ambient signal, or a scripted moment?** If it
   doesn't fit one, it's probably decoration.
3. **If it loops, what is it displacing?** The ambient budget is fixed (DESIGN.md P7).
4. **What is its still state?** Draw it under reduced motion first. If the still version
   doesn't communicate, the moving version is doing work the layout should be doing.

`sig-eq` is the worked example: it added the one missing semantic ("a process is running,
and here is its magnitude"), it swapped rather than stacked, and its resting `scaleY(0.35)`
means the reduced-motion state still reads as an equalizer.

---

## 6. Related

- [DESIGN.md](./DESIGN.md) — palette, type, density budget, honesty vocabulary
- `src/styles/motion.css` — the implementation
- `src/scripts/reveal.ts` — the reveal observer
