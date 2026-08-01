# Alpenglow — the design spec

> The governing document for bishal.ai. `src/styles/tokens.css`, `src/styles/motion.css`,
> `src/assets/scss/base/_root.scss` and ~14 other files cite this file and
> [MOTION.md](./MOTION.md) as the source of truth. Screenshots from
> `.claude/skills/verify-site` are judged against these criteria, not against vibes.
>
> This spec describes what the site **is**, derived from the code. When code and spec
> disagree, that is a bug in one of them — decide which, then fix that one. Don't let them
> drift.

---

## 0. The thesis

**The site is an instrument reading a signal.**

This is not a theme laid over the content; it comes out of the content. Bishal's published
research is electrical signalling between neurons — innexin gap junctions in *C. elegans*
(PLoS Genetics 2019), engineered synapses restoring a damaged circuit (Cell Systems 2021).
So the site's loader is a traveling action potential, the research section is drawn as a
**spike train**, the entire motion vocabulary is namespaced `sig-*`, and the brand gradient
is called `--grad-signal`.

The metaphor is *earned*. That is what makes it defensible, and it is the single most
distinctive thing about this site. Every design decision below serves it.

**The corollary that does the most work:** capabilities are presented as **stations on a
continuous signal path** — never as a grid of cards. A grid of cards is what makes a
personal portfolio read as a product dashboard. When a new section is proposed, the first
question is "where does it sit on the path?", not "what does its card look like?"

---

## 1. Principles

Each is written to adjudicate a real decision. If a proposal can't be checked against
these, the principle is too vague and should be sharpened.

### P1 — One world, three depths, zero themes

```
--void   #0e1124   the night — page canvas
--ink    #181c30   raised panels, cards
--ink-2  #222741   inputs, wells, second raise
```

`:root { color-scheme: dark }` is hardcoded (`tokens.css:12`). There is no light branch, no
toggle, and no `prefers-color-scheme` query anywhere in `src/`.

**Adjudicates:** no second surface family, no light mode, no "card on white". Depth comes
from these three surfaces plus hairlines — not from stacked shadows.

### P2 — Emphasis by glow, never by fill

Weight is signalled with `--glow-crimson` / `--glow-marigold` / `--glow-glacier` /
`--ring-crimson`. Solid fill appears on exactly one element class: the primary pill, always
`--crimson-deep`.

**Adjudicates:** a new emphasis level means more glow, not more paint. There is never a
second filled button colour.

### P3 — Three accents, each with an assigned job

```
--crimson   #d64553   signal start · focus rings · the primary accent
--marigold  #efa33b   signal end · highlights · link hover
--glacier   #4ca8a2   live / success · the cold counterpoint
--paper     #faf7f1   bone — primary text, the light itself
--paper-soft #b0b5ca  secondary text
```

Text on the night canvas uses the AA-safe variants: `--crimson-text #f07a84`,
`--glacier-text #6fc7c1`. `--crimson` and `--glacier` are **not** AA for small text — that
is the entire reason those two tokens exist. Primary text is bone, never pure white.

**Why this matters beyond taste:** a near-black canvas with one bright accent is one of the
most common machine-generated design defaults in circulation. What makes Alpenglow a
*choice* is that it has three accents with distinct semantic jobs plus a bone paper. Diluting
it into "dark theme + accent colour" throws away the differentiation.

**Adjudicates:** a new colour needs a job no existing accent has. There are no decorative
hues.

### P4 — The signal gradient is a line, never a fill

`--grad-signal` (crimson → marigold) appears only as strokes and 2px rules: the loader
spike, the loading-screen trace, the spike-train rail, the pipeline connectors, the trace
terminating into the contact form's submit button, the oscilloscope stroke.

**Adjudicates:** gradient backgrounds and gradient cards are violations. Gradient-clipped
*text* is the single sanctioned exception — see §3.

### P5 — Mono is the voice of data

Three faces, three jobs:

| Role | Face | Used for |
|---|---|---|
| `--font-display` | Bricolage Grotesque Variable | headings — the site makes a claim |
| `--font-body` | Atkinson Hyperlegible | prose — the site is read |
| `--font-mono` | DM Mono | labels, numbers, status — the site is *measured* |

`.label-mono` (uppercase, 500, 0.75rem, `--tracking-label: 0.08em`) carries every eyebrow,
stat, tag and status line — 55 usages. It is the main reason structurally different sections
read as one document.

Section titles are always the same triple:
`clamp(2rem, 4.5vw, 3.25rem)` / `letter-spacing: -0.015em` / `line-height: 1.02`.

**Adjudicates:** a new section header reuses that triple exactly, and its eyebrow is
mono-uppercase — not a coloured badge.

### P6 — Enter once, then hold still

Full entrance vocabulary, and there are no others:

- `sig-arrive` — `translateY(12px)` + fade. The default.
- `sig-settle` — fade only. For heavy compositions.
- `sig-pop` — `scale(0.985) → 1`. Overlays, menus, result cards.

Reveals fire **once** (IntersectionObserver + `WeakSet`, `src/scripts/reveal.ts`), staggered
40ms per item, **capped at 320ms**. Micro-interactions are fixed at three moves and nothing
else: hover `translateY(-1px)`, press `scale(0.96)` @ 80ms, focus
`2px solid var(--crimson)` at 2px offset.

See [MOTION.md](./MOTION.md) for the full vocabulary and the rules for extending it.

### P7 — Ambient motion is rationed

Two continuous loops per viewport, plus the live-dot breath:

1. `Atmosphere.astro` aurora drift (110s / 140s, disabled below 768px)
2. The hero oscilloscope `requestAnimationFrame` loop (IntersectionObserver- and
   `visibilitychange`-paused)
3. `.live-dot` / `sig-live-breathe` — declared in `motion.css` as *"the only infinite ambient
   animation allowed outside the hero oscilloscope"*

**Adjudicates:** a new looping element must **displace** one of these, not become a fourth.
A swap is allowed when the semantic is identical (see `sig-eq` in MOTION.md); an addition is
not.

### P8 — No spinners, no skeletons

The site states this in its own UI (`LiveSignals.astro`: *"this page: no spinners, no
skeletons — just the signal"*). Loading is always `.pulse-line` — a gradient spike travelling
along a 2px rule — with a mono caption. It is a brand asset and it survives into every new
demo.

**Adjudicates:** any component that needs a loading state uses `.pulse-line`. No exceptions,
no third-party spinner components.

### P9 — Proof over claim

Every assertion on this site is live, cited, or runnable. `/notes/choon` publishes a
*failing* result (high-pass, 15.3% recall@1), states its methodology, and marks its NDA
boundary. Demos badge themselves honestly (§4). This candour is a large part of why the site
reads as credible to senior engineers, and a redesign must not sand it off.

**Adjudicates:** no number renders without a run, a paper, or a URL behind it. See §4 and
the evidence gate in the content schema.

### P10 — Accessibility is a floor, not a goal

Enforced by `.claude/skills/verify-site`:

| Gate | Threshold |
|---|---|
| axe violations | **0** |
| Lighthouse accessibility | **100** |
| Lighthouse best-practices | **100** |
| CLS | **0** |
| TBT | **≤ 200ms** |
| Mobile performance | **≥ 85** local (≈90+ on CDN) |

One global `prefers-reduced-motion` block in `motion.css`; the file explicitly forbids
per-component blocks. Content is never hidden from no-JS clients
(`html.no-js [data-reveal] { opacity: 1 }`). Focus is always visible and always distinct
from hover — never collapse the two.

### P11 — The person is the subject

The site opens with a name at `clamp(3.2rem, 11.5vw, 8.5rem)` and speaks in the first person
("I co-own and build Vibeset"). Vibeset is *his company*, presented as evidence of range —
not as the site's subject.

**Adjudicates:** pricing-table grammar, "most chosen" badges, chat-widget framing and other
SaaS-landing-page conventions pull against this. When a pattern would look identical on a
company site, find the version that only a person would ship.

---

## 2. Density budget

The hard constraint is that the site must never feel crowded or read as a product dashboard.
Density is measured, not judged. Baseline at the time of writing: **1,065 visible words,
25 buttons, 21 links, 6 inputs.**

| Constraint | Limit |
|---|---|
| Sections | 5 max, plus ≤2 ambient strips (≤20 words, 0 buttons, 0 inputs) |
| Visible words, whole page | ≤900 (content inside collapsed `<details>` doesn't count) |
| Visible words per section | ≤220 |
| Interactive widgets on `/` | **3**, plus the contact form |
| Chrome buttons (nav, CTAs, form) | ≤8 |
| Demo controls | ≤10 per widget, counted separately |
| Filled `--crimson-deep` CTAs above the fold | exactly 1 |
| CTAs per station | 1 filled + 1 quiet text link |
| Stats per station | ≤3 |
| Consecutive sections using a 3-card row | never 2 in a row |
| Ambient infinite animations | 2 per viewport (P7) |
| Section rhythm | `--space-3xl` |

**Why buttons are counted in two piles.** The original audit capped the page at 16 buttons
flat, measured when 25 of them were pricing tiers, card CTAs and duplicate links — chrome
that asked the visitor to go somewhere. Controls that *drive an instrument* are the content,
not the packaging: a visitor operating the sync workbench is doing the thing the page exists
to show. So chrome is capped hard and demo controls are budgeted per widget. If a demo's
controls ever start reading as chrome, that's a sign the demo isn't demonstrating anything.

**The rhythm rule matters most.** Complaints about density are usually complaints about
rhythm. `--space-3xl` (6–6.75rem) between sections is already generous and must not shrink
to make room for more content. Shrink the content instead.

**Where immersion lives.** The home page is an **index**; the product pages are where depth
belongs. This is what lets the site carry six demonstrations while the landing page gets
*less* dense, not more. A capability that needs room gets a product page or a case note — not
another home-page section.

---

## 3. Typography and labeling rules

- **Display and marketing headlines → sentence case.** "One catalog, three products."
- **UI labels, eyebrows, metadata, buttons → UPPERCASE DM Mono**, `--tracking-label: 0.08em`.
- **Numbers are first-class and always mono.** A stat's *value* is not a label — it is the
  proof. On product pages, values are set at display size with
  `font-variant-numeric: tabular-nums`. `76.9%` should look measured, not mentioned.
- **Two mono sizes, and stop.**
  - `.label-mono` — 0.75rem / 0.08em — eyebrows, stats, status
  - `.label-mono--sm` — 0.65rem / 0.1em — captions, chips, dense annotation

  Improvising a third size is how a labeling system dies.
- **No emoji. Functional glyphs only:** `↳` (leaves the site or opens the app), `▶` (play),
  `✓` (confirmed). Anything else is decoration.
- **One gradient-clipped word per page, at display size only.** `.text-gradient-organic`
  exists for this. Restrict to ≥2rem, where AA requires only 3:1 — gradient-clipped text
  loses contrast guarantees, so it is never used at body size.

---

## 4. Honesty vocabulary

Demos declare their nature in one of exactly three words. This is a design element, not a
disclaimer — it is why the numbers on this site are believed.

| Badge | Means | Example |
|---|---|---|
| **LIVE** | A real call, right now, with measured latency | VibeFinder hitting the production search API |
| **LOCAL** | Real algorithm, real data, running in your browser | The Cue sync workbench recomputing beat alignment |
| **REPLAY** | Recorded real output, played back | Fixtures captured from the production API |

Rules:

1. **Never a fourth word.** "Illustrative", "simulated", "example" all collapse to REPLAY.
2. **REPLAY means recorded, not invented.** Output that was never produced by the real system
   doesn't get a badge — it gets removed or rebuilt.
3. **A badge states what the visitor is looking at**, not what the system can do elsewhere.
4. **Failure is honest, never blank.** When a live path fails, degrade to REPLAY and say so
   ("REPLAY — the API was napping"). Never a spinner, never an empty state, never a lie.

---

## 5. Quoting the products

The three Vibeset products have their own visual identities. The portfolio quotes them; it
does not adopt them.

Each product in the `products` collection carries:

```yaml
accent: '#4CA8A2'                              # Alpenglow-family — load-bearing
quote: { hue: '#1570EF', name: 'Head Nods' }   # the product's real Vibeset hue
```

`--accent` does everything contrast rules govern: borders, focus, CTA fills, text.
`--quote-hue` is licensed for **exactly three non-text uses** on that product's page — a 1px
top rule on the media zone, a gradient stop in the artwork tile, and the glow beneath the
screenshot. Non-text means WCAG contrast never applies, so an out-of-family hue is safe.

**What must never be quoted:** the products' action and focus colours. Cue's lime
`rgb(170,254,42)` is *Vibeset's* signal; crimson is *Bishal's*. Adopting lime as a focus ring
would make the portfolio read as a Vibeset subsite. Same for Choon's neon cyan `#00f3ff`,
which would put a second light source next to crimson on the same canvas.

---

## 6. Related

- [MOTION.md](./MOTION.md) — the animation vocabulary and how to extend it
- `src/styles/tokens.css` — the token implementation
- `.claude/skills/verify-site/SKILL.md` — how these criteria are enforced
