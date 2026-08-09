# Theme rubric — light/dark toggle, front page

Graded from crops in `scripts/verify/out/light-*.png`, not from memory.
Every axis is graded on **both grounds** once dark exists. Pass 0 is light-only
because dark does not exist yet — that absence is itself three of the failures.

Target: nothing below **B**.

## Pass 0 — baseline, before any change

| # | Aspect | What "A" means | Light | Dark |
|---|---|---|---|---|
| 1 | **Light ground is a choice** | Not the near-#F4F1EA AI-default cream | **F** — it is `#f4f1ea`, the default to four decimal places | — |
| 2 | **Dark ground exists** | A real second palette, not an inversion hack | — | **F** — does not exist |
| 3 | **Toggle exists** | A control, findable, not crowding the header | **F** — no toggle machinery anywhere | — |
| 4 | **Default is dark** | No-JS and first paint both land on dark | **F** — `color-scheme: light` hardcoded | — |
| 5 | **No flash of wrong theme** | Theme resolved before first paint | **F** — nothing to resolve | — |
| 6 | **Grounds share one hue family** | Light and dark read as one identity | **F** — warm bone vs cool indigo ink, two worlds | — |
| 7 | **VibeFinder input** | Reads as a modern control | **D** — `#eae7de` khaki on cream; a 1970s form | — |
| 8 | **Marigold on light** | Accent, not mud | **D** — `#b87413` / `#8a570b` are brown mustard | — |
| 9 | **Choon bench table** | Data object, cool and precise | **C−** — beige highlight row, mustard chrome | — |
| 10 | **Cue widget** | Instrument, not stationery | **D+** — "opens on the Peak" in mustard on cream | — |
| 11 | **Front page cohesion** | Designed for this ground | **C** — reads as a dark site with the lights on | — |
| 12 | axe 0 violations | Zero, every route | **A** — 0 across 9 | — |
| 13 | Primary text contrast | ≥ 7:1 | **A** — 15.94 | — |
| 14 | Responsive 320–1440 | No sideways scroll, no clipping | **A** — 8 widths clean | — |
| 15 | Reduced motion | Honoured, nothing animates | **A** | — |
| 16 | Keyboard + focus | Every control reachable, ring visible | **A** | — |

## Pass 0 — added after photographing the rest of the site

Five faults no existing axis covered. Each is graded from a crop, and each is
something I would fix regardless of the toggle.

| # | Aspect | What "A" means | Light | Dark |
|---|---|---|---|---|
| 17 | **Spike-train rail on `/research`** | The signature device appears on the page built for it | **D** — measured: 4 spike elements on `/`, **0** on `/research`. The dedicated research page is a plain document | — |
| 18 | **`/about` portrait blend** | The photograph belongs to the ground | **D+** — the mask dissolves edges into `--void`, which was night when I built it. On bone the bright sky leaves a hard left seam; it reads as a soft eraser, not a blend | — |
| 19 | **Inner-page column** | The page is composed, not left-aligned and abandoned | **C** — `/about` and `/research` run a ~660px column in 1280 and leave ~45% empty. On night that emptiness read as depth; on paper it reads unfinished | — |
| 20 | **Portrait's blue** | Every strong colour is in the palette | **C+** — the sky is a saturated blue that appears in no token, and on bone it is the loudest thing on the page | — |
| 21 | **Spike marks on light** | The rail reads at a glance | **C** — thin `--crimson` strokes at 3.85:1 on bone; present but weak, where on night they carried | — |
| 22 | **Cue media tile** | Signal is a line, never a fill (DESIGN.md P5) | **C−** — a full-bleed crimson→navy gradient rectangle, the heaviest object on the page and a fill of the brand ramp | — |

**Seventeen axes at C or below. Five are F.** The craft floors (12–16) are all A and
must still be A at the end — that is the constraint, not the achievement.

## The diagnosis, in one line

The light theme is not badly executed — it is a *warm* theme wearing a *cool*
brand. `--paper` is indigo `#12152b`; the ground is bone `#f4f1ea`. Those are
opposite temperatures, so every accent had to be re-picked warm to sit on the
bone, which is where the mustard came from. Fix the temperature of the ground
and the mustard problem dissolves rather than being patched.

## Lighthouses (fixed direction, not graded)

- The site is an instrument reading a signal. Dark is the oscilloscope screen;
  light is the chart recorder's paper. The codebase already says this in
  `a7ab861`'s own commit message.
- `--crimson` and `--crimson-deep` are byte-identical across both grounds. They
  are the identity; they do not get re-picked per theme.
- `--stage` is a surface role, not a theme branch. It stays dark in both.
- Proof over claim: contrast is measured and written down, never estimated.


---

# Pass 1 — after the rebuild

`node scripts/verify/themes.mjs` runs axe and an overflow sweep on **both**
grounds across all nine routes; it is a permanent gate now, not a one-off.

| # | Aspect | Light | Dark | Was |
|---|---|---|---|---|
| 1 | Light ground is a choice | **A** — `#eaedf4`, cool, same hue family as the ink | — | F |
| 2 | Dark ground exists | — | **A** — `:root` default, recovered from `a7ab861^` | F |
| 3 | Toggle exists, uncrowded | **A** | **A** — one 34px icon after the GitHub mark | F |
| 4 | Default is dark | **A** | **A** — no attribute means dark; no-JS gets dark | F |
| 5 | No flash of wrong theme | **A** | **A** — inline sync script, measured before paint | F |
| 6 | Grounds share one hue family | **A** | **A** — both indigo, two lightnesses | F |
| 7 | VibeFinder input | **A** — cool `#dfe4ee`, no khaki | **A** | D |
| 8 | Marigold on light | **B+** — burnt orange `#9a4a17`, 5.32; amber kept for marks only | **A** | D |
| 9 | Choon bench table | **A** — cool highlight row | **A** | C− |
| 10 | Cue widget | **B+** — burnt orange reads as a decision | **A** | D+ |
| 11 | Front page cohesion | **A** — one identity, two media | **A** | C |
| 12 | axe 0 violations | **A** — 0 across 9 | **A** — 0 across 9 | A |
| 13 | Primary text contrast | **A** — 15.34 / 17.05 / 14.10 | **A** | A |
| 14 | Responsive, no overflow | **A** — 5 widths | **A** — 5 widths | A |
| 15 | Reduced motion | **A** | **A** | A |
| 16 | Keyboard + focus | **A** | **A** | A |
| 23 | **Mobile navigation is readable** | **A** | **A** | *not an axis in pass 0* |

**Nothing below B+.** The loop is closed for the axes it opened on.

## Fixed on the way, each caught by looking

- **The mobile menu had no background.** Border, radius, padding, no fill — and
  the header is `position: fixed`, so eight transparent items landed on the
  hero. Unreadable on every page, in both grounds, on any phone. Pre-existing,
  and no automated check covers "you can read the navigation".
- **Two filled CTAs were labelled `--paper`.** 15.9:1 on night, **2.3:1** on
  paper. Exactly the failure `a7ab861` predicted in its own commit message.
- **A duplicate id.** `<Navigation>` renders its slot twice, so `id="theme-toggle"`
  shipped twice and only the first was wired. axe no longer flags duplicate ids.
- **`opacity: 0.8` on a passing token.** `--paper-soft` clears 4.5 everywhere;
  multiplying it by 0.8 dropped it to 4.3. If something needs to be quieter than
  a token, that is a token, not an alpha.
- **A translucent disabled fill.** 45% crimson reads as a darker chip on night
  and as broken on paper. Now an opaque muted surface, identical on both.

## Proved, not patched

The first dual-theme run reported six contrast failures, two of them in dark —
where nothing had changed. They were the reveal animation: axe measures rendered
colour, reveals cap at 320ms, and the sweep analysed at 140ms. Adding settle
time cleared all six **with no code change**. Lowering axe's threshold would
have been turning the check off.

## Still open — carried to the next pass

Axes 17–22 from pass 0 are untouched. They are real and none is a theme bug:

- **17** spike-train rail missing on `/research` (0 elements there, 4 on `/`)
- **18** `/about` portrait mask still tuned for a dark ground
- **19** ~45% dead column on `/about` and `/research`
- **20** the portrait's blue is in no token
- **21** spike marks thin on light
- **22** Cue's media tile is a gradient *fill*, against DESIGN.md P5
