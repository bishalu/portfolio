---
name: verify-site
description: Full verification pass for the bishal.ai portfolio — screenshot matrix, console/hydration checks, axe accessibility, live-demo widget drills, function failure drills, and Lighthouse. Use after any visual or functional change, before deploying.
---

# Verify Site

Run the whole suite (or one stage) against a local server or a deploy preview.
Design criteria live in `docs/design/DESIGN.md` + `docs/design/MOTION.md` —
screenshots are judged against those, not vibes.

## Setup (once per session)

```bash
npx playwright install chromium   # if browsers are missing
npm run build
npx netlify dev --port 8888       # dev + functions (live demo proxy works)
# or: npx netlify serve --port 8899  (production build + functions)
```

## Stages

```bash
node scripts/verify/run.mjs shots   [baseUrl]  # 390/768/1280/1920 + reduced-motion, full-page, reveal-aware
node scripts/verify/run.mjs console [baseUrl]  # page errors + hydration across all routes
node scripts/verify/run.mjs a11y    [baseUrl]  # axe on /, /about, /vibeset/choon, /research — must be 0 violations
node scripts/verify/run.mjs widgets [baseUrl]  # VibeFinder round-trip + Choon identify, with element screenshots
node scripts/verify/run.mjs all     [baseUrl]
```

Output lands in `scripts/verify/out/` (gitignored). **Read the screenshots** —
the point is visual judgment against DESIGN.md, not just green exit codes.

## Function failure drills (never-blank guarantee)

```bash
# Replay path: point the proxy at a dead upstream, restart the server, then:
VIBESET_API_BASE=https://example.invalid npx netlify dev --port 8888
curl -s -X POST localhost:8888/api/vibeset-demo -H 'Content-Type: application/json' \
  -d '{"kind":"search","genres":["deep house"],"limit":3}'   # → source:"replay", tracks non-empty
curl -s localhost:8888/api/live-signals                       # → JSON, nulls allowed, never 500
```

## Lighthouse (production build only — dev servers score falsely low)

```bash
npx netlify serve --port 8899
npx lighthouse http://localhost:8899/ --quiet \
  --chrome-flags="--headless --no-sandbox" \
  --only-categories=performance,accessibility,best-practices \
  --form-factor=mobile --screenEmulation.mobile --output=json --output-path=/tmp/lh.json
```

Floors: accessibility **100**, best-practices **100**, CLS **0**, TBT **≤ 200ms**.
Local mobile performance ≥ 85 (≈ 90+ on production CDN).

## Manual spot checks (things scripts can't judge)

- First visit in a fresh session: the loading trace plays once (≤1.6s), is
  click-skippable, and hands off into the hero entrance without a flash.
- Balgo: send a message — oscilloscope gets excited while thinking, pulse-line
  loader (never a spinner), reply cascades in, deep-link chips scroll correctly.
  On a phone width, the chat opens as a full-screen sheet with a close button.
- Choon: play a clip, switch presets mid-play (chain retunes live), identify →
  result card is labeled illustrative.
- Reduced motion (OS setting or DevTools): page fully drawn and functional,
  nothing moves, pulse-lines render as static gradient lines.
