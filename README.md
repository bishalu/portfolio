# bishal.ai

[![Built with Astro](https://astro.badg.es/v2/built-with-astro/small.svg)](https://astro.build)
[![Netlify Status](https://api.netlify.com/api/v1/badges/bd403085-8c0c-47f5-9c1e-a4ee44ef57bd/deploy-status)](https://app.netlify.com/sites/bishalup/deploys)

Personal site for Bishal Upadhyaya — AI systems architect. Three music-tech products under
Vibeset, four peer-reviewed papers, and a set of demonstrations that run against real systems.

**Design spec: [`docs/design/DESIGN.md`](docs/design/DESIGN.md) · Motion spec:
[`docs/design/MOTION.md`](docs/design/MOTION.md).** Those two files govern; the code follows
them. If code and spec disagree, one of them is a bug.

## Stack

- **[Astro 5](https://astro.build/)** — `output: 'server'` with `prerender: true` on every page
  except the chat endpoint. Static speed, one live route.
- **React 19** islands, all `client:visible`. Three of them: `VibeFinder`, `ChoonStressTest`,
  and the Cue sync workbench.
- **Tailwind CSS v4** (CSS-configured — there is no `tailwind.config.js`) + SCSS for component
  styles.
- **No animation library.** The oscilloscope, loading trace and console bloom are hand-written
  Canvas and Web Animations API. This is deliberate — see MOTION.md.
- **Netlify** — adapter, Functions, Blobs, Forms, and a scheduled function.

## Quick start

```bash
npm install
npm run dev          # localhost:4321 — pages only
npx netlify dev      # localhost:8888 — pages + functions (the demos need this)
```

The site builds and runs with no environment variables. Each one lights up an optional
surface; without it, that surface degrades honestly rather than breaking. See
[`.env.example`](.env.example).

| Command | Action |
| :--- | :--- |
| `npm run dev` | Dev server at `localhost:4321` |
| `npm run build` | Production build to `./dist/` |
| `npm run preview` | Preview the build |
| `npx netlify dev --port 8888` | Dev server **with** functions — needed for the live demos |

## Layout

```
docs/design/       DESIGN.md + MOTION.md — the governing spec
src/
  components/      Astro components; .tsx files are the React islands
  content/         MDX collections: products, publications
  layouts/         DefaultLayout
  pages/           File-based routes; pages/api/chat.ts is the one SSR endpoint
  scripts/         reveal.ts — the IntersectionObserver entrance observer
  styles/          tokens.css → motion.css → tailwind.css
  assets/scss/     Base styles, Utopia scales, mixins
netlify/functions/ Demo proxy, live-signals, the leaderboard cron
scripts/verify/    Playwright verification runner
```

## Design system — Alpenglow

Dark only. One deep twilight canvas; crimson→marigold is the light source, glacier the cold
counterpoint, bone paper is what the light lands on.

```
--void #0e1124   --ink #181c30   --ink-2 #222741      canvas → panel → well
--paper #faf7f1  --paper-soft #b0b5ca                  text
--crimson #d64553  --marigold #efa33b  --glacier #4ca8a2   accents
```

Type: **Bricolage Grotesque** (display) · **Atkinson Hyperlegible** (body, chosen for
legibility) · **DM Mono** (labels, numbers, status). Tokens live in
[`src/styles/tokens.css`](src/styles/tokens.css); the reasoning lives in DESIGN.md.

## Verification

Changes are judged against the spec by a script, not by eye:

```bash
npm run build
npx netlify dev --port 8888
node scripts/verify/run.mjs all http://localhost:8888
```

Floors, all enforced: axe **0 violations** · Lighthouse accessibility **100** ·
best-practices **100** · CLS **0** · TBT **≤ 200 ms** · mobile performance ≥ 85.

Full procedure in [`.claude/skills/verify-site/SKILL.md`](.claude/skills/verify-site/SKILL.md),
including the failure drills that prove the demos degrade to replay rather than blanking.

## Accessibility

WCAG 2.2 AA, verified rather than asserted — and the
[accessibility statement](https://bishal.ai/accessibility-statement) lists the known gaps as
well as the passing gates.

## Deployment

Netlify: build `npm run build`, publish `dist`, functions in `netlify/functions`. Environment
variables are set in the Netlify dashboard, never committed.

## License

MIT — see [LICENSE](LICENSE).
