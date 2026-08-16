#!/usr/bin/env node
/**
 * COVERAGE AND REPETITION — the instrument for "some names were always provided
 * while others weren't".
 *
 * WHY IT IS SEPARATE FROM run.mjs. run.mjs answers "when a question has a known
 * right answer, is it in the top 3?" — precision on 85 graded cases, scored
 * against `match.retrieve()`. It cannot see the failure a visitor actually
 * reports, because that failure is about the questions NOT in the case file and
 * about names that never appear for any of them.
 *
 * And it measures a different function. Production assembles the pool through
 * readAsk -> pool() -> rank(); run.mjs calls retrieve(). Both are real, they
 * fail differently, and Success@3 = 0.988 is a statement about the one the page
 * does not use to choose a pool. This harness deliberately walks the shipping
 * path.
 *
 * WHAT IT REPORTS, and every number is a claim someone can check:
 *   reach       how much of the corpus can EVER come back
 *   repetition  how lopsided the appearances are — the visitor's complaint
 *   silence     questions that return nothing
 *
 * The query set is not a benchmark and does not pretend to be. It is a spread
 * of the kinds of thing this page is asked, held fixed so the numbers are
 * comparable across changes. Widen it deliberately, never to make a number move.
 *
 *   node scripts/naam/eval/coverage.mjs
 *   node scripts/naam/eval/coverage.mjs --json      machine-readable
 *   node scripts/naam/eval/coverage.mjs --hubs 20   show more offenders
 */
import { loadAsk, loadRows, loadThesaurus } from './compile.mjs'

const argv = process.argv.slice(2)
const has = (flag) => argv.includes(flag)
const valueOf = (flag, fallback) => {
  const i = argv.indexOf(flag)
  return i >= 0 && argv[i + 1] ? Number(argv[i + 1]) : fallback
}

const rows = loadRows()
const thesaurus = loadThesaurus()
const { ask } = await loadAsk(thesaurus)

/**
 * Six axes, because a set drawn from one axis measures that axis rather than
 * the corpus. `real` is the register the composer actually invites — whole
 * sentences, not keywords — and the chip prompts the page ships with.
 */
const AXES = {
  meaning: ['light', 'water', 'sky', 'fire', 'earth', 'wind', 'moon', 'sun', 'star', 'mountain',
    'river', 'ocean', 'forest', 'lotus', 'dawn', 'dusk', 'rain', 'cloud', 'stone', 'gold'],
  quality: ['calm', 'strong', 'gentle', 'wise', 'brave', 'kind', 'bright', 'pure', 'joyful',
    'patient', 'fierce', 'humble', 'steady', 'clever', 'noble', 'free', 'quiet', 'warm',
    'fearless', 'generous'],
  sound: ['soft', 'short', 'lyrical', 'crisp', 'round', 'open', 'two syllables', 'easy to say',
    'flowing', 'musical', 'deep', 'melodic'],
  domain: ['from the vedas', 'from the sutras', 'a teacher', 'a king', 'a sage', 'a river name',
    'a plant', 'a bird', 'a god', 'a warrior', 'a poet', 'a scholar'],
  use: ['easy abroad', 'easy for grandparents', 'works in english', 'not too common', 'unusual',
    'traditional', 'modern feeling', 'formal', 'friendly'],
  real: [
    'Short names — two syllables, easy to call across a room. Something like Bisa.',
    'Soft names, gentle to say. Something like Samaya.',
    'Names with some strength in them. Something like Virya.',
    'Names that mean water, or the sea. Something like Samudda.',
    'Names that are easy to say abroad as well as at home. Something like Bodha.',
    'Names out of the Sutras. Something like Shamatha.',
    'something calm, about love not war',
    'a name that means light',
    'names that sound soft when you say them out loud',
    'something to do with water or rivers',
    'a strong name but not aggressive',
    'a name a teacher might have',
    'easy for his grandparents in Kathmandu to say',
    'a short name, two syllables',
    'names about the sky or the dawn',
  ],
}
const QUERIES = Object.values(AXES).flat()

const appear = new Map()
const silent = []
for (const q of QUERIES) {
  const ids = ask.readAsk(q, rows).poolIds ?? []
  if (ids.length === 0) silent.push(q)
  for (const id of ids) appear.set(id, (appear.get(id) ?? 0) + 1)
}

const counts = [...appear.values()].sort((a, b) => b - a)
const total = counts.reduce((s, n) => s + n, 0)
const share = (k) => (counts.slice(0, k).reduce((s, n) => s + n, 0) / total) * 100

/** Gini over the WHOLE corpus, so names that never appear count as zeros. */
const all = [...counts, ...Array(Math.max(0, rows.length - appear.size)).fill(0)].sort((a, b) => a - b)
const sum = all.reduce((s, v) => s + v, 0)
const gini = sum === 0 ? 0 : all.reduce((s, v, i) => s + (2 * (i + 1) - all.length - 1) * v, 0) / (all.length * sum)

const byId = new Map(rows.map((r) => [r.id, r]))

/**
 * THE DENOMINATOR IS NOT THE CORPUS, and getting that wrong sends the whole
 * investigation somewhere expensive and wrong.
 *
 * Reach against all 2,098 shipped rows reads 31%, which looks like a crisis and
 * invites a rebuild. But the rows that never come back are not being unfairly
 * excluded — the DOCUMENT declines them. Measured: rows the pool returns are
 * 75.1% evocative-badged; rows it never returns are 15.5% evocative and 84.5%
 * bare-attested, and 24.3% of them have a gloss of the form "name of a man".
 * There are 85 rows whose entire gloss is "name of a man". Driving reach to
 * 100% means dealing those to somebody choosing a name for their son.
 *
 * So the honest measure is reach across the names a family could plausibly
 * want. Against 590 evocative rows the same pool reaches 74.1%, and the real
 * gap is 154 names — of which 139 ALREADY CARRY A THEME. Those are a ranking
 * failure, not a vocabulary failure, and they are the thing to fix.
 *
 * `missedEvocativeThemed` is the gate that cannot be gamed by returning more
 * names: padding the pool with bare-attested rows moves reach, gini and hub all
 * in the flattering direction and leaves this number exactly where it was.
 */
const evocative = rows.filter((r) => r.badges?.evocative)
const reachedEv = evocative.filter((r) => appear.has(r.id))
const missedEvThemed = evocative.filter((r) => !appear.has(r.id) && (r.themes ?? []).length > 0)
const missedEvUnthemed = evocative.filter((r) => !appear.has(r.id) && (r.themes ?? []).length === 0)
const hubs = [...appear.entries()].sort((a, b) => b[1] - a[1])

const report = {
  queries: QUERIES.length,
  corpus: rows.length,
  everReturned: appear.size,
  everReturnedPct: +((appear.size / rows.length) * 100).toFixed(1),
  silent: silent.length,
  maxAppearances: counts[0] ?? 0,
  top10Share: +share(10).toFixed(1),
  gini: +gini.toFixed(3),
  worst: hubs.slice(0, 5).map(([id, n]) => ({ name: byId.get(id)?.latin ?? id, in: n })),
  evocative: evocative.length,
  evocativeReached: reachedEv.length,
  evocativeReachedPct: +((reachedEv.length / evocative.length) * 100).toFixed(1),
  missedEvocativeThemed: missedEvThemed.length,
  missedEvocativeUnthemed: missedEvUnthemed.length,
}

if (has('--json')) {
  console.log(JSON.stringify(report, null, 2))
} else {
  console.log(`\n  queries          ${report.queries}`)
  console.log(`  corpus           ${report.corpus}`)
  console.log(`\n  REACH`)
  console.log(`  ever returned    ${report.everReturned}  (${report.everReturnedPct}% of the corpus)`)
  console.log(`  never returned   ${report.corpus - report.everReturned}`)
  console.log(`  silent queries   ${report.silent}${report.silent ? '  -> ' + silent.slice(0, 4).map((q) => JSON.stringify(q.slice(0, 30))).join(', ') : ''}`)
  console.log(`\n  REPETITION       (the visitor's complaint)`)
  console.log(`  worst name in    ${report.maxAppearances}/${report.queries} pools`)
  console.log(`  top 10 take      ${report.top10Share}% of all appearances`)
  console.log(`  gini             ${report.gini}   (0 even, 1 one name takes everything)`)
  console.log(`\n  REACH THAT MATTERS   (the corpus declines the rest, and is right to)`)
  console.log(`  evocative rows   ${report.evocative}`)
  console.log(`  reached          ${report.evocativeReached}  (${report.evocativeReachedPct}%)`)
  console.log(`  MISSED, themed   ${report.missedEvocativeThemed}   <- a ranking failure. THE GATE.`)
  console.log(`  missed, unthemed ${report.missedEvocativeUnthemed}   <- a vocabulary failure`)
  console.log(`\n  MOST REPEATED`)
  for (const [id, n] of hubs.slice(0, valueOf('--hubs', 10))) {
    const r = byId.get(id)
    console.log(`    ${String(n).padStart(3)}/${report.queries}  ${(r?.latin ?? id).padEnd(14)} ${(r?.gloss ?? '').slice(0, 44)}`)
  }
  console.log()
}
