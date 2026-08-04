/**
 * Score /naam retrieval against the graded query set.
 *
 *   node scripts/naam/eval/run.mjs              score, and diff against the baseline
 *   node scripts/naam/eval/run.mjs --save       write the current numbers as the baseline
 *   node scripts/naam/eval/run.mjs --failures   print every case that missed
 *   node scripts/naam/eval/run.mjs --case moon  dump one case in full
 *
 * Offline: reads names-core.json off disk and calls the real retrieve(). No
 * Bedrock, no dev server, no network, no cost. Runs in about a second, which is
 * the only reason it will actually get run.
 *
 * ─── WHY THESE THREE METRICS ───────────────────────────────────────────────
 *
 * SUCCESS@3 is the one that matters and it is deliberately first. The page deals
 * three cards. A visitor does not see rank 7. If nothing relevant is in the top
 * three, the page failed that person regardless of what the tail looked like.
 *
 * PRECISION@10 is the pool-quality number — the model picks its three from what
 * retrieval hands up, so junk at rank 4-10 is junk the model has to sort through
 * and occasionally picks. It is normalised by min(10, |gold|) so a concept with
 * only two right answers in the whole document can still score 1.0. Without that
 * normalisation "calm" — which has exactly two rows — would be capped at 0.2 and
 * would look like a failure when it was a complete success.
 *
 * MRR is the tie-breaker between two changes that both pass Success@3, and it is
 * the one that moves when ranking improves without coverage changing.
 *
 * NOISE IS SCORED INVERTED. Returning nothing is the pass. A page that answers
 * "asdfgh" with three confident names is worse than one that asks what they
 * meant, and a retrieval change that quietly starts matching gibberish should
 * cost points rather than go unnoticed.
 */
import { writeFileSync, readFileSync, existsSync } from 'node:fs'
import { resolve, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { loadMatch, loadRows, loadThesaurus, REPO } from './compile.mjs'
import { CASES, goldIds } from './cases.mjs'

const HERE = dirname(fileURLToPath(import.meta.url))
const BASELINE = resolve(HERE, 'baseline.json')

const argv = process.argv.slice(2)
const has = (flag) => argv.includes(flag)
const valueOf = (flag) => {
  const i = argv.indexOf(flag)
  return i >= 0 ? argv[i + 1] : null
}

/** How many the page actually shows, and how deep the model's pool goes. */
const SHOWN = 3
const POOL = 10

const rows = loadRows()
const match = await loadMatch()
/** Passed explicitly, exactly as the two call sites pass it, so the harness
 *  cannot pass without the table production runs with. `--no-thesaurus`
 *  measures the before picture on demand. */
const thesaurus = has('--no-thesaurus') ? {} : loadThesaurus()
const byLatin = new Map(rows.map((row) => [row.latin.toLowerCase(), row]))
const byId = new Map(rows.map((row) => [row.id, row]))

/* ── score one case ───────────────────────────────────────────────────────── */

function scoreCase(testCase) {
  const gold = goldIds(testCase, rows, byLatin)
  const hits = match.retrieve(rows, testCase.q, POOL, thesaurus)
  const ranked = hits.map((hit) => hit.row.id)

  if (testCase.tier === 'noise') {
    // Inverted: silence is the correct answer.
    const clean = ranked.length === 0
    return { ...testCase, goldSize: 0, returned: ranked.length, success: clean, precision: clean ? 1 : 0, rr: clean ? 1 : 0, ranked, gold }
  }

  const firstGold = ranked.findIndex((id) => gold.has(id))
  const inTop = ranked.slice(0, POOL).filter((id) => gold.has(id)).length
  return {
    ...testCase,
    goldSize: gold.size,
    returned: ranked.length,
    success: firstGold >= 0 && firstGold < SHOWN,
    // Normalised by what is reachable: a two-row concept can still be perfect.
    precision: inTop / Math.min(POOL, gold.size),
    rr: firstGold >= 0 ? 1 / (firstGold + 1) : 0,
    ranked,
    gold,
  }
}

const results = CASES.map(scoreCase)

/* ── aggregate ────────────────────────────────────────────────────────────── */

const mean = (list, pick) => (list.length === 0 ? 0 : list.reduce((sum, r) => sum + pick(r), 0) / list.length)
const TIERS = ['literal', 'gap', 'phrase', 'noise']

function summarise(list) {
  return {
    n: list.length,
    success: mean(list, (r) => (r.success ? 1 : 0)),
    precision: mean(list, (r) => r.precision),
    mrr: mean(list, (r) => r.rr),
  }
}

const report = { overall: summarise(results) }
for (const tier of TIERS) {
  const list = results.filter((r) => r.tier === tier)
  if (list.length > 0) report[tier] = summarise(list)
}

/* ── print ────────────────────────────────────────────────────────────────── */

const pct = (n) => `${(n * 100).toFixed(1)}%`.padStart(6)
const dim = (s) => `\x1b[2m${s}\x1b[0m`

const single = valueOf('--case')
if (single) {
  const r = results.find((x) => x.q === single || x.q.includes(single))
  if (!r) {
    console.error(`no case matching "${single}"`)
    process.exit(1)
  }
  console.log(`\n"${r.q}"  [${r.tier}]`)
  if (r.note) console.log(dim(`  ${r.note}`))
  console.log(`  gold: ${r.goldSize} rows · returned: ${r.returned} · success@${SHOWN}: ${r.success}`)
  console.log(`  P@${POOL}: ${pct(r.precision)} · RR: ${r.rr.toFixed(3)}\n`)
  r.ranked.forEach((id, i) => {
    const row = byId.get(id)
    const mark = r.gold.has(id) ? '\x1b[32m✓\x1b[0m' : '\x1b[31m·\x1b[0m'
    console.log(`  ${mark} ${String(i + 1).padStart(2)}. ${row.latin.padEnd(14)} ${String(row.gloss).slice(0, 58)}`)
  })
  if (r.tier !== 'noise') {
    const missed = [...r.gold].filter((id) => !r.ranked.includes(id)).slice(0, 8)
    if (missed.length > 0) {
      console.log(dim(`\n  gold rows NOT returned (first ${missed.length}):`))
      for (const id of missed) console.log(dim(`     ${byId.get(id).latin.padEnd(14)} ${String(byId.get(id).gloss).slice(0, 54)}`))
    }
  }
  process.exit(0)
}

const tsize = Object.keys(thesaurus).length
console.log(`\n  /naam retrieval — ${results.length} cases, ${rows.length} rows, ${tsize} thesaurus entries\n`)
console.log(`  ${'tier'.padEnd(9)} ${'n'.padStart(3)}  ${'success@3'.padStart(9)} ${'P@10'.padStart(6)} ${'MRR'.padStart(6)}`)
console.log(`  ${'─'.repeat(40)}`)
for (const tier of TIERS) {
  if (!report[tier]) continue
  const s = report[tier]
  console.log(`  ${tier.padEnd(9)} ${String(s.n).padStart(3)}  ${pct(s.success)}    ${pct(s.precision)} ${pct(s.mrr)}`)
}
console.log(`  ${'─'.repeat(40)}`)
const o = report.overall
console.log(`  ${'overall'.padEnd(9)} ${String(o.n).padStart(3)}  ${pct(o.success)}    ${pct(o.precision)} ${pct(o.mrr)}\n`)

/* Total misses are the headline failure: retrieval returned nothing at all. */
const empty = results.filter((r) => r.tier !== 'noise' && r.returned === 0)
if (empty.length > 0) {
  console.log(`  \x1b[31m${empty.length} queries returned NOTHING:\x1b[0m ${empty.map((r) => `"${r.q}"`).join(', ')}\n`)
}

if (has('--failures')) {
  const bad = results.filter((r) => !r.success)
  console.log(`  ${bad.length} cases missed:\n`)
  for (const r of bad) {
    const head = r.ranked.slice(0, 3).map((id) => byId.get(id).latin).join(', ') || '—'
    console.log(`  [${r.tier}] "${r.q}"`)
    console.log(dim(`      gold ${r.goldSize} · got: ${head}`))
    if (r.note) console.log(dim(`      ${r.note}`))
  }
  console.log()
}

/* ── baseline diff ────────────────────────────────────────────────────────── */

if (has('--save')) {
  writeFileSync(BASELINE, JSON.stringify({ at: new Date().toISOString(), report }, null, 2))
  console.log(`  baseline written to ${BASELINE.replace(REPO + '/', '')}\n`)
} else if (existsSync(BASELINE)) {
  const prev = JSON.parse(readFileSync(BASELINE, 'utf8'))
  console.log(`  vs baseline (${prev.at.slice(0, 16).replace('T', ' ')}):`)
  let regressed = false
  for (const key of ['overall', ...TIERS]) {
    if (!report[key] || !prev.report[key]) continue
    const d = report[key].success - prev.report[key].success
    const dp = report[key].precision - prev.report[key].precision
    if (Math.abs(d) < 1e-9 && Math.abs(dp) < 1e-9) continue
    const arrow = d > 0 ? '\x1b[32m▲\x1b[0m' : d < 0 ? '\x1b[31m▼\x1b[0m' : ' '
    if (d < -1e-9) regressed = true
    console.log(`    ${arrow} ${key.padEnd(9)} success ${d >= 0 ? '+' : ''}${(d * 100).toFixed(1)}pp · P@10 ${dp >= 0 ? '+' : ''}${(dp * 100).toFixed(1)}pp`)
  }
  console.log(regressed ? '\n  \x1b[31mregression\x1b[0m\n' : '\n  no regression\n')
} else {
  console.log(dim('  no baseline — run with --save to record one\n'))
}
