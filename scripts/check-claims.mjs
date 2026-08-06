/**
 * Claims ledger — an internal build-time check, not a public label.
 *
 * Every claim the products make carries a pointer to what backs it. This
 * script enforces one rule and reports the rest:
 *
 *   status: shipped   MUST have evidence. No evidence → the build fails.
 *   status: shipping  Done or all but, with a date. MUST have evidence AND eta.
 *   status: building  Real work in flight, no date. Renders normally, listed here so
 *                     it can't quietly become permanent.
 *
 * Nothing here changes a single word a visitor reads. The point is that when
 * an implementation lands, flipping `building` → `shipped` forces you to
 * attach the evidence in the same commit — so the site's numbers can never
 * drift away from what actually runs.
 *
 * Usage:  node scripts/check-claims.mjs [--quiet]
 * Runs automatically as `prebuild`.
 */
import { readdirSync, readFileSync } from 'node:fs'
import { join } from 'node:path'

const DIR = 'src/content/products'
const quiet = process.argv.includes('--quiet')

/** Minimal front-matter reader — enough for the claims block, no YAML dep. */
function parseClaims(src) {
  const fm = src.match(/^---\n([\s\S]*?)\n---/)
  if (!fm) return []
  const lines = fm[1].split('\n')
  const start = lines.findIndex((l) => l === 'claims:')
  if (start === -1) return []

  const claims = []
  let current = null
  for (const line of lines.slice(start + 1)) {
    if (/^\S/.test(line)) break // dedented to a new top-level key
    const item = line.match(/^ {2}- (\w+):\s*(.*)$/)
    if (item) {
      if (current) claims.push(current)
      current = { [item[1]]: strip(item[2]) }
      continue
    }
    const field = line.match(/^ {4}(\w+):\s*(.*)$/)
    if (field && current) current[field[1]] = strip(field[2])
  }
  if (current) claims.push(current)
  return claims
}

const strip = (v) => v.replace(/^['"]|['"]$/g, '').replace(/^>-\s*$/, '').trim()

const rows = []
for (const file of readdirSync(DIR).filter((f) => f.endsWith('.mdx'))) {
  const product = file.replace(/\.mdx$/, '')
  for (const c of parseClaims(readFileSync(join(DIR, file), 'utf8'))) {
    rows.push({ product, ...c, status: c.status || 'shipped' })
  }
}

const missing = rows.filter((r) => (r.status === 'shipped' || r.status === 'shipping') && !r.evidence)
// A `shipping` row without a date is just `building` wearing optimism, and
// Balgo would relay the optimism to a prospect as a commitment.
const undated = rows.filter((r) => r.status === 'shipping' && !r.eta)
const building = rows.filter((r) => r.status === 'building')
const shipping = rows.filter((r) => r.status === 'shipping')

if (!quiet) {
  const shipped = rows.length - building.length - shipping.length
  console.log(
    `\n  claims ledger — ${rows.length} total · ${shipped} shipped · ${shipping.length} shipping · ${building.length} building`,
  )
  if (shipping.length) {
    console.log('\n  landing soon (Balgo may say so, with the date):')
    for (const r of shipping) console.log(`    · ${r.product}: ${r.claim} — ${r.eta}`)
  }
  if (building.length) {
    console.log('\n  in flight (renders now, evidence pending):')
    for (const r of building) console.log(`    · ${r.product}: ${r.claim}`)
  }
  console.log('')
}

if (undated.length) {
  console.error('\n  FAIL: claims marked `shipping` with no `eta`:\n')
  for (const r of undated) console.error(`    · ${r.product}: ${r.claim}`)
  console.error('\n  Give it an ISO date, or set `status: building` until it has one.\n')
  process.exit(1)
}

if (missing.length) {
  console.error('\n  FAIL: claims marked `shipped` with no evidence pointer:\n')
  for (const r of missing) console.error(`    · ${r.product}: ${r.claim}`)
  console.error('\n  Attach `evidence:` (a run, a file, a paper, or a URL), or set')
  console.error('  `status: building` until it ships.\n')
  process.exit(1)
}
