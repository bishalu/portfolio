/**
 * Generates src/generated/balgo-context.ts from the content collections.
 *
 * Balgo's system prompt used to be a hand-maintained fourth copy of the
 * product facts, and it had drifted — it carried numbers that appeared
 * nowhere else on the site, and named anchors (#vibeset-cue) that don't
 * exist in the DOM.
 *
 * Split of responsibilities: persona, tone and behaviour stay hand-written in
 * src/pages/api/chat.ts. Only FACTS are generated here. If you want to change
 * how Balgo talks, edit chat.ts. If you want to change what Balgo knows, edit
 * the MDX.
 *
 * Runs as part of `prebuild`.
 */
import { readdirSync, readFileSync, writeFileSync, mkdirSync } from 'node:fs'
import { join } from 'node:path'

const OUT = 'src/generated/balgo-context.ts'

/** Front-matter reader for the flat + list-of-objects shapes we use. */
function frontmatter(src) {
  const m = src.match(/^---\n([\s\S]*?)\n---/)
  if (!m) return {}
  const out = {}
  const lines = m[1].split('\n')
  let key = null
  let list = null
  let block = null

  const flush = () => {
    if (key && list) out[key] = list
    if (key && block !== null) out[key] = block.join(' ').trim()
    list = null
    block = null
  }

  for (const line of lines) {
    const top = line.match(/^(\w+):\s*(.*)$/)
    if (top) {
      flush()
      key = top[1]
      const val = top[2].trim()
      if (val === '' || val === '>-' || val === '|') {
        if (val === '') list = []
        else block = []
      } else {
        out[key] = val.replace(/^['"]|['"]$/g, '')
        key = null
      }
      continue
    }
    if (block) {
      block.push(line.trim())
      continue
    }
    if (list) {
      const item = line.match(/^ {2}- (.*)$/)
      if (item) {
        const inline = item[1].match(/^(\w+):\s*(.*)$/)
        list.push(inline ? { [inline[1]]: clean(inline[2]) } : clean(item[1]))
        continue
      }
      const field = line.match(/^ {4}(\w+):\s*(.*)$/)
      if (field && list.length && typeof list[list.length - 1] === 'object') {
        list[list.length - 1][field[1]] = clean(field[2])
      }
    }
  }
  flush()
  return out
}

const clean = (v) => v.replace(/^['"]|['"]$/g, '').trim()

const read = (dir) =>
  readdirSync(dir)
    .filter((f) => f.endsWith('.mdx') && !f.startsWith('_'))
    .map((f) => ({ slug: f.replace(/\.mdx$/, ''), ...frontmatter(readFileSync(join(dir, f), 'utf8')) }))

const products = read('src/content/products').sort((a, b) => Number(a.order) - Number(b.order))
const capabilities = read('src/content/capabilities')
const publications = read('src/content/publications').sort((a, b) => Number(b.year) - Number(a.year))

const lines = []

lines.push("--- SECTION: Vibeset — Bishal's company (Anchor: #vibeset) ---")
lines.push(
  'Vibeset is AI music tooling: three products spanning the music lifecycle — find it, fit it, prove it. Never mention the catalog track count.',
)
for (const p of products) {
  const caps = capabilities.filter((c) => c.product === p.slug).map((c) => c.name)
  const stats = (p.stats || []).map((s) => `${s.label} ${s.value}`).join(', ')
  // The ledger already records what ships and what does not. Balgo never saw
  // that column, so it read the marketing description as fact and told a
  // visitor Choon "returns a signed C2PA manifest" — which the ledger marks
  // `building`, c2pa-python binding not yet wired. Promising a prospect a
  // feature that does not exist is the most expensive mistake this site can
  // make, so the status travels with the facts now.
  //
  // Phrased as UNPROVEN rather than NOT SHIPPED because the two `building`
  // rows mean different things: Choon's C2PA binding is unfinished, while
  // Curation's "Licensed catalog" is licensed but not publicly documented.
  // A single "not yet shipped" label would have had Balgo tell a prospect the
  // catalogue is unlicensed — worse than the bug it was fixing.
  const building = (p.claims || []).filter((c) => c.status === 'building')
  const shipping = (p.claims || []).filter((c) => c.status === 'shipping')
  // Shipped claims carry the evidence pointers — they are the whole reason the
  // ledger exists, and Balgo could not see them. Without this it had the
  // marketing description and the headline stats and nothing underneath, which
  // is how it ended up inventing mechanisms when a prospect pushed.
  const proven = (p.claims || []).filter((c) => (c.status || 'shipped') === 'shipped' && c.evidence)
  const bits = [
    `*   ${p.name} (${p.eyebrow}, Anchor: #vibeset, page: /vibeset/${p.slug}${p.liveUrl ? `, live: ${p.liveUrl}` : ''}): ${p.tagline}.`,
    p.description,
    stats && `Measured: ${stats}.`,
    caps.length && `Capabilities: ${caps.join(', ')}.`,
    proven.length &&
      `PROVEN — each of these has evidence behind it, cite it when it helps: ${proven
        .map((c) => `${c.claim} [${c.evidence}${c.evidenceUrl ? `, see ${c.evidenceUrl}` : ''}]`)
        .join('; ')}.`,
    shipping.length &&
      `CORRECTION TO THE DESCRIPTION ABOVE — anything here is NOT live yet. The description is marketing copy written ahead of the work; this is what is true today. Speak of these in the near future, with the date, and never in the present tense: ${shipping
        .map((c) => `${c.claim} (live ${c.eta}) — ${c.evidence}`)
        .join('; ')}.`,
    building.length &&
      `CORRECTION TO THE DESCRIPTION ABOVE — unproven, never state as fact. If asked, say where each one actually stands: ${building
        .map((c) => `${c.claim} — ${c.evidence}`)
        .join('; ')}.`,
  ].filter(Boolean)
  lines.push(bits.join(' '))
}

lines.push('')
lines.push('--- SECTION: Research (Anchor: #research) ---')
lines.push(`${publications.length} peer-reviewed papers. Deep-link with #paper-<slug>:`)
for (const pub of publications) {
  const applied = pub.appliedIn ? ` Applied in production: ${pub.appliedIn} — ${pub.appliedNote}` : ''
  lines.push(
    `*   ${pub.title} (${pub.year}${pub.journal ? `, ${pub.journal}` : ''}, Anchor: #paper-${pub.slug}).${applied}`,
  )
}

mkdirSync('src/generated', { recursive: true })
writeFileSync(
  OUT,
  `// GENERATED by scripts/gen-balgo-context.mjs — do not edit.
// Facts come from src/content/{products,capabilities,publications}.
// Balgo's persona and tone are hand-written in src/pages/api/chat.ts.
export const BALGO_FACTS = ${JSON.stringify(lines.join('\n'))}
`,
)

console.log(
  `  balgo context — ${products.length} products · ${capabilities.length} capabilities · ${publications.length} papers → ${OUT}`,
)
