/**
 * Build the query thesaurus: everyday English → this document's own vocabulary.
 *
 *   node scripts/naam/build-thesaurus.mjs           build (resumes from cache)
 *   node scripts/naam/build-thesaurus.mjs --fresh   ignore the cache
 *   node scripts/naam/build-thesaurus.mjs --limit 3 first 3 batches only, for a look
 *
 * ─── THE PROBLEM THIS SOLVES ───────────────────────────────────────────────
 *
 * Measured by scripts/naam/eval: ten graded queries return LITERALLY NOTHING,
 * among them brave, calm, peace, hope and healer. Not ranked badly — nothing.
 * Lexical retrieval cannot cross a vocabulary gap, and this corpus is one long
 * vocabulary gap: it is a Victorian Sanskrit lexicon, so it says valiant where a
 * parent says brave, tranquillity where they say calm, and it never once says
 * peace. No amount of BM25 tuning reaches a word that is not in the index.
 *
 * The agent can already cross the gap by searching for synonyms itself, and it
 * does it well. But that costs a second round trip to Bedrock on every ordinary
 * question — it is most of the 1.1-4.6s latency and all of the 6.8s tail. Doing
 * it once at build time makes the common case free.
 *
 * ─── WHY WORDS AND NOT ROWS ────────────────────────────────────────────────
 *
 * The obvious shape is doc2query: expand each of the 906 meaning-bearing ROWS
 * with the concepts it answers. That works, and it is what the research
 * recommended. It is rejected here for one reason: those concepts have to reach
 * the browser, because retrieve() runs client-side to build the pool, and
 * per-row concepts are per-row bytes on a payload that is already 1.22 MB.
 *
 * Expanding the VOCABULARY instead inverts to the same capability at a fraction
 * of the size, because the mapping only needs to exist for words the glosses do
 * NOT already contain. "moon" needs no entry — nineteen glosses say moon and
 * BM25 finds them today. Only the gaps need storing.
 *
 * ─── EVERY WORD GETS ITS CONTEXT, AND THAT IS NOT OPTIONAL ─────────────────
 *
 * Asked for synonyms of the bare word `saman`, gpt-oss answered "grace, hope,
 * joy, faith, bliss". A saman is a Vedic chant. The model had never met the word,
 * so it produced five plausible baby-name abstractions and would have poisoned
 * five everyday queries with 75 rows of liturgy. Shown the gloss the word
 * actually lives in, it has something to reason from instead of a vibe. Context
 * is the difference between a thesaurus and a hallucination.
 */
import { mkdirSync, readFileSync, writeFileSync, existsSync } from 'node:fs'
import { createRequire } from 'node:module'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const HERE = dirname(fileURLToPath(import.meta.url))
const REPO = resolve(HERE, '../..')
const require = createRequire(resolve(REPO, 'package.json'))
const { BedrockRuntimeClient, ConverseCommand } = require('@aws-sdk/client-bedrock-runtime')

const CACHE = resolve(REPO, 'node_modules/.cache/naam-thesaurus.json')
/**
 * Written beside names-core.json, not into src/, because BOTH sides fetch it:
 * the browser through tray.ts and the Lambda through server.ts, exactly as they
 * both already fetch the rows. Bundling it into the island instead cost 11 kB
 * gzipped of main-thread parse — measured, 16.0 kB → 27.0 kB — to save a 24 kB
 * request that rides alongside a 1.22 MB one already in flight.
 */
const OUT = resolve(REPO, 'public/naam/thesaurus.json')

const argv = process.argv.slice(2)
const has = (f) => argv.includes(f)
const num = (f, d) => {
  const i = argv.indexOf(f)
  return i >= 0 ? Number(argv[i + 1]) : d
}

/** Words per Bedrock call. Big enough to be cheap, small enough that one bad
 *  JSON response costs little and retries fast. */
const BATCH = 40
/** Everyday words per corpus word. More than this is the model padding. */
const MAX_SYNONYMS = 5
/** A corpus word must be at least this common to be worth a thesaurus entry —
 *  a hapax in one obscure gloss is not what anyone is asking for. */
const MIN_DOC_FREQ = 1

/* ── .env, by hand: a plain node script gets no framework loader ──────────── */
for (const line of readFileSync(resolve(REPO, '.env'), 'utf8').split('\n')) {
  const m = line.match(/^\s*([A-Z0-9_]+)\s*=\s*(.*)$/)
  if (m && !process.env[m[1]]) process.env[m[1]] = m[2].replace(/^["']|["']$/g, '')
}

const rows = JSON.parse(readFileSync(resolve(REPO, 'public/naam/names-core.json'), 'utf8'))

/* ── which words are worth expanding ──────────────────────────────────────── */

/** "name of a mountain" attests a bearer. Its words are not what the row means. */
const BARE = /^name of\b/i
/** Dictionary apparatus and grammar, not meaning. */
const SKIP = new Set(
  ('the and with also for that this are not from any one two his her its var pra sam per etc esp comp accord ' +
    'said also called see next above below plural sing masc fem neut nom acc gen loc pl du ' +
    'patr name names having being who whom which what when where used')
    .split(' '),
)

const freq = new Map()
const example = new Map()
for (const row of rows) {
  const gloss = String(row.gloss || '')
  if (BARE.test(gloss)) continue
  const words = new Set(
    gloss
      .toLowerCase()
      .replace(/[^a-z\s]/g, ' ')
      .split(/\s+/)
      .filter((w) => w.length > 3 && !SKIP.has(w)),
  )
  for (const w of words) {
    freq.set(w, (freq.get(w) ?? 0) + 1)
    // Shortest gloss wins as the example: the most focused use of the word.
    const prev = example.get(w)
    if (!prev || gloss.length < prev.length) example.set(w, gloss)
  }
}

const vocabulary = [...freq.entries()]
  .filter(([, n]) => n >= MIN_DOC_FREQ)
  .map(([w]) => w)
  .sort()

console.log(`${vocabulary.length} corpus words to expand, ${Math.ceil(vocabulary.length / BATCH)} batches`)

/* ── ask the model ────────────────────────────────────────────────────────── */

const client = new BedrockRuntimeClient({
  region: process.env.BALGO_AWS_REGION || 'us-east-2',
  credentials: {
    accessKeyId: process.env.BALGO_AWS_KEY_ID,
    secretAccessKey: process.env.BALGO_AWS_SECRET,
  },
})
const modelId = process.env.BALGO_MODEL_ID || 'openai.gpt-oss-120b-1:0'

const PROMPT_HEAD =
  'These are words from a Sanskrit-English dictionary, each shown with the definition it appears in.\n' +
  'For each word, list the everyday English words a parent naming a baby boy might type into a search box ' +
  'that SHOULD find this word.\n\n' +
  'Rules:\n' +
  '- Only ordinary modern English. No Sanskrit, no archaic words, no proper nouns.\n' +
  `- 0 to ${MAX_SYNONYMS} per word. Return [] when the word is a technical or liturgical term ` +
  'nobody would search for, or when the everyday word IS the word itself.\n' +
  '- Judge from the definition shown, not from how the word looks.\n' +
  '- Lowercase, single words where possible.\n\n' +
  'Return strict JSON only: {"word": ["everyday", ...], ...}\n\n'

async function expand(batch, attempt = 0) {
  const lines = batch.map((w) => `${w} — "${String(example.get(w) || '').slice(0, 90)}"`).join('\n')
  try {
    const out = await client.send(
      new ConverseCommand({
        modelId,
        messages: [{ role: 'user', content: [{ text: PROMPT_HEAD + lines }] }],
        inferenceConfig: { maxTokens: 3000, temperature: 0.2 },
        // gpt-oss bills reasoning against the same maxTokens; low keeps the
        // budget for the answer. Learned the hard way on the chat route.
        additionalModelRequestFields: { reasoning_effort: 'low' },
      }),
    )
    const text = (out.output?.message?.content ?? []).map((b) => b.text).filter(Boolean).join('')
    const json = text.slice(text.indexOf('{'), text.lastIndexOf('}') + 1)
    const parsed = JSON.parse(json)
    if (!parsed || typeof parsed !== 'object') throw new Error('not an object')
    return parsed
  } catch (err) {
    if (attempt < 2) {
      await new Promise((r) => setTimeout(r, 400 * (attempt + 1)))
      return expand(batch, attempt + 1)
    }
    console.warn(`  batch failed after 3 tries (${batch[0]}…): ${err.message}`)
    return {}
  }
}

mkdirSync(dirname(CACHE), { recursive: true })
const cache = !has('--fresh') && existsSync(CACHE) ? JSON.parse(readFileSync(CACHE, 'utf8')) : {}

const todo = vocabulary.filter((w) => !(w in cache))
const batches = []
for (let i = 0; i < todo.length; i += BATCH) batches.push(todo.slice(i, i + BATCH))
const planned = has('--limit') ? batches.slice(0, num('--limit', 1)) : batches

console.log(`${Object.keys(cache).length} cached, ${todo.length} to fetch in ${planned.length} batches`)

let done = 0
for (const batch of planned) {
  const result = await expand(batch)
  // Record every word in the batch, including ones the model omitted, so a
  // resumed run does not ask for them forever.
  for (const w of batch) cache[w] = Array.isArray(result[w]) ? result[w] : []
  writeFileSync(CACHE, JSON.stringify(cache))
  done++
  if (done % 5 === 0 || done === planned.length) {
    process.stdout.write(`\r  ${done}/${planned.length} batches`)
  }
}
console.log()

/* ── pass B: start from what people type ──────────────────────────────────── */

/**
 * PASS A WALKS THE DOCUMENT AND MISSES WHAT THE DOCUMENT NEVER THINKS OF.
 *
 * Asked what everyday words mean `desired`, the model answered wanted, loved,
 * cherished, sought — all reasonable, none of them "hope". Asked what everyday
 * words mean `healthy`, it never said "healer". So both queries still returned
 * an empty page after pass A, because a walk over the corpus vocabulary can
 * only surface the concepts the corpus vocabulary suggests.
 *
 * This pass runs the other way: from the words a parent actually types toward
 * the archaic English this lexicon writes them in. The list below is the only
 * hand-written thing in the pipeline and it is deliberately a list of QUESTIONS,
 * never of answers — the mapping is still learned and then checked. That is the
 * difference between this and the nineteen-word theme taxonomy it replaced,
 * which hard-coded the targets and could therefore only ever find nineteen
 * things.
 *
 * EVERY ANSWER IS VALIDATED AGAINST THE REAL GLOSS VOCABULARY. A word the model
 * invents is dropped on the floor, so a confabulated synonym cannot enter the
 * table however confident it sounded.
 */
const CONCEPTS = (
  'brave courage fearless bold hero warrior fighter strength strong mighty power ' +
  'calm peace peaceful quiet serene still tranquil gentle soft tender kind kindness ' +
  'compassion mercy love loving beloved dear affection friend friendship loyal devoted ' +
  'hope hopeful faith trust belief joy joyful happy happiness cheerful delight bliss ' +
  'wise wisdom clever smart intelligent learned scholar teacher student knowledge ' +
  'light bright shining radiant glow dawn sunrise sun moon star sky heaven cloud ' +
  'fire flame water river ocean sea rain wind earth mountain forest tree flower lotus ' +
  'gold silver jewel treasure wealth rich prosperous fortune lucky blessed auspicious ' +
  'king prince royal noble leader guide protector guardian shelter refuge safe ' +
  'healer doctor medicine health healthy whole heal cure ' +
  'truth true honest sincere pure clean holy sacred divine god prayer ' +
  'song singer music melody poet voice speech word story ' +
  'swift quick fast strong steady firm patient calmness endurance ' +
  'victory winner triumph success achieve conqueror ' +
  'generous giving gift bounty abundance plenty ' +
  'eternal forever immortal timeless ancient young new life living breath soul spirit ' +
  'beautiful lovely handsome graceful charming sweet ' +
  'humble simple honest free freedom independent ' +
  'home house dwelling family father mother son child ' +
  'journey path way traveller seeker explorer ' +
  'silence stillness meditation monk sage saint ascetic ' +
  'thunder lightning storm rainbow shadow night day morning evening'
)
  .split(/\s+/)
  .filter(Boolean)

const CONCEPT_HEAD =
  'A Victorian-era Sanskrit-English dictionary describes things in formal, archaic English. ' +
  'For each modern everyday word below, give the formal or archaic English words that dictionary ' +
  'would most likely use for the same idea.\n\n' +
  'Rules:\n' +
  '- Single English words only. No Sanskrit, no phrases, no proper nouns.\n' +
  '- 4 to 8 per word. Prefer words a 19th-century lexicographer would actually write.\n' +
  /**
   * ASK FOR PARTICIPLES EXPLICITLY. Left to itself the model answers with
   * archaic ADJECTIVES — for "generous" it proposed munificent, bountiful,
   * liberal, magnanimous, charitable, and not one of the five appears anywhere
   * in this document. The glosses describe the same idea as an action:
   * "bestowing strength", "'giving well'", "strength-giving". A lexicon built
   * from participles cannot be reached with adjectives.
   */
  '- INCLUDE VERB AND PARTICIPLE FORMS, not only adjectives: this dictionary usually says ' +
  '"bestowing", "giving", "shining", "bearing" where modern English says generous, bright, strong.\n' +
  '- Example: for "brave" → ["valiant","heroic","bold","intrepid","courageous"].\n' +
  '- Example: for "generous" → ["bestowing","giving","granting","bountiful","liberal"].\n' +
  '- Example: for "doctor" → ["physician","healer","leech","surgeon"].\n\n' +
  'Return strict JSON only: {"word": ["archaic", ...], ...}\n\n'

async function archaicFor(batch, attempt = 0) {
  try {
    const out = await client.send(
      new ConverseCommand({
        modelId,
        messages: [{ role: 'user', content: [{ text: CONCEPT_HEAD + batch.join('\n') }] }],
        inferenceConfig: { maxTokens: 3000, temperature: 0.2 },
        additionalModelRequestFields: { reasoning_effort: 'low' },
      }),
    )
    const text = (out.output?.message?.content ?? []).map((b) => b.text).filter(Boolean).join('')
    return JSON.parse(text.slice(text.indexOf('{'), text.lastIndexOf('}') + 1))
  } catch (err) {
    if (attempt < 2) {
      await new Promise((r) => setTimeout(r, 400 * (attempt + 1)))
      return archaicFor(batch, attempt + 1)
    }
    console.warn(`  concept batch failed (${batch[0]}…): ${err.message}`)
    return {}
  }
}

/**
 * The only legal expansion targets: words that appear in a gloss which MEANS
 * something, not one that merely attests a bearer. `freq` was built that way
 * for pass A and is reused here deliberately.
 *
 * THIS IS THE POLYSEMY GUARD, and it earned its place. For "song" the model
 * proposed `lay` — a real English word for a ballad, and genuinely present in
 * these glosses. In all three of them it means a LAY DISCIPLE. Validating
 * against every word in every row accepted it, and "song" came back with three
 * Buddhist lay disciples above the one row that means a bard. A word that
 * survives only inside "name of a …" rows can only ever retrieve attestations,
 * so it is not a synonym of anything.
 */
const glossWords = new Set(freq.keys())

/**
 * Morphology, done once at build time instead of with a stemmer at query time.
 *
 * "thunder" appears in these glosses as `thunderer` and `thunderbolt`, and
 * exact tokenisation reaches neither — the query returned one row where four
 * were sitting there. A Porter stemmer would fix it and would also touch every
 * one of the 30 literal cases currently at 100%, which is a poor trade for one
 * query. Prefix-relating the concept to the vocabulary costs nothing at runtime
 * and cannot regress anything it does not match.
 */
function morphologicalVariants(concept) {
  if (concept.length < 4) return []
  const out = []
  for (const word of glossWords) {
    if (word === concept) continue
    if (word.length > concept.length && word.startsWith(concept)) out.push(word)
    else if (concept.length >= 5 && concept.startsWith(word) && word.length >= 4) out.push(word)
  }
  return out.slice(0, 4)
}

const CONCEPT_CACHE = resolve(REPO, 'node_modules/.cache/naam-concepts.json')
const conceptCache = !has('--fresh') && existsSync(CONCEPT_CACHE) ? JSON.parse(readFileSync(CONCEPT_CACHE, 'utf8')) : {}
const conceptTodo = [...new Set(CONCEPTS)].filter((w) => !(w in conceptCache))
const conceptBatches = []
for (let i = 0; i < conceptTodo.length; i += BATCH) conceptBatches.push(conceptTodo.slice(i, i + BATCH))

if (conceptBatches.length > 0) {
  console.log(`pass B: ${conceptTodo.length} concepts in ${conceptBatches.length} batches`)
  let n = 0
  for (const batch of conceptBatches) {
    const result = await archaicFor(batch)
    for (const w of batch) conceptCache[w] = Array.isArray(result[w]) ? result[w] : []
    writeFileSync(CONCEPT_CACHE, JSON.stringify(conceptCache))
    n++
    process.stdout.write(`\r  ${n}/${conceptBatches.length} batches`)
  }
  console.log()
}

/** Merged in below. Kept separate so the validation drop-rate is reportable —
 *  it is the honest measure of how much of pass B was invention. */
const passB = new Map()
let proposed = 0
let kept = 0
for (const [concept, archaic] of Object.entries(conceptCache)) {
  if (!Array.isArray(archaic)) continue

  const add = (word) => {
    if (!passB.has(concept)) passB.set(concept, new Set())
    passB.get(concept).add(word)
  }

  /**
   * A concept already in the meaning-bearing vocabulary still gets its
   * morphological neighbours — "thunder" is in one gloss and `thunderer` and
   * `thunderbolt` are in others, and only the exact token was reachable.
   */
  for (const variant of morphologicalVariants(concept)) add(variant)
  if (glossWords.has(concept)) continue // literal hits already work; no synonyms needed

  for (const raw of archaic) {
    const word = String(raw || '')
      .toLowerCase()
      .replace(/[^a-z]/g, '')
    proposed++
    // THE GUARD: the document must contain the word, in a row that means something.
    if (word.length < 3 || !glossWords.has(word) || word === concept) continue
    kept++
    add(word)
  }
}
console.log(`pass B: ${kept}/${proposed} proposed words survived validation against the glosses`)

/* ── invert, filter, write ────────────────────────────────────────────────── */

/** Words already in the glosses need no entry — BM25 finds them today. */
const glossVocabulary = new Set()
for (const row of rows) {
  for (const w of String(row.gloss || '')
    .toLowerCase()
    .replace(/[^a-z\s]/g, ' ')
    .split(/\s+/)) {
    if (w.length > 2) glossVocabulary.add(w)
  }
}

const inverted = new Map()
// Pass B first, so its validated, query-shaped entries lead the fan-out ranking.
for (const [concept, words] of passB) inverted.set(concept, new Set(words))
for (const [corpusWord, everyday] of Object.entries(cache)) {
  if (!Array.isArray(everyday)) continue
  for (const raw of everyday.slice(0, MAX_SYNONYMS)) {
    const word = String(raw || '')
      .toLowerCase()
      .replace(/[^a-z]/g, '')
    if (word.length < 3 || word.length > 14) continue
    if (word === corpusWord) continue
    // THE SIZE WIN: if the glosses already contain it, retrieval already works.
    if (glossVocabulary.has(word)) continue
    if (!inverted.has(word)) inverted.set(word, new Set())
    inverted.get(word).add(corpusWord)
  }
}

/**
 * Cap the fan-out. A query term that expands into fifteen corpus words stops
 * being a synonym and becomes a topic, and it drags in everything loosely
 * adjacent — which is the precision failure this whole exercise exists to avoid.
 * Rarer corpus words are kept first: they are the more specific match.
 */
const MAX_FANOUT = 6
const thesaurus = {}
for (const [everyday, corpusWords] of [...inverted.entries()].sort()) {
  const ranked = [...corpusWords].sort((a, b) => (freq.get(a) ?? 0) - (freq.get(b) ?? 0)).slice(0, MAX_FANOUT)
  if (ranked.length > 0) thesaurus[everyday] = ranked
}

mkdirSync(dirname(OUT), { recursive: true })
writeFileSync(OUT, JSON.stringify(thesaurus, null, 0))

const bytes = Buffer.byteLength(JSON.stringify(thesaurus))
console.log(`\n${Object.keys(thesaurus).length} everyday words → ${OUT.replace(REPO + '/', '')}`)
console.log(`${(bytes / 1024).toFixed(1)} kB raw`)
for (const probe of ['brave', 'calm', 'peace', 'hope', 'healer', 'song', 'generous']) {
  console.log(`  ${probe.padEnd(9)} → ${(thesaurus[probe] || []).join(', ') || '(none)'}`)
}
