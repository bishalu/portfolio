/**
 * Builds the /naam dataset from the committed text dump of
 * Boy_Name_Candidates.pdf.
 *
 * DESIGN.md P9 — proof over claim. Every name on /naam carries a page number,
 * and the page number has to be worth something: it has to point at a line a
 * reader could check. So the parse is not "good enough", it is exact. The
 * document states its own totals on page 1 and again in every section heading
 * (856 B, 3,786 S, 2,073 V, 6,715 rows, 1,708 attested, 590 evocative, and a
 * count on each of the eighteen letter/syllable/tier buckets). This script
 * reproduces all of them and refuses to emit anything if it can't.
 *
 * DESIGN.md §4 — the honesty vocabulary is three words, so provenance can't be
 * a badge. It is carried by the schema instead: src/types/naam.ts splits every
 * row into document-sourced fields and derived ones. This file is where that
 * line is actually drawn, so the rules below say, for every derived field,
 * exactly how far the interpretation goes.
 *
 *   node scripts/naam/build-dataset.mjs           regenerate everything
 *   node scripts/naam/build-dataset.mjs --check   assert only; exit 1 on drift
 *
 * --check is chained into `prebuild`, so it does no I/O beyond reading the
 * source dump and the three artifacts, and prints nothing unless it fails.
 *
 * The one-time PDF → text step is recorded in scripts/naam/extract-text.py.
 * It has already run; its output is committed and hashed here.
 */
import { createHash } from 'node:crypto'
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const REPO = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..')
const SOURCE_TEXT = join(REPO, 'src/data/naam/source-text.txt')
const SOURCE_PDF = join(REPO, 'Boy_Name_Candidates.pdf')
const OUT_CORE = join(REPO, 'public/naam/names-core.json')
const OUT_REST = join(REPO, 'public/naam/names-rest.json')
const OUT_FACTS = join(REPO, 'src/generated/naam-facts.ts')

/**
 * Frozen input hashes. The PDF is untracked (public repo; publishing 172 pages
 * of the family's research is a decision nobody has made), so a clone can only
 * check the text dump — which is the file this script actually reads.
 */
const SHA_TEXT = 'ef684850d536685f9cf341afcc769cc2c0eda186e0888155888b5335e208b701'
const SHA_PDF = '295e3bb094b70e36e52a198edf3e08ea7da44cfb0a37ffaa578be951d96c5497'

const check = process.argv.includes('--check')
const failures = []
const fail = (msg) => failures.push(msg)
const assertEq = (label, got, want) => {
  if (got !== want) fail(`${label}: got ${got}, expected ${want}`)
}

/* ═══════════════════════════════════════════════════════════════════════════
   1. THE PARSE
   ═══════════════════════════════════════════════════════════════════════════ */

/**
 * A name cell. The optional " / X" is the PDF's own B-variant column, present
 * only in the V section. Badge glyphs trail the name, space separated.
 */
const NAME_RE = /^[A-Z][a-zA-Z]*(?: \/ [A-Z][a-zA-Z]*)?(?:(?: [N+!])| f\?)*$/
/** The Src column: any non-empty ordered subset of V, C, T. */
const SRC_RE = /^(?:V|C|T|VC|VT|CT|VCT)$/
/**
 * A name cell whose badges wrapped onto the next line. Three rows in the
 * document do this (Vichakshus, Vishtambha, Vishpardhas, all on pages 145-150)
 * and they are exactly the reason a naive parse lands on 6,713 instead of
 * 6,715: two vanish entirely, and the third is captured under the name "N".
 */
const BADGE_WRAP_RE = /^(?:[N+!]|f\?)(?: (?:[N+!]|f\?))*$/
/**
 * A name cell whose *name* wrapped, i.e. "Vyavachchhhid /" then "Byavachchhhid".
 * Eleven rows, all long V/B pairs. A naive parse still counts these, but records
 * the B-form as the name and loses the V-form, so the citation stops matching
 * the page.
 */
const NAME_WRAP_RE = /^[A-Z][a-zA-Z]* \/$/
/** The repeated table header, and its own wrapped second line. */
const TABLE_HEAD_RE = /^Name\s+\(/
const TABLE_HEAD_TAIL = 'cleaner-or-B variant)'

const HEADING_LETTER_RE = /^([BSV]) names/
const HEADING_SYLLABLES_RE = /^(\d) syllables?\s+\(/
const HEADING_SHORTLIST_RE = /^Shortlist/
const HEADING_MORE_RE = /^More \(/

const isHeading = (s) =>
  HEADING_LETTER_RE.test(s) || HEADING_SYLLABLES_RE.test(s) || HEADING_SHORTLIST_RE.test(s) || HEADING_MORE_RE.test(s)

/**
 * Strips the page-header triplet (marker + running header + "p. N") but keeps
 * the page number on every line, because the page number is the citation.
 */
function readLines(raw) {
  const all = raw.split('\n')
  const out = []
  let page = 0
  for (let i = 0; i < all.length; i++) {
    const m = /^<<<PAGE (\d+)>>>$/.exec(all[i].trim())
    if (m) {
      page = Number(m[1])
      i += 2 // running header + "p. N"
      continue
    }
    out.push({ text: all[i].trim(), page })
  }
  return out
}

/** Where does a record start at index i? Returns the joined name cell, or null. */
function nameCellAt(lines, i) {
  const at = (k) => (lines[i + k] ? lines[i + k].text : '')
  let name = null
  let consumed = 0
  if (NAME_WRAP_RE.test(at(0)) && NAME_RE.test(`${at(0)} ${at(1)}`)) {
    name = `${at(0)} ${at(1)}`
    consumed = 1
  } else if (NAME_RE.test(at(0))) {
    name = at(0)
  }
  if (name === null) return null
  if (BADGE_WRAP_RE.test(at(consumed + 1)) && SRC_RE.test(at(consumed + 2))) {
    name = `${name} ${at(consumed + 1)}`
    consumed += 1
  }
  if (!SRC_RE.test(at(consumed + 1))) return null
  return { name, sources: at(consumed + 1), consumed: consumed + 1 }
}

function parse(raw) {
  const lines = readLines(raw)
  const records = []
  let letter = null
  let syllables = null
  let tier = null

  for (let i = 0; i < lines.length; i++) {
    const { text, page } = lines[i]
    if (!text) continue

    let m
    if ((m = HEADING_LETTER_RE.exec(text))) {
      letter = m[1] // the V heading reads "V names (shown with B-variant)"
      continue
    }
    if ((m = HEADING_SYLLABLES_RE.exec(text))) {
      syllables = Number(m[1])
      continue
    }
    if (HEADING_SHORTLIST_RE.test(text)) {
      tier = 'shortlist'
      continue
    }
    if (HEADING_MORE_RE.test(text)) {
      tier = 'more'
      continue
    }
    if (TABLE_HEAD_RE.test(text) || text === TABLE_HEAD_TAIL) continue

    const cell = nameCellAt(lines, i)
    if (!cell) continue
    i += cell.consumed

    // Meaning runs until the next record, the next heading, or a repeat of the
    // table header at the top of a page.
    const meaning = []
    for (let j = i + 1; j < lines.length; j++) {
      const s = lines[j].text
      if (!s) continue
      if (TABLE_HEAD_RE.test(s) || s === TABLE_HEAD_TAIL) break
      if (isHeading(s)) break
      if (nameCellAt(lines, j)) break
      meaning.push(s)
      i = j
    }

    records.push({
      nameCell: cell.name,
      sourceCell: cell.sources,
      letter,
      syllables,
      tier,
      page,
      sourceGloss: meaning.join(' '),
    })
  }
  return records
}

/* ═══════════════════════════════════════════════════════════════════════════
   2. DERIVED — the gloss tidy
   ═══════════════════════════════════════════════════════════════════════════ */

/**
 * Hand-checked Devanagari, overriding the transliterator.
 *
 * WHY THIS EXISTS: the document is romanized without diacritics, so vowel
 * length is simply absent from the input. `Bhupati` could be bhupati or
 * bhūpati and nothing in the string says which — but the gloss does
 * ("'earth-protector', a king"), because bhū is earth. The transliterator
 * cannot read glosses, so it renders every vowel short and is wrong on roughly
 * a quarter of the compounds.
 *
 * WHY ONLY THESE: these are the ids in NAAM_FEATURED — the names rendered
 * largest, first, and to the most people. The remaining ~6,700 stay 'derived'
 * and the page says so in its footer. Correcting the visible set is honest;
 * claiming 6,715 hand-checked transliterations would not be.
 *
 * Each entry is read off the row's own gloss. Entries that merely CONFIRM the
 * transliterator are kept deliberately, so the map is a full record of what was
 * looked at rather than only of what was wrong.
 */
const CHECKED_DEVA = {
  // Corrected — the transliterator dropped a long vowel or a retroflex.
  bhagin: { deva: 'भागिन्' }, //  bhāgin, 'having a share' — not bhagin
  bhumitra: { deva: 'भूमित्र' }, //  bhū-mitra, 'earth-friend'
  bhupati: { deva: 'भूपति' }, //  bhū-pati, 'lord of the earth'
  bhutesha: { deva: 'भूतेश' }, //  bhūta-īśa, 'lord of beings'
  svaraj: { deva: 'स्वराज्' }, //  svarāj
  shachisha: { deva: 'शचीश' }, //  śacī-īśa, 'lord of Śacī'
  shubhaksha: { deva: 'शुभाक्ष' }, //  śubha-akṣa, 'auspicious-eyed'
  vishala: { deva: 'विशाल', devaB: 'बिशाल' }, //  viśāla. The father's own name.
  vrishan: { deva: 'वृषन्', devaB: 'बृषन्' }, //  vṛṣan — vocalic ṛ, retroflex ṣ

  // Confirmed correct as derived.
  bhoja: { deva: 'भोज' },
  bhogin: { deva: 'भोगिन्' },
  bhaga: { deva: 'भग' },
  balada: { deva: 'बलद' },
  shuna: { deva: 'शुन' },
  sidhya: { deva: 'सिध्य' },
  saura: { deva: 'सौर' },
  subala: { deva: 'सुबल' },
  snehaja: { deva: 'स्नेहज' }, //  'born from affection' — the mother's name
  vastu: { deva: 'वस्तु', devaB: 'बस्तु' },
  vaktra: { deva: 'वक्त्र', devaB: 'बक्त्र' },
  vida: { deva: 'विद', devaB: 'बिद' },
  vidyesha: { deva: 'विद्येश', devaB: 'बिद्येश' },
  vivarta: { deva: 'विवर्त', devaB: 'बिवर्त' },
  vratesha: { deva: 'व्रतेश', devaB: 'ब्रतेश' },
}

/**
 * Monier-Williams shorthand that is unambiguous in this corpus. Every one of
 * these was checked against all of its occurrences before being enabled.
 */
const SAFE_ABBREVIATIONS = [
  [/\bwh\b/g, 'white'],
  [/\bund\b/g, 'understanding'],
  [/\bkn\b/g, 'knowledge'],
  [/\bsp\b/g, 'speech'],
  [/\bBr\b/g, 'Brahman'],
  [/\bpartic\./g, 'particular'],
  [/\besp\./g, 'especially'],
]

/**
 * Single-letter shorthand is NOT unambiguous here and a blanket expansion would
 * invent meaning, which the brief forbids. In this corpus:
 *   h  is "heaven" in "lord/physician/river of h" and "hundred" in "a h feet"
 *   w  is "water" only where the row is already about rain/cloud/sea; elsewhere
 *      it is "wife", "witness" or "wealth"
 *   m  is "mountains" only in "king/lord of m"; elsewhere it is the masculine
 *      gender marker, or "months", or the letter m
 *   g  is "gods" only in "the g"; "upper g" is a garment
 *   e  is "earth" next to "the" or inside a hyphenated compound
 * So each expansion is gated on the context that disambiguates it, and every
 * other occurrence is left exactly as printed.
 */
const GATED_ABBREVIATIONS = [
  [/\b(lord|physician|river) of h\b/g, '$1 of heaven'],
  [/\ba h\b/g, 'a hundred'],
  [/\b(king|lord) of m\b/g, '$1 of the mountains'],
  [/\bthe g\b/g, 'the gods'],
  [/\bthe e\b/g, 'the earth'],
  [/' e -/g, "' earth-"],
]
/** `w` → water only when the surrounding gloss is already about water. */
const WATER_CONTEXT = /\b(rain|cloud|sea|ocean|river)\b/

/** Deities the brief names. Deliberately short — "soma" is also a plant. */
const DEITIES = ['vishnu', 'krishna', 'shiva', 'indra', 'agni', 'brahma', 'buddha']

/**
 * Citation debris. None of these carry sense; all of them are Monier-Williams
 * apparatus that survived the PDF extraction.
 */
const DEBRIS = [
  /,?\s*,\s*Sch\b\.?/g, //           ", Sch"  — the scholiast
  /\s*\bg\.\s+[A-Za-z/]+\s+di\b/g, // "g. aSvA di"  — gana reference
  /\(\s*accord\.[^)]*\)/gi, //       "( accord. to ...)"
  /\[[^\]]*\]/g, //                  "[ cf. ...]"
  /\bGr\.\s*/g, //                   the grammarians
]

/**
 * Monier-Williams cross-references. These are *inside* the meaning cell but are
 * not meaning: "(for See p. 752, col. 3 ) shining, radiant". Cutting the whole
 * parenthetical would sometimes take a real clause with it — "(called Bihar or
 * Behar from the number of Buddhist monasteries See )" — so the reference is
 * cut out of the parenthetical and whatever was said before it is kept.
 */
function dropSeeRefs(s) {
  let out = s.replace(/\(([^)]*)\)/g, (whole, inner) => {
    if (!/\bsee\b/i.test(inner)) return whole
    const kept = inner
      .replace(/(?:\bfor\s+)?\bsee\b[\s\S]*$/i, '')
      .replace(/[\s,;:.]+$/, '')
      .trim()
    // "(for saksha/na)" is the reference's other half, not a clause.
    if (/^for\s+\S+$/i.test(kept)) return ' '
    return kept.length >= 3 ? `(${kept})` : ' '
  })
  // A trailing bare reference: "name of a man See sattvaki". Only cut it if a
  // real clause survives — rows whose entire meaning cell is a cross-reference
  // ("See p. 758, col. 1") have no gloss to give, and are left verbatim.
  if (/\bSee\b/.test(out)) {
    const cut = out
      .replace(/\s*\bSee\b[\s\S]*$/, '')
      .replace(/[\s,;:.]+$/, '')
      .trim()
    if (cut.length >= 8) out = cut
  }
  return out
}

/**
 * A leading conjugation dump: "cl. 1. A. basate (…; aor. abhasishta; …), to
 * shine, be bright". Everything before the first ", to …" is paradigm, not
 * meaning. If there is no ", to …" the row is paradigm all the way down, and we
 * keep it verbatim rather than guess.
 */
const CONJUGATION_RE = /^cl\.\s*\d.*?[,)]\s*(to\s.+)$/s

const GLOSS_MAX = 120

function tidyGloss(sourceGloss) {
  let s = sourceGloss

  const conj = CONJUGATION_RE.exec(s)
  if (conj) s = conj[1]

  s = dropSeeRefs(s)
  for (const re of DEBRIS) s = s.replace(re, ' ')
  for (const [re, to] of SAFE_ABBREVIATIONS) s = s.replace(re, to)
  for (const [re, to] of GATED_ABBREVIATIONS) s = s.replace(re, to)
  if (WATER_CONTEXT.test(s)) s = s.replace(/\bw\b/g, 'water')

  // Punctuation left over from a two-column PDF: stacked separators, spaces
  // before punctuation, spaces inside quotes and hyphenated compounds.
  s = s
    .replace(/\s+/g, ' ')
    .replace(/([,;:])(\s*[,;:])+/g, '$1')
    .replace(/\s+([,;:.)])/g, '$1')
    .replace(/([("'])\s+/g, '$1')
    .replace(/\s+(['"])/g, '$1')
    .replace(/([a-z])\s+-\s*(?=[a-z])/gi, '$1-')
    .replace(/\(\s*=?\s*\)/g, '')
    .replace(/\s+/g, ' ')
    .replace(/^[\s,;:.)\]]+/, '')
    .replace(/[\s,;:]+$/, '')
    .trim()

  for (const d of DEITIES) {
    s = s.replace(new RegExp(`\\b${d}\\b`, 'g'), d[0].toUpperCase() + d.slice(1))
  }

  // A one-line read. Long comma-lists get cut at a clause boundary; the
  // verbatim sourceGloss is one disclosure away, so nothing is lost.
  if (s.length > GLOSS_MAX) {
    const window = s.slice(0, GLOSS_MAX)
    const cut = Math.max(window.lastIndexOf(';'), window.lastIndexOf(','))
    if (cut >= 40) s = `${window.slice(0, cut).trim()}…`
  }

  // If tidying emptied the cell or ate the sense, keep what the document said.
  if (s.length < 2) return sourceGloss
  return s
}

/* ═══════════════════════════════════════════════════════════════════════════
   3. DERIVED — syllables and Devanagari
   ═══════════════════════════════════════════════════════════════════════════ */

const VOWELS = new Set(['a', 'e', 'i', 'o', 'u'])
/**
 * The only vowel digraphs the corpus actually contains: ai (243), au (151),
 * aa (2), oa (1). Each is one nucleus — dropping any of the four costs
 * accuracy against the document's own syllable counts, and there are no others
 * to add.
 */
const VOWEL_DIGRAPHS = ['ai', 'au', 'aa', 'oa']
/**
 * Consonant units. Longest-match-first; the order matters. `chchhh` is a PDF
 * extraction artifact for -cch- (Vichchhheda = viccheda). `ng` and `ny` are
 * deliberately absent: in this romanization "linga" is l-i-ṅ-g-a, so treating
 * `ng` as one unit would swallow the g.
 */
const CONSONANT_UNITS = ['chchhh', 'chchh', 'chhh', 'ksh', 'chh', 'bh', 'ch', 'dh', 'gh', 'jh', 'kh', 'ph', 'th', 'sh']

/** Digraph-aware tokenizer, shared by the splitter and the transliterator. */
function tokenize(word) {
  const w = word.toLowerCase()
  const units = []
  let i = 0
  while (i < w.length) {
    let hit = null
    for (const d of VOWEL_DIGRAPHS) {
      if (w.startsWith(d, i)) {
        hit = { s: d, vowel: true }
        break
      }
    }
    if (!hit) {
      for (const c of CONSONANT_UNITS) {
        if (w.startsWith(c, i)) {
          hit = { s: c, vowel: false }
          break
        }
      }
    }
    if (!hit) hit = { s: w[i], vowel: VOWELS.has(w[i]) }
    units.push(hit)
    i += hit.s.length
  }
  return units
}

/**
 * Close a syllable after each vowel nucleus; a consonant opens the next
 * syllable if any vowel still follows, and is a coda otherwise.
 * Bhaskara → bha·ska·ra. Vaishtambha → vai·shta·mbha. Bhas → bhas.
 */
function splitSyllables(word) {
  const units = tokenize(word)
  const out = []
  let cur = ''
  let hasNucleus = false
  for (let i = 0; i < units.length; i++) {
    const u = units[i]
    if (u.vowel) {
      if (hasNucleus) {
        out.push(cur)
        cur = ''
      }
      cur += u.s
      hasNucleus = true
      continue
    }
    if (!hasNucleus) {
      cur += u.s
      continue
    }
    let vowelFollows = false
    for (let j = i; j < units.length; j++) {
      if (units[j].vowel) {
        vowelFollows = true
        break
      }
    }
    if (vowelFollows) {
      out.push(cur)
      cur = u.s
      hasNucleus = false
    } else {
      cur += u.s
    }
  }
  if (cur) out.push(cur)
  return out
}

const DEVA_CONSONANT = {
  k: 'क',
  kh: 'ख',
  g: 'ग',
  gh: 'घ',
  ch: 'च',
  chh: 'छ',
  chhh: 'छ',
  chchh: 'च्छ',
  chchhh: 'च्छ',
  j: 'ज',
  jh: 'झ',
  t: 'त',
  th: 'थ',
  d: 'द',
  dh: 'ध',
  n: 'न',
  p: 'प',
  ph: 'फ',
  b: 'ब',
  bh: 'भ',
  m: 'म',
  y: 'य',
  r: 'र',
  l: 'ल',
  v: 'व',
  w: 'व',
  s: 'स',
  sh: 'श',
  h: 'ह',
  ksh: 'क्ष',
}
const DEVA_INDEPENDENT = { a: 'अ', aa: 'आ', i: 'इ', u: 'उ', e: 'ए', ai: 'ऐ', o: 'ओ', au: 'औ', oa: 'ओ' }
const DEVA_MATRA = { a: '', aa: 'ा', i: 'ि', u: 'ु', e: 'े', ai: 'ै', o: 'ो', au: 'ौ', oa: 'ो' }
const VIRAMA = '्'

/**
 * Rule-based transliteration. APPROXIMATE and labelled as such everywhere it is
 * shown: the romanization has already lost vowel length, retroflexion and
 * vocalic ṛ, and none of that is recoverable. Digraphs first, then vowels
 * (independent at a syllable onset, matra after a consonant), then virama for
 * every cluster and every final consonant.
 */
function toDevanagari(word) {
  const units = tokenize(word)
  let out = ''
  let afterConsonant = false
  for (const u of units) {
    if (u.vowel) {
      out += afterConsonant ? (DEVA_MATRA[u.s] ?? '') : (DEVA_INDEPENDENT[u.s] ?? '')
      afterConsonant = false
      continue
    }
    const glyph = DEVA_CONSONANT[u.s]
    if (glyph === undefined) return { deva: null, unknown: u.s }
    if (afterConsonant) out += VIRAMA
    out += glyph
    afterConsonant = true
  }
  if (afterConsonant) out += VIRAMA
  return { deva: out, unknown: null }
}

/* ═══════════════════════════════════════════════════════════════════════════
   4. DERIVED — themes
   ═══════════════════════════════════════════════════════════════════════════ */

/**
 * The lexicon is closed at nineteen and mirrored in src/types/naam.ts, where
 * NAAM_THEMES fixes the bit order for themeMask. The patterns run over the
 * verbatim sourceGloss, never over the tidied one, so a tidy-rule change can
 * never silently re-tag the corpus.
 *
 * The first fifteen are the plan's list. They left 91 of the document's gold
 * rows untagged, almost all of them kings, rivers, "pure", "swift" and "true" —
 * so four themes were added rather than stretching the existing fifteen past
 * what they mean.
 */
const THEME_PATTERNS = {
  light:
    /\b(light|shining|shine|radiant|radiance|splendour|splendor|splendid|brilliant|bright|brightness|lustre|luster|lustrous|glow|glowing|blazing|flame|fire|sun|solar|sunlight|moon|lunar|star|ray|rays|dawn|daybreak|gold|golden|glitter|luminous|illumin|resplendent)\b/i,
  strength:
    /\b(strong|strength|mighty|might|power|powerful|force|forceful|vigour|vigor|vigorous|robust|firm|steadfast|hero|heroic|valour|valor|valiant|brave|bravery|bold|energy|energetic|stout|sturdy|conquer|victorious|victory|triumph|invincible|unconquer)\b/i,
  wisdom:
    /\b(wise|wisdom|wisely|knowledge|knowing|learned|learning|intelligen|intellect|understanding|insight|discern|sage|scholar|teacher|thought|thoughtful|mind|mindful|clever|skil|discrimination|perceiv|perception|prudent|erudite|philosoph|science|lore|instruct)\b/i,
  compassion:
    /\b(compassion|kind|kindness|kindly|merci|mercy|pity|tender|gentle|gentleness|benevolen|generous|generosity|charit|friendly|friendship|affection|love|loving|beloved|dear|grace|gracious|giving|gift|bounti|liberal|sympath)\b/i,
  // Deliberately excludes beautiful / handsome / charming: only 63 rows carry
  // them, no gold row depends on them, and "a joyful name" returning "having
  // beautiful hips" is the kind of slop that makes a filter untrustworthy.
  joy: /\b(joy|joyous|joyful|delight|delightful|happy|happiness|glad|gladness|pleasure|pleasing|pleased|cheer|cheerful|bliss|blissful|rejoic|merry|mirth|festive|festival|exult|enjoy)\b/i,
  auspicious:
    /\b(auspicious|fortunate|fortune|lucky|luck|blessed|blessing|prosper|prosperous|prosperity|welfare|well-being|wealth|wealthy|abundan|plenty|thriv|success|boon|favourable|favorable|propitious|good omen|happy omen)\b/i,
  sky: /\b(sky|heaven|heavenly|celestial|air|aerial|atmosphere|cloud|wind|breeze|space|ether|firmament|horizon|soaring|flying|bird|eagle|swan|hawk|falcon|cosmos|cosmic|infinite|vault)\b/i,
  water:
    /\b(water|watery|river|stream|ocean|sea|lake|pond|rain|raining|rainy|wave|waves|flood|flowing|flows|spring|well|dew|moist|drop|drink|nectar|ambrosia|current|torrent|shore|aquatic)\b/i,
  earth:
    /\b(earth|earthly|ground|soil|land|field|mountain|hill|rock|stone|tree|trees|forest|wood|grove|plant|flower|blossom|fruit|seed|grain|root|leaf|leaves|garden|meadow|terrestrial|region|country|world)\b/i,
  sound:
    /\b(sound|sounding|voice|speech|speak|speaking|song|sing|singing|singer|hymn|chant|chanting|music|musical|melody|tune|note|resound|resonan|echo|roar|thunder|word|words|utter|utterance|recit|praise|eloquen|verse|metre|meter|drum|flute|lute|shout|call)\b/i,
  protection:
    /\b(protect|protection|protector|guard|guardian|shelter|refuge|defend|defence|defense|preserv|saviour|savior|save|rescue|shield|armour|armor|watchman|keeper|ward|sustain|support|upholder|bearer|carrying|nourish)\b/i,
  lotus: /\b(lotus|lotuses|nelumbium|nymphaea|water-lily|waterlily|padma|kamala)\b/i,
  peace:
    /\b(peace|peaceful|calm|calmness|tranquil|tranquillity|tranquility|quiet|still|stillness|serene|repose|rest|restful|composed|composure|patience|patient|content|contentment|appease|pacify|soothe|ease|equanimity|undisturbed|unruffled)\b/i,
  deity:
    /\b(god|gods|goddess|divine|divinity|deity|deities|Vishnu|vishnu|Shiva|shiva|Krishna|krishna|Indra|indra|Agni|agni|Brahma|brahma|Rudra|rudra|Varuna|varuna|Vayu|vayu|Surya|surya|Soma|soma|Skanda|skanda|Ganesha|maruts|adityas|ashvins|vasus|celestial being|epithet of)\b/i,
  monk: /\b(monk|monks|ascetic|ascetics|bhikkhu|bhikshu|sage|rishi|rishis|hermit|recluse|renunc|meditat|dhyana|jhana|nirvana|nibbana|enlighten|awaken|Buddha|buddha|buddhist|sangha|arahant|arhat|dharma|dhamma|sutta|discipline|vow|devotee|worshipper|penance|austerit)\b/i,
  royal:
    /\b(king|kings|kingly|prince|princely|royal|monarch|sovereign|emperor|ruler|rule|reign|lord|chief|chieftain|leader|noble|nobility|throne|crown|imperial|majesty|master|commander|governor)\b/i,
  purity:
    /\b(pure|purity|purifying|purified|clean|cleansing|clear|spotless|stainless|immaculate|untainted|holy|sacred|sanctif|blameless|faultless|innocen|white|bright and clear|crystal|transparent)\b/i,
  swift:
    /\b(swift|swiftly|quick|quickly|fast|rapid|speed|speedy|fleet|nimble|agile|hasten|hasty|running|runner|racing|racer|impetuous|darting|rushing|arrow|steed|horse)\b/i,
  truth:
    /\b(truth|true|truthful|real|reality|genuine|honest|honesty|faith|faithful|sincere|sincerity|right|righteous|just|justice|law|duty|virtue|virtuous|good conduct|integrity|trust|reliable|vow-keeping)\b/i,
}
const THEME_ORDER = Object.keys(THEME_PATTERNS)

/**
 * themeMask is a bitmask over NAAM_THEMES' indices, and NAAM_THEMES lives in
 * src/types/naam.ts because that is where every consumer reads it from. Two
 * lists, one bit order: if they drift, every mask in both JSON files becomes
 * quietly wrong and nothing else would notice. So the build reads the type file
 * and refuses to run on a mismatch.
 */
function assertThemeOrderMatchesTypes() {
  const src = readFileSync(join(REPO, 'src/types/naam.ts'), 'utf8')
  const block = /export const NAAM_THEMES = \[([\s\S]*?)\] as const/.exec(src)
  if (!block) {
    fail('src/types/naam.ts: could not find the NAAM_THEMES array')
    return
  }
  const declared = [...block[1].matchAll(/'([a-z]+)'/g)].map((m) => m[1])
  if (declared.join(',') !== THEME_ORDER.join(',')) {
    fail(
      `theme bit order drift.\n      src/types/naam.ts: ${declared.join(' ')}\n      this file:         ${THEME_ORDER.join(' ')}`,
    )
  }
}

function themesFor(sourceGloss) {
  const out = []
  for (const name of THEME_ORDER) if (THEME_PATTERNS[name].test(sourceGloss)) out.push(name)
  return out
}

/* ═══════════════════════════════════════════════════════════════════════════
   5. DERIVED — B-form quality
   ═══════════════════════════════════════════════════════════════════════════ */

/**
 * How well does the PDF's own B-form read out loud? "Bishala" is fine.
 * "Brish" and "Bli" are not, and a model told to "mention the B-form when it
 * reads well" needs that as a field rather than a guess.
 */
function bFormQuality(bVariant, hardCluster) {
  if (!bVariant) return null
  if (hardCluster) return 'awkward'
  if (/^B[lrvy]/i.test(bVariant)) return 'awkward'
  if (/^B[^aeiouAEIOU]{2}/.test(bVariant)) return 'awkward'
  return 'clean'
}

/* ═══════════════════════════════════════════════════════════════════════════
   6. THE FROZEN EXPECTATIONS
   ═══════════════════════════════════════════════════════════════════════════ */

/** Printed on page 1 of the document. */
const STATED_TOTALS = { total: 6715, attested: 1708, evocative: 590 }
/** Printed in the letter headings. */
const STATED_PER_LETTER = { B: 856, S: 3786, V: 2073 }
/** Printed in the syllable and tier headings, all eighteen of them. */
const STATED_BUCKETS = {
  'B-1-shortlist': 2,
  'B-1-more': 11,
  'B-2-shortlist': 114,
  'B-2-more': 164,
  'B-3-shortlist': 225,
  'B-3-more': 340,
  'S-1-shortlist': 7,
  'S-1-more': 43,
  'S-2-shortlist': 329,
  'S-2-more': 645,
  'S-3-shortlist': 836,
  'S-3-more': 1926,
  'V-1-shortlist': 3,
  'V-1-more': 19,
  'V-2-shortlist': 148,
  'V-2-more': 275,
  'V-3-shortlist': 433,
  'V-3-more': 1195,
}

/**
 * The twenty-four rows where our syllable split disagrees with the document's
 * own count — frozen, so the set can neither grow nor shrink unnoticed.
 *
 * These are the document's error, and the pattern is exact: its counts were
 * computed on a diacritic-stripped ASCII form, so a Pali retroflex ḷ or a
 * Sanskrit vocalic ṛ took its vowel with it. veḷu → "veu" → one nucleus, so the
 * document files a two-syllable name under "1 syllable". Twenty-one are
 * Theravada rows with ḷ; three are Vedic rows with ṛṣi / ṛ.
 *
 * We keep the document's number in `syllables` (it is the document's, and it is
 * what the letter/syllable buckets are built from) and our split in
 * `syllableSplit` (it is what the sound rail draws, and it is what the name
 * actually sounds like). The disagreement is data, not a bug to paper over.
 */
const SYLLABLE_SPLIT_EXCEPTIONS = [
  'balisa',
  'balisika',
  'bharijika',
  'bhattachola',
  'bilara',
  'bubbula',
  'salala',
  'salavagga',
  'sassanala',
  'sauppila',
  'shrautarishi',
  'shrutarishi',
  'sihala',
  'suttagula',
  'valapasa',
  'valava',
  'vegalinga',
  'vekalinga',
  'velagga',
  'veludana',
  'velugumba',
  'veluriya',
  'veluvana',
  'velu',
].sort()

/** Forced into FEATURED. Both parents are already in the document. */
const FEATURED_FORCED = ['snehaja', 'vishala']

/**
 * Rows the prerendered page names directly, which must therefore ship in
 * names-core.json whatever tier the document filed them under.
 *
 * `vyan` is the wall's one documented seed — `Vyan / Byan`, Classical, p.120 —
 * and it is the only seed that is itself a V→B name, which is the entire point
 * of putting it there (src/lib/naam/seeds.ts). The document files it under
 * "More", so a plain tier split left it in names-rest.json: src/pages/naam.astro
 * read only core, `byId.get('vyan')` returned undefined, and the card vanished
 * with no error and no gap in the layout. It also has to be in core for the
 * browser, because the page script resolves the swap control against the core
 * rows it has in memory.
 *
 * A row listed here is emitted in core and withheld from rest, so the two files
 * stay disjoint and `[...core, ...rest]` never duplicates an id.
 */
const CORE_EXTRA_IDS = ['vyan']

const PROVENANCE =
  'Boy Name Candidates: 172 pages, 6,715 names lemma-extracted from the Digital Corpus of Sanskrit ' +
  '(Vedic and Classical) and the Pali suttas via the Digital Pali Dictionary, filtered to B, S and V ' +
  'at 1-3 syllables, each V name carrying the document’s own B-variant. 1,708 are attested as a ' +
  'real name; 590 carry an evocative meaning; 201 are both. Page numbers cite that document.'

/* ═══════════════════════════════════════════════════════════════════════════
   7. BUILD
   ═══════════════════════════════════════════════════════════════════════════ */

const sha256 = (buf) => createHash('sha256').update(buf).digest('hex')

const rawText = readFileSync(SOURCE_TEXT)
const shaText = sha256(rawText)
if (shaText !== SHA_TEXT) {
  fail(
    `source-text.txt sha256 drift: got ${shaText}, expected ${SHA_TEXT}. Re-read extract-text.py before touching anything.`,
  )
}

let shaPdf = SHA_PDF
try {
  shaPdf = sha256(readFileSync(SOURCE_PDF))
  if (shaPdf !== SHA_PDF) fail(`Boy_Name_Candidates.pdf sha256 drift: got ${shaPdf}, expected ${SHA_PDF}`)
} catch {
  // The PDF is untracked. A clone will not have it; that is expected, and the
  // text dump is the file of record anyway.
  shaPdf = SHA_PDF
}

assertThemeOrderMatchesTypes()

const records = parse(rawText.toString('utf8'))

const seenIds = new Map()
const rows = []
const unknownDevanagari = new Map()

for (const rec of records) {
  const badgeTokens = rec.nameCell.match(/(?: [N+!]| f\?)+$/)
  const badgeText = badgeTokens ? badgeTokens[0] : ''
  const base = rec.nameCell.slice(0, rec.nameCell.length - badgeText.length).trim()
  const [latin, bVariant = null] = base.split(' / ')

  const badges = {
    attested: /\bN\b/.test(badgeText),
    evocative: badgeText.includes('+'),
    feminineEnding: badgeText.includes('f?'),
    hardCluster: badgeText.includes('!'),
  }

  const slug = latin.toLowerCase().replace(/[^a-z]/g, '')
  const n = (seenIds.get(slug) ?? 0) + 1
  seenIds.set(slug, n)
  const id = n === 1 ? slug : `${slug}-${n}`

  const gloss = tidyGloss(rec.sourceGloss)
  const themes = themesFor(rec.sourceGloss)
  const { deva, unknown } = toDevanagari(latin)
  if (unknown) unknownDevanagari.set(unknown, (unknownDevanagari.get(unknown) ?? 0) + 1)

  /**
   * The B-form's own Devanagari, from the same transliterator over the
   * document's own bVariant — not string surgery on `deva`.
   *
   * Without it the card contradicts itself for the one reader who can check:
   * every row carries a single Devanagari form transliterated from `latin`,
   * which is the V-form, while `preferB` defaults to true. So a V card read
   * "वस्तु / Bastu" — वस्तु is Vastu, and बस्तु appeared nowhere. The swap is
   * the page's running joke and it was not happening in the one script the
   * joke is about.
   */
  const bDeva = bVariant ? toDevanagari(bVariant) : null
  if (bDeva?.unknown) unknownDevanagari.set(bDeva.unknown, (unknownDevanagari.get(bDeva.unknown) ?? 0) + 1)

  let mask = 0
  for (const t of themes) mask |= 1 << THEME_ORDER.indexOf(t)

  rows.push({
    id,
    latin,
    bVariant,
    letter: rec.letter,
    syllables: rec.syllables,
    tier: rec.tier,
    sources: rec.sourceCell.split(''),
    badges,
    sourceGloss: rec.sourceGloss,
    page: rec.page,
    gloss,
    glossIsVerbatim: gloss === rec.sourceGloss,
    devanagari: CHECKED_DEVA[id]?.deva ?? deva ?? '',
    devanagariB: CHECKED_DEVA[id]?.devaB ?? bDeva?.deva ?? null,
    devanagariConfidence: CHECKED_DEVA[id] ? 'checked' : 'derived',
    syllableSplit: splitSyllables(latin),
    themes,
    themeMask: mask,
    searchKey: [latin, bVariant ?? '', gloss].join(' ').toLowerCase().replace(/\s+/g, ' ').trim(),
    bFormQuality: bFormQuality(bVariant, badges.hardCluster),
  })
}

/* ── assertions ──────────────────────────────────────────────────────────── */

assertEq('total rows', rows.length, STATED_TOTALS.total)
assertEq('attested', rows.filter((r) => r.badges.attested).length, STATED_TOTALS.attested)
assertEq('evocative', rows.filter((r) => r.badges.evocative).length, STATED_TOTALS.evocative)
for (const [letter, want] of Object.entries(STATED_PER_LETTER)) {
  assertEq(`letter ${letter}`, rows.filter((r) => r.letter === letter).length, want)
}
const bucketCounts = {}
for (const r of rows) {
  const k = `${r.letter}-${r.syllables}-${r.tier}`
  bucketCounts[k] = (bucketCounts[k] ?? 0) + 1
}
for (const [k, want] of Object.entries(STATED_BUCKETS)) assertEq(`bucket ${k}`, bucketCounts[k] ?? 0, want)
if (Object.keys(bucketCounts).length !== Object.keys(STATED_BUCKETS).length) fail('unexpected bucket appeared')

const ids = new Set()
for (const r of rows) {
  if (ids.has(r.id)) fail(`duplicate id ${r.id}`)
  ids.add(r.id)
  if (!r.sourceGloss) fail(`empty sourceGloss for ${r.id} (p.${r.page})`)
  if (!(r.page >= 1 && r.page <= 172)) fail(`page out of range for ${r.id}: ${r.page}`)
  if (!r.devanagari) fail(`no devanagari for ${r.id} (${r.latin})`)
  if (r.bVariant && !r.devanagariB) fail(`no devanagariB for ${r.id} (${r.bVariant})`)
  if (!r.bVariant && r.devanagariB !== null) fail(`devanagariB on a row with no B-form: ${r.id}`)
  if (!r.letter || !r.syllables || !r.tier) fail(`unbucketed row ${r.id}`)
  if (r.letter === 'V' && r.bVariant && r.bFormQuality === null) fail(`no bFormQuality for ${r.id}`)
}
if (unknownDevanagari.size) fail(`untransliterable units: ${[...unknownDevanagari.keys()].join(', ')}`)

/* A checked form that names an id which does not exist is a silent no-op: the
   row keeps its derived (wrong) glyphs and nothing says so. Fail instead. And
   a devaB on a row with no B-form means the entry is for the wrong name. */
for (const [id, entry] of Object.entries(CHECKED_DEVA)) {
  const row = rows.find((r) => r.id === id)
  if (!row) fail(`CHECKED_DEVA names an id that is not in the dataset: ${id}`)
  if (entry.devaB && !row.bVariant) fail(`CHECKED_DEVA has a B-form for a non-V row: ${id}`)
  if (!entry.devaB && row.bVariant) fail(`CHECKED_DEVA is missing the B-form for ${id} (${row.bVariant})`)
}

const splitMismatches = rows.filter((r) => r.syllableSplit.length !== r.syllables).map((r) => r.id)
const frozen = SYLLABLE_SPLIT_EXCEPTIONS.join('|')
if (splitMismatches.slice().sort().join('|') !== frozen) {
  const got = new Set(splitMismatches)
  const want = new Set(SYLLABLE_SPLIT_EXCEPTIONS)
  fail(
    'syllableSplit.length !== syllables outside the frozen exception set. ' +
      `new: [${splitMismatches.filter((i) => !want.has(i)).join(', ')}] ` +
      `resolved: [${SYLLABLE_SPLIT_EXCEPTIONS.filter((i) => !got.has(i)).join(', ')}]`,
  )
}

/* ── facts ───────────────────────────────────────────────────────────────── */

const coreExtra = new Set(CORE_EXTRA_IDS)
for (const id of CORE_EXTRA_IDS) {
  const row = rows.find((r) => r.id === id)
  if (!row) fail(`CORE_EXTRA_IDS names ${id}, which is not in the dataset`)
  else if (row.tier === 'shortlist') fail(`CORE_EXTRA_IDS names ${id}, which is already a shortlist row`)
}
const core = rows.filter((r) => r.tier === 'shortlist' || coreExtra.has(r.id))
const rest = rows.filter((r) => r.tier === 'more' && !coreExtra.has(r.id))
assertEq('core + rest', core.length + rest.length, rows.length)
const gold = rows.filter((r) => r.badges.attested && r.badges.evocative)

const perLetter = { B: 0, S: 0, V: 0 }
for (const r of rows) perLetter[r.letter]++
const perTheme = {}
for (const t of THEME_ORDER) perTheme[t] = 0
for (const r of rows) for (const t of r.themes) perTheme[t]++

/**
 * FEATURED, chosen deterministically. Six buckets (B/S/V x 2/3 syllables), four
 * from each, drawn from the gold rows with no hard cluster. Then the two rows
 * that started all of this are forced in, each displacing the last pick from its
 * own bucket so the spread holds.
 *
 * Ranking is not just "shortest gloss". Sorting 201 gold rows by length puts
 * fifteen identical "name of a king" cards on the page, because the shortest
 * gloss in this corpus is almost always a bare attestation with no sense in it.
 * So a row that says something ranks above a row that only says someone was
 * called this, a row with a theme ranks above one without, and only then does
 * length break the tie. Duplicate glosses are dropped outright.
 */
const BARE_ATTESTATION_RE = /^name of\b/i
/**
 * Apparatus that survived the tidy because expanding it would have been a
 * guess: Monier-Williams single-letter shorthand ("' s -king'" — s is serpent),
 * raw IAST-ish transliteration ("brAhmaRa"), and the dictionary's own hedges.
 * Fine in the corpus, wrong on a featured card, so it sorts last rather than
 * being filtered out — a thin bucket still fills.
 */
const MESSY_GLOSS_RE = /(?:^|[\s'"(])[b-hj-z](?=[\s'")\-,.]|$)|[a-z][A-Z]|\(\?\)|\bprob\.|\bperhaps\b/

function pickFeatured() {
  const picked = []
  const usedGloss = new Set()
  for (const letter of ['B', 'S', 'V']) {
    for (const syl of [2, 3]) {
      const bucket = gold
        .filter((r) => r.letter === letter && r.syllables === syl && !r.badges.hardCluster)
        .filter((r) => r.gloss.length >= 8)
        .sort(
          (a, b) =>
            MESSY_GLOSS_RE.test(a.gloss) - MESSY_GLOSS_RE.test(b.gloss) ||
            BARE_ATTESTATION_RE.test(a.gloss) - BARE_ATTESTATION_RE.test(b.gloss) ||
            (a.themes.length === 0) - (b.themes.length === 0) ||
            // Over 60 chars will not fit a card; under 8 says nothing.
            (a.gloss.length > 60) - (b.gloss.length > 60) ||
            a.gloss.length - b.gloss.length ||
            (a.id < b.id ? -1 : 1),
        )
        .filter((r) => {
          const key = r.gloss.toLowerCase()
          if (usedGloss.has(key)) return false
          usedGloss.add(key)
          return true
        })
        .slice(0, 4)
      picked.push(...bucket.map((r) => r.id))
    }
  }
  for (const forcedId of FEATURED_FORCED) {
    if (picked.includes(forcedId)) continue
    const row = rows.find((r) => r.id === forcedId)
    if (!row) {
      fail(`FEATURED must include ${forcedId}, which is not in the dataset`)
      continue
    }
    const sameBucket = picked.filter((id) => {
      const r = rows.find((x) => x.id === id)
      return r.letter === row.letter && r.syllables === row.syllables
    })
    const displaced = sameBucket[sameBucket.length - 1]
    const at = picked.indexOf(displaced)
    if (at >= 0) picked.splice(at, 1, forcedId)
    else picked.push(forcedId)
  }
  return picked
}
const featured = pickFeatured()
assertEq('featured count', featured.length, 24)
for (const forcedId of FEATURED_FORCED) if (!featured.includes(forcedId)) fail(`FEATURED lost ${forcedId}`)
if (PROVENANCE.length > 400) fail(`provenance is ${PROVENANCE.length} chars, cap is 400`)

/* ── emit ────────────────────────────────────────────────────────────────── */

const banner =
  `// GENERATED by scripts/naam/build-dataset.mjs - do not edit.\n` +
  `// Source: Boy_Name_Candidates.pdf (172pp, untracked - the repo is public)\n` +
  `//   pdf  sha256 ${shaPdf}\n` +
  `//   text sha256 ${shaText}  (src/data/naam/source-text.txt)\n` +
  `// Counts below are the document's own stated totals, reproduced by the parse.\n` +
  `// Row type and theme bit order: src/types/naam.ts\n`

const factsSource =
  banner +
  `import type { NaamTheme } from '@/types/naam'\n\n` +
  `export const NAAM_SOURCE_SHA = {\n` +
  `  pdf: '${shaPdf}',\n` +
  `  text: '${shaText}',\n` +
  `} as const\n\n` +
  `export const NAAM_COUNTS = {\n` +
  `  total: ${rows.length},\n` +
  `  core: ${core.length},\n` +
  `  rest: ${rest.length},\n` +
  `  attested: ${rows.filter((r) => r.badges.attested).length},\n` +
  `  evocative: ${rows.filter((r) => r.badges.evocative).length},\n` +
  `  gold: ${gold.length},\n` +
  `  pages: 172,\n` +
  `  perLetter: ${JSON.stringify(perLetter)},\n` +
  `  perTheme: ${JSON.stringify(perTheme)} as Record<NaamTheme, number>,\n` +
  `} as const\n\n` +
  `/** Ids whose syllableSplit.length differs from the document's own count. See src/types/naam.ts. */\n` +
  `export const NAAM_SYLLABLE_SPLIT_EXCEPTIONS: readonly string[] = ${JSON.stringify(SYLLABLE_SPLIT_EXCEPTIONS)}\n\n` +
  `/**\n` +
  ` * Deterministic: 4 gold rows (attested AND evocative, no hard cluster) from each of\n` +
  ` * B/S/V x 2/3 syllables, ranked legible-gloss > says-something > has-a-theme > short,\n` +
  ` * duplicate glosses dropped. Snehaja (p.64) and Vishala/Bishala (p.140) are forced in.\n` +
  ` */\n` +
  `export const NAAM_FEATURED: readonly string[] = ${JSON.stringify(featured)}\n\n` +
  `/** One sentence, <=400 chars. Reused in page copy and in the system prompt - never retyped. */\n` +
  `export const NAAM_PROVENANCE =\n  ${JSON.stringify(PROVENANCE)}\n`

if (check) {
  const same = (path, want) => {
    let got = ''
    try {
      got = readFileSync(path, 'utf8')
    } catch {
      fail(`${path} is missing - run \`npm run naam:build\``)
      return
    }
    if (got !== want) fail(`${path} is stale - run \`npm run naam:build\``)
  }
  same(OUT_CORE, JSON.stringify(core))
  same(OUT_REST, JSON.stringify(rest))
  same(OUT_FACTS, factsSource)
  if (failures.length) {
    console.error('naam dataset check FAILED')
    for (const f of failures) console.error(`  - ${f}`)
    process.exit(1)
  }
  process.exit(0)
}

if (failures.length) {
  console.error('naam dataset build FAILED - nothing written')
  for (const f of failures) console.error(`  - ${f}`)
  process.exit(1)
}

mkdirSync(dirname(OUT_CORE), { recursive: true })
mkdirSync(dirname(OUT_FACTS), { recursive: true })
writeFileSync(OUT_CORE, JSON.stringify(core))
writeFileSync(OUT_REST, JSON.stringify(rest))
writeFileSync(OUT_FACTS, factsSource)

const kb = (path) => `${(readFileSync(path).length / 1024).toFixed(0)} KB`
const goldTagged = gold.filter((r) => r.themes.length > 0).length
console.log(`naam: ${rows.length} rows parsed, all 18 buckets match the document`)
console.log(
  `  core (shortlist)  ${String(core.length).padStart(5)}  ->  ${OUT_CORE.replace(REPO + '/', '')}  ${kb(OUT_CORE)}`,
)
console.log(
  `  rest (more)       ${String(rest.length).padStart(5)}  ->  ${OUT_REST.replace(REPO + '/', '')}  ${kb(OUT_REST)}`,
)
console.log(`  attested ${STATED_TOTALS.attested} · evocative ${STATED_TOTALS.evocative} · gold ${gold.length}`)
console.log(
  `  gloss tidied on ${rows.filter((r) => !r.glossIsVerbatim).length} rows, kept verbatim on ${rows.filter((r) => r.glossIsVerbatim).length}`,
)
console.log(
  `  themes tagged on ${rows.filter((r) => r.themes.length).length} rows; gold coverage ${goldTagged}/${gold.length}`,
)
console.log(
  `  syllableSplit matches the document on ${rows.length - SYLLABLE_SPLIT_EXCEPTIONS.length}/${rows.length} rows (${SYLLABLE_SPLIT_EXCEPTIONS.length} known document undercounts)`,
)
console.log(`  featured ${featured.length}: ${featured.join(' ')}`)
