/**
 * The graded query set for /naam retrieval.
 *
 * WHY THIS FILE IS THE POINT. Every retrieval parameter in match.ts — k1, b, the
 * ×2.4 exact-gloss bonus, the 30% floor, the minimum of six — was tuned by hand
 * against whatever queries happened to occur to me. That is how the floor
 * shipped at 50% and silently cut "moon" from twenty hits to one: it was a
 * number measured against a scorer WITHOUT the exact-gloss bonus, and it was
 * only caught because "moon" is a word I thought to type. Nobody could say what
 * else it broke. This file is the answer to "what else did it break".
 *
 * ─── HOW GOLD IS DECIDED, AND WHY IT IS NOT BM25 ───────────────────────────
 *
 * A gold set derived from the retriever would agree with the retriever forever.
 * So relevance here is decided by two things the retriever never reads:
 *
 *   1. A REGEX OVER THE GLOSS TEXT, written by hand from the corpus's own
 *      vocabulary. For the gap tier this regex deliberately does NOT contain the
 *      query word — that is the whole test. "brave" appears in no gloss in this
 *      document; the rows that answer it say valiant, heroic, bold, courage.
 *   2. THE DOCUMENT'S OWN BADGES, to decide whether a concept match is a name
 *      anyone would actually accept.
 *
 * Rule 2 is not a nicety. "song" matches 75 rows, and 74 of them read "name of a
 * saman" — a Vedic chant, correctly matched and useless as a suggestion for a
 * child. A gold set of 75 chants would reward a retriever for burying Bandin
 * ("a praiser, bard"), which is the one row a person asking for "song" wants. So
 * derived gold drops rows whose gloss merely attests a bearer, unless the case
 * says otherwise. Concept-matched and worth-suggesting are different questions
 * and this corpus separates them sharply.
 *
 * ─── TIERS ─────────────────────────────────────────────────────────────────
 *
 *   literal  the word is in the glosses. Measures ranking.
 *   gap      the word is in NO gloss. Measures whether meaning is reachable at
 *            all. This is the tier the upgrade is for.
 *   phrase   how people actually type. Measures whether the sentence survives
 *            stopword and form-word stripping.
 *   noise    nothing is being asked. Success is returning NOTHING; a confident
 *            answer to gibberish is worse than no answer.
 *
 * Add a case by adding a line. Cases are cheap; being wrong in production is not.
 */

/**
 * A case is `{ q, tier, note }` plus exactly one way of naming the right answers:
 *
 *   gold    explicit latin spellings, for concepts too small or too odd to
 *           express as a pattern. Validated at load — a typo throws.
 *   goldRe  a pattern over the gloss. Everything matching, minus bare
 *           attestations, is gold.
 *   also    extra latin spellings unioned into a goldRe result.
 *   keepBare  set when "name of …" rows genuinely are the answer (deity names,
 *           river names — where being the name of the thing IS the meaning).
 */
export const CASES = [
  /* ── literal: the word is right there, so this is a ranking test ────────── */
  { q: 'moon', tier: 'literal', goldRe: /\bmoon\b/i, note: '19 glosses; Sasi is the one-word gloss' },
  { q: 'sun', tier: 'literal', goldRe: /\bsun\b|\bsolar\b/i },
  { q: 'light', tier: 'literal', goldRe: /\blight\b|\bsplendour\b|\bradian/i },
  { q: 'fire', tier: 'literal', goldRe: /\bfire\b|\bflame\b/i },
  { q: 'water', tier: 'literal', goldRe: /\bwater\b|\bstream\b/i },
  { q: 'ocean', tier: 'literal', goldRe: /\bocean\b|\bsea\b/i },
  { q: 'river', tier: 'literal', goldRe: /\briver\b/i, keepBare: true, note: 'being the name of a river IS the meaning' },
  { q: 'mountain', tier: 'literal', goldRe: /\bmountain\b|\bpeak\b|\bsummit\b/i, keepBare: true },
  { q: 'sky', tier: 'literal', goldRe: /\bsky\b|\bheaven\b|\batmosphere\b|\bair\b/i },
  { q: 'earth', tier: 'literal', goldRe: /\bearth\b|\bsoil\b|\bground\b/i },
  { q: 'tree', tier: 'literal', goldRe: /\btree\b|\bfig-tree\b/i },
  { q: 'lotus', tier: 'literal', goldRe: /\blotus\b/i },
  { q: 'gold', tier: 'literal', goldRe: /\bgold\b|\bgolden\b/i },
  { q: 'king', tier: 'literal', goldRe: /\bking\b|\bsovereign\b|\bmonarch\b/i },
  { q: 'wise', tier: 'literal', goldRe: /\bwise\b|\bwisdom\b|\bintelligen|\bunderstanding\b|\blearned\b/i },
  { q: 'strong', tier: 'literal', goldRe: /\bstrong\b|\bstrength\b|\bmighty\b|\bpowerful\b|\bpower\b/i },
  { q: 'happy', tier: 'literal', goldRe: /\bhappy\b|\bhappiness\b|\bjoy\b|\bglad\b|\bdelight/i },
  { q: 'pure', tier: 'literal', goldRe: /\bpure\b|\bpurity\b|\bspotless\b|\bclean\b/i },
  { q: 'truth', tier: 'literal', goldRe: /\btruth\b|\btrue\b|\breal\b|\bhonest\b/i },
  { q: 'swift', tier: 'literal', goldRe: /\bswift\b|\bquick\b|\bfast\b|\bspeed\b/i },
  { q: 'brilliant', tier: 'literal', goldRe: /\bbrilliant\b|\bshining\b|\bshines\b|\bbright\b/i },
  { q: 'teacher', tier: 'literal', goldRe: /\bteacher\b|\bpreceptor\b|\binstructor\b/i, keepBare: true },
  { q: 'friend', tier: 'literal', goldRe: /\bfriend\b|\bcompanion\b|\ballly?\b/i },
  { q: 'love', tier: 'literal', goldRe: /\blove\b|\baffection\b|\bfond/i },
  { q: 'dawn', tier: 'literal', goldRe: /\bdawn\b|\bdaybreak\b|\bmorning\b/i },
  { q: 'blessed', tier: 'literal', goldRe: /\bblessed\b|\bauspicious\b|\bfortunate\b/i },
  { q: 'protector', tier: 'literal', goldRe: /\bprotect|\bguard|\bshelter\b|\brefuge\b/i },
  { q: 'free', tier: 'literal', goldRe: /\bfree\b|\bfreedom\b|\bliberat/i },
  { q: 'sacred', tier: 'literal', goldRe: /\bsacred\b|\bholy\b/i },
  { q: 'good', tier: 'literal', goldRe: /\bgood\b|\bexcellent\b|\bvirtue\b/i },

  /* ── gap: the word appears in NO gloss. This is the tier the work is for ── */
  {
    q: 'brave',
    tier: 'gap',
    goldRe: /\b(valiant|heroic|hero\b|bold\b|courage)/i,
    note: 'zero glosses contain "brave"; reachable only via valiant/heroic/bold/courage',
  },
  {
    q: 'calm',
    tier: 'gap',
    goldRe: /\b(tranquil|quiet|appease|pacify|composure|equanimity)/i,
    note: 'zero glosses contain "calm". Only two rows exist at all — a sparse concept, not a broken one',
  },
  {
    q: 'peace',
    tier: 'gap',
    goldRe: /\b(tranquil|quiet|appease|pacify|peaceful|equanimity)/i,
    note: 'zero glosses contain "peace"',
  },
  {
    q: 'hope',
    tier: 'gap',
    goldRe: /\b(wish|wished|desired|longing|aspiration)/i,
    note: 'zero glosses contain "hope"',
  },
  {
    q: 'song',
    tier: 'gap',
    gold: ['Bandin'],
    note: '75 rows match hymn/chant/saman and 74 read "name of a saman" — correct and useless. Bandin ("a praiser, bard") is the only real answer, so this case is pinned rather than derived',
  },
  {
    q: 'healer',
    tier: 'gap',
    goldRe: /\b(healthy|health|healing|physician|medicine)/i,
    note: 'zero glosses contain "healer"',
  },
  { q: 'gentle', tier: 'gap', goldRe: /\b(tender|mild|gentleness|kindness|meek)/i, note: 'one gloss has the word; the concept is wider' },
  { q: 'star', tier: 'gap', goldRe: /\b(star|constellation|nakshatra|asterism)/i, keepBare: true, note: 'the word exists twice; the concept is 15 rows, nearly all attestations, so bare rows are kept' },
  { q: 'warrior', tier: 'gap', goldRe: /\b(warrior|hero\b|heroic|valiant|kshatriya|fighting)/i, keepBare: true },
  { q: 'clever', tier: 'gap', goldRe: /\b(intelligen|understanding|wise|shrewd|skilful|skilled)/i },
  { q: 'courage', tier: 'gap', goldRe: /\b(courage|valiant|heroic|bold\b|prowess|exertion)/i },
  {
    q: 'kindness',
    tier: 'gap',
    goldRe: /\b(kindness|compassion|benevolen|friendly|beneficent|gracious)\b|\bkind[;,]/i,
    note: 'NOT /\\bkind\\b/ — that matched "pl. name of a particular kind of brick" and put a brick in the gold set',
  },
  { q: 'victory', tier: 'gap', goldRe: /\b(victor|conquer|triumph|winning)/i },
  { q: 'generous', tier: 'gap', goldRe: /\b(generous|liberal|bountiful|giving|bestow)/i },
  { q: 'noble', tier: 'gap', goldRe: /\b(noble|dignified|illustrious|eminent)/i },
  { q: 'devoted', tier: 'gap', gold: ['Shraddha'], note: 'the whole core set has one row for this' },
  /**
   * NOT CASES, AND THE REASON IS WORTH KEEPING. "patience" and "fearless" are
   * things people genuinely ask for, and this document genuinely does not have
   * them — searched wide (patien|forbear|endur|khanti|kshanti|tolerant|
   * steadfast|persever) the core set yields one row, "hard, solid, firm, strong,
   * steadfast", and fearless yields nothing at all. Scoring retrieval on a
   * concept the corpus lacks measures the corpus, not the retriever. Saying so
   * out loud is the AGENT's job and belongs in an agent test, not here.
   */
  { q: 'radiant', tier: 'gap', goldRe: /\b(radian|shining|splendour|lustre|brilliant|glow)/i },
  { q: 'eternal', tier: 'gap', goldRe: /\b(eternal|everlasting|imperishable|undying|immortal)/i },
  { q: 'guide', tier: 'gap', goldRe: /\b(guide|leader|conductor|instructor)/i },
  { q: 'music', tier: 'gap', goldRe: /\b(music|melody|lute|singer)/i },
  { q: 'voice', tier: 'gap', goldRe: /\b(voice|speech|word\b|utterance)/i },
  { q: 'flower', tier: 'gap', goldRe: /\b(flower|blossom|lotus)/i },
  { q: 'lucky', tier: 'gap', goldRe: /\b(lucky|fortunate|auspicious|prosper)/i },
  { q: 'thunder', tier: 'gap', goldRe: /\b(thunder|lightning|storm)/i },
  { q: 'wind', tier: 'gap', goldRe: /\b(wind|breeze|gale|air\b)/i },
  { q: 'healthy', tier: 'gap', goldRe: /\b(health|healthy|vigor|well-being)/i },
  { q: 'beautiful', tier: 'gap', goldRe: /\b(beautiful|lovely|handsome|adorned|charming)/i },
  { q: 'quiet', tier: 'gap', goldRe: /\b(quiet|tranquil|silent|still)/i },

  /* ── phrase: what people type. Tests stopword and form-word stripping ───── */
  { q: 'something that means light', tier: 'phrase', goldRe: /\blight\b|\bsplendour\b/i },
  { q: 'a name about the ocean', tier: 'phrase', goldRe: /\bocean\b|\bsea\b/i },
  { q: 'any related to the moon?', tier: 'phrase', goldRe: /\bmoon\b/i },
  { q: 'we want something brave', tier: 'phrase', goldRe: /\b(valiant|heroic|hero\b|bold\b|courage)/i },
  { q: 'a short name about light', tier: 'phrase', goldRe: /\blight\b|\bsplendour\b/i, note: '"short" is a FORM word and must not reach the meaning search — it used to retrieve "without, except, short of"' },
  { q: 'two syllables and something to do with the sun', tier: 'phrase', goldRe: /\bsun\b/i, note: '"two" and "syllables" are form words — they used to retrieve "name of two arhats"' },
  { q: 'a name that feels like home', tier: 'phrase', goldRe: /\b(dwelling|residence|abode|house|staying)/i },
  { q: 'something calm and peaceful', tier: 'phrase', goldRe: /\b(tranquil|quiet|appease|pacify)/i },
  { q: 'a warrior name', tier: 'phrase', goldRe: /\b(warrior|hero\b|heroic|valiant)/i, keepBare: true },
  { q: 'names to do with wisdom', tier: 'phrase', goldRe: /\bwise\b|\bwisdom\b|\bintelligen|\bunderstanding\b/i },
  { q: 'i want a name meaning strength', tier: 'phrase', goldRe: /\bstrength\b|\bstrong\b|\bmighty\b|\bpower/i },
  { q: 'something to do with the sky', tier: 'phrase', goldRe: /\bsky\b|\bheaven\b|\bair\b|\batmosphere\b/i },
  { q: 'a gentle sounding name with a good meaning', tier: 'phrase', goldRe: /\b(tender|mild|gentleness|kindness)/i },
  { q: 'what about fire', tier: 'phrase', goldRe: /\bfire\b|\bflame\b/i },
  { q: 'names that mean happy', tier: 'phrase', goldRe: /\bhappy\b|\bhappiness\b|\bjoy\b|\bglad\b/i },
  {
    q: 'a healer, someone who mends people',
    tier: 'phrase',
    goldRe: /\b(health|healthy|physician|medicine|healing)/i,
    keepBare: true,
    note: 'caught end to end, not here: `people` is in 57 glosses meaning a TRIBE, and it buried the real target under ethnonyms — the whole query returned no picks. Regression guard for the apparatus stopwords',
  },
  {
    q: 'a name for a person with a good heart',
    tier: 'phrase',
    goldRe: /\b(kindness|compassion|benevolen|friendly|beneficent|gracious|virtuous)\b|\b(kind|good)[;,]/i,
    note: '`person` is the same trap as `people`. Gold widened after the fact: retrieval answered Sadhu ("good; pleasant; auspicious") and Bhadraka ("good; fine; beneficial"), which ARE what someone means by a good heart — the narrow kindness-only pattern was my error, not the retriever\'s',
  },

  /* ── noise: success is returning NOTHING ────────────────────────────────── */
  { q: 'asdfgh', tier: 'noise' },
  { q: '12345', tier: 'noise' },
  { q: 'qwertyuiop', tier: 'noise' },
  { q: 'zzzzzz', tier: 'noise' },
  { q: 'hello', tier: 'noise', note: 'a greeting is not a query — the agent should push toward a name, not retrieve on it' },
  { q: 'hi there', tier: 'noise' },
  { q: 'ok', tier: 'noise' },
  { q: 'thanks!', tier: 'noise' },
  { q: '???', tier: 'noise' },
  { q: 'lorem ipsum dolor', tier: 'noise' },
]

/** "name of a mountain" names a bearer, not a sense. Same test match.ts uses. */
const BARE_ATTESTATION_RE = /^name of\b/i

/**
 * Resolve a case's gold set to row ids.
 *
 * Throws on an explicit spelling that names no row — a gold set with a typo in
 * it quietly lowers every score forever, and a loud failure at load is much
 * cheaper than a slow drift in the numbers.
 */
export function goldIds(testCase, rows, byLatin) {
  if (testCase.tier === 'noise') return new Set()

  const ids = new Set()
  const add = (latin) => {
    const row = byLatin.get(String(latin).toLowerCase())
    if (!row) throw new Error(`cases.mjs: "${testCase.q}" names ${latin}, which is not a row`)
    ids.add(row.id)
  }

  if (testCase.gold) for (const latin of testCase.gold) add(latin)
  if (testCase.also) for (const latin of testCase.also) add(latin)

  if (testCase.goldRe) {
    for (const row of rows) {
      const gloss = String(row.gloss || '')
      if (!testCase.goldRe.test(gloss)) continue
      if (!testCase.keepBare && BARE_ATTESTATION_RE.test(gloss)) continue
      ids.add(row.id)
    }
  }

  if (ids.size === 0) {
    throw new Error(`cases.mjs: "${testCase.q}" has an empty gold set — the pattern matches nothing`)
  }
  return ids
}
