/**
 * The six capabilities under the three products, and how strongly each product
 * leans on each one.
 *
 * Audited against the product repos rather than recalled, because the first
 * version of this was written from memory and got two rows wrong in the
 * direction that mattered — it had both Cue and Choon merely "using" retrieval
 * when retrieval is central to both. Evidence for every cell:
 *
 *   Curation (vibeset_dj/vibeset_2)
 *     ingest    lean  setlist_metadata_service, setlists_pg, setlist APIs
 *     enrich    lean  test_llm_enrichment_only, test_realistic_enrichment
 *     represent use   embeddings from hosted models (test_noisy_embedding)
 *     train     none  no training code in the repo
 *     retrieve  lean  classify_query_complexity router, hybrid, rerank, diversity
 *     serve     lean  vibe_check A/B framework, finalizer cost collection
 *
 *   Cue (vibeset-video)
 *     ingest    use   partner catalog + audio_cache, ffmpeg; sourced, not scraped
 *     enrich    lean  musical_descriptor_v1 built from genre/mood/instrument/BPM/key
 *     represent lean  Titan V2 embeddings, energy_alignment, nova_video_utils
 *     train     none  no training code; hosted models only
 *     retrieve  lean  pgvector cosine over musical_embedding_v1, rerank branches
 *     serve     lean  run_cost.py, infra/budget.tf, nova-pro-evaluation, rate_limit
 *
 *   Choon (fingerprinting)
 *     ingest    lean  reference catalogue, watermark payloads, C2PA provenance
 *     enrich    use   metadata only; no LLM enrichment path
 *     represent lean  spectral landmarks and learned embeddings
 *     train     lean  the pruning paper, distillation, quantisation — the repo
 *     retrieve  lean  FAISS ANN, temporal alignment, Lowe ratio over RANSAC
 *     serve     lean  eval harnesses, benchmark scripts, Cloud Run /api/ready
 *
 * The shape that falls out is the argument: `train` reaches exactly one product,
 * `retrieve` and `serve` reach all three, and everything else lands in between.
 * One-to-one and one-to-many in the same picture, which is why this is drawn
 * rather than tabulated.
 */

export type Weight = 'lean' | 'use' | 'none'

export interface Capability {
  id: string
  name: string
  gloss: string
  /** Weight per product, in PRODUCTS order. */
  w: readonly [Weight, Weight, Weight]
}

export const PRODUCTS = [
  { id: 'curation', name: 'Curation', href: '/vibeset/curation' },
  { id: 'cue', name: 'Cue', href: '/vibeset/cue' },
  { id: 'choon', name: 'Choon', href: '/vibeset/choon' },
] as const

export const CAPABILITIES: readonly Capability[] = [
  {
    id: 'ingest',
    name: 'Ingest and provenance',
    gloss: 'connectors, crawlers, rights tracked at the source',
    w: ['lean', 'use', 'lean'],
  },
  {
    id: 'enrich',
    name: 'Enrich',
    gloss: 'ETL, LLM labelling, human review in the loop',
    w: ['lean', 'lean', 'use'],
  },
  {
    id: 'represent',
    name: 'Represent',
    gloss: 'embeddings, fingerprints, learned features',
    w: ['use', 'lean', 'lean'],
  },
  {
    id: 'train',
    name: 'Train and compress',
    gloss: 'fine-tuning, pruning, quantisation, latency budgets',
    w: ['none', 'none', 'lean'],
  },
  {
    id: 'retrieve',
    name: 'Retrieve',
    gloss: 'exact, keyword, vector, diversity, rerank',
    w: ['lean', 'lean', 'lean'],
  },
  {
    id: 'serve',
    name: 'Serve and prove',
    gloss: 'inference under a deadline, eval harnesses, cost governance',
    w: ['lean', 'lean', 'lean'],
  },
]

export const WEIGHT_LABEL: Record<Weight, string> = {
  lean: 'leans on',
  use: 'uses',
  none: 'does not use',
}
