/**
 * Choon's neural channel against the published Table 1 of
 * "Robust Neural Audio Fingerprinting using Music Foundation Models" (2025).
 *
 * Paper rows: fingerprinting/docs/TABLE_1_1.md, from the published Table 1.
 *
 * Choon row: a run of scripts/evaluate_paper11_track_exact.py against
 * data/processed/benchmarks/paper11_5k_8s — 5,000 references, 19,991 reference
 * anchors, 500 query tracks, 5,500 queries, 8s segments, scorer
 * `softmax_pool|4view_overlap|k2|t0.15`, checkpoint
 * experiments/run_20260704_102735_saug_student/checkpoints/best_hardening_recall.pt
 * (epoch 26, 27.66M parameters). Do not retype these from memory — rerun it.
 *
 * The earlier version of this table used the 2,000-reference proxy and the 48M
 * ep76 checkpoint, which understated the model twice over: wrong checkpoint,
 * and an index scale the paper rows were not measured at.
 *
 * The corpus caveat that sat here was also wrong. The paper evaluates on 5,000
 * FMA reference tracks (SCALING_66K_STRATEGY.md:29) — the same corpus and the
 * same reference count as this run. What actually differs is query length: the
 * paper uses 10-second queries, this uses 8-second. Longer queries carry more
 * evidence per match, so they are easier. The difference runs against Choon,
 * not for it, which is why it is stated rather than buried.
 *
 * The Choon row is the NEURAL CHANNEL ALONE. That is the whole point of the
 * comparison and it is the thing easiest to get wrong: the shipped identifier
 * is a two-tier system whose classical channel carries most of the accuracy, so
 * putting the fused number next to a single-model baseline would be comparing a
 * system to a component. The fused number is materially higher and it is not
 * used here.
 *
 * Same corpus, same reference count, same eleven conditions. Shorter queries.
 * Stating that is the difference between a benchmark and a claim.
 */

export interface BenchRow {
  model: string
  /** Parameter count in millions. null where no count is published. */
  params: number | null
  /** Printed size — carries "~" where the paper only gives an approximation. */
  paramsLabel: string
  /** Track-level Recall@1, percent. */
  recall: number
  /** Recall@1 on re-encoded audio, percent. */
  encoded: number
  ours?: boolean
  note?: string
}

export const BENCH: readonly BenchRow[] = [
  { model: 'MuQ-Large, unfrozen', params: 300, paramsLabel: '~300M', recall: 88.18, encoded: 96.0 },
  {
    model: 'Choon — neural channel only',
    params: 27.7,
    paramsLabel: '27.7M',
    recall: 85.4,
    encoded: 97.6,
    ours: true,
  },
  { model: 'MuQ-Large, frozen', params: 300, paramsLabel: '~300M', recall: 83.91, encoded: 90.0 },
  { model: 'MERT, unfrozen', params: 95, paramsLabel: '95M', recall: 74.27, encoded: 44.0 },
  { model: 'MERT, frozen', params: 95, paramsLabel: '95M', recall: 70.91, encoded: 38.0 },
  { model: 'BEATs, unfrozen', params: 90, paramsLabel: '~90M', recall: 70.27, encoded: 33.0 },
  { model: 'GraFPrint', params: null, paramsLabel: 'not published', recall: 67.82, encoded: 17.0 },
  { model: 'NAFP', params: 19.22, paramsLabel: '19.2M', recall: 63.45, encoded: 10.0 },
  { model: 'Dejavu', params: null, paramsLabel: 'classical', recall: 49.18, encoded: 3.0 },
]

/**
 * The landing-page subset. Four rows, chosen so each one answers a different
 * question a reader has:
 *
 *   MuQ-Large  what is ahead of it, and how big is that
 *   Choon      us
 *   MERT       the nearest larger model it beats — 3.4× the parameters
 *   NAFP       the nearest model of comparable size, 8 points below
 *
 * The five omitted rows are all below MERT and change no conclusion; the full
 * table is on /vibeset/choon. Picking four is an editorial choice and the panel
 * links the rest rather than pretending these are all of them.
 */
export const BENCH_HOME: readonly BenchRow[] = BENCH.filter((r) =>
  ['MuQ-Large, unfrozen', 'Choon — neural channel only', 'MERT, unfrozen', 'NAFP'].includes(r.model),
)

export const BENCH_SOURCE = {
  title: 'Robust Neural Audio Fingerprinting using Music Foundation Models',
  where: 'arXiv:2511.05399, November 2025 — Table 1',
  url: 'https://arxiv.org/abs/2511.05399',
  pdf: 'https://openreview.net/pdf/ea1aa2f13377131ac8653e70f015ef224fd9e55d.pdf',
  theirSetup: '5,000 FMA references · 10-second queries',
  ourSetup: '5,000 FMA references · 5,500 queries · 8-second queries',
}
