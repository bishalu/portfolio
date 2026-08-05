/**
 * Choon's neural channel against the published Table 1 of
 * "Robust Neural Audio Fingerprinting using Music Foundation Models" (2025).
 *
 * Source of record: fingerprinting/docs/TABLE_1_1.md, which is itself derived
 * from data/processed/benchmarks/paper11_scoring_sweep.csv (winning scorer
 * `softmax_pool|4view_overlap|k2|t0.15`) and the paper PDF. Do not retype these
 * from memory — regenerate that doc and copy it across.
 *
 * The Choon row is the NEURAL CHANNEL ALONE. That is the whole point of the
 * comparison and it is the thing easiest to get wrong: the shipped identifier
 * is a two-tier system whose classical channel carries most of the accuracy, so
 * putting the fused number next to a single-model baseline would be comparing a
 * system to a component. The fused number is materially higher and it is not
 * used here.
 *
 * Not a like-for-like reproduction, and the panel says so: the paper's rows are
 * its own track-level evaluation; the Choon row is this repo's reduced-scale
 * proxy — 2,000 reference tracks, 400 queries, 8-second segments. Same
 * conditions, smaller index. Stating that is the difference between a benchmark
 * and a claim.
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
  { model: 'MuQ-Large, frozen', params: 300, paramsLabel: '~300M', recall: 83.91, encoded: 90.0 },
  {
    model: 'Choon — neural channel only',
    params: 48,
    paramsLabel: '48M',
    recall: 77.23,
    encoded: 95.5,
    ours: true,
  },
  { model: 'MERT, unfrozen', params: 95, paramsLabel: '95M', recall: 74.27, encoded: 44.0 },
  { model: 'MERT, frozen', params: 95, paramsLabel: '95M', recall: 70.91, encoded: 38.0 },
  { model: 'BEATs, unfrozen', params: 90, paramsLabel: '~90M', recall: 70.27, encoded: 33.0 },
  { model: 'GraFPrint', params: null, paramsLabel: 'not published', recall: 67.82, encoded: 17.0 },
  { model: 'NAFP', params: 19.22, paramsLabel: '19.2M', recall: 63.45, encoded: 10.0 },
  { model: 'Dejavu', params: null, paramsLabel: 'classical', recall: 49.18, encoded: 3.0 },
]

export const BENCH_SOURCE = {
  paper: 'Robust Neural Audio Fingerprinting using Music Foundation Models (2025), Table 1',
  url: 'https://openreview.net/pdf/ea1aa2f13377131ac8653e70f015ef224fd9e55d.pdf',
  ourSetup: '2,000 reference tracks · 400 queries · 8-second segments',
}
