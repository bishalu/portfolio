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
 * and an index scale the paper rows were not measured at. The paper's numbers
 * are at 5k, so this is now the same scale. What still differs is the corpus —
 * ours is FMA — and the panel says so.
 *
 * The Choon row is the NEURAL CHANNEL ALONE. That is the whole point of the
 * comparison and it is the thing easiest to get wrong: the shipped identifier
 * is a two-tier system whose classical channel carries most of the accuracy, so
 * putting the fused number next to a single-model baseline would be comparing a
 * system to a component. The fused number is materially higher and it is not
 * used here.
 *
 * Same eleven conditions and the same 5,000-reference scale, but a different
 * corpus. Stating that is the difference between a benchmark and a claim.
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

export const BENCH_SOURCE = {
  paper: 'Robust Neural Audio Fingerprinting using Music Foundation Models (2025), Table 1',
  url: 'https://openreview.net/pdf/ea1aa2f13377131ac8653e70f015ef224fd9e55d.pdf',
  ourSetup: '5,000 references · 5,500 queries · 8-second segments',
}
