/** Static export v2. Raw recordings keep the original experiment row fields. */
interface MethodSummary {
  num_instances: number; accuracy: number; baseline_accuracy: number;
  paired_accuracy_diff: number; rescued_count: number; base_wrong_count: number;
  degraded_count: number; base_correct_count: number; mean_recurrent_calls: number;
  cluster_ci_95_low: number; cluster_ci_95_high: number;
  cluster_mcnemar_p_value?: number; num_unique_problems: number; ledger_mismatches: number;
}
interface RecordedRun {
  run_id: string; label: string; split: string; record_count: number;
  problem_ids: string[]; num_problems: number; seeds: number[];
  methods: Record<string, MethodSummary>; trace_url: string; trace_sha256: string;
  source_trace_sha256: string; manifest_url: string; ledger_mismatches: number; status?: string;
}
interface EvidenceItem { label: string; source?: string; url: string; sha256: string; historical: boolean }
interface Observatory {
  schema_version: number; meta?: Record<string, unknown>; runs: Record<string, RecordedRun>;
  evidence: EvidenceItem[]; files: Record<string, {sha256: string; bytes: number; source: string}>;
}
interface ExcursionTelemetry {
  step: number; alpha?: number; effective_alpha?: number; beta?: number; token_scope?: string;
  curvature_frob?: number; skipped_unstable?: boolean; clipped?: boolean;
  donor_branch_id?: string; receiving_state_hash?: string; visit_state_hash?: string;
  returned_shift_vs_normal_frob?: number; branch_relative_distance?: number;
  update_visit_frob?: number; post_receiving_state_hash?: string;
}
interface PolicyDecision {
  action_type?: string; action?: "visit" | "no_visit"; reason?: string; action_id?: string; donor_index?: number;
  predicted_advantage?: number; action_scores?: Record<string, number>;
  state_features?: {step?: number};
}
interface StepSnapshot {
  step: number; branch_entropy?: Record<string, number>; branch_margin?: Record<string, number>;
  branch_rms?: Record<string, number>; ab_relative_distance?: number;
  policy_decision?: PolicyDecision | null; excursion_telemetry?: ExcursionTelemetry | null;
}
interface TokenLedgerEntry {
  token_index?: number; recurrent_block_calls: number; decoded_token?: string; token_id?: number;
  shared_calls?: number; primary_branch_calls?: number; donor_calls?: number; visit_calls?: number;
  continuation_calls?: number; chosen_depth?: number; exited_early_step?: number | null;
  step_snapshots?: StepSnapshot[]; controller_action_type?: string; selected_donor?: string;
  effective_alpha?: number; effective_beta?: number; policy_reason?: string;
  action_scores?: Record<string, number>; rejected_actions?: Record<string, unknown>;
}
interface RecordedTrace {
  problem_id: string; family: string; split: string; seed: number; method: string;
  prompt: string; gold_answer: string; extracted_answer?: string; decoded_text: string;
  is_correct: boolean; recurrent_block_calls: number; scoring_recurrent_calls?: number;
  wall_time_ms: number; derived_parallel_selection?: boolean; alpha?: number; beta?: number;
  visit_step?: number; token_scope?: string; curvature_frob?: number; curvature_relative?: number;
  returned_shift_vs_normal_frob?: number; branch_relative_distance?: number; clipped?: boolean;
  skipped_unstable?: boolean; policy_action?: string; policy_reason?: string;
  controller_action_type?: string; selected_donor?: string; early_exit_step?: number | null;
  controller_predicted_advantage?: number; controller_action_scores?: Record<string, number>;
  gold_logprob?: number; delta_gold_logprob?: number; donor_b_answer?: string;
  token_ledger?: TokenLedgerEntry[]; step_snapshots?: StepSnapshot[];
  pca_coords?: Record<string, number[][]>; pca_explained_variance?: number[];
  [key: string]: unknown;
}
interface MathJaxApi { typesetPromise?: (elements?: HTMLElement[]) => Promise<unknown> }
interface Window { TEBDIL_DATA?: Observatory; MathJax?: MathJaxApi }
