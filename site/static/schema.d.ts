/** Static export v2. Optional trace fields stay absent on older recordings. */
interface MethodSummary {
  num_instances: number;
  accuracy: number;
  baseline_accuracy: number;
  paired_accuracy_diff: number;
  rescued_count: number;
  base_wrong_count: number;
  degraded_count: number;
  base_correct_count: number;
  mean_recurrent_calls: number;
  cluster_ci_95_low: number;
  cluster_ci_95_high: number;
  num_unique_problems: number;
  ledger_mismatches: number;
}
interface RecordedRun {
  run_id: string;
  label: string;
  split: string;
  record_count: number;
  num_problems: number;
  seeds: number[];
  methods: Record<string, MethodSummary>;
  trace_url: string;
  trace_sha256: string;
  source_trace_sha256: string;
  manifest_url: string;
  ledger_mismatches: number;
}
interface Observatory {
  schema_version: number;
  runs: Record<string, RecordedRun>;
  evidence: {label: string; url: string; sha256: string; historical: boolean}[];
  files: Record<string, {sha256: string; bytes: number; source: string}>;
}
interface Window { TEBDIL_DATA?: Observatory }
interface ExcursionTelemetry {
  step: number;
  donor_branch_id?: string;
  effective_alpha: number;
  beta: number;
  curvature_frob?: number;
  skipped_unstable?: boolean;
  receiving_state_hash?: string;
  visit_state_hash?: string;
  returned_shift_vs_normal_frob?: number;
}
interface PolicyDecision {
  action_type: string;
  reason?: string;
  state_features?: {step?: number};
}
interface StepSnapshot {
  step: number;
  branch_entropy?: Record<string, number>;
  branch_margin?: Record<string, number>;
  ab_relative_distance?: number;
  policy_decision?: PolicyDecision;
  excursion_telemetry?: ExcursionTelemetry;
}
interface TokenLedgerEntry {
  recurrent_block_calls: number;
  decoded_token?: string;
  token_id?: number;
  step_snapshots?: StepSnapshot[];
  controller_action_type?: string;
  selected_donor?: string;
  effective_alpha?: number;
  effective_beta?: number;
  policy_reason?: string;
  action_scores?: Record<string, number>;
  rejected_actions?: Record<string, unknown>;
}
interface RecordedTrace {
  problem_id: string;
  seed: number;
  method: string;
  prompt: string;
  gold_answer: string;
  extracted_answer?: string;
  decoded_text: string;
  is_correct: boolean;
  recurrent_block_calls: number;
  wall_time_ms: number;
  derived_parallel_selection?: boolean;
  token_ledger?: TokenLedgerEntry[];
  step_snapshots?: StepSnapshot[];
  pca_coords?: Record<string, number[][]>;
  pca_explained_variance?: number[];
}
