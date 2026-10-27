"""Evaluation runner, metrics, bootstrap tests, and trace logging."""

from tebdil.evaluation.bootstrap import mcnemar_exact_test, paired_bootstrap_ci
from tebdil.evaluation.metrics import (
    LogitComparisonMetrics,
    MethodPairedSummary,
    compare_logits,
    compute_token_logprob,
    summarize_paired_records,
)
from tebdil.evaluation.runner import (
    ByteLevelToyTokenizer,
    load_adapter_and_tokenizer,
    run_experiment_from_config,
)
from tebdil.evaluation.trace import (
    RunManifest,
    append_trace_record,
    create_run_manifest,
    read_trace_records,
    save_run_manifest,
    write_trace_records,
)

__all__ = [
    "ByteLevelToyTokenizer",
    "LogitComparisonMetrics",
    "MethodPairedSummary",
    "RunManifest",
    "append_trace_record",
    "compare_logits",
    "compute_token_logprob",
    "create_run_manifest",
    "load_adapter_and_tokenizer",
    "mcnemar_exact_test",
    "paired_bootstrap_ci",
    "read_trace_records",
    "run_experiment_from_config",
    "save_run_manifest",
    "summarize_paired_records",
    "write_trace_records",
]
