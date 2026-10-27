"""TEBDIL Observatory: Interactive Streamlit + Plotly interface for latent excursion traces."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"


def load_available_runs() -> List[str]:
    if not RESULTS_DIR.exists():
        return []
    runs = []
    for d in sorted(RESULTS_DIR.iterdir()):
        if d.is_dir() and (d / "traces.jsonl").exists():
            runs.append(d.name)
    return runs


def load_run_data(run_id: str) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    run_dir = RESULTS_DIR / run_id
    summary_path = run_dir / "summary.json"
    traces_path = run_dir / "traces.jsonl"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    records: List[Dict[str, Any]] = []
    if traces_path.exists():
        for line in traces_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    return summary, records


def main() -> None:
    st.set_page_config(
        page_title="TEBDIL Observatory",
        layout="wide",
    )
    st.title("TEBDIL Observatory — Cooperative Latent Excursions in LoopLMs")
    st.caption(
        "Empirical trace inspector for recurrent hidden-state trajectories, excursion curvature C_t, "
        "rescue/degradation trade-offs, and matched-compute Pareto frontiers. All metrics are loaded directly "
        "from verified experiment artifacts in `results/`."
    )

    runs = load_available_runs()
    if not runs:
        st.warning("No completed experiment runs found in `results/`. Run `python -m tebdil.cli run --config configs/phase1_smoke.yaml` first.")
        return

    # Left panel: Run, problem, seed, and method selection
    col_left, col_center, col_right = st.columns([1.1, 1.8, 1.3])

    with col_left:
        st.subheader("1. Problem & Branch Selector")
        selected_run = st.selectbox("Experiment Run (`run_id`)", runs, index=len(runs) - 1)
        summary, records = load_run_data(selected_run)

        manifest = summary.get("manifest", {})
        st.markdown(
            f"**Model:** `{manifest.get('model_id', 'unknown')}`  \n"
            f"**Revision:** `{manifest.get('model_revision', 'unknown')[:12]}`  \n"
            f"**Split:** `{manifest.get('split', 'unknown')}` · **Steps:** `{manifest.get('recurrent_steps', 4)}`"
        )

        problem_ids = sorted({str(r["problem_id"]) for r in records})
        selected_pid = st.selectbox("Problem ID", problem_ids)
        prob_recs = [r for r in records if str(r["problem_id"]) == selected_pid]
        seeds = sorted({int(r["seed"]) for r in prob_recs})
        selected_seed = st.selectbox("Seed", seeds)

        instance_recs = {
            str(r["method"]): r
            for r in prob_recs
            if int(r["seed"]) == selected_seed
        }
        available_methods = list(instance_recs.keys())
        default_idx = available_methods.index("E1") if "E1" in available_methods else 0
        selected_method = st.selectbox("Intervention Method", available_methods, index=default_idx)

        rec = instance_recs[selected_method]
        s0_rec = instance_recs.get("S0", rec)

        st.markdown("#### Prompt")
        st.code(rec.get("prompt", ""), language="text")
        st.markdown(
            f"- **Gold Answer:** `{rec.get('gold_answer')}`\n"
            f"- **Baseline A (`S0`) Output:** `{s0_rec.get('extracted_answer')}` "
            f"({'VALID' if s0_rec.get('is_correct') else 'WRONG'})\n"
            f"- **Donor B Output:** `{rec.get('donor_b_answer', 'N/A')}` "
            f"({'VALID' if rec.get('donor_b_correct') else 'WRONG'})\n"
            f"- **`{selected_method}` Output:** `{rec.get('extracted_answer')}` "
            f"({'VALID' if rec.get('is_correct') else 'WRONG'})"
        )

        if not s0_rec.get("is_correct") and rec.get("is_correct"):
            st.success("Outcome: RESCUED (A wrong -> A' correct)")
        elif s0_rec.get("is_correct") and not rec.get("is_correct"):
            st.error("Outcome: DEGRADED (A correct -> A' wrong)")
        elif rec.get("is_correct"):
            st.info("Outcome: RETAINED CORRECT (A correct -> A' correct)")
        else:
            st.warning("Outcome: UNRESCUED (A wrong -> A' wrong)")

    # Center panel: Stepwise full-dimensional metrics + 2D PCA projection with explicit caveat
    with col_center:
        st.subheader("2. Latent Trajectory & Stepwise Dynamics")
        snaps = rec.get("step_snapshots", [])
        if snaps:
            step_rows = []
            for s in snaps:
                step_rows.append(
                    {
                        "Recurrent Step": s["step"],
                        "||H_B - H_A||_F (Full-Dim)": s.get("ab_distance_frob", 0.0),
                        "Relative Divergence": s.get("ab_relative_distance", 0.0),
                        "Cosine Similarity": s.get("ab_cosine_similarity", 1.0),
                        "Branch A RMS": s.get("branch_rms", {}).get("A", 0.0),
                        "Branch A Entropy": s.get("branch_entropy", {}).get("A", 0.0),
                        "Branch A Margin": s.get("branch_margin", {}).get("A", 0.0),
                    }
                )
            df_steps = pd.DataFrame(step_rows)
            fig_div = go.Figure()
            fig_div.add_trace(
                go.Scatter(
                    x=df_steps["Recurrent Step"],
                    y=df_steps["||H_B - H_A||_F (Full-Dim)"],
                    mode="lines+markers",
                    name="||H_B - H_A||_F",
                )
            )
            fig_div.add_trace(
                go.Scatter(
                    x=df_steps["Recurrent Step"],
                    y=df_steps["Branch A Entropy"],
                    mode="lines+markers",
                    name="Branch A Logit Entropy",
                    yaxis="y2",
                )
            )
            fig_div.update_layout(
                title="Full-Dimensional Branch Distance & Intermediate Entropy by Step",
                xaxis_title="Recurrent Step t",
                yaxis_title="Frobenius Distance ||H_B - H_A||_F",
                yaxis2=dict(title="Entropy (nats)", overlaying="y", side="right"),
                height=300,
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig_div, use_container_width=True)

        pca_coords = rec.get("pca_coords", {})
        evr = rec.get("pca_explained_variance", [0.0, 0.0])
        if pca_coords:
            fig_pca = go.Figure()
            for branch_id, pts in pca_coords.items():
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                labels = [f"{branch_id}:t={i+1}" for i in range(len(pts))]
                fig_pca.add_trace(
                    go.Scatter(
                        x=xs,
                        y=ys,
                        mode="lines+markers+text",
                        text=labels,
                        textposition="top center",
                        name=f"Branch {branch_id}",
                    )
                )
            ev1 = evr[0] * 100.0 if len(evr) > 0 else 0.0
            ev2 = evr[1] * 100.0 if len(evr) > 1 else 0.0
            fig_pca.update_layout(
                title=f"2D PCA Projection of Last-Token States (PC1: {ev1:.1f}%, PC2: {ev2:.1f}%)",
                xaxis_title=f"PC1 ({ev1:.1f}% var)",
                yaxis_title=f"PC2 ({ev2:.1f}% var)",
                height=320,
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig_pca, use_container_width=True)
            st.caption(
                f"**Methodological Note:** 2D linear PCA captures {ev1 + ev2:.1f}% of variance across the "
                "plotted points. Visual proximity in 2D does NOT prove full-dimensional attractor basin membership; "
                "rely on exact Frobenius norms and curvature metrics in the right panel."
            )

    # Right panel: Intervention telemetry, curvature C_t, and run-wide Pareto comparison
    with col_right:
        st.subheader("3. Excursion & Curvature Telemetry")
        st.metric("Curvature Residual ||C_t||_F", f"{rec.get('curvature_frob', 0.0):.4f}")
        st.metric("Relative Nonlinearity ||C_t|| / ||V_A||", f"{rec.get('curvature_relative', 0.0):.4f}")
        st.metric("Gold Answer Log-Prob (Delta vs S0)", f"{rec.get('gold_logprob', 0.0):.3f}", f"{rec.get('delta_gold_logprob', 0.0):+.3f}")
        st.metric("Recurrent Calls / Wall Time", f"{rec.get('recurrent_block_calls', 0)} calls", f"{rec.get('wall_time_ms', 0.0):.0f} ms")

        st.markdown(
            f"- **Parameters:** `alpha={rec.get('alpha')}`, `beta={rec.get('beta')}`, `visit_step={rec.get('visit_step')}`\n"
            f"- **Scope:** `{rec.get('token_scope')}` · **Trust Clipping:** `{rec.get('clipped')}`\n"
            f"- **Post-Step RMS:** `{rec.get('post_rms_a', 0.0):.4f}` · **Finite:** `{rec.get('post_finite')}`\n"
            f"- **Policy Action:** `{rec.get('policy_action')}` (`{rec.get('policy_reason')}`)"
        )

    st.divider()
    st.subheader("4. Run-Wide Rescue, Degradation, and Matched-Compute Comparison")
    methods_summary = summary.get("rescue_matrix", {}).get("methods", {})
    if methods_summary:
        table_rows = []
        for m_name, m_data in methods_summary.items():
            table_rows.append(
                {
                    "Method": m_name,
                    "Accuracy": round(m_data["accuracy"], 4),
                    "Paired Diff vs S0": round(m_data["paired_accuracy_diff"], 4),
                    "95% CI Low": round(m_data["ci_95_low"], 4),
                    "95% CI High": round(m_data["ci_95_high"], 4),
                    "Rescue Rate": f"{m_data['rescue_rate']:.3f} ({m_data['rescued_count']}/{m_data['base_wrong_count']})",
                    "Degradation Rate": f"{m_data['degradation_rate']:.3f} ({m_data['degraded_count']}/{m_data['base_correct_count']})",
                    "Oracle Rescue": f"{m_data['oracle_rescue_rate']:.3f} ({m_data['oracle_rescued_count']}/{m_data['oracle_eligible_count']})",
                    "Mean ||C_t||_F": round(m_data["mean_curvature_frob"], 4),
                    "Mean Calls": round(m_data["mean_recurrent_calls"], 2),
                }
            )
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True)

        pareto_pts = summary.get("pareto_frontier", {}).get("points", [])
        if pareto_pts:
            df_p = pd.DataFrame(pareto_pts)
            fig_par = px.scatter(
                df_p,
                x="mean_recurrent_calls",
                y="accuracy",
                color="is_oracle",
                text="method",
                title="Accuracy vs Recurrent Block Evaluations (Compute Frontier)",
                labels={
                    "mean_recurrent_calls": "Mean Recurrent Core Calls per Problem",
                    "accuracy": "Task Accuracy (Formal Validator)",
                },
            )
            fig_par.update_traces(textposition="top center", marker=dict(size=11))
            st.plotly_chart(fig_par, use_container_width=True)


if __name__ == "__main__":
    main()
