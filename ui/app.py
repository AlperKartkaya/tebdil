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


@st.cache_data(ttl="5m", max_entries=20)
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
        page_title="TEBDIL observatory",
        layout="wide",
    )
    st.title("TEBDIL observatory — Self-directed latent relocation in LoopLMs")
    st.caption(
        "Empirical trace inspector for recurrent hidden-state trajectories, learned meta-controller decisions "
        "(CONTINUE / EXCURSION / EXIT), excursion curvature C_t, donor selection, and matched-compute Pareto frontiers. "
        "All metrics are loaded directly from verified experiment artifacts in `results/`."
    )

    runs = load_available_runs()
    if not runs:
        st.warning(
            "No completed experiment runs found in `results/`. Run `python -m tebdil.cli run --config configs/phase1_smoke.yaml` first.",
            title="Missing artifacts",
        )
        return

    col_left, col_center, col_right = st.columns([1.1, 1.8, 1.3])

    with col_left:
        st.subheader("1. Problem and branch selector")
        selected_run = st.selectbox("Experiment run (`run_id`)", runs, index=len(runs) - 1)
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
        default_idx = (
            available_methods.index("E3_ridge")
            if "E3_ridge" in available_methods
            else (available_methods.index("E1") if "E1" in available_methods else 0)
        )
        selected_method = st.selectbox("Intervention / controller method", available_methods, index=default_idx)

        rec = instance_recs[selected_method]
        s0_rec = instance_recs.get("S0", rec)

        st.markdown("#### Prompt")
        st.code(rec.get("prompt", ""), language="text")
        st.markdown(
            f"- **Gold answer:** `{rec.get('gold_answer')}`\n"
            f"- **Baseline A (`S0`) output:** `{s0_rec.get('extracted_answer')}` "
            f"({'VALID' if s0_rec.get('is_correct') else 'WRONG'})\n"
            f"- **Donor B output:** `{rec.get('donor_b_answer', 'N/A')}` "
            f"({'VALID' if rec.get('donor_b_correct') else 'WRONG'})\n"
            f"- **Donor C output:** `{rec.get('donor_c_answer', 'N/A')}` "
            f"({'VALID' if rec.get('donor_c_correct') else 'WRONG'})\n"
            f"- **`{selected_method}` output:** `{rec.get('extracted_answer')}` "
            f"({'VALID' if rec.get('is_correct') else 'WRONG'})"
        )

        if not s0_rec.get("is_correct") and rec.get("is_correct"):
            st.success("A wrong -> A' correct", title="Outcome: RESCUED")
        elif s0_rec.get("is_correct") and not rec.get("is_correct"):
            st.error("A correct -> A' wrong", title="Outcome: DEGRADED")
        elif rec.get("is_correct"):
            st.info("A correct -> A' correct", title="Outcome: RETAINED CORRECT")
        else:
            st.warning("A wrong -> A' wrong", title="Outcome: UNRESCUED")

    with col_center:
        st.subheader("2. Latent trajectory and stepwise dynamics")
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
                title="Full-dimensional branch distance and intermediate entropy by step",
                xaxis_title="Recurrent Step t",
                yaxis_title="Frobenius Distance ||H_B - H_A||_F",
                yaxis2=dict(title="Entropy (nats)", overlaying="y", side="right"),
                height=300,
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig_div, width="stretch")

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
                title=f"2D PCA projection of last-token states (PC1: {ev1:.1f}%, PC2: {ev2:.1f}%)",
                xaxis_title=f"PC1 ({ev1:.1f}% var)",
                yaxis_title=f"PC2 ({ev2:.1f}% var)",
                height=320,
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig_pca, width="stretch")
            st.caption(
                f"**Methodological note:** 2D linear PCA captures {ev1 + ev2:.1f}% of variance across the "
                "plotted points. Visual proximity in 2D does NOT prove full-dimensional attractor basin membership; "
                "rely on exact Frobenius norms and curvature metrics in the right panel."
            )

    with col_right:
        st.subheader("3. Controller and excursion telemetry")
        with st.container(border=True):
            st.markdown("##### Self-directed controller decision")
            st.markdown(
                f"- **Selected action:** `{rec.get('controller_action_type', 'CONTINUE')}` (`{rec.get('policy_action')}`)\n"
                f"- **Decision reason:** `{rec.get('policy_reason')}`\n"
                f"- **Selected donor:** `{rec.get('selected_donor', 'B')}` · **Early exit step:** `{rec.get('early_exit_step')}`\n"
                f"- **Predicted advantage:** `{rec.get('controller_advantage', 0.0):+.4f}` · "
                f"**Overhead:** `{rec.get('controller_overhead_ms', 0.0):.2f} ms`"
            )
            action_scores = rec.get("controller_action_scores", {})
            if action_scores:
                df_scores = pd.DataFrame(
                    [
                        {"Action": k, "Predicted Advantage": round(float(v), 4)}
                        for k, v in sorted(action_scores.items(), key=lambda x: x[1], reverse=True)
                    ]
                )
                st.dataframe(df_scores, hide_index=True)

            state_feats = rec.get("controller_state_features", {})
            if state_feats:
                with st.expander("Observed label-free state features (z_t)"):
                    st.json(state_feats)

        st.metric("Curvature residual ||C_t||_F", f"{rec.get('curvature_frob', 0.0):.4f}")
        st.metric("Relative nonlinearity ||C_t|| / ||V_A||", f"{rec.get('curvature_relative', 0.0):.4f}")
        st.metric(
            "Gold answer log-prob (delta vs S0)",
            f"{rec.get('gold_logprob', 0.0):.3f}",
            f"{rec.get('delta_gold_logprob', 0.0):+.3f}",
        )
        st.metric(
            "Recurrent calls / wall time",
            f"{rec.get('recurrent_block_calls', 0)} calls",
            f"{rec.get('wall_time_ms', 0.0):.0f} ms",
        )

        st.markdown(
            f"- **Parameters:** `alpha={rec.get('alpha')}`, `beta={rec.get('beta')}`, `visit_step={rec.get('visit_step')}`\n"
            f"- **Scope:** `{rec.get('token_scope')}` · **Subspace:** `{rec.get('subspace_fraction', 1.0)}` · "
            f"**Gate modulated:** `{rec.get('gate_modulated', False)}`\n"
            f"- **Trust clipping:** `{rec.get('clipped')}` · **Post-step RMS:** `{rec.get('post_rms_a', 0.0):.4f}` · "
            f"**Finite:** `{rec.get('post_finite')}`"
        )

    st.divider()
    st.subheader("4. Run-wide rescue, degradation, and matched-compute comparison")
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
                    "Excursion Rate": round(m_data.get("excursion_rate", 0.0), 3),
                    "Early Exit Rate": round(m_data.get("early_exit_rate", 0.0), 3),
                    "Mean ||C_t||_F": round(m_data["mean_curvature_frob"], 4),
                    "Mean Calls": round(m_data["mean_recurrent_calls"], 2),
                }
            )
        st.dataframe(pd.DataFrame(table_rows), hide_index=True)

        pareto_pts = summary.get("pareto_frontier", {}).get("points", [])
        if pareto_pts:
            df_p = pd.DataFrame(pareto_pts)
            fig_par = px.scatter(
                df_p,
                x="mean_recurrent_calls",
                y="accuracy",
                color="is_oracle",
                text="method",
                title="Accuracy vs recurrent block evaluations (compute frontier)",
                labels={
                    "mean_recurrent_calls": "Mean Recurrent Core Calls per Problem",
                    "accuracy": "Task Accuracy (Formal Validator)",
                },
            )
            fig_par.update_traces(textposition="top center", marker=dict(size=11))
            st.plotly_chart(fig_par, width="stretch")


if __name__ == "__main__":
    main()
