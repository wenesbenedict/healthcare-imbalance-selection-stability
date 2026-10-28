"""Streamlit front-end for the synthetic selection-stability demonstrator."""
from __future__ import annotations

import streamlit as st

from demo_core import DemoConfig, run_demo

st.set_page_config(page_title="Selection Stability Audit", page_icon="🧪", layout="wide")
st.title("Selection Stability Audit")
st.caption("Synthetic-data prototype for the proposed conference-paper methodology")
st.warning(
    "SYNTHETIC DEMONSTRATION ONLY. The displayed metrics illustrate the workflow; "
    "they are not clinical evidence and must not be reported as results from healthcare datasets."
)

with st.sidebar:
    st.header("Experiment settings")
    training_size = st.select_slider("Training-set size", options=[100, 250, 500], value=250)
    imbalance_ratio = st.select_slider("Majority : minority ratio", options=[2, 5, 10], value=5)
    noise_percent = st.select_slider("Training-label noise", options=[0, 5, 10], value=5, format_func=lambda x: f"{x}%")
    repeats = st.slider("Repeated training samples", min_value=5, max_value=50, value=20, step=5)
    seed = st.number_input("Random seed", min_value=0, max_value=1_000_000, value=42, step=1)
    run_button = st.button("Run synthetic demonstration", type="primary", use_container_width=True)

if run_button:
    cfg = DemoConfig(
        training_size=int(training_size),
        imbalance_ratio=int(imbalance_ratio),
        noise_percent=int(noise_percent),
        repeats=int(repeats),
        seed=int(seed),
    )
    with st.spinner("Running repeated inner-CV selection and clean-test evaluation..."):
        try:
            st.session_state["result"] = run_demo(cfg)
        except Exception as exc:
            st.error(f"Demonstration failed: {exc}")
            st.stop()

result = st.session_state.get("result")
if result is None:
    st.info("Choose settings in the sidebar and click **Run synthetic demonstration**.")
    st.markdown(
        "### What this prototype demonstrates\n"
        "1. Draw repeated small training samples at a chosen imbalance ratio.\n"
        "2. Corrupt training labels at a controllable level while keeping the test set clean.\n"
        "3. Select among no handling, class weighting, and random oversampling using inner cross-validation.\n"
        "4. Report how often the strategy selector agrees with itself across resamples.\n"
        "5. Compare the selected-per-resample policy with a fixed reference chosen using inner-CV scores only."
    )
else:
    cfg = result["config"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Selection agreement", f"{result['agreement']:.1%}")
    c2.metric("Normalised entropy", f"{result['normalized_entropy']:.3f}")
    c3.metric("Fixed reference", result["fixed_strategy"])
    c4.metric("Fixed-reference AP gap", f"{result['fixed_reference_gap']:+.3f}")

    st.markdown(
        f"**Condition:** n = {cfg.training_size}; target ratio 1:{cfg.imbalance_ratio} "
        f"(majority:minority); simulated training-label noise = {cfg.noise_percent}%; "
        f"repeats = {cfg.repeats}; approximate minority count = {result['minority_count']}."
    )
    st.caption(result["noise_definition"])

    left, right = st.columns(2)
    with left:
        st.subheader("Strategy selection frequency")
        frequency_df = result["summary"][["Strategy", "Selection frequency"]].set_index("Strategy")
        st.bar_chart(frequency_df)
    with right:
        st.subheader("Mean score across repeated training samples")
        score_df = result["summary"].set_index("Strategy")[["Mean inner-CV AP", "Mean clean-test AP"]]
        st.bar_chart(score_df)

    st.subheader("Strategy-level summary")
    st.dataframe(result["summary"].style.format({
        "Selection frequency": "{:.1%}",
        "Mean inner-CV AP": "{:.3f}",
        "Mean clean-test AP": "{:.3f}",
        "Mean clean-test recall @ 0.5": "{:.3f}",
        "Mean clean-test precision @ 0.5": "{:.3f}",
    }), use_container_width=True, hide_index=True)

    st.subheader("Repeat-level selections")
    st.dataframe(result["runs"], use_container_width=True, hide_index=True)
    st.download_button(
        "Download strategy summary (CSV)",
        result["summary"].to_csv(index=False).encode("utf-8"),
        file_name="synthetic_demo_summary.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download repeat-level data (CSV)",
        result["runs"].to_csv(index=False).encode("utf-8"),
        file_name="synthetic_demo_runs.csv",
        mime="text/csv",
    )
    st.info(
        "Interpretation: high agreement means the selector repeatedly chooses the same strategy; "
        "it does not prove that strategy is clinically optimal. A positive fixed-reference gap means "
        "the selected-per-resample policy scored below the fixed reference on this synthetic test set."
    )
