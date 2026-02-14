import sys
import json
import base64
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import run as run_pipeline  # noqa: E402


def _st_image_fixed(path: Path, caption: str, height_px: int = 340) -> None:
    """Render an image in a fixed-height container so a 2×2 grid stays aligned."""
    if not path.exists():
        st.warning(f"Missing figure: {path.name}")
        return

    b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
    st.markdown(
        f"""
        <div style="border:1px solid rgba(49,51,63,0.15); border-radius:12px; padding:10px;">
          <img src="data:image/png;base64,{b64}"
               style="width:100%; height:{height_px}px; object-fit:contain; display:block;" />
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(caption)


st.set_page_config(page_title="Detector Reliability Report Card", layout="wide")
st.title("Detector Reliability Report Card")
st.caption("Calibration + abstention + decision-safe UI for human vs AI vs post-edited AI detection.")

DEFAULT_INPUT = PROJECT_ROOT / "data" / "raw" / "ai_human_detection.csv"
OUT_DIR = PROJECT_ROOT / "outputs"
FIG_DIR = PROJECT_ROOT / "reports" / "figures"

with st.sidebar:
    st.header("Pipeline")
    uploaded = st.file_uploader("Upload CSV (optional)", type=["csv"])
    input_path = st.text_input("Or CSV path", value=str(DEFAULT_INPUT))
    target_cov = st.slider("Target auto-decision coverage", 0.1, 0.95, 0.70, 0.05)
    calibration = st.selectbox("Calibration method", ["sigmoid", "isotonic"], index=0)
    run_btn = st.button("Run / Refresh")

effective_input = Path(input_path)
if uploaded is not None:
    tmp = PROJECT_ROOT / "data" / "raw" / "uploaded.csv"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_bytes(uploaded.getbuffer())
    effective_input = tmp

if run_btn:
    with st.spinner("Training + evaluating..."):
        run_pipeline(
            input_path=str(effective_input),
            out_dir=str(OUT_DIR),
            figures_dir=str(FIG_DIR),
            calibration_method=str(calibration),
            recommend_target_coverage=float(target_cov),
        )
    st.success("Done! Outputs regenerated.")

metrics_path = OUT_DIR / "metrics_overall.json"
policy_path = OUT_DIR / "abstention_policy.json"
preds_path = OUT_DIR / "test_predictions.csv"
curve_path = OUT_DIR / "coverage_curve.csv"

if not metrics_path.exists():
    st.info("Run the pipeline from the sidebar to generate the report card.")
    st.stop()

metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
policy = json.loads(policy_path.read_text(encoding="utf-8")) if policy_path.exists() else {}
preds = pd.read_csv(preds_path) if preds_path.exists() else pd.DataFrame()
curve = pd.read_csv(curve_path) if curve_path.exists() else pd.DataFrame()

tab_report, tab_curve, tab_triage, tab_notes = st.tabs(
    ["Report Card", "Coverage Curve", "Triage UI", "Notes"]
)

with tab_report:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy (test)", f'{metrics["accuracy"]:.3f}')
    c2.metric("Macro F1 (test)", f'{metrics["macro_f1"]:.3f}')
    c3.metric("ECE (lower better)", f'{metrics["ece"]:.3f}')
    c4.metric("Brier (lower better)", f'{metrics["brier"]:.3f}')

    st.subheader("Figures")

    # Row 1
    r1 = st.columns(2, gap="large")
    with r1[0]:
        _st_image_fixed(FIG_DIR / "confusion_matrix.png", "Confusion matrix", height_px=340)
    with r1[1]:
        _st_image_fixed(FIG_DIR / "coverage_vs_accuracy.png", "Coverage vs performance", height_px=340)

    # Row 2
    r2 = st.columns(2, gap="large")
    with r2[0]:
        _st_image_fixed(FIG_DIR / "reliability_diagram.png", "Reliability diagram", height_px=340)
    with r2[1]:
        _st_image_fixed(FIG_DIR / "probability_histograms.png", "Confidence histogram", height_px=340)

    if policy:
        st.subheader("Recommended abstention policy")
        st.json(policy)

with tab_curve:
    st.subheader("Coverage vs Accuracy / Macro F1")
    if not curve.empty:
        fig = px.line(curve, x="coverage", y=["accuracy", "macro_f1"], markers=True)
        st.plotly_chart(fig, width="stretch")

        fig2 = px.line(curve, x="threshold", y=["coverage", "accuracy", "macro_f1"], markers=True)
        st.plotly_chart(fig2, width="stretch")
    else:
        st.info("Coverage curve not found.")

with tab_triage:
    st.subheader("Paste text → see decision-safe output format")
    labels = metrics.get("labels", [])
    text = st.text_area("Text", height=180, placeholder="Paste or type text here...")

    st.caption(
        "Note: this demo uses the saved test prediction table to show the output format. "
        "For real inference, persist the trained model (joblib) and load it here."
    )
    if preds.empty:
        st.info("Predictions table not found. Run pipeline.")
    else:
        if text.strip():
            preds["len"] = preds["text"].astype(str).str.len()
            qlen = len(text.strip())
            row = preds.iloc[(preds["len"] - qlen).abs().argsort()[:1]].iloc[0]
            probs = {lab: float(row[f"p_{lab}"]) for lab in labels}
            pred_label = str(row["pred_label"])
            conf = float(row["confidence"])
            disagree = int(row.get("disagree_word_char", 0))

            thr = float(policy.get("recommended_threshold", 0.7))
            abstain = (conf < thr) or (disagree == 1 and conf < min(0.99, thr + 0.05))

            col1, col2, col3 = st.columns(3)
            col1.metric("Predicted label", pred_label)
            col2.metric("Confidence", f"{conf:.3f}")
            col3.metric("Decision", "ABSTAIN → review" if abstain else "AUTO-DECIDE")

            st.plotly_chart(
                px.bar(
                    x=list(probs.keys()),
                    y=list(probs.values()),
                    labels={"x": "class", "y": "probability"},
                ),
                width="stretch",
            )
            st.caption(
                f"Rule: abstain if confidence < {thr:.2f} OR (model disagreement and confidence < {min(0.99, thr+0.05):.2f})."
            )
        else:
            st.info("Paste some text to see the output format.")

with tab_notes:
    st.markdown(
        """
### Decision safety notes
- **Accuracy ≠ trust.** A model can be accurate but overconfident.
- **ECE** measures calibration error on confidence bins (lower is better).
- **Coverage** is a real product metric: if you abstain too much, you lose usability.

### Make it production-grade
- Persist the trained model (joblib) and run true inference in the UI.
- Add slice audits: performance by language/domain/edit_level.
- Add drift monitoring: score distribution shifts over time.
        """.strip()
    )
