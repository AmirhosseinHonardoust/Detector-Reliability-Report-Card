from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.io import read_csv, write_csv, write_json
from src.clean import clean_df
from src.split import make_splits, SplitConfig
from src.features import FeatureConfig
from src.models import ModelConfig, build_word_model, build_char_model
from src.metrics import compute_overall, coverage_curve
from src.reporting import plot_confusion, plot_reliability, plot_coverage, plot_confidence_hist

def _encode_labels(y: pd.Series):
    labels = sorted(y.unique().tolist())
    mapping = {lab: i for i, lab in enumerate(labels)}
    inv = {i: lab for lab, i in mapping.items()}
    return y.map(mapping).to_numpy(), labels, mapping, inv

def run(
    input_path: str,
    out_dir: str = "outputs",
    figures_dir: str = "reports/figures",
    random_state: int = 42,
    calibration_method: str = "sigmoid",
    recommend_target_coverage: float = 0.7,
) -> dict:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = Path(figures_dir); fig_dir.mkdir(parents=True, exist_ok=True)

    df = clean_df(read_csv(input_path))
    splits = make_splits(df, SplitConfig(random_state=random_state))
    train, val, test = splits["train"], splits["val"], splits["test"]

    y_train, labels, mapping, inv = _encode_labels(train["label"])
    y_val = val["label"].map(mapping).to_numpy()
    y_test = test["label"].map(mapping).to_numpy()

    fcfg = FeatureConfig()
    mcfg = ModelConfig(calibrate=True, calibration_method=calibration_method)

    word_model = build_word_model(fcfg, mcfg)
    char_model = build_char_model(fcfg, mcfg)

    word_model.fit(train["text"], y_train)
    char_model.fit(train["text"], y_train)

    w_val_proba = word_model.predict_proba(val["text"]); w_val_pred = w_val_proba.argmax(axis=1)
    c_val_proba = char_model.predict_proba(val["text"]); c_val_pred = c_val_proba.argmax(axis=1)

    w_f1 = float(f1_score(y_val, w_val_pred, average="macro"))
    c_f1 = float(f1_score(y_val, c_val_pred, average="macro"))

    primary = "word" if w_f1 >= c_f1 else "char"
    primary_model = word_model if primary == "word" else char_model
    other_model = char_model if primary == "word" else word_model

    proba = primary_model.predict_proba(test["text"])
    pred = proba.argmax(axis=1)
    conf = proba.max(axis=1)

    other_pred = other_model.predict_proba(test["text"]).argmax(axis=1)
    disagree = (pred != other_pred)

    overall = compute_overall(y_test, pred, proba, labels)
    overall.update({
        "primary_model": primary,
        "val_macro_f1_word": w_f1,
        "val_macro_f1_char": c_f1,
    })

    thresholds = np.linspace(0.0, 0.99, 40)
    curve = coverage_curve(y_test, proba, thresholds)

    cand = curve[curve["coverage"] >= recommend_target_coverage].dropna()
    if len(cand) == 0:
        rec = curve.dropna().iloc[-1]
    else:
        rec = cand.sort_values(["accuracy", "coverage"], ascending=[False, False]).iloc[0]

    policy = {
        "recommended_threshold": float(rec["threshold"]),
        "target_coverage": float(recommend_target_coverage),
        "estimated_coverage": float(rec["coverage"]),
        "estimated_accuracy": float(rec["accuracy"]),
        "estimated_macro_f1": float(rec["macro_f1"]),
        "abstain_rule": "abstain if max_proba < threshold OR (disagree_across_models and max_proba < threshold+0.05)",
    }

    # Save
    proba_df = pd.DataFrame(proba, columns=[f"p_{lab}" for lab in labels])
    out_pred = pd.concat([test.reset_index(drop=True)[["text","label"]], proba_df], axis=1)
    out_pred["pred_label"] = [inv[i] for i in pred]
    out_pred["confidence"] = conf
    out_pred["disagree_word_char"] = disagree.astype(int)
    write_csv(out_pred, out_dir/"test_predictions.csv")
    write_csv(curve, out_dir/"coverage_curve.csv")

    split_summary = {
        "n_total": int(len(df)),
        "n_train": int(len(train)),
        "n_val": int(len(val)),
        "n_test": int(len(test)),
        "label_counts_total": df["label"].value_counts().to_dict(),
        "label_counts_train": train["label"].value_counts().to_dict(),
        "label_counts_val": val["label"].value_counts().to_dict(),
        "label_counts_test": test["label"].value_counts().to_dict(),
        "labels": labels,
    }
    write_json(split_summary, out_dir/"splits_summary.json")
    write_json(overall, out_dir/"metrics_overall.json")
    write_json(policy, out_dir/"abstention_policy.json")

    plot_confusion(np.array(overall["confusion_matrix"]), labels, fig_dir/"confusion_matrix.png")
    plot_reliability(y_test, proba, fig_dir/"reliability_diagram.png")
    plot_coverage(curve, fig_dir/"coverage_vs_accuracy.png")
    plot_confidence_hist(proba, fig_dir/"probability_histograms.png")

    return {"out_dir": str(out_dir), "figures_dir": str(fig_dir), "policy": policy, "primary_model": primary, "labels": labels}

def main() -> None:
    parser = argparse.ArgumentParser(description="Detector Reliability Report Card pipeline")
    parser.add_argument("--input", required=True, help="Path to CSV")
    parser.add_argument("--out", default="outputs", help="Output directory")
    parser.add_argument("--figures", default="reports/figures", help="Figures directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--calibration", default="sigmoid", choices=["sigmoid","isotonic"], help="Calibration method")
    parser.add_argument("--target-coverage", type=float, default=0.7, help="Target coverage for recommended threshold")
    args = parser.parse_args()

    res = run(
        input_path=args.input,
        out_dir=args.out,
        figures_dir=args.figures,
        random_state=args.seed,
        calibration_method=args.calibration,
        recommend_target_coverage=args.target_coverage,
    )

    print("\nDone! Reliability report card created.", flush=True)
    print(f"Outputs: {res['out_dir']}", flush=True)
    print(f"Figures: {res['figures_dir']}", flush=True)
    print(f"Primary model: {res['primary_model']}", flush=True)
    print(f"Recommended threshold: {res['policy']['recommended_threshold']:.2f} (coverage≈{res['policy']['estimated_coverage']:.2f})\n", flush=True)

if __name__ == "__main__":
    main()
