<div align="center">

# Detector Reliability Report Card

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Calibration-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![Task](https://img.shields.io/badge/Task-AI%20vs%20Human%20Detection-purple)
![Focus](https://img.shields.io/badge/Focus-Calibration%20%26%20Reliability-6f42c1)
![Metrics](https://img.shields.io/badge/Metrics-ECE%20%7C%20Brier%20%7C%20MacroF1-yellow)
![License](https://img.shields.io/badge/License-MIT-informational)
[![CI](https://github.com/AmirhosseinHonardoust/Detector-Reliability-Report-Card/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/AmirhosseinHonardoust/Detector-Reliability-Report-Card/actions/workflows/ci.yml)

</div>

A decision-safe workflow for **AI vs human vs post-edited AI** text detection that goes beyond accuracy with **probability calibration**, a learned **abstention threshold**, **coverage analysis**, **reliability figures**, and a **Streamlit review dashboard**.

> **Important:** This project is a **portfolio and research demo**, not a production content-detection system.
>
> The models are simple TF-IDF baselines on a small example dataset. The metrics, thresholds, and abstention policy are designed to demonstrate a professional reliability workflow, not to make real moderation or integrity decisions without expert validation, monitoring, and review.

---

## Table of Contents

- [Project Overview](#project-overview)
- [What This Project Does](#what-this-project-does)
- [What This Project Does Not Do](#what-this-project-does-not-do)
- [Key Features](#key-features)
- [System Workflow](#system-workflow)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Data Format](#data-format)
- [How the Pipeline Works](#how-the-pipeline-works)
- [Metrics Explained](#metrics-explained)
- [Abstention and Coverage](#abstention-and-coverage)
- [Visual Reports](#visual-reports)
- [Streamlit Dashboard](#streamlit-dashboard)
- [Recommended Threshold](#recommended-threshold)
- [Decision Safety](#decision-safety)
- [Testing and CI](#testing-and-ci)
- [Code Quality](#code-quality)
- [Limitations](#limitations)
- [Responsible Use](#responsible-use)
- [Future Improvements](#future-improvements)
- [Tech Stack](#tech-stack)
- [Troubleshooting](#troubleshooting)
- [Author](#author)
- [License](#license)

---

## Project Overview

Most detector projects stop at accuracy. Real detection workflows need more, because a wrong decision is costly:

- flagging humans as AI (false positives) harms trust and policy enforcement
- missing AI (false negatives) weakens moderation and integrity workflows
- post-edited AI is especially tricky and can resemble both classes

Accuracy alone does not tell you whether to trust the model on any single decision. This project produces a **Reliability Report Card** (calibration + performance), learns a **recommended abstention threshold** that targets a chosen **auto-decision coverage**, and ships a **Streamlit dashboard** that presents everything in a decision-ready format.

The goal is to show how a detector can be turned into a **decision-support system**, not just a metric in a notebook.

---

## What This Project Does

This project can:

- Load and normalize a text/label dataset with automatic column detection
- Build word-level and character-level TF-IDF baseline models
- Calibrate predicted probabilities (sigmoid or isotonic)
- Select the primary model using validation macro-F1
- Evaluate accuracy, macro-F1, ECE, and Brier score on held-out test data
- Sweep confidence thresholds to build a coverage-vs-performance curve
- Recommend an abstention threshold for a target auto-decision coverage
- Save machine-readable metrics, policy, curves, and predictions
- Generate reliability figures (confusion matrix, reliability diagram, coverage, confidence)
- Provide a Streamlit dashboard for report-card review and triage
- Run unit tests and a GitHub Actions CI quality gate

---

## What This Project Does Not Do

This project does **not**:

- Detect AI-generated text in production
- Use a large, representative, or adversarial dataset
- Guarantee detection decisions are fair, robust, or deployable
- Replace human reviewers or policy teams
- Provide real-time or streaming inference
- Include drift monitoring or automatic retraining
- Persist a trained model for live single-text inference (the dashboard triage tab demonstrates UI format using saved test predictions)

A production detector would need stronger models, governance, live monitoring, adversarial testing, and human escalation workflows.

---

## Key Features

- **Automatic column detection** for text and label columns, robust across pandas 2 and 3
- **Word + character TF-IDF baselines** that are fast and explainable
- **Probability calibration** (sigmoid / isotonic) for trustworthy confidence
- **Validation-based model selection** by macro-F1
- **Calibration metrics** (ECE, Brier) alongside accuracy and macro-F1
- **Coverage curve** sweeping thresholds for the coverage-vs-performance tradeoff
- **Abstention policy artifact** with a recommended threshold for target coverage
- **Machine-readable artifacts** (JSON/CSV) for analysis, audits, and the dashboard
- **Reliability figures** for the report card
- **Streamlit dashboard** for review and triage
- **Unit tests and GitHub Actions CI** (ruff, black, mypy, pytest on Python 3.10–3.12)

---

## System Workflow

```text
Text / label dataset (CSV)
        ↓
Column detection + cleaning
        ↓
Stratified train/val/test split
        ↓
Word + char TF-IDF model pipelines
        ↓
Probability calibration (sigmoid / isotonic)
        ↓
Validation macro-F1 → primary model selection
        ↓
Test metrics (accuracy, macro-F1, ECE, Brier)
        ↓
Threshold sweep → coverage curve → recommended abstention policy
        ↓
Artifacts + figures → Streamlit report card & triage
```

---

## Project Structure

```text
Detector-Reliability-Report-Card/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── app/
│   └── app.py
│
├── data/
│   └── raw/
│       └── ai_human_detection.csv
│
├── outputs/
│   ├── metrics_overall.json
│   ├── abstention_policy.json
│   ├── coverage_curve.csv
│   └── test_predictions.csv
│
├── reports/
│   └── figures/
│       ├── confusion_matrix.png
│       ├── reliability_diagram.png
│       ├── coverage_vs_accuracy.png
│       └── probability_histograms.png
│
├── src/
│   ├── __init__.py
│   ├── pipeline.py
│   ├── io.py
│   ├── clean.py
│   ├── split.py
│   ├── features.py
│   ├── models.py
│   ├── metrics.py
│   └── reporting.py
│
├── tests/
│   ├── test_clean.py
│   ├── test_metrics.py
│   ├── test_split.py
│   └── test_pipeline_smoke.py
│
├── pyproject.toml
├── requirements.txt
├── .gitignore
├── README.md
└── LICENSE
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/AmirhosseinHonardoust/Detector-Reliability-Report-Card.git
cd Detector-Reliability-Report-Card
```

### 2. Create a Virtual Environment

On Windows CMD:

```cmd
python -m venv .venv
.venv\Scripts\activate
```

On macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

---

## Quick Start

Run the pipeline to generate report artifacts:

```bash
python -m src.pipeline --input data/raw/ai_human_detection.csv
```

Expected outcome:

- `outputs/` populated with JSON/CSV artifacts
- `reports/figures/` populated with PNG plots
- terminal prints a recommended threshold + estimated coverage (example: threshold ≈ 0.61, coverage ≈ 0.71)

Launch the dashboard:

```bash
streamlit run app/app.py
```

> **Reproducibility note:** runs are deterministic for a fixed random seed and dependency set, but calibration internals can shift slightly between scikit-learn versions, so ECE/Brier and the committed `outputs/*` may differ marginally across environments. Pin exact versions for byte-identical artifacts.

---

## Data Format

At minimum the pipeline needs:

- a **text column** (the content)
- a **label column** (ground-truth class)

Typical labels used by this project:

- `human`
- `ai`
- `post_edited_ai`

Column names are detected automatically: `text` / `content` / `sentence` for text and `label` / `class` / `human_or_ai` / `target` for labels, with a length/cardinality fallback when names differ. Post-edited AI behaves like an in-between distribution, confusing with `ai` when edits are light and with `human` when edits are heavy, which is why macro-F1 and confusion analysis are emphasized.

---

## How the Pipeline Works

<div align="center">
        
| Step | What happens |
|---|---|
| A. Clean | Detect text/label columns, normalize labels, drop empty text |
| B. Split | Stratified train/val/test split preserving class balance |
| C. Model | Train word-level and character-level TF-IDF baselines |
| D. Calibrate | Reshape probabilities so "0.8" behaves like ~80% correct |
| E. Select | Pick the primary model by validation macro-F1 |
| F. Evaluate | Accuracy, macro-F1, ECE, and Brier on held-out test data |
| G. Sweep | Evaluate many thresholds to build the coverage curve |
| H. Save | Write metrics, policy, curves, predictions, and figures |
</div>

Baselines are used because they are quick to train, easy to debug, and a strong reference point before heavier models. Calibration matters because raw scores are often miscalibrated, and thresholding on miscalibrated confidence is dangerous.

---

## Metrics Explained

<div align="center">

| Metric | What it measures | Why it matters here |
|---|---|---|
| Accuracy | Overall correctness rate | Can hide class imbalance and which mistakes matter |
| Macro-F1 | F1 per class, averaged equally | Prevents an "it's accurate" conclusion driven by the easy class |
| ECE | Gap between confidence and empirical accuracy | High ECE means the confidence numbers are misleading |
| Brier | Squared error of predicted probabilities | Rewards both correctness and probability sharpness |
</div>

For ECE and Brier, **lower is better**. A model can be accurate but miscalibrated (dangerous thresholds) or moderately accurate but well-calibrated (safer abstention behavior); decision-safe deployment needs both.

Example results from the included example dataset:

<div align="center">

| Metric | Example value |
|---|---|
| Accuracy | 0.739 |
| Macro-F1 | 0.617 |
| ECE | 0.068 |
| Brier | 0.349 |
| Recommended threshold | 0.66 |
| Estimated coverage | 0.725 |
| Estimated accuracy at threshold | 0.820 |
</div>

> These values come from a small example dataset and should not be read as real-world detection performance. They are reproducible with the default seed but may shift slightly across scikit-learn/NumPy versions.

---

## Abstention and Coverage

**Coverage** is the fraction of cases the model auto-decides. A higher threshold keeps only confident predictions (coverage drops); a lower threshold decides more often (coverage rises). Performance is evaluated on the **decided subset**, which usually improves as the threshold rises because only easy, high-confidence examples remain.

The real deployment tradeoff:

- High coverage → less review cost, more wrong auto-decisions
- Low coverage → safer auto-decisions, higher review burden

This project makes that tradeoff measurable and explicit, and recommends a threshold for your chosen target coverage. The abstain rule is: abstain if `max_proba < threshold`, or if the word and char models disagree and `max_proba < threshold + 0.05`.

---

## Visual Reports

### Calibration and performance

<div align="center">

| Confusion Matrix | Reliability Diagram |
|---|---|
| <img width="460" alt="confusion_matrix" src="https://github.com/user-attachments/assets/21d20d07-fc52-474a-b70c-540d62a75f7c" /> | <img width="460" alt="reliability_diagram" src="https://github.com/user-attachments/assets/39a1a125-ff0f-479a-babc-470978914769" /> |
| **Analysis:** Rows are true labels, columns predicted. The diagonal is correct; off-diagonal cells reveal where post-edited AI is confused with `ai` (light edits) or `human` (heavy edits). | **Analysis:** Points below the diagonal indicate overconfidence (high risk for thresholding); points above indicate the model is safer than it claims. |
</div>

### Coverage and confidence

<div align="center">

| Coverage vs Performance | Confidence Histogram |
|---|---|
| <img width="460" alt="coverage_vs_accuracy" src="https://github.com/user-attachments/assets/734ee3d9-41ec-4cb4-8656-28b13d457afb" /> | <img width="460" alt="probability_histograms" src="https://github.com/user-attachments/assets/f46b2ead-9d26-4ec1-aba1-0c9ed0354316" /> |
| **Analysis:** Pick a target coverage your workflow can handle and read off the expected accuracy/macro-F1, or fix a minimum macro-F1 and read the coverage you must accept. | **Analysis:** The distribution of max probability per sample shows how often the model is uncertain and whether a threshold will sharply change coverage. |
</div>

---

## Streamlit Dashboard

Launch the app:

```bash
streamlit run app/app.py
```

The dashboard turns offline artifacts into a clean decision interface across four tabs.

### Report Card

<div align="center">

<img width="900" alt="Report Card tab" src="https://github.com/user-attachments/assets/79a8bfa3-4560-427a-996f-f78984fd3112" />
</div>

One screen that answers "should we trust this model?": top-line metrics (accuracy, macro-F1, ECE, Brier) and a 2×2 grid of the figures, plus the recommended abstention policy.

### Coverage Curve

<div align="center">

<img width="900" alt="Coverage Curve tab" src="https://github.com/user-attachments/assets/cab423ea-3f01-4d29-b258-8463f82fe542" />
</div>

Explore threshold tradeoffs interactively: what threshold gives ~70% auto-decisions, how much performance is lost as coverage rises, and where diminishing returns begin.

### Triage UI

<div align="center">

<img width="900" alt="Triage UI tab" src="https://github.com/user-attachments/assets/e7bb5eeb-76fa-4dcb-b795-da2bad587447" />
</div>

Shows what a decision-safe output looks like: predicted class, confidence, auto-decide vs abstain, and a probability breakdown. This tab uses saved test predictions to demonstrate UI format; real single-text inference would require persisting and loading the trained model.

### Notes

<div align="center">

<img width="900" alt="Notes tab" src="https://github.com/user-attachments/assets/00e54a7a-0bf2-4820-9e7c-a3695088514d" />
</div>

Documents the policy philosophy and upgrade path: accuracy is not trust, calibration and ECE, coverage as a product metric, and recommended next steps.

---

## Recommended Threshold

The pipeline prints a recommended threshold for your target coverage. For example, "threshold = 0.61, coverage ≈ 0.71" means that auto-deciding when confidence ≥ 0.61 auto-decides about 71% of cases on similar data, sending the remaining ~29% to review.

Coverage estimates are only valid if future data resembles the evaluation data, calibration remains stable, and the class mix does not drift heavily. That is why drift monitoring is part of the production-grade upgrade path.

---

## Decision Safety

This project avoids common failure patterns:

- **"The model is 80% accurate so we trust it."** Not safe; accuracy can mask overconfidence and minority-class failures.
- **"Set threshold to 0.9 and ship."** Not safe unless calibration supports it; an overconfident 0.9 is not truly 90% reliable.
- **"We'll just review random samples."** Not efficient; abstention focuses review on uncertain cases where humans add the most value.

---

## Testing and CI

Run the quality gate locally (config lives in `pyproject.toml`):

```bash
pip install ruff black mypy pytest
ruff check src app tests
black --check src app tests
mypy src app tests
pytest -q
```

The GitHub Actions workflow runs the same gate on Python 3.10, 3.11, and 3.12 for every push and pull request, covering:

- linting (ruff)
- formatting (black)
- type checking (mypy)
- unit tests and an end-to-end smoke test (pytest)

CI is defined in:

```text
.github/workflows/ci.yml
```

---

## Code Quality

The project separates responsibilities across small, single-purpose modules:

<div align="center">

| Module | Purpose |
|---|---|
| `src/pipeline.py` | Orchestration: train → evaluate → save artifacts/plots |
| `src/io.py` | CSV/JSON read and write helpers |
| `src/clean.py` | Column detection + text/label normalization |
| `src/split.py` | Stratified train/val/test split |
| `src/features.py` | Word/char TF-IDF vectorizer configs |
| `src/models.py` | Baseline + calibrated model builders |
| `src/metrics.py` | Accuracy, macro-F1, ECE, Brier, coverage curve |
| `src/reporting.py` | Figure generation |
| `app/app.py` | Streamlit dashboard |
</div>

---

## Limitations

This project has important limitations:

- The dataset is small and illustrative, not representative
- Results do not prove real-world AI-text detection performance
- Models are TF-IDF baselines, not state-of-the-art detectors
- The triage tab is a demo and does not run live single-text inference
- No drift monitoring, streaming inference, or retraining is included
- No fairness, robustness, or adversarial review is included
- The abstention policy is an example, not an approved business rule

The project is strongest as a portfolio demonstration of reliability-workflow design.

---

## Responsible Use

This repository is intended for:

- learning calibration and abstention workflows
- demonstrating coverage-vs-performance tradeoffs
- practicing model evaluation and reporting
- portfolio demonstration

It should not be used as-is for:

- real content-moderation or integrity decisions
- penalizing users or accounts
- academic-integrity enforcement
- any high-stakes automated decision

Any real deployment would require stronger models, expert review, monitoring, and governance.

---

## Future Improvements

- Persist the trained model and add live single-text inference in the dashboard
- Add slice audits (language, domain, length, post-edit intensity)
- Add drift monitoring (confidence, class mix, calibration over time)
- Add cost-aware abstention (minimize error cost + review cost)
- Add stronger models and richer features
- Add a model card and data statement

---

## Tech Stack

- Python
- pandas
- NumPy
- scikit-learn
- matplotlib
- Streamlit
- Plotly
- pytest
- ruff / black / mypy
- GitHub Actions

---

## Troubleshooting

**Streamlit `use_container_width` deprecation.** Use `width="stretch"` in `st.image()` and `st.plotly_chart()`. This repo's layout is designed around `width="stretch"` so figures align cleanly.

**"Run pipeline first."** If the dashboard reports missing outputs, click **Run / Refresh** in the sidebar or run the pipeline from the command line.

**Outputs not updating.** Ensure `out_dir` and `figures_dir` are correct and the app points to the same project root.

---

## Author

**Amir Honardoust**

GitHub: [@AmirhosseinHonardoust](https://github.com/AmirhosseinHonardoust)

---

## License

This project is intended for educational and portfolio purposes.

If you use or modify this project, please keep the responsible-use notes and limitations clear.
