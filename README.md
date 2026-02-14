<p align="center">
  <h1 align="center">Detector Reliability Report Card </h1>
    <p align="center">
<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)
![Task](https://img.shields.io/badge/Task-AI%20vs%20Human%20Detection-purple)
![Reliability](https://img.shields.io/badge/Focus-Calibration%20%26%20Reliability-6f42c1)
![Abstention](https://img.shields.io/badge/Feature-Abstention%20Policy-orange)
![Metrics](https://img.shields.io/badge/Metrics-ECE%20%7C%20Brier%20%7C%20MacroF1-yellow)
![License](https://img.shields.io/badge/License-MIT-informational)

</div>

### Calibration • Abstention • Decision-safe UI for AI vs Human vs Post-Edited AI detection

Most detector projects stop at **accuracy**. This repo goes one step further and answers the question you actually need in a workflow:

> **When the model says “I’m confident”, should we trust it and what should we do when it’s not?**

This project produces a **Reliability Report Card** (calibration + performance), learns a **recommended abstention threshold** to hit a target **auto-decision coverage**, and ships a **Streamlit dashboard** that presents everything in a clean, decision-ready format.

---

## Table of contents
- [What problem this solves](#what-problem-this-solves)
- [Key ideas](#key-ideas)
- [What the pipeline produces](#what-the-pipeline-produces)
- [Repository structure](#repository-structure)
- [Quickstart](#quickstart)
- [Data format](#data-format)
- [How the pipeline works](#how-the-pipeline-works)
- [Metrics explained](#metrics-explained)
- [Abstention and coverage explained](#abstention-and-coverage-explained)
- [Figures explained (each one, deeply)](#figures-explained-each-one-deeply)
- [Streamlit dashboard explained (tab-by-tab)](#streamlit-dashboard-explained-tab-by-tab)
- [Recommended threshold: what it means](#recommended-threshold-what-it-means)
- [Decision safety: what this UI prevents](#decision-safety-what-this-ui-prevents)
- [How to make it production-grade](#how-to-make-it-production-grade)
- [Troubleshooting](#troubleshooting)

---

## What problem this solves

Detectors are often deployed in environments where a wrong decision is costly:
- flagging humans as AI (false positives) can harm trust or policy enforcement
- missing AI (false negatives) may weaken moderation or integrity workflows
- post-edited AI is especially tricky: it can resemble both classes

**Accuracy alone doesn’t tell you whether to trust the model** on any specific decision.

This repo introduces two “real-world” requirements:

1) **Calibration**
   - If the model outputs “0.80 confidence,” it should be correct about ~80% of the time (on similar data).
   - If it’s **overconfident**, thresholding becomes dangerous: you think you’re safe when you’re not.

2) **Abstention**
   - A safe detector doesn’t need to auto-decide on everything.
   - It can **abstain** (send uncertain cases to human review) to reduce harmful errors.
   - But abstaining too often reduces usability and increases review cost.

So the product question becomes:

> **What confidence threshold gives us the best tradeoff between coverage (auto-decisions) and correctness?**

That’s exactly what the report card + dashboard answer.

---

## Key ideas

### 1) Prediction quality is not the same as probability quality
A model can be:
- **accurate but miscalibrated** (probabilities are misleading)
- **well-calibrated but not very accurate** (it “knows what it doesn’t know,” but still struggles)

You need both for decision-safe deployment.

### 2) Coverage is a first-class metric
Coverage = fraction of cases the model decides automatically.
- High coverage means fewer reviews but more exposure to wrong decisions.
- Lower coverage means safer auto-decisions but more review load.

### 3) “Decision-safe” means the model can say “I don’t know”
This repo operationalizes that with:
- a recommended confidence threshold
- a clear rule in the UI showing why the system abstained

---

## What the pipeline produces

After running the pipeline, you get:

### Saved metrics and policies (machine-readable)
- `outputs/metrics_overall.json`
  - accuracy, macro_f1, ECE, Brier score, labels, etc.
- `outputs/abstention_policy.json`
  - recommended threshold
  - expected coverage at that threshold
  - (optionally) rule notes

### Saved curves (for analysis + UI)
- `outputs/coverage_curve.csv`
  - threshold → coverage → accuracy → macro_f1 (and potentially more)

### Saved predictions (for UI demo and audits)
- `outputs/test_predictions.csv`
  - predicted label, confidence
  - per-class probabilities `p_<label>`
  - optional “disagreement” features used in the abstain rule

### Figures for the report card
Saved to: `reports/figures/`
- confusion matrix
- reliability diagram
- coverage vs performance
- confidence histogram

---

## Repository structure

```text
detector-reliability-report-card/
├─ app/
│  └─ app.py                      # Streamlit dashboard
├─ src/
│  └─ pipeline.py                 # Training + evaluation + artifacts + plots
├─ data/
│  └─ raw/
│     └─ ai_human_detection.csv   # Example input
├─ outputs/
│  ├─ metrics_overall.json
│  ├─ abstention_policy.json
│  ├─ coverage_curve.csv
│  └─ test_predictions.csv
└─ reports/
   ├─ figures/
   │  ├─ confusion_matrix.png
   │  ├─ reliability_diagram.png
   │  ├─ coverage_vs_accuracy.png
   │  └─ probability_histograms.png
   └─ screenshots/                # Optional: UI screenshots for README
      ├─ ui_report_card.png
      ├─ ui_coverage_curve.png
      ├─ ui_triage_ui.png
      └─ ui_notes.png
````

---

## Quickstart

### 1) Install

```bash
pip install -r requirements.txt
```

### 2) Run pipeline (generate report artifacts)

```bash
python -m src.pipeline --input data/raw/ai_human_detection.csv
```

Expected outcome:

* `outputs/` populated with JSON/CSV artifacts
* `reports/figures/` populated with PNG plots
* terminal prints a recommended threshold + estimated coverage (example: threshold≈0.61, coverage≈0.71)

### 3) Launch Streamlit

```bash
streamlit run app/app.py
```

---

## Data format

At minimum, the pipeline needs:

* a **text column** (the content)
* a **label column** (ground truth class)

Typical labels used by this project:

* `human`
* `ai`
* `post_edited_ai`

**Why 3-class matters:**
Post-edited AI behaves like an “in-between” distribution. It often creates:

* confusion with `ai` when edits are light
* confusion with `human` when edits are heavy

That’s why macro-F1 and confusion analysis are emphasized.

> If your dataset uses different column names, preprocess to match the expected schema (or update pipeline mapping).

---

## How the pipeline works

This section explains *what the pipeline is doing conceptually*, not just “it trains a model.”

### Step A | Train a baseline classifier

The pipeline trains a simple, explainable baseline (fast and strong enough for report-card purposes).
You may see “Primary model: char” in logs, indicating a character-based representation/model variant.

Why baseline models?

* quick to train and iterate
* easy to debug
* provide a strong reference point before heavier models

### Step B | Produce probabilities (not just labels)

Decision safety requires probabilities because:

* thresholds operate on probabilities (confidence)
* calibration evaluates probability quality

### Step C | Calibrate probabilities

Raw ML scores are often miscalibrated.
Calibration reshapes predicted probabilities so that “0.8” behaves like “~80% correct.”

This repo supports common calibration choices:

* **sigmoid (Platt scaling)**: stable, good default
* **isotonic**: more flexible but can overfit on small calibration sets

### Step D | Evaluate classification quality (accuracy / macro-F1)

Standard performance metrics on held-out test data.

### Step E | Evaluate calibration (ECE, Brier)

This is the “trust layer”:

* calibration answers whether confidence aligns with reality
* ECE/Brier quantify that alignment

### Step F | Sweep thresholds to build the abstention curve

The pipeline evaluates many confidence thresholds:

* for each threshold `t`, auto-decide only when `confidence ≥ t`
* compute coverage and metrics on the decided subset

This produces:

* a curve that shows **coverage vs performance**
* a recommended threshold based on a target coverage

### Step G | Save artifacts and figures

Everything is written out so you can:

* reproduce results
* compare runs (diff JSON/CSV)
* use figures in reports and posts
* power the dashboard without re-training every time

---

## Metrics explained

### Accuracy

**What it measures:** overall correctness rate.
**What it hides:** class imbalance and which mistakes matter.

If one class dominates, accuracy may look fine even if you fail on minority classes.

### Macro F1

**What it measures:** F1 per class averaged equally across classes.
**Why it matters here:** AI / human / post-edited classes often have different difficulty and prevalence.

Macro F1 helps prevent a misleading “it’s accurate!” conclusion that’s driven by the easiest class.

### Calibration (why probability trust matters)

#### Reliability

A model is well-calibrated if:

* among all predictions made at ~0.70 confidence, about 70% are correct

You can have:

* high accuracy but terrible calibration (dangerous thresholds)
* moderate accuracy but excellent calibration (safer abstention behavior)

#### ECE (Expected Calibration Error)

ECE bins predictions by confidence:

* bin 0.6–0.7, 0.7–0.8, etc.
* for each bin compare:

  * mean confidence
  * empirical accuracy

Then it averages the absolute gap weighted by bin size.

**Interpretation:**

* lower is better
* high ECE means “confidence numbers are lying”

#### Brier Score

Brier measures the squared error of predicted probabilities.
It rewards:

* accurate predictions
* probabilities that are close to the true outcome

**Interpretation:**

* lower is better
* sensitive to both correctness and probability sharpness

---

## Abstention and coverage explained

### Coverage

Coverage = fraction of samples that the model **auto-decides**.

If you set threshold high:

* you keep only very confident predictions
* coverage drops

If you set threshold low:

* you decide more often
* coverage increases

### Performance under abstention

When abstaining, you evaluate performance on the *decided subset*.

This usually increases as threshold rises (coverage drops), because:

* the model keeps only easy/high-confidence examples

### The real deployment tradeoff

* High coverage → less review cost, more wrong auto-decisions
* Low coverage → safer auto-decisions, higher review burden

This repo makes that tradeoff measurable and explicit.

---

## Figures explained (each one, deeply)

All figures are in `reports/figures/`.

---

### 1) Confusion matrix

**File:** `reports/figures/confusion_matrix.png`

<img width="1020" height="850" alt="confusion_matrix" src="https://github.com/user-attachments/assets/21d20d07-fc52-474a-b70c-540d62a75f7c" />

**What it shows**

* Rows: true label
* Columns: predicted label
* Each cell: count of examples

**How to read it**

* Diagonal: correct predictions
* Off-diagonal: confusions

**Why it’s critical for 3-class detection**
In AI vs human vs post-edited AI:

* `post_edited_ai → ai` may mean edits are subtle or model sees AI artifacts
* `post_edited_ai → human` may mean edits “wash out” artifacts
* `ai → human` may indicate generator style resembles human writing

**Actionable use**

* decide where to invest improvement:

  * collect more post-edited examples
  * add features for edit markers
  * adjust class weights or thresholds per class (advanced)

---

### 2) Reliability diagram (confidence vs accuracy)

**File:** `reports/figures/reliability_diagram.png`

<img width="1020" height="1020" alt="reliability_diagram" src="https://github.com/user-attachments/assets/39a1a125-ff0f-479a-babc-470978914769" />

**What it shows**

* x: predicted confidence (averaged within a bin)
* y: empirical accuracy (within that bin)
* dashed diagonal: perfect calibration

**Interpretation**

* Points below diagonal → overconfidence (high risk)
* Points above diagonal → underconfidence (model is safer than it claims)

**Why it matters**
If your decision rule uses “confidence ≥ 0.8,” but the model is overconfident:

* you may think you’re auto-deciding safely
* but your true correctness might be far lower than expected

This figure helps validate whether thresholding is trustworthy.

---

### 3) Coverage vs performance under abstention

**File:** `reports/figures/coverage_vs_accuracy.png`

<img width="1190" height="850" alt="coverage_vs_accuracy" src="https://github.com/user-attachments/assets/734ee3d9-41ec-4cb4-8656-28b13d457afb" />

**What it shows**

* x: coverage (fraction auto-decided)
* y: performance (accuracy, macro-F1) on auto-decided subset

**How to use it**

* choose a target coverage your workflow can handle
* check the resulting expected performance

**Example reasoning**

* If you can only review 30% of items:

  * you need ~70% coverage
  * find what accuracy/macro-F1 you get there
* If you require minimum macro-F1 of 0.70:

  * find what coverage you must accept

This connects model behavior to operational constraints.

---

### 4) Confidence histogram (max probability)

**File:** `reports/figures/probability_histograms.png`

<img width="1190" height="850" alt="probability_histograms" src="https://github.com/user-attachments/assets/f46b2ead-9d26-4ec1-aba1-0c9ed0354316" />

**What it shows**

* distribution of max predicted probability per sample (“confidence”)

**Why it’s useful**

* reveals how often the model is uncertain
* shows whether a threshold will dramatically change coverage
* helps detect suspicious confidence behavior:

  * everything near 1.0 can be a sign of overconfidence (check calibration)
  * everything midrange suggests the model lacks separability

**Practical threshold insight**
A good threshold often lies where:

* you drop the “uncertain mass”
* without destroying coverage

---

## Streamlit dashboard explained (tab-by-tab)

The dashboard turns offline artifacts into a clean decision interface.

---

### Tab 1 | Report Card

<img width="1861" height="904" alt="Screenshot 2026-02-14 at 15-22-31 Detector Reliability Report Card" src="https://github.com/user-attachments/assets/79a8bfa3-4560-427a-996f-f78984fd3112" />

Purpose: **one screen that answers: “should we trust this model?”**

What it shows:

* top-line metrics:

  * accuracy, macro-F1 (quality)
  * ECE, Brier (trust)
* a clean **2×2 grid** of figures:

  1. confusion matrix
  2. coverage vs performance
  3. reliability diagram
  4. confidence histogram
* recommended abstention policy JSON (if present)

Why the 2×2 grid matters:

* it prevents “scroll blindness”
* lets you compare diagnostics side-by-side
* keeps the view clean (same width and aligned)

---

### Tab 2 | Coverage Curve

<img width="1904" height="935" alt="Screenshot 2026-02-14 at 15-23-14 Detector Reliability Report Card" src="https://github.com/user-attachments/assets/cab423ea-3f01-4d29-b258-8463f82fe542" />

Purpose: **explore threshold tradeoffs interactively**

It typically includes:

* performance vs coverage
* threshold vs coverage/metrics

What it helps you decide:

* “What threshold gives me ~70% auto-decisions?”
* “How much performance do I lose if I increase coverage?”
* “Where are diminishing returns?”

---

### Tab 3 | Triage UI

<img width="1867" height="491" alt="Screenshot 2026-02-14 at 15-23-28 Detector Reliability Report Card" src="https://github.com/user-attachments/assets/e7bb5eeb-76fa-4dcb-b795-da2bad587447" />

Purpose: **show what a decision-safe output would look like to a user/operator**

What it demonstrates:

* predicted class
* confidence
* auto-decide vs abstain decision
* a probability breakdown bar chart

Important note (current behavior):

* this demo uses saved test predictions to demonstrate UI format
* for real inference:

  * persist the trained model
  * load it in the app
  * run prediction on pasted text

---

### Tab 4 | Notes

<img width="952" height="444" alt="Screenshot 2026-02-14 at 15-23-37 Detector Reliability Report Card" src="https://github.com/user-attachments/assets/00e54a7a-0bf2-4820-9e7c-a3695088514d" />

Purpose: **document the policy philosophy + upgrade path**

It explains:

* accuracy ≠ trust
* calibration and ECE
* coverage as a real product metric
* recommended next upgrades

---

## Recommended threshold: what it means

The pipeline prints a “Recommended threshold” based on your chosen target coverage.

**Interpretation:**

* “Threshold = 0.61, coverage ≈ 0.71” means:

  * if you auto-decide when confidence ≥ 0.61
  * you will auto-decide about 71% of cases (on similar data)
  * the remaining ~29% should go to review

**Important limitation**
Coverage estimates are only valid if:

* future data resembles evaluation data
* calibration remains stable
* class mix doesn’t drift heavily

That’s why drift monitoring is part of production-grade upgrades.

---

## Decision safety: what this UI prevents

This project avoids common failure patterns:

### “The model is 80% accurate so we trust it”

Not safe.
80% accuracy can still mean:

* severe overconfidence
* unacceptable errors on minority classes
* catastrophic errors on specific slices

### “Set threshold to 0.9 and ship”

Not safe unless calibration supports it.
If the model is overconfident, 0.9 is not truly 90% reliable.

### “We’ll just review random samples”

Not efficient.
Abstention focuses review on **uncertain** cases where humans add the most value.

---

## How to make it production-grade

If you want this to become a real internal tool:

### 1) Real inference in the UI

* save trained model artifact (joblib)
* load it in `app.py`
* run prediction on input text

### 2) Slice audits (must-have)

Track performance/calibration by:

* language
* topic/domain
* content length
* “post-edited intensity” bins

### 3) Drift monitoring (must-have)

Monitor over time:

* confidence distribution drift
* class mix drift
* calibration drift (ECE over time)

### 4) Cost-aware abstention (advanced)

Choose threshold by minimizing:

* error cost (wrong auto-decisions)
* review cost (abstentions)

---

## Troubleshooting

### Streamlit warning: `use_container_width` deprecation

If you see:

> “Please replace `use_container_width` with `width`...”

Fix:

* use `width="stretch"` in `st.image()` and `st.plotly_chart()`

This repo’s UI layout is designed to use `width="stretch"` so figures align cleanly.

### “Run pipeline first”

If the dashboard says outputs are missing:

* click **Run / Refresh** in the sidebar
* or run the pipeline from the command line

### Outputs not updating

* ensure your `out_dir` and `figures_dir` are correct
* confirm the app points to the same project root
