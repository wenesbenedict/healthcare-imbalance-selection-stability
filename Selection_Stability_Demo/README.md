# Selection Stability Audit — Synthetic Demonstrator

This is the working demonstrator accompanying a **proposed research protocol**. It is not a validated healthcare model and does not use real patient records.

## What it demonstrates

- A synthetic binary dataset with a user-selected majority:minority ratio.
- Small training samples of 100, 250, or 500 rows.
- Training-label corruption at 0%, 5%, or 10% using random binomial paired flips, leaving the held-out test labels clean. The number of flips may be zero when class counts are low.
- Repeated strategy selection using inner stratified cross-validation and average precision (AP).
- Three baseline strategies: no imbalance handling, class-weighted logistic regression, and random oversampling.
- Selection agreement with a Wilson 95% interval, normalised selection entropy with a 2,000-resample bootstrap interval, threshold-free clean-test AP/ROC-AUC, training-only F1 threshold recall/precision, and a paired-bootstrap interval for the fixed-reference performance gap.

**Important:** All displayed scores are from synthetic data created by `sklearn.datasets.make_classification`. They are only a demonstration that the workflow runs. Do not copy them into the conference paper as healthcare findings.

## Run on Windows (PowerShell)

1. Extract this folder.
2. Open PowerShell in the extracted folder.
3. Create and activate a virtual environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

4. Install dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

5. Start the interactive demonstrator:

```powershell
streamlit run app.py
```

Streamlit prints a local URL (normally `http://localhost:8501`). Open it in your browser.

## Command-line fallback

The core demonstrator runs without Streamlit:

```powershell
python demo_cli.py --training-size 250 --imbalance-ratio 5 --noise 5 --repeats 100
```

The command writes `demo_summary.csv` and `demo_summary_runs.csv` in the current directory. Try a second condition such as:

```powershell
python demo_cli.py --training-size 100 --imbalance-ratio 10 --noise 10 --repeats 100 --seed 43 --output condition_2.csv
```

Use `python demo_cli.py --feasibility-table` to print the supported size/ratio grid,
and `python demo_cli.py --version` to print the demonstrator version. Runs below 100
repeats print a warning because their uncertainty intervals are wide.

## Metric definitions

- **Selection agreement:** fraction of repeated samples selecting the most frequently chosen strategy.
- **Tie-breaking:** when the best inner-CV mean AP values are within 0.005 AP, the simplest strategy wins in this fixed order: **No handling**, then **Class weights**, then **Random oversampling**. Each repeat records whether this rule was applied.
- **Threshold:** by default, the threshold maximising F1 on inner-CV predictions for the selected strategy is chosen from training data only (`training_f1`). `fixed_0.5` is retained as an explicit comparison mode. The clean test set is never used to choose a threshold.
- **Normalised entropy:** entropy of strategy-selection frequencies divided by `log(number_of_strategies)`; values near zero indicate concentrated selections, while values near one indicate selections spread across strategies. Its interval is a 2,000-resample bootstrap interval over repeats.
- **AP:** average precision, used as a precision-recall ranking metric.
- **Fixed-reference gap:** mean paired clean-test AP difference per repeat (fixed reference minus selected strategy), with a 2,000-resample paired bootstrap 95% interval. Positive values indicate the repeated-selection policy scored below that fixed reference for this synthetic condition.

## Caveats

- The population is synthetic and does not capture healthcare-specific sampling, label error, confounding, missingness, or clinical workflow.
- The same clean synthetic test set is used only for repeated evaluation; it is never used to select strategies. This is adequate for demonstrating program flow, not a substitute for the paper's planned nested multi-dataset evaluation.
- AP and ROC-AUC are the primary threshold-free test metrics. Recall and precision use the training-only F1 threshold by default.
- The fixed reference is selected using pooled inner-CV scores, not the clean test labels.
- Any real-data study must follow the paper's protocol, confirm dataset licences/provenance, predefine all conditions, and report uncertainty across datasets.
