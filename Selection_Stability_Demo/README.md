# Selection Stability Audit — Synthetic Demonstrator

This is the working demonstrator accompanying a **proposed research protocol**. It is not a validated healthcare model and does not use real patient records.

## What it demonstrates

- A synthetic binary dataset with a user-selected majority:minority ratio.
- Small training samples of 100, 250, or 500 rows.
- Training-label corruption at 0%, 5%, or 10% using paired flips, leaving the held-out test labels clean.
- Repeated strategy selection using inner stratified cross-validation and average precision (AP).
- Three baseline strategies: no imbalance handling, class-weighted logistic regression, and random oversampling.
- Selection agreement, normalised selection entropy, clean-test AP, minority recall/precision, and a fixed-reference performance gap.

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
python demo_cli.py --training-size 250 --imbalance-ratio 5 --noise 5 --repeats 20
```

The command writes `demo_summary.csv` and `demo_summary_runs.csv` in the current directory. Try a second condition such as:

```powershell
python demo_cli.py --training-size 100 --imbalance-ratio 10 --noise 10 --repeats 20 --seed 43 --output condition_2.csv
```

## Metric definitions

- **Selection agreement:** fraction of repeated samples selecting the most frequently chosen strategy.
- **Normalised entropy:** entropy of strategy-selection frequencies divided by `log(number_of_strategies)`; values near zero indicate concentrated selections, while values near one indicate selections spread across strategies.
- **AP:** average precision, used as a precision-recall ranking metric.
- **Fixed-reference gap:** mean clean-test AP of the fixed strategy selected by pooled inner-CV scores minus mean clean-test AP of the strategy selected separately in each repeat. Positive values indicate the repeated-selection policy scored below that fixed reference for this synthetic condition.

## Caveats

- The population is synthetic and does not capture healthcare-specific sampling, label error, confounding, missingness, or clinical workflow.
- The same clean synthetic test set is used only for repeated evaluation; it is never used to select strategies. This is adequate for demonstrating program flow, not a substitute for the paper's planned nested multi-dataset evaluation.
- The default probability threshold of 0.5 is used only to illustrate recall and precision. The main strategy-selection metric is AP.
- The fixed reference is selected using pooled inner-CV scores, not the clean test labels.
- Any real-data study must follow the paper's protocol, confirm dataset licences/provenance, predefine all conditions, and report uncertainty across datasets.
