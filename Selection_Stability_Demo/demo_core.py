"""Synthetic-only demonstrator for auditing imbalance-strategy selection stability.

This module is educational/prototyping code. It does not load clinical data and its
outputs must not be reported as healthcare study results.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

STRATEGIES = ("No handling", "Class weights", "Random oversampling")


@dataclass
class DemoConfig:
    training_size: int = 250
    imbalance_ratio: int = 5
    noise_percent: int = 5
    repeats: int = 20
    seed: int = 42
    population_size: int = 12000


def _make_population(config: DemoConfig):
    """Create a fixed synthetic population with a clean, specified class ratio."""
    majority_weight = config.imbalance_ratio / (config.imbalance_ratio + 1.0)
    minority_weight = 1.0 / (config.imbalance_ratio + 1.0)
    X, y = make_classification(
        n_samples=config.population_size,
        n_features=12,
        n_informative=7,
        n_redundant=2,
        n_repeated=0,
        n_classes=2,
        n_clusters_per_class=2,
        weights=[majority_weight, minority_weight],
        flip_y=0.0,
        class_sep=1.0,
        random_state=config.seed,
    )
    # The held-out set remains uncorrupted and is never used in strategy selection.
    X_pool, X_test, y_pool, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=config.seed + 1,
    )
    return X_pool, y_pool, X_test, y_test


def _draw_training_sample(
    X_pool: np.ndarray,
    y_pool: np.ndarray,
    config: DemoConfig,
    seed: int,
):
    """Draw a training sample with a controlled total size and imbalance ratio."""
    rng = np.random.default_rng(seed)
    n_minority = int(round(config.training_size / (config.imbalance_ratio + 1)))
    n_minority = max(3, n_minority)
    n_majority = config.training_size - n_minority

    idx_majority = np.flatnonzero(y_pool == 0)
    idx_minority = np.flatnonzero(y_pool == 1)
    if n_majority > len(idx_majority) or n_minority > len(idx_minority):
        raise ValueError(
            "Requested training condition exceeds the generated population. "
            "Lower training size or imbalance ratio."
        )

    chosen_majority = rng.choice(idx_majority, size=n_majority, replace=False)
    chosen_minority = rng.choice(idx_minority, size=n_minority, replace=False)
    indices = np.concatenate([chosen_majority, chosen_minority])
    rng.shuffle(indices)
    X_train = X_pool[indices].copy()
    y_clean = y_pool[indices].copy()
    y_noisy = y_clean.copy()

    # CLIMB-inspired symmetric corruption: flip k minority labels to majority and
    # the same number of majority labels to minority, preserving the class count.
    if config.noise_percent > 0:
        minority_indices = np.flatnonzero(y_noisy == 1)
        majority_indices = np.flatnonzero(y_noisy == 0)
        n_flip = int(rng.binomial(len(minority_indices), config.noise_percent / 100.0))
        if n_flip > 0:
            to_majority = rng.choice(minority_indices, size=n_flip, replace=False)
            to_minority = rng.choice(majority_indices, size=n_flip, replace=False)
            y_noisy[to_majority] = 0
            y_noisy[to_minority] = 1
    return X_train, y_noisy, y_clean


def _random_oversample(X: np.ndarray, y: np.ndarray, seed: int):
    """Randomly duplicate minority examples to match the majority count."""
    rng = np.random.default_rng(seed)
    idx0 = np.flatnonzero(y == 0)
    idx1 = np.flatnonzero(y == 1)
    if len(idx0) == 0 or len(idx1) == 0:
        raise ValueError("Both classes are required for this demonstration.")
    if len(idx0) == len(idx1):
        return X, y
    minority_indices = idx1 if len(idx1) < len(idx0) else idx0
    majority_count = max(len(idx0), len(idx1))
    n_extra = majority_count - len(minority_indices)
    extra = rng.choice(minority_indices, size=n_extra, replace=True)
    new_idx = np.concatenate([np.arange(len(y)), extra])
    rng.shuffle(new_idx)
    return X[new_idx], y[new_idx]


def _fit_predict_proba(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    strategy: str,
    seed: int,
):
    """Fit a consistent logistic-regression learner for one strategy."""
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xev = scaler.transform(X_eval)

    class_weight = "balanced" if strategy == "Class weights" else None
    if strategy == "Random oversampling":
        Xtr, y_fit = _random_oversample(Xtr, y_train, seed)
    else:
        y_fit = y_train

    model = LogisticRegression(
        max_iter=700,
        solver="liblinear",
        class_weight=class_weight,
        random_state=seed,
    )
    model.fit(Xtr, y_fit)
    return model.predict_proba(Xev)[:, 1]


def _inner_cv_score(
    X: np.ndarray,
    y: np.ndarray,
    strategy: str,
    seed: int,
) -> float:
    """Mean average precision across three stratified inner folds."""
    counts = np.bincount(y.astype(int), minlength=2)
    n_splits = min(3, int(counts.min()))
    if n_splits < 2:
        raise ValueError("At least two examples per class are required for inner CV.")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores: list[float] = []
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        p = _fit_predict_proba(
            X[train_idx], y[train_idx], X[val_idx], strategy, seed + fold + 1
        )
        scores.append(float(average_precision_score(y[val_idx], p)))
    return float(np.mean(scores))


def run_demo(config: DemoConfig) -> dict[str, Any]:
    """Run repeated inner-CV selection and evaluate on a clean synthetic test set."""
    if config.training_size < 30:
        raise ValueError("Training size must be at least 30 for this demonstration.")
    if config.imbalance_ratio not in (2, 5, 10):
        raise ValueError("Imbalance ratio must be one of 2, 5, or 10.")
    if config.noise_percent not in (0, 5, 10):
        raise ValueError("Noise must be one of 0%, 5%, or 10%.")
    if config.repeats < 3:
        raise ValueError("Use at least three repeated samples.")

    X_pool, y_pool, X_test, y_test = _make_population(config)
    selected: list[str] = []
    inner_scores_by_run: list[dict[str, float]] = []
    test_scores_by_run: list[dict[str, float]] = []
    test_recalls_by_run: list[dict[str, float]] = []
    test_precisions_by_run: list[dict[str, float]] = []

    for repeat in range(config.repeats):
        run_seed = config.seed + 1000 + repeat * 17
        X_train, y_train, _y_original = _draw_training_sample(
            X_pool, y_pool, config, run_seed
        )

        inner_scores = {
            strategy: _inner_cv_score(
                X_train, y_train, strategy, run_seed + 500 + i
            )
            for i, strategy in enumerate(STRATEGIES)
        }
        # Stable deterministic tie-break follows STRATEGIES order.
        chosen = max(STRATEGIES, key=lambda name: inner_scores[name])
        selected.append(chosen)
        inner_scores_by_run.append(inner_scores)

        test_scores: dict[str, float] = {}
        test_recalls: dict[str, float] = {}
        test_precisions: dict[str, float] = {}
        for i, strategy in enumerate(STRATEGIES):
            probabilities = _fit_predict_proba(
                X_train, y_train, X_test, strategy, run_seed + 800 + i
            )
            pred = (probabilities >= 0.5).astype(int)
            test_scores[strategy] = float(average_precision_score(y_test, probabilities))
            test_recalls[strategy] = float(recall_score(y_test, pred, zero_division=0))
            test_precisions[strategy] = float(precision_score(y_test, pred, zero_division=0))
        test_scores_by_run.append(test_scores)
        test_recalls_by_run.append(test_recalls)
        test_precisions_by_run.append(test_precisions)

    counts = Counter(selected)
    frequency = {s: counts.get(s, 0) / config.repeats for s in STRATEGIES}
    positive_freq = [p for p in frequency.values() if p > 0]
    entropy = -sum(p * np.log(p) for p in positive_freq)
    normalized_entropy = float(entropy / np.log(len(STRATEGIES)))
    agreement = float(max(frequency.values()))

    # The fixed reference is selected using pooled inner-CV scores only. Test labels
    # are not used to select it; the clean test set is used only for evaluation.
    pooled_inner = {
        s: float(np.mean([run[s] for run in inner_scores_by_run]))
        for s in STRATEGIES
    }
    fixed_strategy = max(STRATEGIES, key=lambda name: pooled_inner[name])

    selected_test_scores = [scores[name] for scores, name in zip(test_scores_by_run, selected)]
    mean_selected_test_ap = float(np.mean(selected_test_scores))
    mean_fixed_test_ap = float(np.mean([scores[fixed_strategy] for scores in test_scores_by_run]))
    fixed_reference_gap = mean_fixed_test_ap - mean_selected_test_ap

    summary_rows = []
    for strategy in STRATEGIES:
        summary_rows.append(
            {
                "Strategy": strategy,
                "Selection frequency": frequency[strategy],
                "Mean inner-CV AP": pooled_inner[strategy],
                "Mean clean-test AP": float(np.mean([r[strategy] for r in test_scores_by_run])),
                "Mean clean-test recall @ 0.5": float(np.mean([r[strategy] for r in test_recalls_by_run])),
                "Mean clean-test precision @ 0.5": float(np.mean([r[strategy] for r in test_precisions_by_run])),
            }
        )

    run_rows = []
    for i, strategy in enumerate(selected):
        run_rows.append(
            {
                "Repeat": i + 1,
                "Selected strategy": strategy,
                "No handling - inner AP": inner_scores_by_run[i]["No handling"],
                "Class weights - inner AP": inner_scores_by_run[i]["Class weights"],
                "Random oversampling - inner AP": inner_scores_by_run[i]["Random oversampling"],
                "Selected strategy - clean test AP": test_scores_by_run[i][strategy],
                "Fixed reference strategy": fixed_strategy,
            }
        )

    minority_count = int(round(config.training_size / (config.imbalance_ratio + 1)))
    return {
        "config": config,
        "agreement": agreement,
        "normalized_entropy": normalized_entropy,
        "frequency": frequency,
        "fixed_strategy": fixed_strategy,
        "mean_selected_test_ap": mean_selected_test_ap,
        "mean_fixed_test_ap": mean_fixed_test_ap,
        "fixed_reference_gap": float(fixed_reference_gap),
        "summary": pd.DataFrame(summary_rows),
        "runs": pd.DataFrame(run_rows),
        "minority_count": minority_count,
        "majority_count": config.training_size - minority_count,
        "noise_definition": (
            "q% of minority labels are randomly flipped to majority, with an equal "
            "number of majority labels flipped to minority; labels are corrupted in "
            "the training sample only. Rounding is stochastic across repeats."
        ),
        "disclaimer": (
            "Synthetic demonstration only. Results are not clinical evidence and must "
            "not be presented as findings from real healthcare datasets."
        ),
    }
