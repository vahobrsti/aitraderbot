"""
Incremental-value analysis via logistic regression.

Answers:
  - Which metric is strongest standalone?
  - Which adds the most incremental value?
  - Is any metric mainly a confirmer rather than an anchor?

Uses only simple, interpretable models (LogisticRegression)
with TimeSeriesSplit cross-validation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

# ────────────────────────────────────────────────────────────────────
# Feature preparation
# ────────────────────────────────────────────────────────────────────
_METRIC_COLS = {
    "mdia": "mdia_bucket",
    "whales": "whale_bucket",
    "mvrv_ls": "mvrv_ls_bucket",
}

_MODEL_SETS: list[tuple[str, list[str]]] = [
    ("mdia", ["mdia"]),
    ("whales", ["whales"]),
    ("mvrv_ls", ["mvrv_ls"]),
    ("mdia+whales", ["mdia", "whales"]),
    ("mdia+mvrv_ls", ["mdia", "mvrv_ls"]),
    ("whales+mvrv_ls", ["whales", "mvrv_ls"]),
    ("mdia+whales+mvrv_ls", ["mdia", "whales", "mvrv_ls"]),
]


# ────────────────────────────────────────────────────────────────────
# Single model evaluation
# ────────────────────────────────────────────────────────────────────
def _evaluate_model(
    X_raw: pd.DataFrame,
    y: pd.Series,
    metrics: list[str],
    n_splits: int = 5,
) -> dict:
    """Fit logistic regression with TimeSeriesSplit, return per-fold metrics.
    
    Encoding is done inside each fold to prevent future-category leakage.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_aucs = []
    fold_losses = []
    coef_list = []

    cols = [_METRIC_COLS[m] for m in metrics]

    for train_idx, test_idx in tscv.split(X_raw):
        X_train_str = X_raw.iloc[train_idx][cols]
        X_test_str = X_raw.iloc[test_idx][cols]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Skip if only one class in training or test
        if y_train.nunique() < 2 or y_test.nunique() < 2:
            continue
            
        # Encode features strictly inside the fold
        X_train = pd.get_dummies(X_train_str, columns=cols, drop_first=True).astype(float)
        X_test = pd.get_dummies(X_test_str, columns=cols, drop_first=True).astype(float)
        
        # Align test features to training features (drop unseen, fill missing with 0)
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0.0)

        model = LogisticRegression(
            max_iter=1000, solver="lbfgs", random_state=42
        )
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, proba)
        ll = log_loss(y_test, proba)
        fold_aucs.append(auc)
        fold_losses.append(ll)
        coef_list.append(dict(zip(X_train.columns, model.coef_[0])))

    if not fold_aucs:
        return {
            "mean_auc": np.nan,
            "std_auc": np.nan,
            "mean_logloss": np.nan,
            "per_fold_auc": [],
            "coefficients": [],
        }

    return {
        "mean_auc": float(np.mean(fold_aucs)),
        "std_auc": float(np.std(fold_aucs)),
        "mean_logloss": float(np.mean(fold_losses)),
        "per_fold_auc": [float(a) for a in fold_aucs],
        "coefficients": coef_list,
    }


# ────────────────────────────────────────────────────────────────────
# Permutation importance (Fold-based)
# ────────────────────────────────────────────────────────────────────
def _permutation_importance_cv(
    X_raw: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    n_repeats: int = 5,
) -> dict[str, float]:
    """Compute permutation importance across TimeSeriesSplit folds.

    Returns metric_name → mean AUC drop when that metric's columns
    are shuffled in the test set.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    importance_dicts = []
    
    # We use all available metrics for the full model importance
    all_metrics = list(_METRIC_COLS.keys())
    cols = list(_METRIC_COLS.values())

    for train_idx, test_idx in tscv.split(X_raw):
        X_train_str = X_raw.iloc[train_idx][cols]
        X_test_str = X_raw.iloc[test_idx][cols]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        if y_train.nunique() < 2 or y_test.nunique() < 2:
            continue
            
        # Encode features strictly inside the fold
        X_train = pd.get_dummies(X_train_str, columns=cols, drop_first=True).astype(float)
        X_test = pd.get_dummies(X_test_str, columns=cols, drop_first=True).astype(float)
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0.0)

        model = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42)
        model.fit(X_train, y_train)
        baseline_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

        fold_imp = {}
        for metric_name in all_metrics:
            bucket_col = _METRIC_COLS[metric_name]
            
            # Note: We must check if the metric is even present in the aligned columns
            # But the simplest way to permute is to shuffle the raw string column
            # and then re-encode, aligning with the training columns.
            rng = np.random.RandomState(42)
            drops = []
            
            for _ in range(n_repeats):
                X_test_str_perm = X_test_str.copy()
                X_test_str_perm[bucket_col] = rng.permutation(X_test_str_perm[bucket_col].values)
                
                # Re-encode and re-align
                X_perm_encoded = pd.get_dummies(X_test_str_perm, columns=cols, drop_first=True).astype(float)
                X_perm_encoded = X_perm_encoded.reindex(columns=X_train.columns, fill_value=0.0)
                
                perm_auc = roc_auc_score(y_test, model.predict_proba(X_perm_encoded)[:, 1])
                drops.append(baseline_auc - perm_auc)
            
            fold_imp[metric_name] = float(np.mean(drops))
            
        importance_dicts.append(fold_imp)

    if not importance_dicts:
        return {}

    # Average drops across folds
    final_imp = {}
    for metric_name in all_metrics:
        vals = [d.get(metric_name, 0) for d in importance_dicts if metric_name in d]
        if vals:
            final_imp[metric_name] = float(np.mean(vals))

    return final_imp


# ────────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────────
def run_model_comparison(
    research_df: pd.DataFrame,
    target_col: str = "label_good_move_long",
    n_splits: int = 5,
) -> dict:
    """Run all 7 model variants and return comparison results.

    Parameters
    ----------
    research_df : DataFrame
        Output of ``build_research_table``.
    target_col : str
        Binary label column to predict.
    n_splits : int
        Number of TimeSeriesSplit folds.

    Returns
    -------
    dict with keys:
        model_results : list[dict]
            One entry per model variant with name, auc, logloss, etc.
        feature_importance : dict
            Permutation importance for each metric (from the full model).
        summary : dict
            best_standalone, best_incremental, confirmer_metrics.
    """
    # Drop rows with missing target
    clean = research_df.dropna(subset=[target_col]).copy()
    
    # Guard n_splits against small datasets
    n_splits = min(n_splits, max(2, len(clean) // 10))
    if len(clean) < 10 or n_splits < 2:
        return {"model_results": [], "feature_importance": {}, "summary": {}}
        
    y = clean[target_col].astype(int)

    # Run each model variant
    model_results = []
    for name, metrics in _MODEL_SETS:
        result = _evaluate_model(clean, y, metrics, n_splits=n_splits)
        result["model_name"] = name
        result["metrics_used"] = metrics
        model_results.append(result)

    # Permutation importance from CV on the full model
    importance = _permutation_importance_cv(clean, y, n_splits=n_splits)

    # Build summary
    singles = [r for r in model_results if len(r["metrics_used"]) == 1]
    valid_singles = [s for s in singles if not np.isnan(s["mean_auc"])]
    if valid_singles:
        best_single = max(valid_singles, key=lambda r: r["mean_auc"])
        best_standalone = best_single["model_name"]
    else:
        best_standalone = None

    # Incremental value: which metric adds the most when going from 2→3?
    full_auc = next(
        (r["mean_auc"] for r in model_results if r["model_name"] == "mdia+whales+mvrv_ls"),
        np.nan,
    )
    pairs = [r for r in model_results if len(r["metrics_used"]) == 2]
    incremental = {}
    for pair in pairs:
        missing = [m for m in ("mdia", "whales", "mvrv_ls") if m not in pair["metrics_used"]]
        if missing and not np.isnan(full_auc) and not np.isnan(pair["mean_auc"]):
            gain = full_auc - pair["mean_auc"]
            if gain > 0:  # Must have positive incremental value to be considered
                incremental[missing[0]] = gain

    best_incremental = max(incremental, key=incremental.get) if incremental else None

    # Confirmer: metric with lowest standalone AUC but positive incremental
    confirmer_metrics = []
    if valid_singles and incremental:
        worst_single = min(valid_singles, key=lambda r: r["mean_auc"])
        if worst_single["model_name"] in incremental:
            # We already enforced gain > 0 above, so if it's in the dict, it's a confirmer
            confirmer_metrics.append(worst_single["model_name"])

    return {
        "model_results": model_results,
        "feature_importance": importance,
        "summary": {
            "best_standalone": best_standalone,
            "best_incremental": best_incremental,
            "incremental_values": incremental,
            "confirmer_metrics": confirmer_metrics,
        },
    }
