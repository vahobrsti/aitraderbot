# ml/training.py
"""
ML training pipeline for long/short prediction models.

Improvements:
- Global feature_cols computed once from full dataset
- Decision lag option to avoid lookahead bias
- Feature mode (pure/hybrid) to control inclusion of handcrafted signals
- Top-k precision metrics with baseline for sparse trading evaluation
- Per-split dropna to avoid global row loss
- Label stats for regime understanding
"""

from pathlib import Path
from enum import Enum
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_score


class FeatureMode(Enum):
    """Feature selection modes"""
    PURE = "pure"      # Exclude all handcrafted signals and fusion features
    HYBRID = "hybrid"  # Include regime features, exclude trading signals


# Columns to always exclude from ML features
LABEL_COLS = {"label_good_move_long", "label_good_move_short"}

# Prefixes to exclude in PURE mode (all handcrafted)
PURE_EXCLUDE_PREFIXES = (
    "signal_option_",  # Trading signals (literally your rules)
    "fusion_",         # Fusion engine outputs
    "sent_regime_",    # Sentiment regime flags
    "mdia_regime_",    # MDIA regime flags
    "whale_regime_",   # Whale regime flags
    "mvrv_ls_regime_", # MVRV LS regime flags
)

# Prefixes to exclude in HYBRID mode (only trading signals)
HYBRID_EXCLUDE_PREFIXES = (
    "signal_option_",  # Trading signals (literally your rules)
    "fusion_",         # Fusion engine outputs
)


def load_feature_dataset(csv_path: Path) -> pd.DataFrame:
    """Load feature CSV with date index"""
    df = pd.read_csv(csv_path, parse_dates=["date"]).set_index("date")
    df = df.sort_index()
    return df


def get_feature_cols(df: pd.DataFrame, mode: FeatureMode = FeatureMode.HYBRID) -> list[str]:
    """
    Get feature columns based on mode.
    
    This is computed ONCE from the full dataset to ensure consistency.
    """
    exclude_prefixes = PURE_EXCLUDE_PREFIXES if mode == FeatureMode.PURE else HYBRID_EXCLUDE_PREFIXES
    
    feature_cols = [
        c for c in df.columns
        if c not in LABEL_COLS and not any(c.startswith(p) for p in exclude_prefixes)
    ]
    
    return feature_cols


def apply_decision_lag(df: pd.DataFrame, feature_cols: list[str], lag: int = 1) -> pd.DataFrame:
    """
    Apply decision lag to features to avoid lookahead bias.
    
    If trading decision is made on day T, features should be from day T-lag.
    NOTE: Does NOT drop NaN rows here - that's done per split/label.
    """
    if lag == 0:
        return df
    
    df = df.copy()
    df[feature_cols] = df[feature_cols].shift(lag)
    return df  # Don't dropna globally - do it per split


def drop_na_for_task(df: pd.DataFrame, feature_cols: list[str], label_col: str) -> pd.DataFrame:
    """Drop rows with NaN in features or label for this specific task"""
    cols_to_check = feature_cols + [label_col]
    return df.dropna(subset=cols_to_check)


def split_X_y(df: pd.DataFrame, feature_cols: list[str], label_col: str):
    """Split dataframe into features and labels using pre-computed feature_cols"""
    X = df[feature_cols]
    y = df[label_col]
    return X, y


def split_train_val_test_by_date(df: pd.DataFrame):
    """
    Train:  up to 2023-12-31
    Val:    2024-01-01 to 2024-12-31
    Test:   2025-01-01 onwards
    """
    df = df.sort_index()

    train = df.loc[: "2023-12-31"]
    val   = df.loc["2024-01-01":"2024-12-31"]
    test  = df.loc["2025-01-01":]

    return train, val, test


def print_label_stats(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, label_col: str):
    """Print label statistics for each split to understand base rates"""
    print(f"\n--- LABEL STATS ({label_col}) ---")
    for name, d in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        mean = d[label_col].mean()
        n = len(d)
        pos = d[label_col].sum()
        print(f"  {name:5s}: n={n:4d}  positive={pos:3.0f}  base_rate={mean:.3f}")
    print()


def compute_top_k_precision(y_true: np.ndarray, y_proba: np.ndarray, k_pct: float) -> float:
    """
    Compute precision for top k% of predictions.
    
    This is more relevant for sparse trading than overall AUC.
    """
    n = len(y_true)
    k = max(1, int(n * k_pct / 100))
    
    # Get indices of top k predictions
    top_k_idx = np.argsort(y_proba)[-k:]
    
    # Compute precision on top k
    top_k_true = y_true.iloc[top_k_idx] if hasattr(y_true, 'iloc') else y_true[top_k_idx]
    precision = top_k_true.sum() / k
    
    return precision


def print_model_metrics(name: str, y_val: pd.Series, y_val_proba: np.ndarray,
                        y_test: pd.Series, y_test_proba: np.ndarray):
    """Print comprehensive model metrics including top-k precision with baseline"""
    val_auc = roc_auc_score(y_val, y_val_proba)
    test_auc = roc_auc_score(y_test, y_test_proba)
    
    # Baseline = just the positive rate (random selection precision)
    val_baseline = y_val.mean()
    test_baseline = y_test.mean()
    
    print(f"\n[{name}] === METRICS ===")
    print(f"  Baseline (base rate):  Val={val_baseline:.3f}  Test={test_baseline:.3f}")
    print(f"  AUC:                   Val={val_auc:.3f}  Test={test_auc:.3f}")
    
    # Top-k precision with baseline comparison
    print(f"\n  --- Top-K Precision (vs baseline) ---")
    for k in [1, 2, 5, 10]:
        val_prec = compute_top_k_precision(y_val, y_val_proba, k)
        test_prec = compute_top_k_precision(y_test, y_test_proba, k)
        # Lift = precision / baseline
        val_lift = val_prec / val_baseline if val_baseline > 0 else 0
        test_lift = test_prec / test_baseline if test_baseline > 0 else 0
        print(f"  Top {k:2d}%: Val={val_prec:.3f} (lift={val_lift:.2f}x)  Test={test_prec:.3f} (lift={test_lift:.2f}x)")
    
    # Hit rate at high probability threshold
    print(f"\n  --- Threshold Hit Rates ---")
    for thresh in [0.5, 0.6, 0.7]:
        val_hits = (y_val[y_val_proba >= thresh]).sum()
        val_signals = (y_val_proba >= thresh).sum()
        test_hits = (y_test[y_test_proba >= thresh]).sum()
        test_signals = (y_test_proba >= thresh).sum()
        
        val_rate = val_hits / val_signals if val_signals > 0 else 0
        test_rate = test_hits / test_signals if test_signals > 0 else 0
        
        print(f"  P>={thresh}: Val={val_rate:.2f} ({val_hits:.0f}/{val_signals})  Test={test_rate:.2f} ({test_hits:.0f}/{test_signals})")
    
    return val_auc, test_auc


def train_long_model_with_holdout(
    csv_path: Path,
    model_out: Path,
    mode: FeatureMode = FeatureMode.HYBRID,
    decision_lag: int = 0,
) -> None:
    """Train long prediction model"""
    label_col = "label_good_move_long"
    
    print(f"\n{'='*60}")
    print(f"Training LONG model | mode={mode.value} | lag={decision_lag}")
    print('='*60)
    
    # Load full dataset
    df = load_feature_dataset(csv_path)
    
    # Compute feature columns ONCE from full dataset
    feature_cols = get_feature_cols(df, mode)
    print(f"Using {len(feature_cols)} features (mode={mode.value})")
    
    # Apply decision lag if specified (doesn't drop rows)
    if decision_lag > 0:
        df = apply_decision_lag(df, feature_cols, decision_lag)
        print(f"Applied decision lag of {decision_lag} day(s)")
    
    # Split data
    train_df, val_df, test_df = split_train_val_test_by_date(df)
    
    # Drop NaN per split (not globally)
    train_df = drop_na_for_task(train_df, feature_cols, label_col)
    val_df = drop_na_for_task(val_df, feature_cols, label_col)
    test_df = drop_na_for_task(test_df, feature_cols, label_col)
    
    print(f"Train: {train_df.index.min().date()} → {train_df.index.max().date()}  ({len(train_df)} rows)")
    print(f"Val:   {val_df.index.min().date()}   → {val_df.index.max().date()}    ({len(val_df)} rows)")
    print(f"Test:  {test_df.index.min().date()}  → {test_df.index.max().date()}   ({len(test_df)} rows)")
    
    # Print label stats to understand base rates
    print_label_stats(train_df, val_df, test_df, label_col)

    X_train, y_train = split_X_y(train_df, feature_cols, label_col)
    X_val, y_val = split_X_y(val_df, feature_cols, label_col)
    X_test, y_test = split_X_y(test_df, feature_cols, label_col)

    # Train on TRAIN + VAL
    X_train_val = pd.concat([X_train, X_val])
    y_train_val = pd.concat([y_train, y_val])

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=20,
        class_weight='balanced',  # Correct for label imbalance
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train_val, y_train_val)

    # Get predictions
    y_val_proba = model.predict_proba(X_val)[:, 1]
    y_test_proba = model.predict_proba(X_test)[:, 1]
    
    # Print metrics
    val_auc, test_auc = print_model_metrics("LONG", y_val, y_val_proba, y_test, y_test_proba)

    # Save model with correct feature_cols
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_names": feature_cols,  # Now computed from full df
            "feature_mode": mode.value,
            "decision_lag": decision_lag,
            "val_auc": float(val_auc),
            "test_auc": float(test_auc),
        },
        model_out,
    )
    print(f"\n[LONG] Saved model to {model_out}")


def train_short_model_with_holdout(
    csv_path: Path,
    model_out: Path,
    mode: FeatureMode = FeatureMode.HYBRID,
    decision_lag: int = 0,
) -> None:
    """Train short prediction model"""
    label_col = "label_good_move_short"
    
    print(f"\n{'='*60}")
    print(f"Training SHORT model | mode={mode.value} | lag={decision_lag}")
    print('='*60)
    
    # Load full dataset
    df = load_feature_dataset(csv_path)
    
    # Compute feature columns ONCE from full dataset
    feature_cols = get_feature_cols(df, mode)
    print(f"Using {len(feature_cols)} features (mode={mode.value})")
    
    # Apply decision lag if specified (doesn't drop rows)
    if decision_lag > 0:
        df = apply_decision_lag(df, feature_cols, decision_lag)
        print(f"Applied decision lag of {decision_lag} day(s)")
    
    # Split data
    train_df, val_df, test_df = split_train_val_test_by_date(df)
    
    # Drop NaN per split (not globally)
    train_df = drop_na_for_task(train_df, feature_cols, label_col)
    val_df = drop_na_for_task(val_df, feature_cols, label_col)
    test_df = drop_na_for_task(test_df, feature_cols, label_col)
    
    print(f"Train: {train_df.index.min().date()} → {train_df.index.max().date()}  ({len(train_df)} rows)")
    print(f"Val:   {val_df.index.min().date()}   → {val_df.index.max().date()}    ({len(val_df)} rows)")
    print(f"Test:  {test_df.index.min().date()}  → {test_df.index.max().date()}   ({len(test_df)} rows)")
    
    # Print label stats to understand base rates
    print_label_stats(train_df, val_df, test_df, label_col)

    X_train, y_train = split_X_y(train_df, feature_cols, label_col)
    X_val, y_val = split_X_y(val_df, feature_cols, label_col)
    X_test, y_test = split_X_y(test_df, feature_cols, label_col)

    # Train on TRAIN + VAL
    X_train_val = pd.concat([X_train, X_val])
    y_train_val = pd.concat([y_train, y_val])

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=20,
        class_weight='balanced',  # Correct for label imbalance
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train_val, y_train_val)

    # Get predictions
    y_val_proba = model.predict_proba(X_val)[:, 1]
    y_test_proba = model.predict_proba(X_test)[:, 1]
    
    # Print metrics
    val_auc, test_auc = print_model_metrics("SHORT", y_val, y_val_proba, y_test, y_test_proba)

    # Save model with correct feature_cols
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_names": feature_cols,  # Now computed from full df
            "feature_mode": mode.value,
            "decision_lag": decision_lag,
            "val_auc": float(val_auc),
            "test_auc": float(test_auc),
        },
        model_out,
    )
    print(f"\n[SHORT] Saved model to {model_out}")


# ============================================================================
# WALK-FORWARD VALIDATION
# ============================================================================

def get_walk_forward_folds(val_months: int = 6):
    """
    Generate walk-forward validation folds.
    
    Each fold: train up to cutoff, validate on next `val_months` months.
    Test (2025) is never touched during validation.
    
    Returns list of (train_end, val_start, val_end) date tuples.
    """
    folds = [
        ("2023-03-31", "2023-04-01", "2023-09-30"),  # Fold 1
        ("2023-06-30", "2023-07-01", "2023-12-31"),  # Fold 2
        ("2023-09-30", "2023-10-01", "2024-03-31"),  # Fold 3
        ("2023-12-31", "2024-01-01", "2024-06-30"),  # Fold 4
        ("2024-03-31", "2024-04-01", "2024-09-30"),  # Fold 5
        ("2024-06-30", "2024-07-01", "2024-12-31"),  # Fold 6
    ]
    return folds


def evaluate_fold(df: pd.DataFrame, feature_cols: list[str], label_col: str,
                  train_end: str, val_start: str, val_end: str) -> dict:
    """
    Train and evaluate a single walk-forward fold.
    
    Returns metrics dict for this fold.
    """
    # Split data
    train_df = df.loc[:train_end]
    val_df = df.loc[val_start:val_end]
    
    # Drop NaN per split
    train_df = drop_na_for_task(train_df, feature_cols, label_col)
    val_df = drop_na_for_task(val_df, feature_cols, label_col)
    
    if len(train_df) == 0 or len(val_df) == 0:
        return None
    
    X_train, y_train = split_X_y(train_df, feature_cols, label_col)
    X_val, y_val = split_X_y(val_df, feature_cols, label_col)
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=20,
        class_weight='balanced',  # Correct for label imbalance
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train, y_train)
    
    # Predict
    y_val_proba = model.predict_proba(X_val)[:, 1]
    
    # Compute metrics
    auc = roc_auc_score(y_val, y_val_proba)
    baseline = y_val.mean()
    
    metrics = {
        'train_end': train_end,
        'val_start': val_start,
        'val_end': val_end,
        'n_train': len(train_df),
        'n_val': len(val_df),
        'baseline': baseline,
        'auc': auc,
    }
    
    # Top-k precision
    for k in [1, 2, 5, 10]:
        prec = compute_top_k_precision(y_val, y_val_proba, k)
        lift = prec / baseline if baseline > 0 else 0
        metrics[f'top_{k}_prec'] = prec
        metrics[f'top_{k}_lift'] = lift
    
    return metrics


def run_walk_forward_validation(
    csv_path: Path,
    label_col: str,
    mode: FeatureMode = FeatureMode.HYBRID,
    decision_lag: int = 1,
) -> list[dict]:
    """
    Run walk-forward validation across multiple folds.
    
    Returns list of metrics dicts, one per fold.
    """
    print(f"\n{'='*60}")
    print(f"WALK-FORWARD VALIDATION | {label_col} | mode={mode.value} | lag={decision_lag}")
    print('='*60)
    
    # Load and prepare data
    df = load_feature_dataset(csv_path)
    feature_cols = get_feature_cols(df, mode)
    
    if decision_lag > 0:
        df = apply_decision_lag(df, feature_cols, decision_lag)
    
    print(f"Using {len(feature_cols)} features")
    print(f"Data range: {df.index.min().date()} → {df.index.max().date()}")
    
    # Get folds
    folds = get_walk_forward_folds()
    all_metrics = []
    
    print(f"\nRunning {len(folds)} folds...")
    print("-" * 80)
    
    for i, (train_end, val_start, val_end) in enumerate(folds, 1):
        metrics = evaluate_fold(df, feature_cols, label_col, train_end, val_start, val_end)
        
        if metrics is None:
            print(f"  Fold {i}: SKIPPED (no data)")
            continue
        
        all_metrics.append(metrics)
        
        # Print fold summary
        print(f"  Fold {i}: Train→{train_end} | Val[{val_start}→{val_end}] | "
              f"n={metrics['n_val']:3d} | base={metrics['baseline']:.3f} | "
              f"AUC={metrics['auc']:.3f} | Top5%={metrics['top_5_prec']:.3f} (lift={metrics['top_5_lift']:.2f}x)")
    
    # Compute averages
    if all_metrics:
        print("-" * 80)
        print("\n--- AVERAGE ACROSS FOLDS ---")
        
        avg_auc = np.mean([m['auc'] for m in all_metrics])
        avg_baseline = np.mean([m['baseline'] for m in all_metrics])
        
        print(f"  Avg Baseline: {avg_baseline:.3f}")
        print(f"  Avg AUC:      {avg_auc:.3f}")
        
        for k in [1, 2, 5, 10]:
            avg_prec = np.mean([m[f'top_{k}_prec'] for m in all_metrics])
            avg_lift = np.mean([m[f'top_{k}_lift'] for m in all_metrics])
            std_lift = np.std([m[f'top_{k}_lift'] for m in all_metrics])
            print(f"  Avg Top {k:2d}%: prec={avg_prec:.3f} lift={avg_lift:.2f}x (±{std_lift:.2f})")
    
    return all_metrics


def train_with_walk_forward(
    csv_path: Path,
    model_out: Path,
    label_col: str,
    mode: FeatureMode = FeatureMode.HYBRID,
    decision_lag: int = 1,
) -> None:
    """
    Full walk-forward training pipeline:
    1. Run walk-forward validation to assess stability
    2. Train final model on all pre-2025 data
    3. Evaluate once on locked 2025 test set
    4. Save model
    """
    name = "LONG" if "long" in label_col else "SHORT"
    
    # Step 1: Walk-forward validation
    fold_metrics = run_walk_forward_validation(csv_path, label_col, mode, decision_lag)
    
    # Step 2: Final training on all pre-2025 data
    print(f"\n{'='*60}")
    print(f"FINAL MODEL | {name} | Train→2024-12-31 | Test=2025")
    print('='*60)
    
    df = load_feature_dataset(csv_path)
    feature_cols = get_feature_cols(df, mode)
    
    if decision_lag > 0:
        df = apply_decision_lag(df, feature_cols, decision_lag)
    
    # Train on everything up to 2024
    train_df = df.loc[:"2024-12-31"]
    test_df = df.loc["2025-01-01":]
    
    train_df = drop_na_for_task(train_df, feature_cols, label_col)
    test_df = drop_na_for_task(test_df, feature_cols, label_col)
    
    print(f"Final Train: {train_df.index.min().date()} → {train_df.index.max().date()} ({len(train_df)} rows)")
    print(f"Final Test:  {test_df.index.min().date()} → {test_df.index.max().date()} ({len(test_df)} rows)")
    
    X_train, y_train = split_X_y(train_df, feature_cols, label_col)
    X_test, y_test = split_X_y(test_df, feature_cols, label_col)
    
    # Train final model
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=20,
        class_weight='balanced',  # Correct for label imbalance
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train, y_train)
    
    # Evaluate on test
    y_test_proba = model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_test_proba)
    test_baseline = y_test.mean()
    
    print(f"\n--- FINAL TEST METRICS ({name}) ---")
    print(f"  Baseline: {test_baseline:.3f}")
    print(f"  AUC:      {test_auc:.3f}")
    
    for k in [1, 2, 5, 10]:
        prec = compute_top_k_precision(y_test, y_test_proba, k)
        lift = prec / test_baseline if test_baseline > 0 else 0
        print(f"  Top {k:2d}%:  prec={prec:.3f} lift={lift:.2f}x")
    
    # Compute average validation metrics for comparison
    if fold_metrics:
        avg_val_auc = np.mean([m['auc'] for m in fold_metrics])
        avg_val_lift = np.mean([m['top_5_lift'] for m in fold_metrics])
        print(f"\n  (Avg Val AUC={avg_val_auc:.3f} vs Test AUC={test_auc:.3f})")
        print(f"  (Avg Val Lift@5%={avg_val_lift:.2f}x vs Test Lift@5%={lift:.2f}x)")
    
    # Save model
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_names": feature_cols,
            "feature_mode": mode.value,
            "decision_lag": decision_lag,
            "test_auc": float(test_auc),
            "walk_forward_metrics": fold_metrics,
        },
        model_out,
    )
    print(f"\n[{name}] Saved model to {model_out}")

