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
- Probability calibration for meaningful thresholds
- Purge/embargo for walk-forward to prevent label leakage
- PR-AUC and coverage metrics
- Alternative feature selectors (mutual info) to avoid double-dipping
"""

from datetime import datetime
from pathlib import Path
from enum import Enum
from typing import Optional, Literal

import warnings

import joblib
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils.class_weight import compute_sample_weight

# Default label horizon (should match feature_builder.py)
DEFAULT_LABEL_HORIZON_DAYS = 14

# Purge/embargo should match label horizon to prevent leakage
DEFAULT_PURGE_DAYS = DEFAULT_LABEL_HORIZON_DAYS
DEFAULT_EMBARGO_DAYS = DEFAULT_LABEL_HORIZON_DAYS


class FeatureMode(Enum):
    """Feature selection modes"""
    PURE = "pure"      # Exclude all handcrafted signals and fusion features
    HYBRID = "hybrid"  # Include regime features, exclude trading signals


# Type aliases
ModelType = Literal["rf", "extra_trees", "hist_gb"]
SelectorMethod = Literal["rf", "mi"]


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
    Only includes numeric columns to avoid silent failures.
    """
    exclude_prefixes = PURE_EXCLUDE_PREFIXES if mode == FeatureMode.PURE else HYBRID_EXCLUDE_PREFIXES
    
    feature_cols = [
        c for c in df.columns
        if c not in LABEL_COLS 
        and not any(c.startswith(p) for p in exclude_prefixes)
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    
    return feature_cols


def select_top_features(
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    n_features: int = 50,
    method: SelectorMethod = "mi",
) -> list[str]:
    """
    Select top N features by importance score.
    
    Args:
        method: "mi" for mutual information (default, avoids RF double-dipping)
                "rf" for Random Forest importance (legacy)
    
    Uses mutual information by default to avoid RF "double-dipping" bias
    when the final model is also RF.
    """
    if method == "mi":
        # Mutual information - avoids inductive bias with final model
        mi_scores = mutual_info_classif(X_train, y_train, random_state=42, n_jobs=-1)
        importance_df = pd.DataFrame({
            'feature': X_train.columns,
            'importance': mi_scores
        }).sort_values('importance', ascending=False)
    else:
        # RF importance (legacy) - can double-dip if final model is RF
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=4,
            min_samples_leaf=30,
            class_weight='balanced',
            n_jobs=-1,
            random_state=42,
        )
        rf.fit(X_train, y_train)
        importance_df = pd.DataFrame({
            'feature': X_train.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
    
    # Return top N feature names
    top_features = importance_df.head(n_features)['feature'].tolist()
    return top_features


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


def compute_top_k_precision(
    y_true: np.ndarray, 
    y_proba: np.ndarray, 
    k_pct: float, 
    min_k: int = 25
) -> tuple[float, int]:
    """
    Compute precision for top k% of predictions.
    
    This is more relevant for sparse trading than overall AUC.
    Uses min_k to ensure stable metrics on small datasets.
    
    Returns:
        tuple of (precision, k_actual) - k_actual shows if min_k was triggered
    """
    n = len(y_true)
    k_calculated = int(n * k_pct / 100)
    k_actual = max(min_k, k_calculated)  # At least min_k samples
    k_actual = min(k_actual, n)  # But not more than we have
    
    # Get indices of top k predictions
    top_k_idx = np.argsort(y_proba)[-k_actual:]
    
    # Compute precision on top k
    top_k_true = y_true.iloc[top_k_idx] if hasattr(y_true, 'iloc') else y_true[top_k_idx]
    precision = float(top_k_true.mean())  # Use mean for clarity
    
    return precision, k_actual


def compute_threshold_metrics(y_true: pd.Series, y_proba: np.ndarray) -> dict:
    """
    Compute metrics at various probability thresholds.
    
    Returns dict with coverage (% of days signaled) and precision at each threshold.
    """
    metrics = {}
    for thresh in [0.5, 0.6, 0.7]:
        mask = y_proba >= thresh
        n_signals = mask.sum()
        coverage = n_signals / len(y_true)
        
        if n_signals > 0:
            hits = y_true[mask].sum()
            precision = hits / n_signals
        else:
            hits = 0
            precision = 0.0
        
        metrics[str(thresh)] = {
            'n_signals': int(n_signals),
            'coverage': float(coverage),
            'hits': int(hits),
            'precision': float(precision),
        }
    return metrics


def compute_comprehensive_metrics(
    y_true: pd.Series, 
    y_proba: np.ndarray,
    prefix: str = ""
) -> dict:
    """Compute all metrics for a prediction set."""
    baseline = y_true.mean()
    
    metrics = {
        f'{prefix}baseline': baseline,
        f'{prefix}auc': roc_auc_score(y_true, y_proba),
        f'{prefix}pr_auc': average_precision_score(y_true, y_proba),
    }
    
    # Top-k precision with actual k
    for k_pct in [1, 2, 5, 10]:
        prec, k_actual = compute_top_k_precision(y_true, y_proba, k_pct)
        lift = prec / baseline if baseline > 0 else 0
        metrics[f'{prefix}top_{k_pct}_prec'] = prec
        metrics[f'{prefix}top_{k_pct}_lift'] = lift
        metrics[f'{prefix}top_{k_pct}_k_actual'] = k_actual
    
    # Coverage at thresholds
    thresh_metrics = compute_threshold_metrics(y_true, y_proba)
    for thresh, m in thresh_metrics.items():
        metrics[f'{prefix}thresh_{thresh}_coverage'] = m['coverage']
        metrics[f'{prefix}thresh_{thresh}_precision'] = m['precision']
    
    return metrics


def print_model_metrics(name: str, y_val: pd.Series, y_val_proba: np.ndarray,
                        y_test: pd.Series, y_test_proba: np.ndarray):
    """Print comprehensive model metrics including top-k precision with baseline"""
    val_auc = roc_auc_score(y_val, y_val_proba)
    test_auc = roc_auc_score(y_test, y_test_proba)
    
    # PR-AUC (more honest for imbalanced data)
    val_pr_auc = average_precision_score(y_val, y_val_proba)
    test_pr_auc = average_precision_score(y_test, y_test_proba)
    
    # Baseline = just the positive rate (random selection precision)
    val_baseline = y_val.mean()
    test_baseline = y_test.mean()
    
    print(f"\n[{name}] === METRICS ===")
    print(f"  Baseline (base rate):  Val={val_baseline:.3f}  Test={test_baseline:.3f}")
    print(f"  AUC:                   Val={val_auc:.3f}  Test={test_auc:.3f}")
    print(f"  PR-AUC:                Val={val_pr_auc:.3f}  Test={test_pr_auc:.3f}")
    
    # Top-k precision with baseline comparison
    print(f"\n  --- Top-K Precision (vs baseline) ---")
    for k in [1, 2, 5, 10]:
        val_prec, val_k = compute_top_k_precision(y_val, y_val_proba, k)
        test_prec, test_k = compute_top_k_precision(y_test, y_test_proba, k)
        # Lift = precision / baseline
        val_lift = val_prec / val_baseline if val_baseline > 0 else 0
        test_lift = test_prec / test_baseline if test_baseline > 0 else 0
        
        # Show if min_k was triggered
        val_note = f"k={val_k}" if val_k != int(len(y_val) * k / 100) else ""
        test_note = f"k={test_k}" if test_k != int(len(y_test) * k / 100) else ""
        
        print(f"  Top {k:2d}%: Val={val_prec:.3f} (lift={val_lift:.2f}x{val_note})  Test={test_prec:.3f} (lift={test_lift:.2f}x{test_note})")
    
    # Hit rate at high probability threshold
    print(f"\n  --- Threshold Hit Rates (coverage + precision) ---")
    for thresh in [0.5, 0.6, 0.7]:
        val_signals = (y_val_proba >= thresh).sum()
        val_hits = (y_val[y_val_proba >= thresh]).sum()
        val_coverage = val_signals / len(y_val)
        
        test_signals = (y_test_proba >= thresh).sum()
        test_hits = (y_test[y_test_proba >= thresh]).sum()
        test_coverage = test_signals / len(y_test)
        
        val_rate = val_hits / val_signals if val_signals > 0 else 0
        test_rate = test_hits / test_signals if test_signals > 0 else 0
        
        print(f"  P>={thresh}: Val={val_rate:.2f} ({val_hits:.0f}/{val_signals}, cov={val_coverage:.1%})  "
              f"Test={test_rate:.2f} ({test_hits:.0f}/{test_signals}, cov={test_coverage:.1%})")
    
    return val_auc, test_auc


def get_model(model_type: ModelType = "rf", **params) -> object:
    """
    Create a classifier based on model type.
    
    Args:
        model_type: "rf" (RandomForest), "extra_trees", or "hist_gb"
    """
    common_params = {
        'n_jobs': -1,
        'random_state': 42,
    }
    
    if model_type == "rf":
        return RandomForestClassifier(
            n_estimators=params.get('n_estimators', 300),
            max_depth=params.get('max_depth', 6),
            min_samples_leaf=params.get('min_samples_leaf', 20),
            class_weight='balanced',
            **common_params,
        )
    elif model_type == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=params.get('n_estimators', 300),
            max_depth=params.get('max_depth', 6),
            min_samples_leaf=params.get('min_samples_leaf', 20),
            class_weight='balanced',
            **common_params,
        )
    elif model_type == "hist_gb":
        # HistGradientBoosting doesn't support class_weight - use sample_weight in fit()
        return HistGradientBoostingClassifier(
            max_iter=params.get('n_estimators', 300),
            max_depth=params.get('max_depth', 6),
            min_samples_leaf=params.get('min_samples_leaf', 20),
            random_state=42,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def fit_model(model, X, y, model_type: ModelType = "rf"):
    """
    Fit model with appropriate handling for each model type.
    
    For HistGradientBoosting, computes sample_weight since it doesn't support class_weight.
    """
    if model_type == "hist_gb":
        sample_weight = compute_sample_weight(class_weight="balanced", y=y)
        model.fit(X, y, sample_weight=sample_weight)
    else:
        model.fit(X, y)
    return model


def get_calibration_method(n_samples: int) -> str:
    """
    Choose calibration method based on sample size.
    
    - sigmoid (Platt scaling): better for small samples, less prone to overfitting
    - isotonic: more flexible but needs more data (1500+ samples)
    """
    return "sigmoid" if n_samples < 1500 else "isotonic"


def get_time_series_cv(n_splits: int = 5, gap: int = DEFAULT_LABEL_HORIZON_DAYS) -> tuple:
    """
    Create TimeSeriesSplit with gap for calibration CV.
    
    Falls back to no-gap if sklearn version doesn't support it.
    
    Returns:
        tuple of (TimeSeriesSplit, gap_used) where gap_used is actual gap or 0 if fallback
    """
    try:
        return TimeSeriesSplit(n_splits=n_splits, gap=gap), gap
    except TypeError:
        print(f"  WARNING: TimeSeriesSplit(gap=...) not supported; using no-gap CV.")
        return TimeSeriesSplit(n_splits=n_splits), 0


# ============================================================================
# UNIFIED BINARY MODEL TRAINING (DRY - replaces separate long/short functions)
# ============================================================================

def train_binary_model_with_holdout(
    csv_path: Path,
    model_out: Path,
    label_col: str,
    mode: FeatureMode = FeatureMode.HYBRID,
    decision_lag: int = 0,
    calibrate: bool = True,
    model_type: ModelType = "rf",
) -> None:
    """
    Train a binary prediction model with proper holdout validation.
    
    This is the unified function for both LONG and SHORT models.
    
    Args:
        label_col: "label_good_move_long" or "label_good_move_short"
        calibrate: If True, calibrate probabilities on val set
        model_type: "rf", "extra_trees", or "hist_gb"
    """
    name = "LONG" if "long" in label_col else "SHORT"
    
    print(f"\n{'='*60}")
    print(f"Training {name} model | mode={mode.value} | lag={decision_lag} | calibrate={calibrate}")
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

    # PHASE 1: Train on TRAIN only, evaluate on VAL (true holdout)
    print(f"\n  [Phase 1] Training {model_type} on TRAIN, evaluating on VAL (true holdout)...")
    model_val = get_model(model_type)
    fit_model(model_val, X_train, y_train, model_type)
    
    # Get raw probabilities
    y_val_proba_raw = model_val.predict_proba(X_val)[:, 1]
    
    # Calibrate if requested - choose method based on sample size
    calibration_method = None
    if calibrate:
        cal_method = get_calibration_method(len(y_val))
        print(f"  [Phase 1] Calibrating probabilities on VAL ({cal_method}, n={len(y_val)})...")
        # Use cv='prefit' with warning suppression (FrozenEstimator not available in all versions)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*cv='prefit'.*deprecated.*")
            calibrator = CalibratedClassifierCV(model_val, method=cal_method, cv="prefit")
            calibrator.fit(X_val, y_val)
        y_val_proba = calibrator.predict_proba(X_val)[:, 1]
        calibration_method = cal_method
        
        # Show calibration effect
        print(f"    Raw proba range: [{y_val_proba_raw.min():.3f}, {y_val_proba_raw.max():.3f}]")
        print(f"    Calibrated range: [{y_val_proba.min():.3f}, {y_val_proba.max():.3f}]")
    else:
        y_val_proba = y_val_proba_raw
    
    # PHASE 2: Retrain on TRAIN+VAL, evaluate on TEST (final model)
    # CRITICAL: Must also calibrate Phase-2 model for TEST thresholds to be meaningful
    print(f"  [Phase 2] Retraining {model_type} on TRAIN+VAL, evaluating on TEST...")
    X_train_val = pd.concat([X_train, X_val])
    y_train_val = pd.concat([y_train, y_val])
    
    model_final = get_model(model_type)
    fit_model(model_final, X_train_val, y_train_val, model_type)
    
    # Calibrate Phase-2 using TimeSeriesSplit CV on TRAIN+VAL (time-aware with gap, no leakage)
    if calibrate:
        cal_method = get_calibration_method(len(y_train_val))
        tscv, gap_used = get_time_series_cv(n_splits=5, gap=DEFAULT_LABEL_HORIZON_DAYS)
        print(f"  [Phase 2] Calibrating with TimeSeriesSplit(5, gap={gap_used}) on TRAIN+VAL ({cal_method})...")
        calibrated_final = CalibratedClassifierCV(
            model_final, 
            method=cal_method, 
            cv=tscv
        )
        # Need to fit HGB with sample_weight if applicable
        if model_type == "hist_gb":
            sample_weight = compute_sample_weight(class_weight="balanced", y=y_train_val)
            calibrated_final.fit(X_train_val, y_train_val, sample_weight=sample_weight)
        else:
            calibrated_final.fit(X_train_val, y_train_val)
        y_test_proba = calibrated_final.predict_proba(X_test)[:, 1]
        model_to_save = calibrated_final
        calibration_method = f"{cal_method}_tscv5_gap{gap_used}"
    else:
        y_test_proba = model_final.predict_proba(X_test)[:, 1]
        model_to_save = model_final
    
    # Print metrics (now both VAL and TEST are comparable!)
    val_auc, test_auc = print_model_metrics(name, y_val, y_val_proba, y_test, y_test_proba)
    
    # Compute feature importance for saving
    if hasattr(model_final, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': model_final.feature_importances_
        }).sort_values('importance', ascending=False)
    else:
        importance_df = pd.DataFrame({'feature': feature_cols, 'importance': [0] * len(feature_cols)})
    
    # Compute quantile cutoffs from TRAIN+VAL (not TEST - avoid test leakage)
    y_trainval_proba = model_to_save.predict_proba(X_train_val)[:, 1]
    quantiles = {
        'q90': float(np.quantile(y_trainval_proba, 0.90)),
        'q95': float(np.quantile(y_trainval_proba, 0.95)),
        'q99': float(np.quantile(y_trainval_proba, 0.99)),
    }

    # Save FINAL model (trained on TRAIN+VAL, calibrated) with rich artifacts
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model_to_save,
            "feature_names": feature_cols,
            "feature_mode": mode.value,
            "decision_lag": decision_lag,
            "model_type": model_type,
            "calibration_method": calibration_method,
            "val_auc": float(val_auc),
            "test_auc": float(test_auc),
            # Rich artifacts for debugging
            "train_base_rate": float(y_train.mean()),
            "val_base_rate": float(y_val.mean()),
            "test_base_rate": float(y_test.mean()),
            "thresholds_summary": compute_threshold_metrics(y_test, y_test_proba),
            "proba_quantiles": quantiles,  # For production top-K cutoffs
            "feature_importance": importance_df.head(50).to_dict('records'),
            "trained_at": datetime.now().isoformat(),
        },
        model_out,
    )
    print(f"\n[{name}] Saved model to {model_out}")


# Convenience wrappers for backward compatibility
def train_long_model_with_holdout(
    csv_path: Path,
    model_out: Path,
    mode: FeatureMode = FeatureMode.HYBRID,
    decision_lag: int = 0,
) -> None:
    """Train long prediction model (wrapper for backward compatibility)"""
    train_binary_model_with_holdout(
        csv_path=csv_path,
        model_out=model_out,
        label_col="label_good_move_long",
        mode=mode,
        decision_lag=decision_lag,
        calibrate=True,
    )


def train_short_model_with_holdout(
    csv_path: Path,
    model_out: Path,
    mode: FeatureMode = FeatureMode.HYBRID,
    decision_lag: int = 0,
) -> None:
    """Train short prediction model (wrapper for backward compatibility)"""
    train_binary_model_with_holdout(
        csv_path=csv_path,
        model_out=model_out,
        label_col="label_good_move_short",
        mode=mode,
        decision_lag=decision_lag,
        calibrate=True,
    )


# ============================================================================
# WALK-FORWARD VALIDATION WITH PURGE/EMBARGO
# ============================================================================

def get_walk_forward_folds(
    start_date: str = "2021-01-01",
    end_date: str = "2024-12-31",
    train_step_months: int = 3,
    val_months: int = 6,
) -> list[tuple[str, str, str]]:
    """
    Generate walk-forward validation folds dynamically.
    
    Each fold: train up to cutoff, validate on next `val_months` months.
    Test (2025) is never touched during validation.
    
    Args:
        start_date: Start of first validation window
        end_date: Don't create folds validating past this date
        train_step_months: Roll training forward by this many months each fold
        val_months: Length of validation window
    
    Returns list of (train_end, val_start, val_end) date tuples.
    """
    folds = []
    current = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    
    while current + relativedelta(months=val_months) <= end:
        train_end = current - relativedelta(days=1)  # Day before val starts
        val_start = current
        val_end = current + relativedelta(months=val_months) - relativedelta(days=1)
        
        # Don't exceed end_date
        if val_end > end:
            val_end = end
        
        folds.append((
            train_end.strftime("%Y-%m-%d"),
            val_start.strftime("%Y-%m-%d"),
            val_end.strftime("%Y-%m-%d"),
        ))
        
        current += relativedelta(months=train_step_months)
    
    return folds


def evaluate_fold(
    df: pd.DataFrame, 
    feature_cols: list[str], 
    label_col: str,
    train_end: str, 
    val_start: str, 
    val_end: str,
    n_features: Optional[int] = None,
    purge_days: int = DEFAULT_PURGE_DAYS,
    embargo_days: int = DEFAULT_EMBARGO_DAYS,
    model_type: ModelType = "rf",
    selector_method: SelectorMethod = "mi",
) -> dict:
    """
    Train and evaluate a single walk-forward fold with purge/embargo.
    
    Args:
        n_features: If set, select top N features on training data before fitting.
        purge_days: Remove last N days of training to prevent label overlap with val.
                    Should match label horizon (default: DEFAULT_LABEL_HORIZON_DAYS).
        embargo_days: Skip first N days of validation after train end.
        model_type: Type of classifier to use.
        selector_method: "mi" for mutual info, "rf" for RF importance.
    
    Returns metrics dict for this fold.
    """
    # Apply purge: remove last purge_days from training
    train_end_ts = pd.Timestamp(train_end)
    purge_cutoff = train_end_ts - pd.Timedelta(days=purge_days)
    train_df = df.loc[:purge_cutoff]
    
    # Apply embargo: skip first embargo_days of validation
    val_start_ts = pd.Timestamp(val_start)
    embargo_start = val_start_ts + pd.Timedelta(days=embargo_days)
    val_df = df.loc[embargo_start:val_end]
    
    # Drop NaN per split
    train_df = drop_na_for_task(train_df, feature_cols, label_col)
    val_df = drop_na_for_task(val_df, feature_cols, label_col)
    
    if len(train_df) == 0 or len(val_df) == 0:
        return None
    
    X_train, y_train = split_X_y(train_df, feature_cols, label_col)
    X_val, y_val = split_X_y(val_df, feature_cols, label_col)
    
    # Optional feature selection
    selected_cols = feature_cols
    if n_features and n_features < len(feature_cols):
        selected_cols = select_top_features(X_train, y_train, n_features, method=selector_method)
        X_train = X_train[selected_cols]
        X_val = X_val[selected_cols]
    
    # Train model
    model = get_model(model_type)
    fit_model(model, X_train, y_train, model_type)
    
    # Predict
    y_val_proba = model.predict_proba(X_val)[:, 1]
    
    # Compute metrics
    auc = roc_auc_score(y_val, y_val_proba)
    pr_auc = average_precision_score(y_val, y_val_proba)
    baseline = y_val.mean()
    
    metrics = {
        'train_end': train_end,
        'purge_cutoff': purge_cutoff.strftime("%Y-%m-%d"),
        'val_start': val_start,
        'val_end': val_end,
        'n_train': len(train_df),
        'n_val': len(val_df),
        'n_features': len(selected_cols),
        'baseline': baseline,
        'auc': auc,
        'pr_auc': pr_auc,
    }
    
    # Top-k precision with actual k
    for k in [1, 2, 5, 10]:
        prec, k_actual = compute_top_k_precision(y_val, y_val_proba, k)
        lift = prec / baseline if baseline > 0 else 0
        metrics[f'top_{k}_prec'] = prec
        metrics[f'top_{k}_lift'] = lift
        metrics[f'top_{k}_k_actual'] = k_actual
    
    # Coverage at thresholds
    for thresh in [0.5, 0.6, 0.7]:
        coverage = (y_val_proba >= thresh).mean()
        metrics[f'coverage_{thresh}'] = coverage
    
    return metrics


def run_walk_forward_validation(
    csv_path: Path,
    label_col: str,
    mode: FeatureMode = FeatureMode.HYBRID,
    decision_lag: int = 1,
    n_features: Optional[int] = None,
    purge_days: int = DEFAULT_PURGE_DAYS,
    embargo_days: int = DEFAULT_EMBARGO_DAYS,
    model_type: ModelType = "rf",
    selector_method: SelectorMethod = "mi",
) -> list[dict]:
    """
    Run walk-forward validation across multiple folds with purge/embargo.
    
    Args:
        n_features: If set, select top N features per fold to reduce overfitting.
        purge_days: Remove last N days of training to prevent label overlap.
        embargo_days: Skip first N days of validation.
        model_type: Type of classifier.
        selector_method: "mi" or "rf" for feature selection.
    
    Returns list of metrics dicts, one per fold.
    """
    feat_str = f"top {n_features}" if n_features else "all"
    print(f"\n{'='*60}")
    print(f"WALK-FORWARD VALIDATION | {label_col}")
    print(f"  mode={mode.value} | lag={decision_lag} | features={feat_str}")
    print(f"  purge={purge_days}d | embargo={embargo_days}d | model={model_type} | selector={selector_method}")
    print('='*60)
    
    # Load and prepare data
    df = load_feature_dataset(csv_path)
    feature_cols = get_feature_cols(df, mode)
    
    if decision_lag > 0:
        df = apply_decision_lag(df, feature_cols, decision_lag)
    
    print(f"Available features: {len(feature_cols)} (using {n_features or 'all'})")
    print(f"Data range: {df.index.min().date()} → {df.index.max().date()}")
    
    # Get folds dynamically
    folds = get_walk_forward_folds()
    all_metrics = []
    
    print(f"\nRunning {len(folds)} folds (purge={purge_days}d)...")
    print("-" * 100)
    
    for i, (train_end, val_start, val_end) in enumerate(folds, 1):
        metrics = evaluate_fold(
            df, feature_cols, label_col, 
            train_end, val_start, val_end, 
            n_features,
            purge_days=purge_days,
            embargo_days=embargo_days,
            model_type=model_type,
            selector_method=selector_method,
        )
        
        if metrics is None:
            print(f"  Fold {i}: SKIPPED (no data)")
            continue
        
        all_metrics.append(metrics)
        
        # Print fold summary with purge info
        print(f"  Fold {i}: Train→{metrics['purge_cutoff']} (purged) | Val[{val_start}→{val_end}] | "
              f"n={metrics['n_val']:3d} | base={metrics['baseline']:.3f} | "
              f"AUC={metrics['auc']:.3f} | PR-AUC={metrics['pr_auc']:.3f} | Top5%={metrics['top_5_prec']:.3f} (lift={metrics['top_5_lift']:.2f}x)")
    
    # Compute averages
    if all_metrics:
        print("-" * 100)
        print("\n--- AVERAGE ACROSS FOLDS ---")
        
        avg_auc = np.mean([m['auc'] for m in all_metrics])
        std_auc = np.std([m['auc'] for m in all_metrics])
        avg_pr_auc = np.mean([m['pr_auc'] for m in all_metrics])
        avg_baseline = np.mean([m['baseline'] for m in all_metrics])
        
        print(f"  Avg Baseline: {avg_baseline:.3f}")
        print(f"  Avg AUC:      {avg_auc:.3f} (±{std_auc:.3f})")
        print(f"  Avg PR-AUC:   {avg_pr_auc:.3f}")
        
        for k in [1, 2, 5, 10]:
            avg_prec = np.mean([m[f'top_{k}_prec'] for m in all_metrics])
            avg_lift = np.mean([m[f'top_{k}_lift'] for m in all_metrics])
            std_lift = np.std([m[f'top_{k}_lift'] for m in all_metrics])
            print(f"  Avg Top {k:2d}%: prec={avg_prec:.3f} lift={avg_lift:.2f}x (±{std_lift:.2f})")
        
        # Coverage summary
        print("\n  --- Avg Coverage @ Thresholds ---")
        for thresh in [0.5, 0.6, 0.7]:
            avg_cov = np.mean([m[f'coverage_{thresh}'] for m in all_metrics])
            print(f"  P>={thresh}: {avg_cov:.1%} of days signaled")
    
    return all_metrics


def train_with_walk_forward(
    csv_path: Path,
    model_out: Path,
    label_col: str,
    mode: FeatureMode = FeatureMode.HYBRID,
    decision_lag: int = 1,
    n_features: Optional[int] = None,
    purge_days: int = DEFAULT_PURGE_DAYS,
    calibrate: bool = True,
    model_type: ModelType = "rf",
    selector_method: SelectorMethod = "mi",
) -> None:
    """
    Full walk-forward training pipeline:
    1. Run walk-forward validation to assess stability
    2. Train final model on all pre-2025 data
    3. Evaluate once on locked 2025 test set
    4. Save model with rich artifacts
    
    Args:
        n_features: If set, select top N features to reduce overfitting.
        purge_days: Days to purge from training in walk-forward.
        calibrate: If True, calibrate probabilities.
        model_type: Type of classifier.
        selector_method: Feature selection method.
    """
    name = "LONG" if "long" in label_col else "SHORT"
    
    # Step 1: Walk-forward validation
    fold_metrics = run_walk_forward_validation(
        csv_path, label_col, mode, decision_lag, n_features,
        purge_days=purge_days,
        model_type=model_type,
        selector_method=selector_method,
    )
    
    # Step 2: Final training on all pre-2025 data
    print(f"\n{'='*60}")
    print(f"FINAL MODEL | {name} | Train→2024-12-31 | Test=2025")
    print(f"  calibrate={calibrate} | model={model_type}")
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
    
    # Optional feature selection on full training set
    selected_cols = feature_cols
    if n_features and n_features < len(feature_cols):
        print(f"Selecting top {n_features} features using {selector_method}...")
        selected_cols = select_top_features(X_train, y_train, n_features, method=selector_method)
        X_train = X_train[selected_cols]
        X_test = X_test[selected_cols]
        feature_cols = selected_cols  # Update for saving
    
    # Train final model
    model = get_model(model_type)
    model.fit(X_train, y_train)
    
    # Optionally calibrate using time-aware CV on training data
    calibration_method = None
    if calibrate:
        cal_method = get_calibration_method(len(y_train))
        tscv, gap_used = get_time_series_cv(n_splits=5, gap=DEFAULT_LABEL_HORIZON_DAYS)
        print(f"Calibrating with TimeSeriesSplit(5, gap={gap_used}) ({cal_method})...")
        calibrated_model = CalibratedClassifierCV(model, method=cal_method, cv=tscv)
        calibrated_model.fit(X_train, y_train)
        y_test_proba = calibrated_model.predict_proba(X_test)[:, 1]
        calibration_method = f"{cal_method}_tscv5_gap{gap_used}"
        model = calibrated_model  # Use calibrated model for saving
    else:
        y_test_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    test_auc = roc_auc_score(y_test, y_test_proba)
    test_pr_auc = average_precision_score(y_test, y_test_proba)
    test_baseline = y_test.mean()
    
    print(f"\n--- FINAL TEST METRICS ({name}) ---")
    print(f"  Baseline: {test_baseline:.3f}")
    print(f"  AUC:      {test_auc:.3f}")
    print(f"  PR-AUC:   {test_pr_auc:.3f}")
    
    for k in [1, 2, 5, 10]:
        prec, k_actual = compute_top_k_precision(y_test, y_test_proba, k)
        lift = prec / test_baseline if test_baseline > 0 else 0
        k_note = f" (k={k_actual})" if k_actual != int(len(y_test) * k / 100) else ""
        print(f"  Top {k:2d}%:  prec={prec:.3f} lift={lift:.2f}x{k_note}")
    
    # Threshold summary
    thresh_summary = compute_threshold_metrics(y_test, y_test_proba)
    print(f"\n  --- Threshold Summary ---")
    for thresh, m in thresh_summary.items():
        print(f"  P>={thresh}: precision={m['precision']:.2f} ({m['hits']}/{m['n_signals']}, coverage={m['coverage']:.1%})")
    
    # Compute average validation metrics for comparison
    if fold_metrics:
        avg_val_auc = np.mean([m['auc'] for m in fold_metrics])
        avg_val_lift = np.mean([m['top_5_lift'] for m in fold_metrics])
        print(f"\n  (Avg Val AUC={avg_val_auc:.3f} vs Test AUC={test_auc:.3f})")
        print(f"  (Avg Val Lift@5%={avg_val_lift:.2f}x vs Test Lift@5%={lift:.2f}x)")
    
    # Feature importance
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
    elif hasattr(model, 'calibrated_classifiers_') and hasattr(model.calibrated_classifiers_[0].estimator, 'feature_importances_'):
        # For calibrated models, get importance from base estimator
        avg_importance = np.mean([
            c.estimator.feature_importances_ 
            for c in model.calibrated_classifiers_
        ], axis=0)
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': avg_importance
        }).sort_values('importance', ascending=False)
    else:
        importance_df = pd.DataFrame({'feature': feature_cols, 'importance': [0] * len(feature_cols)})
    
    # Save model with rich artifacts
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_names": feature_cols,
            "feature_mode": mode.value,
            "decision_lag": decision_lag,
            "n_features": n_features,
            "model_type": model_type,
            "calibration_method": calibration_method,
            "selector_method": selector_method,
            "purge_days": purge_days,
            "test_auc": float(test_auc),
            "test_pr_auc": float(test_pr_auc),
            "walk_forward_metrics": fold_metrics,
            # Rich debugging artifacts
            "train_base_rate": float(y_train.mean()),
            "test_base_rate": float(test_baseline),
            "thresholds_summary": thresh_summary,
            "feature_importance": importance_df.head(50).to_dict('records'),
            "trained_at": datetime.now().isoformat(),
        },
        model_out,
    )
    print(f"\n[{name}] Saved model to {model_out}")


# ============================================================================
# PRODUCTION TRAINING (all data, no holdout)
# ============================================================================

def train_production_model(
    csv_path: Path,
    model_out: Path,
    label_col: str,
    mode: FeatureMode = FeatureMode.HYBRID,
    decision_lag: int = 1,
    n_features: Optional[int] = None,
    model_type: ModelType = "rf",
    selector_method: SelectorMethod = "mi",
) -> None:
    """
    Train a production model on ALL available data.
    
    Use this for deployment after you've validated your approach
    with walk-forward testing. No holdout = use all information.
    """
    name = "LONG" if "long" in label_col else "SHORT"
    
    print(f"\n{'='*60}")
    print(f"PRODUCTION MODEL | {name} | Train on ALL data")
    print(f"  model={model_type} | selector={selector_method}")
    print('='*60)
    
    # Load and prepare data
    df = load_feature_dataset(csv_path)
    feature_cols = get_feature_cols(df, mode)
    
    if decision_lag > 0:
        df = apply_decision_lag(df, feature_cols, decision_lag)
    
    # Use ALL data for training
    df = drop_na_for_task(df, feature_cols, label_col)
    
    print(f"Training period: {df.index.min().date()} → {df.index.max().date()}")
    print(f"Total rows: {len(df)}")
    print(f"Positive rate: {df[label_col].mean():.3f}")
    print(f"Available features: {len(feature_cols)}")
    
    X, y = split_X_y(df, feature_cols, label_col)
    
    # Optional feature selection
    selected_cols = feature_cols
    if n_features and n_features < len(feature_cols):
        print(f"Selecting top {n_features} features using {selector_method}...")
        selected_cols = select_top_features(X, y, n_features, method=selector_method)
        X = X[selected_cols]
        feature_cols = selected_cols
        print(f"Using {len(feature_cols)} features")
    
    # Train final production model
    model = get_model(model_type)
    model.fit(X, y)
    
    # Feature importance summary
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
    else:
        importance_df = pd.DataFrame({'feature': feature_cols, 'importance': [0] * len(feature_cols)})
    
    print(f"\n--- Top 10 Features ---")
    for _, row in importance_df.head(10).iterrows():
        print(f"  {row['importance']:.4f}  {row['feature']}")
    
    # Save model with metadata
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_names": feature_cols,
            "feature_mode": mode.value,
            "decision_lag": decision_lag,
            "n_features": n_features,
            "model_type": model_type,
            "selector_method": selector_method,
            "training_mode": "production",
            "train_start": str(df.index.min().date()),
            "train_end": str(df.index.max().date()),
            "train_rows": len(df),
            "positive_rate": float(df[label_col].mean()),
            "trained_at": datetime.now().isoformat(),
            "feature_importances": importance_df.head(50).to_dict('records'),
        },
        model_out,
    )
    print(f"\n[{name}] Saved PRODUCTION model to {model_out}")
