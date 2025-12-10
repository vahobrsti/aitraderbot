# features/ml/training.py

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


def load_feature_dataset(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["date"]).set_index("date")
    df = df.sort_index()
    return df


def _split_X_y(df: pd.DataFrame, label_col: str):
    y = df[label_col]
    feature_cols = [
        c for c in df.columns
        if c not in ["label_good_move_long", "label_good_move_short"]
    ]
    X = df[feature_cols]
    return X, y, feature_cols


def split_train_val_test_by_date(df: pd.DataFrame):
    """
    Train:  up to 2023-12-31
    Val:    2024-01-01 to 2024-12-31
    Test:   2025-01-01 onwards
    """
    df = df.sort_index()

    train = df.loc[: "2024-06-30"]
    val   = df.loc["2024-07-01":"2024-12-31"]
    test  = df.loc["2025-01-01":]

    print(f"Train: {train.index.min().date()} → {train.index.max().date()}  ({len(train)} rows)")
    print(f"Val:   {val.index.min().date()}   → {val.index.max().date()}    ({len(val)} rows)")
    print(f"Test:  {test.index.min().date()}  → {test.index.max().date()}   ({len(test)} rows)")

    return train, val, test


def train_long_model_with_holdout(
    csv_path: Path,
    model_out: Path,
) -> None:
    df = load_feature_dataset(csv_path)
    train_df, val_df, test_df = split_train_val_test_by_date(df)

    X_train, y_train, feature_cols = _split_X_y(train_df, "label_good_move_long")
    X_val,   y_val,   _            = _split_X_y(val_df,   "label_good_move_long")
    X_test,  y_test,  _            = _split_X_y(test_df,  "label_good_move_long")

    # Train on TRAIN + VAL
    X_train_val = pd.concat([X_train, X_val])
    y_train_val = pd.concat([y_train, y_val])

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=20,
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train_val, y_train_val)

    val_auc  = roc_auc_score(y_val,  model.predict_proba(X_val)[:, 1])
    test_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    print(f"[LONG] Val AUC:  {val_auc:.3f}")
    print(f"[LONG] Test AUC: {test_auc:.3f}")

    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_names": feature_cols,
            "val_auc": float(val_auc),
            "test_auc": float(test_auc),
        },
        model_out,
    )
    print(f"[LONG] Saved model to {model_out}")


def train_short_model_with_holdout(
    csv_path: Path,
    model_out: Path,
) -> None:
    df = load_feature_dataset(csv_path)
    train_df, val_df, test_df = split_train_val_test_by_date(df)

    X_train, y_train, feature_cols = _split_X_y(train_df, "label_good_move_short")
    X_val,   y_val,   _            = _split_X_y(val_df,   "label_good_move_short")
    X_test,  y_test,  _            = _split_X_y(test_df,  "label_good_move_short")

    X_train_val = pd.concat([X_train, X_val])
    y_train_val = pd.concat([y_train, y_val])

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=20,
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train_val, y_train_val)

    val_auc  = roc_auc_score(y_val,  model.predict_proba(X_val)[:, 1])
    test_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    print(f"[SHORT] Val AUC:  {val_auc:.3f}")
    print(f"[SHORT] Test AUC: {test_auc:.3f}")

    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_names": feature_cols,
            "val_auc": float(val_auc),
            "test_auc": float(test_auc),
        },
        model_out,
    )
    print(f"[SHORT] Saved model to {model_out}")
