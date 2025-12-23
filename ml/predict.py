# ml/predict.py
"""
Thin prediction interface for the signals app.
This module exposes a stable API that signals can import without
coupling to the training internals.
"""

from pathlib import Path
from typing import Optional, Union

import joblib
import pandas as pd


class ModelBundle:
    """Wrapper for a loaded ML model with its feature names."""
    
    def __init__(self, model, feature_names: list):
        self.model = model
        self.feature_names = feature_names
    
    def predict_proba(self, df: pd.DataFrame) -> float:
        """Get probability for class 1 (positive) for a single row."""
        if len(df) != 1:
            raise ValueError("predict_proba expects a single-row DataFrame")
        return float(self.model.predict_proba(df[self.feature_names])[:, 1][0])
    
    def score_dataframe(self, df: pd.DataFrame) -> pd.Series:
        """Score entire dataframe, returning probability column."""
        return pd.Series(
            self.model.predict_proba(df[self.feature_names])[:, 1],
            index=df.index
        )


def load_model(path: Union[str, Path]) -> ModelBundle:
    """
    Load a trained model bundle from disk.
    
    Args:
        path: Path to the joblib file.
    
    Returns:
        ModelBundle with model and feature_names.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    
    bundle = joblib.load(path)
    return ModelBundle(
        model=bundle["model"],
        feature_names=bundle["feature_names"]
    )


# Convenience loaders for the standard models
def load_long_model(path: str = "models/long_model.joblib") -> ModelBundle:
    """Load the long prediction model."""
    return load_model(path)


def load_short_model(path: str = "models/short_model.joblib") -> ModelBundle:
    """Load the short prediction model."""
    return load_model(path)
