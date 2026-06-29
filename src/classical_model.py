"""
src/classical_model.py
Employee Feedback Intelligence Platform — classical ML model (LinearSVC + TF-IDF).
Loads pre-trained model saved from Notebook 03.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import List

import joblib
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

MODEL_PATH = Path("models/classical_model.pkl")


# ── Availability check ────────────────────────────────────────────────────────

def is_model_available(path: str = str(MODEL_PATH)) -> bool:
    return Path(path).exists()


# ── Load ──────────────────────────────────────────────────────────────────────

def load_model(path: str = str(MODEL_PATH)) -> Pipeline:
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Classical model not found at `{path}`. "
            "Run Notebook 03 and save the model first."
        )
    return joblib.load(path)


# ── Predict ───────────────────────────────────────────────────────────────────

def predict(pipeline: Pipeline, texts: List[str]) -> List[str]:
    return list(pipeline.predict(texts))