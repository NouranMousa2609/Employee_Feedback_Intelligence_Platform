"""
src/model_comparison.py
Employee Feedback Intelligence Platform — model comparison utilities.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def extract_metrics(results: Optional[Dict], model_name: str) -> Optional[Dict]:
    if results is None:
        return None
    return {
        "model"    : model_name,
        "accuracy" : results.get("accuracy", 0),
        "f1"       : results.get("f1_macro", 0),
        "precision": results.get("report", {}).get("macro avg", {}).get("precision", 0),
        "recall"   : results.get("report", {}).get("macro avg", {}).get("recall", 0),
    }


def compare_from_results(
    classical_results: Optional[Dict],
    transformer_results: Optional[Dict],
    **kwargs,
) -> pd.DataFrame:
    rows = []
    if classical_results:
        rows.append({
            "model"    : "LinearSVC (TF-IDF)",
            "accuracy" : classical_results.get("accuracy", 0),
            "f1"       : classical_results.get("f1_macro", 0),
            "precision": classical_results.get("report", {}).get("macro avg", {}).get("precision", 0),
            "recall"   : classical_results.get("report", {}).get("macro avg", {}).get("recall", 0),
        })
    if transformer_results:
        rows.append({
            "model"    : "DistilBERT (fine-tuned)",
            "accuracy" : transformer_results.get("accuracy", 0),
            "f1"       : transformer_results.get("f1_macro", 0),
            "precision": 0,
            "recall"   : 0,
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def save_comparison_report(
    comparison_df: pd.DataFrame,
    classical_results: Optional[Dict],
    transformer_results: Optional[Dict],
    path: str,
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    report = {
        "comparison_table": comparison_df.to_dict(orient="records"),
        "classical"   : {
            "accuracy"        : classical_results.get("accuracy")   if classical_results else None,
            "f1_macro"        : classical_results.get("f1_macro")   if classical_results else None,
            "confusion_matrix": classical_results.get("confusion_matrix") if classical_results else None,
            "labels"          : classical_results.get("labels")     if classical_results else None,
        },
        "transformer" : {
            "accuracy" : transformer_results.get("accuracy") if transformer_results else None,
            "f1_macro" : transformer_results.get("f1_macro") if transformer_results else None,
        },
    }
    with open(path, "w") as f:
        json.dump(report, f, indent=2)


def load_comparison_report(path: str) -> Optional[Dict]:
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None
