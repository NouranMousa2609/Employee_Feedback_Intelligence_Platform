"""tests/test_model_comparison.py"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import pytest
import pandas as pd
from src.model_comparison import compare_from_results, extract_metrics

MOCK_CLASSICAL = {
    "accuracy": 0.82, "f1_macro": 0.80,
    "report": {"macro avg": {"precision": 0.81, "recall": 0.79}},
}
MOCK_TRANSFORMER = {
    "accuracy": 0.88, "f1_macro": 0.86,
    "report": {"macro avg": {"precision": 0.87, "recall": 0.85}},
}

def test_extract_metrics_returns_dict():
    m = extract_metrics(MOCK_CLASSICAL, "LinearSVC")
    assert m["model"]    == "LinearSVC"
    assert m["accuracy"] == 0.82
    assert m["f1"]       == 0.80

def test_extract_metrics_none_input():
    assert extract_metrics(None, "test") is None

def test_compare_from_results_returns_df():
    df = compare_from_results(MOCK_CLASSICAL, MOCK_TRANSFORMER)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "model" in df.columns
    assert "accuracy" in df.columns

def test_compare_only_classical():
    df = compare_from_results(MOCK_CLASSICAL, None)
    assert len(df) == 1

def test_compare_none_inputs():
    df = compare_from_results(None, None)
    assert df.empty
