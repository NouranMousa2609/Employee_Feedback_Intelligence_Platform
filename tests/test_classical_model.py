"""tests/test_classical_model.py"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import pytest
import pandas as pd
from src.classical_model import train, predict, save_model, load_model

SAMPLE_DF = pd.DataFrame({
    "cleaned_text": [
        "great company work life balance amazing",
        "terrible management toxic environment",
        "average okay nothing special neutral",
        "love the culture benefits excellent",
        "poor pay bad leadership worst",
        "decent job reasonable expectations",
    ] * 30,
    "sentiment": (["Positive", "Negative", "Neutral"] * 2) * 30,
})

def test_train_returns_required_keys():
    result = train(SAMPLE_DF)
    for key in ["pipeline", "accuracy", "f1_macro", "report", "confusion_matrix", "labels"]:
        assert key in result

def test_train_accuracy_in_range():
    result = train(SAMPLE_DF)
    assert 0.0 <= result["accuracy"] <= 1.0
    assert 0.0 <= result["f1_macro"] <= 1.0

def test_predict_returns_list():
    result  = train(SAMPLE_DF)
    preds   = predict(result["pipeline"], ["great company", "terrible experience"])
    assert isinstance(preds, list)
    assert len(preds) == 2
    assert all(p in ["Positive", "Neutral", "Negative"] for p in preds)

def test_save_and_load_model(tmp_path):
    result = train(SAMPLE_DF)
    path   = str(tmp_path / "model.pkl")
    save_model(result["pipeline"], path)
    loaded = load_model(path)
    preds  = predict(loaded, ["great work"])
    assert preds[0] in ["Positive", "Neutral", "Negative"]
