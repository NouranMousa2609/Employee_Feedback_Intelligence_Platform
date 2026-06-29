"""
src/transformer_model.py
Employee Feedback Intelligence Platform — DistilBERT inference.

Uses the fine-tuned model saved in models/transformer/.
Training was done offline (Kaggle notebook 04_transformer_model.ipynb).
This module provides inference-only functionality.

Usage
-----
    from src.transformer_model import predict, predict_one

    results = predict(["Great place to work!", "Salary is too low."])
    # → [{"label": "Positive", "score": 0.97}, {"label": "Negative", "score": 0.91}]
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, DistilBertTokenizerFast

logger = logging.getLogger(__name__)

# ── Constants (must match training config) ────────────────────────────────────

DEFAULT_MODEL_PATH = "models/transformer"
MAX_LEN            = 128

LABEL2ID: Dict[str, int] = {"Negative": 0, "Neutral": 1, "Positive": 2}
ID2LABEL: Dict[int, str] = {0: "Negative", 1: "Neutral", 2: "Positive"}

# ── In-process model cache ────────────────────────────────────────────────────

_cache: Dict[str, Dict] = {}


def _load_model(model_path: str):
    """
    Load tokenizer and model from *model_path*, caching in memory after the first call.

    Parameters
    ----------
    model_path : directory containing config.json, tokenizer files, and model weights.

    Returns
    -------
    (tokenizer, model, device)

    Raises
    ------
    FileNotFoundError if config.json is absent from the directory.
    """
    if model_path in _cache:
        cached = _cache[model_path]
        return cached["tokenizer"], cached["model"], cached["device"]

    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"No config.json found in '{model_path}'.\n"
            "Ensure the full models/transformer/ folder is present:\n"
            "  config.json, tokenizer.json, tokenizer_config.json,\n"
            "  model.safetensors (or pytorch_model.bin)."
        )

    logger.info("Loading tokenizer from '%s'…", model_path)
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)

    logger.info("Loading model weights from '%s'…", model_path)
    model  = AutoModelForSequenceClassification.from_pretrained(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    _cache[model_path] = {"tokenizer": tokenizer, "model": model, "device": device}
    logger.info("DistilBERT loaded on %s.", device.upper())
    return tokenizer, model, device


# ── Public API ────────────────────────────────────────────────────────────────

def predict(
    texts: List[str],
    model_path: str = DEFAULT_MODEL_PATH,
    batch_size: int = 64,
) -> List[Dict]:
    """
    Run sentiment inference on a list of texts.

    Parameters
    ----------
    texts      : raw strings to classify (no preprocessing required).
    model_path : path to the fine-tuned model folder.
    batch_size : number of samples per forward pass (reduce if OOM on CPU).

    Returns
    -------
    List of dicts, one per input text:
        {"label": "Positive" | "Neutral" | "Negative", "score": float}

    Example
    -------
    >>> results = predict(["Amazing team culture!", "Poor work-life balance."])
    >>> results[0]
    {'label': 'Positive', 'score': 0.9712}
    """
    if not texts:
        logger.warning("predict() called with an empty list — returning [].")
        return []

    tokenizer, model, device = _load_model(model_path)

    enc = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=MAX_LEN,
        return_tensors="pt",
    )

    ds = TensorDataset(enc["input_ids"], enc["attention_mask"])
    dl = DataLoader(ds, batch_size=batch_size)

    all_preds:  List[int]   = []
    all_scores: List[float] = []

    with torch.no_grad():
        for input_ids, attention_mask in dl:
            logits = model(
                input_ids.to(device),
                attention_mask=attention_mask.to(device),
            ).logits
            probs  = torch.softmax(logits, dim=1)
            preds  = torch.argmax(probs, dim=1).cpu().numpy()
            scores = probs.max(dim=1).values.cpu().numpy()
            all_preds.extend(preds)
            all_scores.extend(scores)

    logger.info("Classified %d text(s) via DistilBERT.", len(texts))

    return [
        {"label": ID2LABEL[int(p)], "score": round(float(s), 4)}
        for p, s in zip(all_preds, all_scores)
    ]


def predict_one(text: str, model_path: str = DEFAULT_MODEL_PATH) -> Dict:
    """
    Classify a single text string.

    Returns
    -------
    dict: {"label": str, "score": float}
    """
    return predict([text], model_path=model_path)[0]


def is_model_available(model_path: str = DEFAULT_MODEL_PATH) -> bool:
    """Return True if the model directory contains the required config.json."""
    return (Path(model_path) / "config.json").exists()


def train(*args, **kwargs):
    """
    Training stub — the fine-tuned model is pre-saved in models/transformer/.

    To retrain, run the Kaggle notebook: 04_transformer_model.ipynb
    """
    raise NotImplementedError(
        "Training is disabled in this module.\n"
        "The fine-tuned model is saved in 'models/transformer/'.\n"
        "Use predict() or predict_one() for inference.\n"
        "To retrain, run 04_transformer_model.ipynb on Kaggle."
    )


# ── Quick smoke test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    samples = [
        "Management is very supportive and the team culture is excellent.",
        "The salary is average — nothing special.",
        "Terrible work environment, I dread coming in every morning.",
    ]

    if not is_model_available():
        print(f"[WARN] Model not found at '{DEFAULT_MODEL_PATH}'. Skipping smoke test.")
    else:
        results = predict(samples)
        print("\n── Smoke-test results ────────────────────────────────")
        for text, res in zip(samples, results):
            print(f"  [{res['label']:8s}  {res['score']:.2f}]  {text[:65]}")