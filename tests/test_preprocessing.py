"""tests/test_preprocessing.py"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import pytest
import pandas as pd
from src.preprocessing import (
    clean_text, tokenize, remove_stopwords, lemmatize,
    apply_cleaning, handle_missing, map_sentiment,
    detect_text_column, detect_date_column,
)

# ── clean_text ────────────────────────────────────────
def test_clean_text_lowercase():
    assert clean_text("Hello WORLD") == "hello world"

def test_clean_text_removes_html():
    assert "<b>" not in clean_text("<b>bold</b>")

def test_clean_text_removes_urls():
    assert "http" not in clean_text("visit https://example.com today")

def test_clean_text_removes_punctuation():
    assert "!" not in clean_text("Great job!!")

def test_clean_text_handles_none():
    assert clean_text(None) == ""
    assert clean_text("") == ""

# ── map_sentiment ─────────────────────────────────────
def test_map_sentiment_negative():
    assert map_sentiment(1) == "Negative"
    assert map_sentiment(2) == "Negative"

def test_map_sentiment_neutral():
    assert map_sentiment(3) == "Neutral"

def test_map_sentiment_positive():
    assert map_sentiment(4) == "Positive"
    assert map_sentiment(5) == "Positive"

def test_map_sentiment_invalid():
    assert map_sentiment("abc") == "Neutral"

# ── tokenize / stopwords / lemmatize ─────────────────
def test_tokenize():
    tokens = tokenize("good company culture")
    assert isinstance(tokens, list)
    assert len(tokens) >= 3

def test_remove_stopwords():
    tokens = ["the", "great", "and", "company"]
    result = remove_stopwords(tokens)
    assert "the" not in result
    assert "great" in result

def test_lemmatize():
    result = lemmatize(["running", "companies"])
    assert "running" not in result or "run" in result

# ── DataFrame helpers ─────────────────────────────────
def test_handle_missing_drops_empty_rows():
    df = pd.DataFrame({"feedback_text": ["good", None, "bad"]})
    result = handle_missing(df, "feedback_text")
    assert len(result) == 2

def test_apply_cleaning_creates_columns():
    df = pd.DataFrame({"feedback_text": ["Great work life balance"]})
    result = apply_cleaning(df, "feedback_text")
    assert "cleaned_text" in result.columns
    assert "word_count" in result.columns
    assert "char_count" in result.columns

def test_detect_text_column():
    df = pd.DataFrame({"review": ["hello"]})
    assert detect_text_column(df) == "review"

def test_detect_date_column():
    df = pd.DataFrame({"date": ["2024-01-01"], "other": [1]})
    assert detect_date_column(df) == "date"
