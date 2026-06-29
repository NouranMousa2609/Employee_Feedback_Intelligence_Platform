"""
src/preprocessing.py
Employee Feedback Intelligence Platform — text preprocessing utilities.
"""

import re
import warnings
from typing import Optional, Tuple

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

warnings.filterwarnings("ignore")

for pkg in ["punkt", "stopwords", "wordnet", "omw-1.4", "punkt_tab"]:
    nltk.download(pkg, quiet=True)

STOP_WORDS = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

LABEL_COL_CANDIDATES = ["sentiment_label", "sentiment", "label", "rating"]
TEXT_COL_CANDIDATES  = ["feedback_text", "feedback", "text", "review", "clean_text"]
DATE_COL_CANDIDATES  = ["date", "review_date", "created_at", "timestamp"]


# ── Helpers ──────────────────────────────────────────────────────────────────

def detect_label_column(df: pd.DataFrame) -> Optional[str]:
    for col in LABEL_COL_CANDIDATES:
        if col in df.columns:
            return col
    return None


def detect_date_column(df: pd.DataFrame) -> Optional[str]:
    for col in DATE_COL_CANDIDATES:
        if col in df.columns:
            return col
    return None


def detect_text_column(df: pd.DataFrame) -> Optional[str]:
    for col in TEXT_COL_CANDIDATES:
        if col in df.columns:
            return col
    return None


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def validate_uploaded_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[str]]:
    text_col = detect_text_column(df)
    if text_col is None:
        return df, (
            "Could not find a text column. Expected one of: "
            + ", ".join(TEXT_COL_CANDIDATES)
        )
    if text_col != "feedback_text":
        df = df.rename(columns={text_col: "feedback_text"})
    return df, None


def handle_missing(df: pd.DataFrame, text_col: str = "feedback_text") -> pd.DataFrame:
    df = df.copy()
    if text_col in df.columns:
        df[text_col] = df[text_col].fillna("")
    return df.dropna(subset=[text_col]).reset_index(drop=True)


# ── Text cleaning ─────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)          # HTML tags
    text = re.sub(r"http\S+|www\S+", " ", text)   # URLs
    text = re.sub(r"[^a-z\s]", " ", text)         # keep letters only
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list:
    return word_tokenize(text)


def remove_stopwords(tokens: list) -> list:
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 1]


def lemmatize(tokens: list) -> list:
    return [lemmatizer.lemmatize(t) for t in tokens]


def apply_cleaning(df: pd.DataFrame, text_col: str = "feedback_text") -> pd.DataFrame:
    df = df.copy()
    df["cleaned_text"] = (
        df[text_col]
        .apply(clean_text)
        .apply(lambda t: " ".join(lemmatize(remove_stopwords(tokenize(t)))))
    )
    df["word_count"]     = df["cleaned_text"].str.split().str.len()
    df["char_count"]     = df["cleaned_text"].str.len()
    df["avg_word_length"] = df["cleaned_text"].apply(
        lambda t: (sum(len(w) for w in t.split()) / max(len(t.split()), 1))
    )
    return df


def map_sentiment(rating) -> str:
    try:
        r = float(rating)
    except (ValueError, TypeError):
        return "Neutral"
    if r <= 2:
        return "Negative"
    elif r == 3:
        return "Neutral"
    return "Positive"


def full_pipeline(
    df: pd.DataFrame,
    text_col: str = "feedback_text",
    label_col: Optional[str] = None,
) -> pd.DataFrame:
    df = handle_missing(df, text_col)
    df = apply_cleaning(df, text_col)
    if label_col and label_col in df.columns:
        df[label_col] = df[label_col].astype(str)
    return df
