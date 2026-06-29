# Employee Feedback Intelligence Platform — Project Report

---

## 1. Problem Statement

Organizations collect thousands of employee and customer feedback entries but lack scalable tools to extract meaningful sentiment insights from them. Manual review is slow and inconsistent. This project builds an end-to-end AI pipeline that:

- Ingests raw feedback CSV files
- Automatically cleans and preprocesses text
- Classifies sentiment as **Positive**, **Neutral**, or **Negative**
- Compares a classical ML model against a fine-tuned transformer
- Serves results through an interactive web dashboard

---

## 2. Dataset

**Source:** Glassdoor Job Reviews — [kaggle.com/datasets/davidgauthier/glassdoor-job-reviews](https://www.kaggle.com/datasets/davidgauthier/glassdoor-job-reviews)

| Property | Detail |
|---|---|
| Size | ~850,000 reviews |
| Text fields | `pros`, `cons`, `advice-to-mgmt` |
| Label source | `overall-ratings` (1–5 stars) |
| Label mapping | 1–2 → Negative · 3 → Neutral · 4–5 → Positive |
| Date range | Multi-year (trend analysis enabled) |


## 3. Approach

### 3.1 Preprocessing Pipeline

```
pros + cons  →  clean_text()  →  tokenize()  →  remove_stopwords()  →  lemmatize()  →  cleaned_text
```

Steps implemented in `src/preprocessing.py`:

1. **Combine** `pros` and `cons` into a single `feedback_text` field
2. **Clean:** lowercase, strip HTML tags, URLs, and punctuation
3. **Tokenize:** NLTK `word_tokenize`
4. **Stopword removal:** NLTK English stopwords
5. **Lemmatize:** WordNetLemmatizer
6. **Label mapping:** star rating → 3-class sentiment label

### 3.2 Classical ML Model (`src/classical_model.py`)

- **Vectorizer:** TF-IDF (30k features, bigrams, sublinear TF)
- **Classifier:** LinearSVC with GridSearchCV over C ∈ {0.01, 0.1, 1, 5, 10}
- **CV strategy:** StratifiedKFold (5 folds), scoring = macro-F1
- **Split:** 70% train / 15% val / 15% test

### 3.3 Transformer Model (`src/transformer_model.py`)

- **Base model:** `distilbert-base-uncased` (66M parameters)
- **Fine-tuning:** HuggingFace `Trainer` API
- **Training config:** 3 epochs, batch=8, lr=2e-5, warmup_ratio=0.1, weight_decay=0.01
- **Early stopping:** patience=2 on `eval_f1_macro`
- **Sample:** 20,000 stratified reviews (to keep training time reasonable)

---

## 4. Results

### 4.1 Classical Model (LinearSVC + TF-IDF)

| Metric | Score |
|---|---|
| Test Accuracy | *fill after training* |
| Test Macro-F1 | *fill after training* |
| Best C | *fill after GridSearch* |
| Training Time | < 5 minutes (CPU) |

### 4.2 Transformer Model (DistilBERT)

| Metric | Score |
|---|---|
| Test Accuracy | *fill after training* |
| Test Macro-F1 | *fill after training* |
| Epochs | 3 |
| Training Time | ~30–60 min (CPU) / ~10 min (GPU) |

### 4.3 Comparison

| Model | Accuracy | Macro-F1 | GPU Required | Training Time |
|---|---|---|---|---|
| LinearSVC (TF-IDF) | — | — | No  | Fast |
| DistilBERT (fine-tuned) | — | — | Recommended | Slow |

*Fill in actual numbers after running Notebooks 03 and 04.*

---

## 5. Limitations

1. **Class imbalance:** Positive reviews dominate (~60%). Neutral class has the lowest F1 score due to being sandwiched between the other two.
2. **Sample size for transformer:** Training on 20k rows (not the full 850k) to keep runtime feasible — accuracy would likely improve with full data and GPU.
3. **Domain specificity:** The model is trained on Glassdoor data. Performance may degrade on very different feedback styles (e.g., customer support tickets).
4. **No aspect-based analysis:** The system gives a single sentiment per review, not per aspect (pay, culture, management, etc.).
5. **Static model:** No online learning — the model does not update as new feedback arrives.

---

## 6. Potential Improvements

| Improvement | Impact |
|---|---|
| Train transformer on full dataset with GPU | Higher accuracy |
| Aspect-based sentiment analysis (ABSA) | Richer insights per topic |
| BERTopic for topic modeling | Discover hidden themes |
| Multilingual model (`xlm-roberta`) | Support non-English reviews |
| Active learning loop | Label uncertain samples efficiently |
| FastAPI REST endpoint | Programmatic access for integrations |
| Streamlit Cloud / HuggingFace Spaces deployment | Public demo |

---

## 7. Project Structure

```
feedback-intelligence/
├── data/raw/              ← Original CSV (gitignored)
├── data/processed/        ← Cleaned outputs
├── notebooks/             ← 01–05 Jupyter notebooks
├── src/                   ← Reusable Python modules
├── app/app.py             ← Streamlit web app
├── models/                ← Saved models (gitignored)
├── tests/                 ← Unit tests
├── assets/                ← Diagrams and screenshots
└── reports/report.md      ← This file
```

