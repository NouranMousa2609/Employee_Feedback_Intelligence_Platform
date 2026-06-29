# 🧠 Employee Feedback Intelligence Platform

> **NLP + Classical ML + Transformers** — End-to-end pipeline that ingests employee/customer feedback CSVs, cleans and analyses them, trains and compares two sentiment models, and serves everything through an interactive Streamlit dashboard.

---

## 📁 Project Structure

```
feedback-intelligence/
│
├── 📄 README.md                          # Project overview, setup, usage
├── 📄 requirements.txt                   # Pinned dependencies
├── 📄 .gitignore                         # Exclude data, models, __pycache__
├── 📄 .env.example                       # Environment variable template
│
├── 📂 data/
│   ├── glassdoor_reviews.csv             # Original CSV (never modified)
│   └── processed/                        # Cleaned, tokenized outputs
│
├── 📂 notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_classical_model.ipynb
│   ├── 04_transformer_model.ipynb
│   └── 05_model_comparison.ipynb
│
├── 📂 src/
│   ├── __init__.py
│   ├── preprocessing.py                  # Data loading, cleaning, tokenization, lemmatization
│   ├── classical_model.py                # TF-IDF + linearSVC
│   ├── transformer_model.py              # DistilBERT predictions
│   └── visualizations.py                # Plotly charts + WordCloud helpers
│
├── 📂 models/                            # Saved after training (gitignored)
│   ├── classical_model.joblib
│   └── transformer/                      # HuggingFace checkpoint folder
│       ├── config.json
│       ├── tokenizer.json
│       └── pytorch_model.bin
│
├── 📂 app/
│   └── app.py                            # Streamlit entry point
│
├── 📂 tests/
│   ├── test_preprocessing.py
│   ├── test_classical_model.py
│   └── test_transformer_model.py
│
├── 📂 assets/
│   ├── architecture.svg                  # Architecture diagram
│   └── screenshots/                      # App screenshots for README
│
└── 📂 reports/
    └── report.pdf                        # Final written report
```

### Key rules this structure follows

**`data/raw/` is sacred** — the original CSV is never touched or modified. All cleaning outputs go to `data/processed/`.

**`src/` is the brain** — all reusable logic lives here. Both notebooks and the app import from it, so no code is duplicated.

**`notebooks/` tells the story** — numbered sequentially, meant to be read top to bottom 

**`app/` is separate from `src/`** — the web layer is kept apart from business logic.

**`models/` is gitignored** — model files are too large for GitHub. Use Git LFS or regenerate them locally after cloning.



## 🗂 USED Dataset

**Glassdoor Job Reviews** — [kaggle.com/datasets/davidgauthier/glassdoor-job-reviews](https://www.kaggle.com/datasets/davidgauthier/glassdoor-job-reviews)


## ⚙️ Setup Instructions

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/feedback-intelligence.git
cd feedback-intelligence
```


### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
- Go to [Kaggle Glassdoor Job Reviews](https://www.kaggle.com/datasets/davidgauthier/glassdoor-job-reviews)
- Download and place the CSV in `data/raw/glassdoor_reviews.csv`

### 4. Run the app
```bash
streamlit run app/app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---


## 🧠 ML Pipeline

### Classical Model
```
Raw text
  → clean_text()       # lowercase, strip HTML/URLs/punctuation
  → tokenize()         # NLTK word_tokenize
  → remove_stopwords()
  → lemmatize()        # WordNetLemmatizer
  → TfidfVectorizer    # 30k features, bigrams, sublinear TF
  → LogisticRegression / RandomForest
```

### Transformer Model
```
Raw text
  → DistilBertTokenizerFast (max_len=128)
  → DistilBertForSequenceClassification (fine-tuned, 3 classes)
  → AdamW + linear warmup scheduler
  → 3 epochs on 20k sample
```

---

## 📊 Concepts Covered

| Concept | Where |
|---|---|
| Python | All modules |
| Pandas | `preprocessing.py` |
| Data Cleaning | `preprocessing.py` |
| Visualization | `visualizations.py` (Plotly, WordCloud, Matplotlib) |
| Classification | `classical_model.py` |
| NLP | `preprocessing.py`, `classical_model.py` |
| Transformers | `transformer_model.py` (DistilBERT) |
| Deployment | `app.py` (Streamlit) |

