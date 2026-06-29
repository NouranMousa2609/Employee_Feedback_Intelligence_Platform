"""
app/app.py — Employee Feedback Intelligence Platform
Run with:  streamlit run app/app.py
"""

import os, sys, json, tempfile
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
from src.transformer_model import predict
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
st.set_page_config(page_title="Feedback Intelligence", page_icon="🧠",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.metric-card { background:#f8fafc; border:1px solid #e2e8f0; border-radius:12px;
               padding:20px 24px; text-align:center; margin-bottom:8px; }
.metric-card .label { font-size:13px; color:#64748b; font-weight:500; }
.metric-card .value { font-size:28px; font-weight:700; color:#1e293b; }
.badge-Positive{background:#dcfce7;color:#166534;padding:4px 12px;border-radius:20px;font-weight:600;font-size:15px;}
.badge-Neutral {background:#fef3c7;color:#92400e;padding:4px 12px;border-radius:20px;font-weight:600;font-size:15px;}
.badge-Negative{background:#fee2e2;color:#991b1b;padding:4px 12px;border-radius:20px;font-weight:600;font-size:15px;}
#MainMenu, footer { visibility: hidden; }
</style>""", unsafe_allow_html=True)

@st.cache_resource
def get_preprocessing():
    from src.preprocessing import (apply_cleaning, handle_missing, map_sentiment,
                                   detect_text_column, detect_date_column)
    return apply_cleaning, handle_missing, map_sentiment, detect_text_column, detect_date_column

@st.cache_resource
def get_viz():
    import src.visualization as viz
    return viz

@st.cache_resource
def get_comparison():
    from src.model_comparison import compare_from_results
    return compare_from_results

for key in ["df", "classical_result", "transformer_result"]:
    if key not in st.session_state:
        st.session_state[key] = None

with st.sidebar:
    st.markdown("## 🧠 Feedback Intelligence")
    st.caption("NLP · Classical ML · Transformers")
    st.divider()
    page = option_menu(
        menu_title=None,
        options=["Upload & Preview", "Dashboard", "Train Models", "Compare & Predict"],
        icons=["cloud-upload", "bar-chart-line", "cpu", "lightning-charge"],
        default_index=0,
        styles={"nav-link-selected": {"background-color": "#6366f1"}},
    )
    st.divider()
    if st.session_state["df"] is not None:
        df = st.session_state["df"]
        label_col = "sentiment_label" if "sentiment_label" in df.columns else "sentiment"
        st.caption(f"📊 **{len(df):,}** reviews loaded")
        counts = df[label_col].value_counts()
        for s, c in [("Positive", "#22c55e"), ("Neutral", "#f59e0b"), ("Negative", "#ef4444")]:
            pct = counts.get(s, 0) / len(df) * 100
            st.markdown(f"<span style='color:{c}'>●</span> {s}: **{pct:.1f}%**",
                        unsafe_allow_html=True)

# ═══════════════════════════════ PAGE 1 ═══════════════════════════════════════
if page == "Upload & Preview":
    st.title("📂 Upload Feedback Data")
    uploaded   = st.file_uploader("Choose a CSV file", type=["csv"])
    rating_col = st.text_input("Rating column (1–5 stars)", value="overall-ratings")

    if uploaded:
        tmp = os.path.join(tempfile.gettempdir(), "uploaded_feedback.csv")
        with open(tmp, "wb") as f:
            f.write(uploaded.getbuffer())

        apply_cleaning, handle_missing, map_sentiment, detect_text_col, detect_date_col = get_preprocessing()
        with st.spinner("Loading and preprocessing…"):
            try:
                raw = pd.read_csv(tmp, low_memory=False).drop_duplicates().reset_index(drop=True)
                text_col = detect_text_col(raw)
                if text_col is None:
                    if "pros" in raw.columns and "cons" in raw.columns:
                        raw["feedback_text"] = (raw["pros"].fillna("") + " " + raw["cons"].fillna("")).str.strip()
                    else:
                        st.error("Cannot detect text column."); st.stop()
                elif text_col != "feedback_text":
                    raw = raw.rename(columns={text_col: "feedback_text"})

                if rating_col and rating_col in raw.columns:
                    raw[rating_col] = pd.to_numeric(raw[rating_col], errors="coerce")
                    raw = raw.dropna(subset=[rating_col]).reset_index(drop=True)
                    raw["sentiment_label"] = raw[rating_col].apply(map_sentiment)
                elif "sentiment" in raw.columns:
                    raw["sentiment_label"] = raw["sentiment"]
                else:
                    st.error("Cannot find rating or sentiment column."); st.stop()

                date_col = detect_date_col(raw)
                if date_col:
                    raw["date"] = pd.to_datetime(raw[date_col], errors="coerce")

                df = apply_cleaning(handle_missing(raw, "feedback_text"), "feedback_text")
                st.session_state["df"] = df
                st.success(f"✅ Loaded **{len(df):,}** rows.")
            except Exception as e:
                st.error(f"Error: {e}"); st.stop()

        df     = st.session_state["df"]
        counts = df["sentiment_label"].value_counts()
        c1, c2, c3, c4 = st.columns(4)
        for col, label, val in [
            (c1, "Total Reviews", f"{len(df):,}"),
            (c2, "✅ Positive",   f"{counts.get('Positive', 0):,}"),
            (c3, "⚠️ Neutral",    f"{counts.get('Neutral', 0):,}"),
            (c4, "❌ Negative",   f"{counts.get('Negative', 0):,}"),
        ]:
            col.markdown(f'<div class="metric-card"><div class="label">{label}</div>'
                         f'<div class="value">{val}</div></div>', unsafe_allow_html=True)
        st.divider()

        # ── Missing values ──────────────────────────────────────────────────
        miss = df.isnull().sum()
        miss = miss[miss > 0]
        st.subheader("Missing Values")
        if miss.empty:
            st.success("None 🎉")
        else:
            st.dataframe(
                pd.DataFrame({
                    "count": miss,
                    "%": (miss / len(df) * 100).round(2),
                }),
                use_container_width=True,
            )

        # ── Sample rows ─────────────────────────────────────────────────────
        st.subheader("Sample Rows")
        show = [c for c in ["feedback_text", "cleaned_text", "sentiment_label", "word_count", "date"]
                if c in df.columns]
        st.dataframe(df[show].head(20), use_container_width=True)
        st.download_button(
            "⬇️ Download cleaned CSV",
            df[show].to_csv(index=False).encode(),
            "cleaned_feedback.csv",
            "text/csv",
        )

# ═══════════════════════════════ PAGE 2 ═══════════════════════════════════════
elif page == "Dashboard":
    st.title("📊 Feedback Dashboard")
    if st.session_state["df"] is None:
        st.warning("Upload data first."); st.stop()
    df        = st.session_state["df"].copy()
    viz       = get_viz()
    label_col = "sentiment_label" if "sentiment_label" in df.columns else "sentiment"
    text_col  = "cleaned_text"    if "cleaned_text"    in df.columns else "feedback_text"

    with st.expander("🔍 Filters"):
        sel = st.multiselect("Sentiment", ["Positive", "Neutral", "Negative"],
                             default=["Positive", "Neutral", "Negative"])
        df  = df[df[label_col].isin(sel)]
        st.caption(f"{len(df):,} reviews")

    c1, c2 = st.columns(2)
    c1.plotly_chart(viz.sentiment_distribution_chart(df, label_col), use_container_width=True)
    c2.plotly_chart(viz.word_frequency_chart(df, text_col), use_container_width=True)

    if "date" in df.columns:
        st.plotly_chart(viz.sentiment_trend_chart(df, "date", label_col), use_container_width=True)
    dept = viz.department_sentiment_chart(df, label_col)
    if dept:
        st.plotly_chart(dept, use_container_width=True)

# ═══════════════════════════════ PAGE 3 ═══════════════════════════════════════
elif page == "Train Models":
    st.title("🤖 Train Models")
    if st.session_state["df"] is None:
        st.warning("Upload data first."); st.stop()
    df        = st.session_state["df"]
    viz       = get_viz()
    label_col = "sentiment_label" if "sentiment_label" in df.columns else "sentiment"
    text_col  = "cleaned_text"    if "cleaned_text"    in df.columns else "feedback_text"

    # Classical
    st.subheader("1️⃣  Classical ML — TF-IDF + LinearSVC")
    if st.button("🚀 Load Classical Model"):
        from src.classical_model import load_model
        from sklearn.metrics import accuracy_score, f1_score, confusion_matrix as sk_cm, classification_report
        with st.spinner("Loading model and running inference…"):
            pipeline = load_model("models/classical_model.pkl")
            y_true   = df[label_col].tolist()
            y_pred   = list(pipeline.predict(df["feedback_text"].tolist()))
            acc      = accuracy_score(y_true, y_pred)
            f1       = f1_score(y_true, y_pred, average="macro", zero_division=0)
            cm       = sk_cm(y_true, y_pred, labels=["Negative", "Neutral", "Positive"]).tolist()
            report   = classification_report(y_true, y_pred, output_dict=True)
        result = {
            "pipeline": pipeline, "accuracy": round(acc, 4), "f1_macro": round(f1, 4),
            "confusion_matrix": cm, "report": report,
            "labels": ["Negative", "Neutral", "Positive"],
            "y_test": y_true, "y_pred": y_pred, "best_C": "pretrained",
        }
        st.session_state["classical_result"] = result
        st.success(f"✅  Accuracy: {acc:.2%}")
        c1, c2 = st.columns(2)
        c1.plotly_chart(viz.confusion_matrix_heatmap(result["confusion_matrix"], result["labels"]), use_container_width=True)
        c2.dataframe(pd.DataFrame(result["report"]).T.round(3), use_container_width=True)

    st.divider()

    # Transformer
    st.subheader("2️⃣  Transformer — DistilBERT (HuggingFace Trainer)")
    if st.button("🚀 Load Transformer Model"):
        with st.spinner("Loading pretrained DistilBERT..."):
            results = predict(df["feedback_text"].tolist())
            y_pred = [r["label"] for r in results]
            y_true = df[label_col].tolist()
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average="macro")
            t_result = {
                "accuracy": acc,
                "f1_macro": f1,
                "y_test": y_true,
                "y_pred": y_pred,
                "labels": ["Negative", "Neutral", "Positive"],
                "report": classification_report(y_true,y_pred,output_dict=True)
                }
            st.session_state["transformer_result"] = t_result
            st.success(
                f"✅ Accuracy: {acc:.2%}"
                )
            cm = confusion_matrix(
                y_true,
                y_pred,
                labels=t_result["labels"])
            st.plotly_chart(
                viz.confusion_matrix_heatmap(
                    cm.tolist(),
                    t_result["labels"],
                    "Transformer"
                    ),
                use_container_width=True
                )
# ═══════════════════════════════ PAGE 4 ═══════════════════════════════════════
elif page == "Compare & Predict":
    st.title("⚡ Compare Models & Live Predict")
    viz        = get_viz()
    compare_fn = get_comparison()
    cr         = st.session_state.get("classical_result")
    tr         = st.session_state.get("transformer_result")

    if cr or tr:
        st.subheader("Model Comparison")
        compare_df = compare_fn(cr, tr)
        if not compare_df.empty:
            st.plotly_chart(viz.metrics_comparison_chart(compare_df), use_container_width=True)
            st.dataframe(compare_df.set_index("model").round(4), use_container_width=True)
        if cr and tr:
            diff   = tr["accuracy"] - cr["accuracy"]
            winner = "🏆 Transformer (DistilBERT)" if diff > 0 else "🏆 Classical (LinearSVC)"
            c1, c2, c3 = st.columns(3)
            c1.metric("Classical accuracy",   f"{cr['accuracy']:.2%}")
            c2.metric("Transformer accuracy", f"{tr['accuracy']:.2%}", f"{diff:+.2%}")
            c3.markdown(f"**{winner}**")
    else:
        st.info("Train both models on **Train Models** to see comparisons.")

    st.divider()
    st.subheader("🔮 Live Prediction")
    user_text  = st.text_area("Enter feedback:", height=120,
                               placeholder="e.g. Great culture but salary is below market…")
    pred_model = st.radio("Model", ["Classical", "Transformer"], horizontal=True)

    if st.button("🔍 Predict") and user_text.strip():
        if pred_model == "Classical":
            if cr is None:
                st.warning("Train the classical model first.")
            else:
                from src.preprocessing import clean_text, tokenize, remove_stopwords, lemmatize
                cleaned = " ".join(lemmatize(remove_stopwords(tokenize(clean_text(user_text)))))
                pred    = cr["pipeline"].predict([cleaned])[0]
                st.markdown(f"**Sentiment:** <span class='badge-{pred}'>{pred}</span>",
                            unsafe_allow_html=True)
        else:
            if tr is None:
                st.warning("Fine-tune the transformer first.")
            else:
                from src.transformer_model import predict as bert_predict
                res  = bert_predict([user_text], model_path="models/transformer")
                pred = res[0]["label"]
                st.markdown(f"**Sentiment:** <span class='badge-{pred}'>{pred}</span>",
                            unsafe_allow_html=True)
                if "score" in res[0]:
                    st.caption(f"Confidence: {res[0]['score']:.1%}")