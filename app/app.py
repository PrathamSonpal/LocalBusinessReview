# app.py — Clean final UI (no sidebar, no debug output)
import os
import joblib
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer

st.set_page_config(page_title="Local Business Review Analyzer", layout="wide")

MODEL_SENTIMENT = "sentiment_model.joblib"
VEC_SENTIMENT   = "sentiment_tfidf_vectorizer.joblib"
MODEL_RATING    = "rating_model_LogisticRegression.joblib"
VEC_RATING      = "rating_tfidf_vectorizer.joblib"

# -------------------------
# Load model utilities
# -------------------------
def safe_load(path):
    try:
        if not os.path.exists(path):
            return None
        return joblib.load(path)
    except Exception:
        return None

@st.cache_resource(show_spinner=False)
def load_models():
    return (
        safe_load(MODEL_SENTIMENT),
        safe_load(VEC_SENTIMENT),
        safe_load(MODEL_RATING),
        safe_load(VEC_RATING)
    )

sent_model, sent_vec, rating_model, rating_vec = load_models()

# -------------------------
# TF-IDF safe transform helper
# -------------------------
def safe_transform(vec_obj, texts):
    """Try TF-IDF transform, fallback to CountVectorizer(vocabulary) quietly."""
    try:
        return vec_obj.transform(texts), False
    except Exception:
        vocab = getattr(vec_obj, "vocabulary_", None)
        if vocab and isinstance(vocab, dict):
            fallback = CountVectorizer(vocabulary=vocab)
            return fallback.transform(texts), True
        raise RuntimeError("Vectorizer transform failed; vocabulary missing.")

# -------------------------
# Prediction functions
# -------------------------
def pred_sentiment(texts):
    if sent_model is None or sent_vec is None:
        raise RuntimeError("Sentiment model/files missing.")
    X, fallback = safe_transform(sent_vec, texts)
    y_pred = sent_model.predict(X)
    try:
        proba = sent_model.predict_proba(X)
        classes = list(sent_model.classes_)
        probs = [dict(zip(classes, p)) for p in proba]
    except:
        probs = [None] * len(y_pred)
    return y_pred, probs, fallback

def pred_rating(texts):
    if rating_model is None or rating_vec is None:
        raise RuntimeError("Rating model/files missing.")
    X, fallback = safe_transform(rating_vec, texts)
    y_pred = rating_model.predict(X)
    try:
        proba = rating_model.predict_proba(X)
        classes = list(rating_model.classes_)
        probs = [dict(zip(classes, p)) for p in proba]
    except:
        probs = [None] * len(y_pred)
    return y_pred, probs, fallback

# ------------------------------------------------
#                 PAGE UI (NO SIDEBAR)
# ------------------------------------------------
st.markdown("<h1 style='text-align:center;'>📊 Local Business Review Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align:center;'>Ahmedabad & Bangalore</h3>", unsafe_allow_html=True)
st.write("")
st.write("This tool predicts **sentiment** (positive / neutral / negative) and **star rating (1–5)** from customer reviews using NLP models.")

# ------------------------------------------------
# Single Review Prediction
# ------------------------------------------------
st.markdown("## 🔍 Single Review Prediction")

review = st.text_area("Paste a customer review:", height=160)

if st.button("Predict"):
    if not review.strip():
        st.warning("Please enter a review.")
    else:
        try:
            s_label, s_probs, s_fallback = pred_sentiment([review])
            r_label, r_probs, r_fallback = pred_rating([review])
        except Exception as e:
            st.error("Prediction failed. Model files may be incompatible.")
        else:
            if s_fallback or r_fallback:
                st.info("⚠️ TF-IDF unavailable — using fallback transformer.")

            st.markdown("### 🔎 Prediction Result")
            st.write(f"**Sentiment:** `{s_label[0]}`")
            if s_probs[0]:
                st.write("Sentiment probabilities:")
                st.json(s_probs[0])

            st.write(f"**Predicted Rating:** ⭐ {r_label[0]}")
            if r_probs[0]:
                dfp = pd.DataFrame.from_dict(r_probs[0], orient='index', columns=['prob']).reset_index()
                dfp.columns = ['Rating', 'Probability']
                dfp = dfp.sort_values('Probability', ascending=False)
                st.table(dfp)

# ------------------------------------------------
# Batch CSV Prediction
# ------------------------------------------------
st.markdown("## 📂 Batch Prediction (CSV Upload)")

up = st.file_uploader("Upload a CSV containing review text", type=["csv"])

if up:
    try:
        df = pd.read_csv(up)
    except Exception:
        st.error("Could not read CSV file.")
        df = None

    if df is not None:
        st.write("### Preview")
        st.dataframe(df.head())

        columns = df.columns.tolist()
        default_col = 0
        for i, c in enumerate(columns):
            if "review" in c.lower() or "text" in c.lower():
                default_col = i; break

        col = st.selectbox("Select the review column:", columns, index=default_col)

        if st.button("Run Batch Prediction"):
            texts = df[col].astype(str).fillna("").tolist()
            try:
                s_lbl, s_proba, s_fb = pred_sentiment(texts)
                r_lbl, r_proba, r_fb = pred_rating(texts)
            except Exception:
                st.error("Batch prediction failed.")
            else:
                if s_fb or r_fb:
                    st.info("⚠️ TF-IDF unavailable — fallback used.")

                df['pred_sentiment'] = s_lbl
                df['pred_sentiment_proba'] = [str(p) if p else "" for p in s_proba]
                df['pred_rating'] = r_lbl
                df['pred_rating_proba'] = [str(p) if p else "" for p in r_proba]

                st.success("Predictions added!")
                st.dataframe(df.head())

                st.download_button(
                    "Download results",
                    df.to_csv(index=False).encode("utf-8"),
                    "predictions.csv",
                    "text/csv"
                )

st.markdown("---")
st.caption("Developed as part of a Data Science Group Project — 2025")
