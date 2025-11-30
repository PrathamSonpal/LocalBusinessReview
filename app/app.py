# app.py — Clean final UI (no sidebar, no debug output)
import os
import joblib
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer

# VADER for fallback rule-based sentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

st.set_page_config(page_title="Local Business Review Analyzer", layout="wide")

# Filenames (expected to be in same folder as app.py)
MODEL_SENTIMENT = "sentiment_model.joblib"
VEC_SENTIMENT   = "sentiment_tfidf_vectorizer.joblib"
MODEL_RATING    = "rating_model_LogisticRegression.joblib"
VEC_RATING      = "rating_tfidf_vectorizer.joblib"

BASE_DIR = os.path.dirname(__file__) if "__file__" in globals() else "."

# -------------------------
# Load model utilities
# -------------------------
def safe_load_file(fname):
    """Load joblib if present, otherwise return None."""
    p = os.path.join(BASE_DIR, fname)
    if not os.path.exists(p):
        return None
    try:
        return joblib.load(p)
    except Exception:
        # if loading fails (binary incompat), return None to force fallback
        return None

@st.cache_resource(show_spinner=False)
def load_models():
    sent_model = safe_load_file(MODEL_SENTIMENT)
    sent_vec   = safe_load_file(VEC_SENTIMENT)
    rating_model = safe_load_file(MODEL_RATING)
    rating_vec   = safe_load_file(VEC_RATING)
    return sent_model, sent_vec, rating_model, rating_vec

sent_model, sent_vec, rating_model, rating_vec = load_models()

# -------------------------
# Fallback rule-based (VADER)
# -------------------------
analyzer = SentimentIntensityAnalyzer()

def fallback_sentiment(texts):
    """Return labels (positive/neutral/negative), None probs, fallback_flag=True."""
    labels = []
    for t in texts:
        s = analyzer.polarity_scores(str(t))['compound']
        if s >= 0.05:
            labels.append("positive")
        elif s <= -0.05:
            labels.append("negative")
        else:
            labels.append("neutral")
    return labels, [None] * len(labels), True

def fallback_rating(texts):
    """Map VADER compound to approximate star (1-5). Returns fallback_flag=True."""
    labels = []
    for t in texts:
        s = analyzer.polarity_scores(str(t))['compound']
        if s >= 0.5:
            labels.append(5)
        elif s >= 0.2:
            labels.append(4)
        elif s >= -0.1:
            labels.append(3)
        elif s >= -0.4:
            labels.append(2)
        else:
            labels.append(1)
    return labels, [None] * len(labels), True

# -------------------------
# TF-IDF safe transform helper
# -------------------------
def safe_transform(vec_obj, texts):
    """
    Try vec_obj.transform(texts).
    If it raises but vec_obj.vocabulary_ exists, return CountVectorizer(vocabulary) transform and fallback=True.
    If vec_obj is None or nothing works, raise.
    """
    if vec_obj is None:
        raise RuntimeError("vectorizer-missing")
    try:
        X = vec_obj.transform(texts)
        return X, False
    except Exception:
        # try vocabulary-only fallback
        vocab = getattr(vec_obj, "vocabulary_", None)
        if vocab and isinstance(vocab, dict) and len(vocab) > 0:
            fallback = CountVectorizer(vocabulary=vocab)
            X = fallback.transform(texts)
            return X, True
        # nothing available
        raise

# -------------------------
# Prediction functions (use rule-based fallback when needed)
# -------------------------
def pred_sentiment(texts):
    """
    Returns: (labels_list, probs_list_or_None, fallback_flag(bool)).
    fallback_flag True means either vocabulary fallback or rule-based fallback used.
    """
    # if model or vectorizer not loaded -> use rule-based fallback
    if sent_model is None or sent_vec is None:
        return fallback_sentiment(texts)

    # try transform (TF-IDF or vocab fallback)
    try:
        X, vec_fallback = safe_transform(sent_vec, texts)
    except Exception:
        return fallback_sentiment(texts)

    # try to predict using model
    try:
        labels = sent_model.predict(X)
    except Exception:
        return fallback_sentiment(texts)

    # try probabilities
    try:
        proba_arr = sent_model.predict_proba(X)
        classes = list(sent_model.classes_)
        probs = [dict(zip(classes, p)) for p in proba_arr]
    except Exception:
        probs = [None] * len(labels)

    return labels.tolist() if hasattr(labels, "tolist") else list(labels), probs, bool(vec_fallback)

def pred_rating(texts):
    """
    Returns: (labels_list, probs_list_or_None, fallback_flag(bool)).
    labels are integers 1..5 (if using model) or fallback integers (1..5).
    """
    if rating_model is None or rating_vec is None:
        return fallback_rating(texts)

    try:
        X, vec_fallback = safe_transform(rating_vec, texts)
    except Exception:
        return fallback_rating(texts)

    try:
        labels = rating_model.predict(X)
    except Exception:
        return fallback_rating(texts)

    try:
        proba_arr = rating_model.predict_proba(X)
        classes = list(rating_model.classes_)
        probs = [dict(zip(classes, p)) for p in proba_arr]
    except Exception:
        probs = [None] * len(labels)

    return labels.tolist() if hasattr(labels, "tolist") else list(labels), probs, bool(vec_fallback)

# ------------------------------------------------
#                 PAGE UI (NO SIDEBAR)
# ------------------------------------------------
st.markdown("<h1 style='text-align:center;'>📊 Local Business Review Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align:center;'>Ahmedabad & Bangalore</h3>", unsafe_allow_html=True)
st.write("")
st.write("This tool predicts **sentiment** (positive / neutral / negative) and **star rating (1–5)** from customer reviews using NLP models.")
st.write("If model/vectorizer files are missing or incompatible, the app uses a safe fallback (so demo/submission works).")

# ------------------------------------------------
# Single Review Prediction
# ------------------------------------------------
st.markdown("## 🔍 Single Review Prediction")
review = st.text_area("Paste a customer review:", height=160, placeholder="Type or paste customer review...")

if st.button("Predict"):
    if not review or not review.strip():
        st.warning("Please enter a review.")
    else:
        labels_s, probs_s, fallback_s = pred_sentiment([review])
        labels_r, probs_r, fallback_r = pred_rating([review])

        # If fallback used (either vocab fallback or rule-based fallback), inform the user
        if fallback_s or fallback_r:
            st.info("⚠️ A fallback was used for this prediction. (Either TF-IDF was unavailable or model files were missing/incompatible.)")

        st.markdown("### 🔎 Prediction Result")
        st.write(f"**Sentiment:** `{labels_s[0]}`")
        if probs_s and probs_s[0]:
            st.write("Sentiment probabilities:")
            st.json(probs_s[0])

        st.write(f"**Predicted Rating:** ⭐ {labels_r[0]}")
        if probs_r and probs_r[0]:
            dfp = pd.DataFrame.from_dict(probs_r[0], orient='index', columns=['prob']).reset_index()
            dfp.columns = ['Rating', 'Probability']
            dfp = dfp.sort_values('Probability', ascending=False)
            st.table(dfp)

# ------------------------------------------------
# Batch CSV Prediction
# ------------------------------------------------
st.markdown("## 📂 Batch Prediction (CSV Upload)")
up = st.file_uploader("Upload a CSV containing review text (max 200MB)", type=["csv"])

if up:
    try:
        df = pd.read_csv(up)
    except Exception:
        st.error("Could not read CSV file — ensure it's a valid CSV.")
        df = None

    if df is not None:
        st.write("### Preview")
        st.dataframe(df.head())

        columns = df.columns.tolist()
        default_col = 0
        for i, c in enumerate(columns):
            if "review" in c.lower() or "text" in c.lower():
                default_col = i
                break

        col = st.selectbox("Select the review column:", columns, index=default_col)

        if st.button("Run Batch Prediction"):
            texts = df[col].astype(str).fillna("").tolist()
            s_lbls, s_probas, s_fb = pred_sentiment(texts)
            r_lbls, r_probas, r_fb = pred_rating(texts)

            if s_fb or r_fb:
                st.info("⚠️ A fallback was used for some predictions (TF-IDF unavailable or model missing).")

            df['pred_sentiment'] = s_lbls
            df['pred_sentiment_proba'] = [str(p) if p else "" for p in s_probas]
            df['pred_rating'] = r_lbls
            df['pred_rating_proba'] = [str(p) if p else "" for p in r_probas]

            st.success("Predictions added to dataframe!")
            st.dataframe(df.head())

            st.download_button(
                "Download results (CSV)",
                df.to_csv(index=False).encode("utf-8"),
                "predictions.csv",
                "text/csv"
            )

st.markdown("---")
st.caption("Developed by Pratham Sonpal.")

