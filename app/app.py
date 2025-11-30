# app.py — Clean final UI with robust model loading + safe fallbacks
import os
import joblib
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import traceback

st.set_page_config(page_title="Local Business Review Analyzer", layout="wide")

# --- CONSTANTS: expected filenames (we will search both app dir and repo root) ---
FILENAMES = {
    "sent_model": "sentiment_model.joblib",
    "sent_vec": "sentiment_tfidf_vectorizer.joblib",
    "rate_model": "rating_model_LogisticRegression.joblib",
    "rate_vec": "rating_tfidf_vectorizer.joblib",
}

# --- Helper paths: resolve app dir and repo root ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # e.g. /mount/src/.../app
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, os.pardir))

def candidate_paths(name):
    """Return candidate absolute paths to look for a given filename key."""
    fn = FILENAMES[name]
    return [
        os.path.join(BASE_DIR, fn),
        os.path.join(REPO_ROOT, fn),
    ]

# --- Model load + status container ---
class ModelBundle:
    def __init__(self):
        self.sent_model = None
        self.sent_vec = None
        self.rate_model = None
        self.rate_vec = None
        self.load_errors = {}  # key -> exception string
        self.found_paths = {}  # key -> path used

BUNDLE = ModelBundle()

def try_load(path):
    """Try to joblib.load and capture exceptions (return object or raise)."""
    obj = joblib.load(path)
    return obj

@st.cache_resource(show_spinner=False)
def load_models():
    mb = ModelBundle()
    # For each expected file, try candidate paths
    for key in FILENAMES.keys():
        loaded = None
        exc_info = None
        for p in candidate_paths(key):
            try:
                if os.path.exists(p):
                    obj = try_load(p)
                    loaded = obj
                    mb.found_paths[key] = p
                    break
            except Exception as e:
                exc_info = traceback.format_exc()
                # continue to try other candidate path
        if loaded is None:
            if exc_info:
                mb.load_errors[key] = exc_info
        # assign to bundle attribute
        if key == "sent_model":
            mb.sent_model = loaded
        elif key == "sent_vec":
            mb.sent_vec = loaded
        elif key == "rate_model":
            mb.rate_model = loaded
        elif key == "rate_vec":
            mb.rate_vec = loaded
    return mb

# Load once (cached)
BUNDLE = load_models()

# --- Small hidden debug expander (only open if you need details) ---
with st.expander("⚠️ Internal load/debug info (expand only if needed)"):
    st.write("Base app folder:", BASE_DIR)
    st.write("Repo root:", REPO_ROOT)
    for k in FILENAMES.keys():
        st.write(k, "expected filename:", FILENAMES[k])
        st.write("found path:", BUNDLE.found_paths.get(k, "Not found"))
        if k in BUNDLE.load_errors:
            st.text_area(f"{k} load error", BUNDLE.load_errors[k], height=200)

# --- utilities: safe transform + fallback handling ---
def safe_transform(vec_obj, texts):
    """
    Try to use vectorizer.transform(); if it fails (common: 'idf not fitted' or binary issues),
    fallback to CountVectorizer(vocabulary=vec_obj.vocabulary_) quietly.
    Returns (X, used_fallback:boolean).
    """
    if vec_obj is None:
        raise RuntimeError("Vectorizer object is None.")
    try:
        X = vec_obj.transform(texts)
        return X, False
    except Exception as e:
        # fallback path
        vocab = getattr(vec_obj, "vocabulary_", None)
        if vocab and isinstance(vocab, dict):
            fallback_vec = CountVectorizer(vocabulary=vocab)
            X = fallback_vec.transform(texts)
            return X, True
        # nothing to fallback to
        raise

# --- Fallback rule-based models (VADER) ---
vader = SentimentIntensityAnalyzer()
def vader_sentiment_labels(texts):
    labels = []
    for t in texts:
        s = vader.polarity_scores(str(t))["compound"]
        if s >= 0.05:
            labels.append("positive")
        elif s <= -0.05:
            labels.append("negative")
        else:
            labels.append("neutral")
    return labels

def vader_rating_estimate(texts):
    out = []
    for t in texts:
        s = vader.polarity_scores(str(t))["compound"]
        if s >= 0.5: out.append(5)
        elif s >= 0.2: out.append(4)
        elif s >= -0.1: out.append(3)
        elif s >= -0.4: out.append(2)
        else: out.append(1)
    return out

# --- Prediction wrapper that tries model path, else VADER fallback ---
def predict_sentiment(texts):
    """Return (labels_list, probs_list_or_None, used_fallback_bool, source_str)"""
    if BUNDLE.sent_model is None or BUNDLE.sent_vec is None:
        # Model missing — use VADER fallback
        return vader_sentiment_labels(texts), [None]*len(texts), True, "vader_fallback_missing"
    # try model path
    try:
        X, used_vocab_fallback = safe_transform(BUNDLE.sent_vec, texts)
        preds = BUNDLE.sent_model.predict(X)
        try:
            proba = BUNDLE.sent_model.predict_proba(X)
            # map to dict per sample
            classes = list(BUNDLE.sent_model.classes_)
            probs = [dict(zip(classes, p)) for p in proba]
        except Exception:
            probs = [None] * len(preds)
        return list(preds), probs, used_vocab_fallback, "model"
    except Exception as e:
        # on any transform/predict error -> fallback to VADER
        return vader_sentiment_labels(texts), [None]*len(texts), True, "vader_fallback_error"

def predict_rating(texts):
    """Return (labels_list, probs_list_or_None, used_fallback_bool, source_str)"""
    if BUNDLE.rate_model is None or BUNDLE.rate_vec is None:
        return vader_rating_estimate(texts), [None]*len(texts), True, "vader_fallback_missing"
    try:
        X, used_vocab_fallback = safe_transform(BUNDLE.rate_vec, texts)
        preds = BUNDLE.rate_model.predict(X)
        try:
            proba = BUNDLE.rate_model.predict_proba(X)
            classes = list(BUNDLE.rate_model.classes_)
            probs = [dict(zip(classes, p)) for p in proba]
        except Exception:
            probs = [None] * len(preds)
        # ensure integer stars (sometimes preds are numpy types)
        preds = [int(p) for p in preds]
        return preds, probs, used_vocab_fallback, "model"
    except Exception as e:
        return vader_rating_estimate(texts), [None]*len(texts), True, "vader_fallback_error"

# -----------------------
# UI (clean, no sidebar)
# -----------------------
st.markdown("<h1 style='text-align:center;'>📊 Local Business Review Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center; margin-top:-12px;'>Ahmedabad & Bangalore — Sentiment & Rating</h4>", unsafe_allow_html=True)
st.write("")
st.write("Paste a single review or upload a CSV (review column). The app will predict sentiment (positive/neutral/negative) and an estimated star rating (1–5).")

# show a compact model status strip
def compact_status():
    def ok(x): return "🟢" if x else "🔴"
    st.markdown(
        f"**Model status — Sentiment:** {ok(BUNDLE.sent_model)} {ok(BUNDLE.sent_vec)}   •   "
        f"**Rating:** {ok(BUNDLE.rate_model)} {ok(BUNDLE.rate_vec)}"
    )
    if any(k in BUNDLE.load_errors for k in BUNDLE.load_errors):
        st.caption("Some model files had load errors — app may use fallback. Expand the debug panel for details.")

compact_status()
st.markdown("---")

# Single review
st.subheader("1) Single review prediction")
single = st.text_area("Paste a review here:", height=140, placeholder="Customer review text...")
if st.button("Predict"):
    if not single or not single.strip():
        st.warning("Enter review text to analyze.")
    else:
        texts = [single.strip()]
        s_labels, s_probs, s_fallback, s_src = predict_sentiment(texts)
        r_labels, r_probs, r_fallback, r_src = predict_rating(texts)

        if s_fallback or r_fallback:
            st.info("⚠️ A fallback was used for this prediction (TF-IDF/model unavailable or incompatible). Result is still provided for demo purposes.")

        st.markdown("### Result")
        st.write(f"**Sentiment:** `{s_labels[0]}`")
        if s_probs[0]:
            st.write("Sentiment probabilities:")
            st.json(s_probs[0])
        st.write(f"**Predicted rating:** ⭐ {r_labels[0]}")
        if r_probs[0]:
            dfp = pd.DataFrame.from_dict(r_probs[0], orient='index', columns=["prob"]).reset_index()
            dfp.columns = ["rating", "probability"]
            dfp["rating"] = dfp["rating"].astype(str)
            dfp = dfp.sort_values("probability", ascending=False)
            st.table(dfp)

st.markdown("---")

# Batch CSV
st.subheader("2) Batch prediction (CSV upload)")
uploaded = st.file_uploader("Upload a CSV with a review text column", type=["csv"])
if uploaded:
    try:
        df = pd.read_csv(uploaded)
    except Exception:
        st.error("Failed to read CSV. Ensure it's a valid CSV file.")
        df = None

    if df is not None:
        st.write("Preview (first 5 rows):")
        st.dataframe(df.head())
        cols = df.columns.tolist()
        default_index = 0
        for i, c in enumerate(cols):
            if "review" in c.lower() or "text" in c.lower():
                default_index = i
                break
        review_col = st.selectbox("Select review text column", cols, index=default_index)
        if st.button("Run batch predictions"):
            texts = df[review_col].astype(str).fillna("").tolist()
            s_labels, s_probs, s_fallback, s_src = predict_sentiment(texts)
            r_labels, r_probs, r_fallback, r_src = predict_rating(texts)
            if s_fallback or r_fallback:
                st.info("⚠️ Fallback used for some or all rows (TF-IDF/model unavailable).")
            df["pred_sentiment"] = s_labels
            df["pred_sentiment_proba"] = [str(p) if p else "" for p in s_probs]
            df["pred_rating"] = r_labels
            df["pred_rating_proba"] = [str(p) if p else "" for p in r_probs]
            st.success("Predictions added to dataframe.")
            st.dataframe(df.head())
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download predictions CSV", csv_bytes, file_name="predictions.csv")

st.markdown("---")
st.caption("If predictions show fallback or 'TF-IDF unavailable', either (1) upload the four joblib files into the app folder or (2) re-save the joblib files in an environment whose numpy/scikit-learn versions match your deployment environment.")

# small demo tester (quick)
if st.checkbox("Show small demo examples"):
    demo = pd.DataFrame({
        "review_text": [
            "Amazing service, staff were friendly and very helpful!",
            "Terrible experience — dirty and rude staff. Will not return.",
            "Okay experience, average pricing and decent service.",
            "Loved it! Reasonable price and prompt.",
            "Too slow and overpriced."
        ]
    })
    st.dataframe(demo)
    if st.button("Run demo predictions"):
        texts = demo["review_text"].tolist()
        s_labels, s_probs, s_fallback, s_src = predict_sentiment(texts)
        r_labels, r_probs, r_fallback, r_src = predict_rating(texts)
        demo["pred_sentiment"] = s_labels
        demo["pred_rating"] = r_labels
        st.dataframe(demo)

# End
