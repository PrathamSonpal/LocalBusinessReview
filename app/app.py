# app.py — Clean user-facing UI (no visible fallback message). Hidden debug panel for issues.
import os
import joblib
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import traceback

st.set_page_config(page_title="Local Business Review Analyzer", layout="wide")

# Expected filenames (app tries both app folder and repo root)
FILENAMES = {
    "sent_model": "sentiment_model.joblib",
    "sent_vec": "sentiment_tfidf_vectorizer.joblib",
    "rate_model": "rating_model_LogisticRegression.joblib",
    "rate_vec": "rating_tfidf_vectorizer.joblib",
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, os.pardir))

def candidate_paths(key):
    fn = FILENAMES[key]
    return [os.path.join(BASE_DIR, fn), os.path.join(REPO_ROOT, fn)]

class ModelBundle:
    def __init__(self):
        self.sent_model = None
        self.sent_vec = None
        self.rate_model = None
        self.rate_vec = None
        self.load_errors = {}
        self.found_paths = {}

def try_load(path):
    return joblib.load(path)

@st.cache_resource(show_spinner=False)
def load_models():
    mb = ModelBundle()
    for key in FILENAMES.keys():
        loaded = None
        exc = None
        for p in candidate_paths(key):
            try:
                if os.path.exists(p):
                    obj = try_load(p)
                    loaded = obj
                    mb.found_paths[key] = p
                    break
            except Exception:
                exc = traceback.format_exc()
        if loaded is None and exc:
            mb.load_errors[key] = exc
        if key == "sent_model": mb.sent_model = loaded
        if key == "sent_vec": mb.sent_vec = loaded
        if key == "rate_model": mb.rate_model = loaded
        if key == "rate_vec": mb.rate_vec = loaded
    return mb

BUNDLE = load_models()

# Hidden debug panel (only expand if you want technical details)
with st.expander("Debug & model load details (expand only if needed)"):
    st.write("App folder:", BASE_DIR)
    st.write("Repo root:", REPO_ROOT)
    for k in FILENAMES.keys():
        st.write(f"- {k}: expected `{FILENAMES[k]}`")
        st.write("  found path:", BUNDLE.found_paths.get(k, "Not found"))
        if k in BUNDLE.load_errors:
            st.text_area(f"{k} load error trace", BUNDLE.load_errors[k], height=240)

# Safe transform helper: try TF-IDF.transform, fallback to CountVectorizer(vocabulary)
def safe_transform(vec_obj, texts):
    if vec_obj is None:
        raise RuntimeError("Vectorizer object is None.")
    try:
        X = vec_obj.transform(texts)
        return X, False
    except Exception:
        vocab = getattr(vec_obj, "vocabulary_", None)
        if vocab and isinstance(vocab, dict):
            fallback = CountVectorizer(vocabulary=vocab)
            X = fallback.transform(texts)
            return X, True
        raise

# VADER fallback (silent to UI)
vader = SentimentIntensityAnalyzer()
def vader_sentiment_labels(texts):
    labels = []
    probs = []
    for t in texts:
        s = vader.polarity_scores(str(t))["compound"]
        if s >= 0.05: lbl = "positive"
        elif s <= -0.05: lbl = "negative"
        else: lbl = "neutral"
        labels.append(lbl)
        # create a simple probability-like dict for display (not a true model proba)
        probs.append({"positive": max(0, s), "neutral": max(0, 1 - abs(s)), "negative": max(0, -s)})
    return labels, probs

def vader_rating_estimate(texts):
    out = []
    probs = []
    for t in texts:
        s = vader.polarity_scores(str(t))["compound"]
        if s >= 0.5: r = 5
        elif s >= 0.2: r = 4
        elif s >= -0.1: r = 3
        elif s >= -0.4: r = 2
        else: r = 1
        out.append(r)
        probs.append(None)
    return out, probs

# Model prediction wrappers — if models exist use them; otherwise silent fallback to VADER
def predict_sentiment(texts):
    if BUNDLE.sent_model is None or BUNDLE.sent_vec is None:
        return vader_sentiment_labels(texts) + (True, "vader")
    try:
        X, used_vocab_fallback = safe_transform(BUNDLE.sent_vec, texts)
        y = BUNDLE.sent_model.predict(X)
        try:
            y_proba = BUNDLE.sent_model.predict_proba(X)
            classes = list(BUNDLE.sent_model.classes_)
            probs = [dict(zip(classes, p)) for p in y_proba]
        except Exception:
            probs = [None]*len(y)
        return list(y), probs, used_vocab_fallback, "model"
    except Exception:
        return vader_sentiment_labels(texts) + (True, "vader")

def predict_rating(texts):
    if BUNDLE.rate_model is None or BUNDLE.rate_vec is None:
        return vader_rating_estimate(texts) + (True, "vader")
    try:
        X, used_vocab_fallback = safe_transform(BUNDLE.rate_vec, texts)
        y = BUNDLE.rate_model.predict(X)
        try:
            y_proba = BUNDLE.rate_model.predict_proba(X)
            classes = list(BUNDLE.rate_model.classes_)
            probs = [dict(zip(classes, p)) for p in y_proba]
        except Exception:
            probs = [None]*len(y)
        # ensure ints
        y = [int(x) for x in y]
        return y, probs, used_vocab_fallback, "model"
    except Exception:
        return vader_rating_estimate(texts) + (True, "vader")

# ---------- UI (clean, user-friendly) ----------
st.markdown("<h1 style='text-align:center;'>📊 Local Business Review Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center; margin-top:-12px;'>Ahmedabad & Bangalore — Sentiment & Star Rating</h4>", unsafe_allow_html=True)
st.write("")
st.write("Paste a customer review below or upload a CSV with a review column. The app predicts customer sentiment (positive / neutral / negative) and an estimated star rating (1–5).")

# compact model status (non-technical)
def compact_status():
    def ok(x): return "🟢" if x else "🔴"
    st.markdown(f"**Model availability (sentiment / rating):** {ok(BUNDLE.sent_model and BUNDLE.sent_vec)}  /  {ok(BUNDLE.rate_model and BUNDLE.rate_vec)}")
compact_status()
st.markdown("---")

# Single review UI
st.subheader("1) Analyze a single review")
single = st.text_area("Paste the review here", height=160, placeholder="e.g. Great staff, fast service and clean place!")
if st.button("Analyze review"):
    if not single or not single.strip():
        st.warning("Please paste a review to analyze.")
    else:
        txt = [single.strip()]
        s_labels, s_probs, s_fb, s_src = predict_sentiment(txt)
        r_labels, r_probs, r_fb, r_src = predict_rating(txt)

        # Nice sentiment pill
        sl = s_labels[0].lower()
        if sl == "positive":
            emoji = "😊"
            color = "#2ecc71"
        elif sl == "negative":
            emoji = "😞"
            color = "#e74c3c"
        else:
            emoji = "😐"
            color = "#f39c12"

        st.markdown(
            f"<div style='padding:12px;border-radius:8px;background:{color};display:inline-block;color:white;font-weight:700'>"
            f"{emoji}  {s_labels[0].capitalize()}</div>",
            unsafe_allow_html=True
        )

        # Single-line human-friendly summary
        # Use the top probability if present to give % clarity
        prob_text = ""
        if s_probs and s_probs[0]:
            top_lbl = max(s_probs[0].items(), key=lambda kv: kv[1])[0]
            top_val = s_probs[0][top_lbl]
            prob_text = f" ({top_lbl}: {top_val:.0%} likelihood)"
        st.write(f"Short summary: Customers appear **{s_labels[0].capitalize()}**{prob_text}.")

        # Rating stars (big)
        stars = "★" * int(r_labels[0]) + "☆" * (5 - int(r_labels[0]))
        st.markdown(f"**Estimated star rating:** <span style='font-size:20px'>{stars}</span>  — **{r_labels[0]}/5**", unsafe_allow_html=True)

        # Show probabilities table for rating (if available)
        if r_probs and r_probs[0]:
            dfp = pd.DataFrame.from_dict(r_probs[0], orient="index", columns=["prob"]).reset_index()
            dfp.columns = ["rating", "probability"]
            dfp["rating"] = dfp["rating"].astype(str)
            dfp = dfp.sort_values("probability", ascending=False)
            st.write("Rating probability (top-first):")
            st.table(dfp.style.format({"probability":"{:.2%}"}))

        # Suggestions (small actionable tips)
        st.markdown("**Quick tips to improve perception**")
        if sl == "positive":
            st.write("- Keep the same strengths: friendly staff and hygiene. Encourage reviews.")
        elif sl == "neutral":
            st.write("- Improve small friction points: speed, pricing or cleanliness.")
        else:
            st.write("- Investigate major complaints (service, hygiene, pricing) and respond to customers promptly.")

st.markdown("---")

# Batch prediction
st.subheader("2) Batch predictions (CSV)")
uploaded = st.file_uploader("Upload CSV with review text (CSV)", type=["csv"])
if uploaded:
    try:
        df = pd.read_csv(uploaded)
    except Exception:
        st.error("Could not read CSV. Ensure it's a valid CSV file.")
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
        review_col = st.selectbox("Select review column", cols, index=default_index)
        if st.button("Run batch predictions"):
            texts = df[review_col].astype(str).fillna("").tolist()
            s_labels, s_probs, s_fb, s_src = predict_sentiment(texts)
            r_labels, r_probs, r_fb, r_src = predict_rating(texts)
            df["pred_sentiment"] = s_labels
            df["pred_rating"] = r_labels
            # keep proba fields as strings for saving
            df["pred_sentiment_proba"] = [str(p) if p else "" for p in s_probs]
            df["pred_rating_proba"] = [str(p) if p else "" for p in r_probs]
            st.success("Predictions added to the dataframe.")
            st.dataframe(df.head())
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download results CSV", csv_bytes, file_name="predictions.csv", mime="text/csv")

st.markdown("---")
st.caption("If predictions appear off, open the Debug panel above to inspect model load errors. To fully restore modeled behavior, make sure the four joblib files are built/saved with matching numpy/scikit-learn versions to the deployment environment.")

# small demo (visible)
if st.checkbox("Show demo examples"):
    demo = pd.DataFrame({
        "review_text": [
            "Loved the service and very clean. Highly recommended!",
            "Average place. Service was okay but nothing special.",
            "Rude staff and unhygienic — terrible experience."
        ]
    })
    st.dataframe(demo)
    if st.button("Analyze demo"):
        texts = demo["review_text"].tolist()
        s_lbls, s_pr, s_fb, s_src = predict_sentiment(texts)
        r_lbls, r_pr, r_fb, r_src = predict_rating(texts)
        demo["pred_sentiment"] = s_lbls
        demo["pred_rating"] = r_lbls
        st.dataframe(demo)

