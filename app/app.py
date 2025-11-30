# app.py — Minimal, production-looking UI (no debug banners)
import os
import joblib
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer

st.set_page_config(page_title="Local Business Review Analyzer", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_SENTIMENT = os.path.join(BASE_DIR, "sentiment_model.joblib")
VEC_SENTIMENT   = os.path.join(BASE_DIR, "sentiment_tfidf_vectorizer.joblib")
MODEL_RATING    = os.path.join(BASE_DIR, "rating_model_LogisticRegression.joblib")
VEC_RATING      = os.path.join(BASE_DIR, "rating_tfidf_vectorizer.joblib")

def safe_load(path):
    """Load a joblib file; return None if missing or load fails (silently)."""
    if not os.path.exists(path):
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None

@st.cache_resource(show_spinner=False)
def load_models():
    sent_model = safe_load(MODEL_SENTIMENT)
    sent_vec = safe_load(VEC_SENTIMENT)
    rating_model = safe_load(MODEL_RATING)
    rating_vec = safe_load(VEC_RATING)
    return sent_model, sent_vec, rating_model, rating_vec

sent_model, sent_vec, rating_model, rating_vec = load_models()

# --- Sidebar: compact status only ---
with st.sidebar:
    st.header("Model status")
    def tick(x): return "✅ Loaded" if x is not None else "❌ Missing"
    st.markdown(f"**Sentiment model:** {tick(sent_model)}")
    st.markdown(f"**Sentiment vectorizer:** {tick(sent_vec)}")
    st.markdown(f"**Rating model:** {tick(rating_model)}")
    st.markdown(f"**Rating vectorizer:** {tick(rating_vec)}")
    st.markdown("---")
    st.markdown("Notes:")
    st.markdown("- Put the 4 `.joblib` files in the `app/` folder before deploy.")
    st.markdown("- If any file is missing, upload and redeploy.")

# --------------------
# Transform helper with vocabulary-only fallback (quiet)
# --------------------
def safe_transform(vec_obj, texts):
    """Try transform; if TF-IDF 'not fitted' style failure happens,
       fallback to CountVectorizer(vocabulary=vec_obj.vocabulary_) if available.
       Returns (X, fallback_used_bool).
    """
    try:
        X = vec_obj.transform(texts)
        return X, False
    except Exception as e:
        vocab = getattr(vec_obj, "vocabulary_", None)
        if vocab and isinstance(vocab, dict) and len(vocab) > 0:
            fallback = CountVectorizer(vocabulary=vocab)
            X = fallback.transform(texts)
            return X, True
        raise

# --------------------
# Prediction functions
# --------------------
def pred_sentiment(texts):
    if sent_vec is None or sent_model is None:
        raise RuntimeError("Sentiment model or vectorizer not loaded.")
    X, fallback_used = safe_transform(sent_vec, texts)
    y_pred = sent_model.predict(X)
    try:
        y_proba = sent_model.predict_proba(X)
        classes = list(sent_model.classes_)
        probs = [dict(zip(classes, p)) for p in y_proba]
    except Exception:
        probs = [None] * len(y_pred)
    return y_pred, probs, fallback_used

def pred_rating(texts):
    if rating_vec is None or rating_model is None:
        raise RuntimeError("Rating model or vectorizer not loaded.")
    X, fallback_used = safe_transform(rating_vec, texts)
    y_pred = rating_model.predict(X)
    try:
        y_proba = rating_model.predict_proba(X)
        classes = list(rating_model.classes_)
        probs = [dict(zip(classes, p)) for p in y_proba]
    except Exception:
        probs = [None] * len(y_pred)
    return y_pred, probs, fallback_used

# --------------------
# Page UI
# --------------------
st.title("Local Business Review Analyzer — Ahmedabad & Bangalore")
st.markdown("Predict sentiment (positive/neutral/negative) and star rating (1–5) from review text.")

# Single-review prediction
st.header("1) Single review prediction")
review_input = st.text_area("Paste a review here", height=140, placeholder="Type or paste customer review...")

if st.button("Predict (Single)"):
    if not review_input or not review_input.strip():
        st.warning("Please paste a review to predict.")
    else:
        try:
            sent_label, sent_probs, sent_fallback = pred_sentiment([review_input.strip()])
            rating_label, rating_probs, rating_fallback = pred_rating([review_input.strip()])
        except RuntimeError as e:
            st.error(str(e))
        except Exception as e:
            st.error("Prediction failed: " + str(e))
        else:
            # single user-friendly fallback warning if used
            if sent_fallback or rating_fallback:
                st.warning("TF-IDF transform failed; a vocabulary-only fallback was used for this prediction (may slightly change scores).")
            st.markdown("**Prediction result**")
            st.write(f"**Predicted sentiment:** `{sent_label[0]}`")
            if sent_probs[0]:
                st.write("Sentiment probabilities:")
                st.json(sent_probs[0])
            st.write(f"**Predicted rating:** `{rating_label[0]}`")
            if rating_probs[0]:
                proba_df = pd.DataFrame.from_dict(rating_probs[0], orient='index', columns=['prob']).reset_index()
                proba_df.columns = ['rating', 'probability']
                proba_df['rating'] = proba_df['rating'].astype(str)
                proba_df = proba_df.sort_values('probability', ascending=False)
                st.table(proba_df)

# Batch predictions
st.header("2) Batch predictions (CSV)")
st.markdown("Upload a CSV with a review column — app will add `pred_sentiment`, `pred_rating` and probability columns.")
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error("Error reading CSV: " + str(e))
        df = None

    if df is not None:
        st.write("Preview:")
        st.dataframe(df.head())
        cols = df.columns.tolist()
        default_idx = 0
        for i,c in enumerate(cols):
            if "review" in c.lower() or "text" in c.lower():
                default_idx = i; break
        review_col = st.selectbox("Which column has review text?", cols, index=default_idx)

        if st.button("Run batch predictions"):
            texts = df[review_col].astype(str).fillna("").tolist()
            if len([t for t in texts if t.strip()]) == 0:
                st.warning("No non-empty reviews found in selected column.")
            else:
                try:
                    sent_labels, sent_probs, sent_fallback = pred_sentiment(texts)
                    rating_labels, rating_probs, rating_fallback = pred_rating(texts)
                except RuntimeError as e:
                    st.error(str(e))
                except Exception as e:
                    st.error("Batch prediction failed: " + str(e))
                else:
                    if sent_fallback or rating_fallback:
                        st.warning("TF-IDF transform failed for some predictions; vocabulary-only fallback was used.")
                    df["pred_sentiment"] = sent_labels
                    df["pred_sentiment_proba"] = [str(p) if p is not None else "" for p in sent_probs]
                    df["pred_rating"] = rating_labels
                    df["pred_rating_proba"] = [str(p) if p is not None else "" for p in rating_probs]
                    st.success("Predictions added to dataframe!")
                    st.dataframe(df.head())
                    csv_bytes = df.to_csv(index=False).encode("utf-8")
                    st.download_button("Download predictions CSV", csv_bytes, file_name="predictions_with_model.csv", mime="text/csv")

st.markdown("---")
st.caption("Ensure joblib files are compatible with deployed Python / numpy / scikit-learn versions to avoid TF-IDF fallback.")
