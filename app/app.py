import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from io import BytesIO

st.set_page_config(page_title="Local Business Review Analyzer", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_SENTIMENT = os.path.join(BASE_DIR, "sentiment_model.joblib")
VEC_SENTIMENT   = os.path.join(BASE_DIR, "sentiment_tfidf_vectorizer.joblib")
MODEL_RATING    = os.path.join(BASE_DIR, "rating_model_LogisticRegression.joblib")
VEC_RATING      = os.path.join(BASE_DIR, "rating_tfidf_vectorizer.joblib")

def safe_load(path):
    try:
        if not os.path.exists(path):
            # helpful error for missing file
            raise FileNotFoundError(path)
        return joblib.load(path)
    except FileNotFoundError:
        st.error(f"File not found: {path}")
    except Exception as e:
        st.error(f"Error loading {path}: {e}")
    return None

@st.cache_resource(show_spinner=False)
def load_models():
    sent_model = safe_load(MODEL_SENTIMENT)
    sent_vec = safe_load(VEC_SENTIMENT)
    rating_model = safe_load(MODEL_RATING)
    rating_vec = safe_load(VEC_RATING)
    return sent_model, sent_vec, rating_model, rating_vec

sent_model, sent_vec, rating_model, rating_vec = load_models()

# --- DEBUG: show file listings to help Streamlit Cloud troubleshooting ---
with st.expander("Debug: show file listings (click to expand)"):
    try:
        st.write("Repo working dir:", os.getcwd())
        st.write("Files in repo root:", sorted(os.listdir(".")))
    except Exception as e:
        st.write("Could not list repo root:", e)
    try:
        st.write("App folder (BASE_DIR):", BASE_DIR)
        st.write("Files in app folder:", sorted(os.listdir(BASE_DIR)))
    except Exception as e:
        st.write("Could not list app folder:", e)
# -----------------------------------------------------------------------

# --- UI ---
st.title("Local Business Review Analyzer — Ahmedabad & Bangalore")
st.markdown(
    """
    Predict sentiment and star rating (1-5) from review text.
    - Paste a single review or upload CSV with a review column.
    - Models & vectorizers should be placed in the same folder as `app.py`.
    """
)

with st.sidebar:
    st.header("Model status")
    def good(x): return "✅" if x is not None else "❌"
    st.write(f"Sentiment model: {good(sent_model)}")
    st.write(f"Sentiment vectorizer: {good(sent_vec)}")
    st.write(f"Rating model: {good(rating_model)}")
    st.write(f"Rating vectorizer: {good(rating_vec)}")
    st.markdown("---")
    st.markdown("**Notes**\n- If any file shows ❌, upload the matching joblib files into the app folder.\n- Filenames expected in this app: `sentiment_model.joblib`, `sentiment_tfidf_vectorizer.joblib`, `rating_model_LogisticRegression.joblib`, `rating_tfidf_vectorizer.joblib`.")

def pred_sentiment(texts):
    """Return predicted label and probability (if available)."""
    if sent_vec is None or sent_model is None:
        raise RuntimeError("Sentiment model or vectorizer not loaded.")
    X = sent_vec.transform(texts)
    # predict label
    y_pred = sent_model.predict(X)
    # try predict_proba
    try:
        y_proba = sent_model.predict_proba(X)
        # create dict of probabilities with classes
        class_names = list(sent_model.classes_)
        probs = [dict(zip(class_names, p)) for p in y_proba]
    except Exception:
        probs = [None] * len(y_pred)
    return y_pred, probs

def pred_rating(texts):
    """Return predicted rating (1-5) and probability distribution (if available)."""
    if rating_vec is None or rating_model is None:
        raise RuntimeError("Rating model or vectorizer not loaded.")
    X = rating_vec.transform(texts)
    y_pred = rating_model.predict(X)
    try:
        y_proba = rating_model.predict_proba(X)
        classes = list(rating_model.classes_)
        probs = [dict(zip(classes, p)) for p in y_proba]
    except Exception:
        probs = [None] * len(y_pred)
    return y_pred, probs

st.subheader("1) Predict for a single review")
review_input = st.text_area("Paste a review here", height=140)
col1, col2 = st.columns(2)
with col1:
    if st.button("Predict (Single)"):
        if not review_input or not review_input.strip():
            st.warning("Please paste a review to predict.")
        else:
            try:
                sent_label, sent_probs = pred_sentiment([review_input.strip()])
                rating_label, rating_probs = pred_rating([review_input.strip()])
            except RuntimeError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Prediction failed: {e}")
            else:
                st.markdown("**Prediction result**")
                st.write(f"**Predicted sentiment:** `{sent_label[0]}`")
                if sent_probs[0]:
                    st.write("Sentiment probabilities:")
                    st.json(sent_probs[0])
                st.write(f"**Predicted rating:** `{rating_label[0]}`")
                if rating_probs[0]:
                    # format rating proba sorted descending
                    proba_df = pd.DataFrame.from_dict(rating_probs[0], orient='index', columns=['prob']).reset_index()
                    proba_df.columns = ['rating', 'probability']
                    proba_df['rating'] = proba_df['rating'].astype(str)
                    proba_df = proba_df.sort_values('probability', ascending=False)
                    st.table(proba_df)

st.subheader("2) Batch predictions (CSV)")
st.markdown("Upload a CSV containing a column with review text. The app will add `pred_sentiment`, `pred_sentiment_proba`, `pred_rating`, `pred_rating_proba` columns.")

uploaded_file = st.file_uploader("Upload CSV file (or leave empty to skip batch)", type=["csv"])
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        df = None

    if df is not None:
        st.write("CSV preview (first 5 rows):")
        st.dataframe(df.head())

        # let user pick review column
        cols = df.columns.tolist()
        default_col = None
        for c in cols:
            if "review" in c.lower() or "text" in c.lower():
                default_col = c
                break
        review_col = st.selectbox("Select the review text column", cols, index=cols.index(default_col) if default_col else 0)

        if st.button("Run batch predictions"):
            texts = df[review_col].astype(str).fillna("").tolist()
            # drop empty
            if len([t for t in texts if t.strip()]) == 0:
                st.warning("No non-empty reviews found in selected column.")
            else:
                try:
                    sent_labels, sent_probs = pred_sentiment(texts)
                    rating_labels, rating_probs = pred_rating(texts)
                except RuntimeError as e:
                    st.error(str(e))
                except Exception as e:
                    st.error(f"Batch prediction failed: {e}")
                else:
                    df["pred_sentiment"] = sent_labels
                    # convert prob dicts to JSON strings for saving
                    df["pred_sentiment_proba"] = [str(p) if p is not None else "" for p in sent_probs]
                    df["pred_rating"] = rating_labels
                    df["pred_rating_proba"] = [str(p) if p is not None else "" for p in rating_probs]
                    st.success("Predictions added to dataframe!")
                    st.dataframe(df.head())

                    # allow download
                    csv_bytes = df.to_csv(index=False).encode("utf-8")
                    st.download_button("Download predictions CSV", csv_bytes, file_name="predictions_with_model.csv", mime="text/csv")

st.markdown("---")
st.subheader("Quick tips / troubleshooting")
st.markdown("""
- If any model file is missing, upload the joblib files into the same folder as `app.py`.
- If the app loads but predictions raise `No model` errors, confirm the model and vectorizer variable names used when you saved them match those expected here.
- For deploying on Streamlit Cloud: include the 4 joblib files and `requirements.txt` in repo, set the app path to `app/app.py`.
""")

st.markdown("---")
if st.checkbox("Create a tiny demo dataset (5 rows)"):
    demo = pd.DataFrame({
        "business_name": ["Demo A","Demo B","Demo C","Demo D","Demo E"],
        "review_text": [
            "Excellent staff and quick service, loved it!",
            "Terrible experience, unhygienic and rude staff.",
            "Average, nothing special.",
            "Great atmosphere and value for money.",
            "Too expensive and not worth the wait."
        ]
    })
    st.dataframe(demo)
    if st.button("Predict demo (this requires models)"):
        try:
            sl, sp = pred_sentiment(demo["review_text"].tolist())
            rl, rp = pred_rating(demo["review_text"].tolist())
            demo["pred_sentiment"] = sl
            demo["pred_rating"] = rl
            st.dataframe(demo)
        except RuntimeError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"Demo prediction failed: {e}")
