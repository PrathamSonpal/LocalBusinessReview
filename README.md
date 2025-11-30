# Local Business Review Analyzer
### Sentiment & Star Rating Prediction for Ahmedabad & Bangalore

---

## Project Summary
This project predicts **sentiment** (positive / neutral / negative) and **star rating (1–5)** from customer reviews using **NLP + Machine Learning**.

It includes:

- A trained ML sentiment classification model
- A trained ML rating prediction model
- A fallback rule-based system to avoid app crashes
- A Streamlit web app
- CSV batch prediction feature

This project was made as part of a **Data Science group project**, developed by **Pratham Sonpal**.

---

## Dataset
The dataset contains reviews from two Indian cities:

- Ahmedabad  
- Bangalore

After cleaning:

- ~2,000 rows per category (gyms, salons, restaurants) — combined ~6,000 rows  
- Columns: `business_name`, `review_text`, `rating`, `city`, `category`

---

## Data Cleaning Steps

- Lowercasing
- Removing punctuation, emojis and URLs
- Stopword removal
- Lemmatization
- Handling missing values
- Rating normalization

---

## Machine Learning Models

### Sentiment Model
- Algorithm: Logistic Regression  
- Features: TF-IDF (20,000 features)  
- Classes: `positive`, `neutral`, `negative`  
- Typical accuracy: ~94% (on cleaned test set)

### Rating Model
- Algorithm: Logistic Regression  
- Features: TF-IDF (25,000 features)  
- Output: Rating 1–5  
- Typical accuracy: ~93–94% (on cleaned test set)

---

## Fallback System
To avoid the app breaking due to environment/model version mismatches:

- If TF-IDF transform fails, fallback to `CountVectorizer` using saved vocabulary.
- If ML model is incompatible, fallback to rule-based VADER sentiment (for sentiment) and mapped rating heuristic.
- These fallbacks keep the app usable (results marked as fallback).

---

## Streamlit Web App Features

### Single Review Prediction
- Paste one review and get:
  - Predicted sentiment (and probabilities)
  - Predicted star rating (and probability distribution)

### Batch Prediction (CSV)
- Upload a CSV with reviews
- App auto-detects review column (or you select it)
- App returns a CSV with added columns:
  - `pred_sentiment`
  - `pred_sentiment_proba`
  - `pred_rating`
  - `pred_rating_proba`

---

## Tech Stack

- Python 3.10+
- pandas, numpy
- scikit-learn
- TF-IDF Vectorizer
- VADER Sentiment Analyzer
- Streamlit
- joblib

---
