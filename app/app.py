import streamlit as st
import joblib
import re
import nltk
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK data
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("punkt")

# Absolute base path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "pkl", "sentiment_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "pkl", "tfidf_vectorizer.pkl")

# Load model and vectorizer
model = joblib.load(MODEL_PATH)
tfidf = joblib.load(VECTORIZER_PATH)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

st.title("🛍️ Product Review Sentiment Analyzer")
st.write("Enter a product review to predict sentiment")

user_input = st.text_area("Review Text")

if st.button("Predict Sentiment"):
    if not user_input.strip():
        st.warning("Please enter a review")
    else:
        clean_text = preprocess(user_input)
        vectorized_text = tfidf.transform([clean_text])
        prediction = model.predict(vectorized_text)[0]

        if prediction == 1:
            st.success("✅ Positive Review")
        else:
            st.error("❌ Negative Review")
