import streamlit as st
import joblib
import re
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ---------------------------
# NLTK setup (run once)
# ---------------------------
nltk.data.path.append(os.path.expanduser("~/nltk_data"))

# ---------------------------
# Load model & vectorizer
# ---------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model = joblib.load(os.path.join(BASE_DIR, "pkl", "sentiment_model.pkl"))
tfidf = joblib.load(os.path.join(BASE_DIR, "pkl", "tfidf_vectorizer.pkl"))

# ---------------------------
# Text preprocessing
# ---------------------------
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Sentiment Analyzer", page_icon="🛍️")

st.title("🛍️ Product Review Sentiment Analyzer")
st.write("Enter a product review to predict sentiment")

user_input = st.text_area("Review Text")

if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a review")
    else:
        clean_text = preprocess(user_input)
        vectorized_text = tfidf.transform([clean_text])
        prediction = model.predict(vectorized_text)[0]

        if prediction == 1:
            st.success("✅ Positive Review")
        else:
            st.error("❌ Negative Review")
