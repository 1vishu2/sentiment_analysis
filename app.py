import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

# Load model & vectorizer
model = joblib.load(r"E:\sentiment_model.pkl")
tfidf = joblib.load(r"E:\tfidf_vectorizer.pkl")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

# Streamlit UI
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
