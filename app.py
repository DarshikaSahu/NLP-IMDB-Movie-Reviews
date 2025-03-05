import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Ensure necessary NLTK data is downloaded
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Load pre-trained vectorizer & model
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("sentiment.pkl", "rb") as f:
    model = pickle.load(f)

# Text Preprocessing Function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
    text = re.sub(r'<br />', '', text)  # Remove HTML line breaks

    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    words = [w for w in words if w not in stop_words]

    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(w) for w in words]

    return ' '.join(words)

# Streamlit UI
st.title("🎬 IMDB Sentiment Analysis")
user_input = st.text_area("Enter a review:", "")

if st.button("Predict Sentiment"):
    if user_input.strip():
        processed_text = preprocess_text(user_input)
        vectorized_text = vectorizer.transform([processed_text])  # Transform input
        prediction = model.predict(vectorized_text)
        sentiment = "😊 Positive" if prediction[0] == 1 else "😢 Negative"
        st.success(f"Predicted Sentiment: **{sentiment}**")
    else:
        st.warning("Please enter a review to analyze.")

# Visualization: Word Clouds
st.subheader("☁️ Word Cloud for Positive & Negative Words")
if st.button("Generate Word Clouds"):
    try:
        df = pd.read_csv('IMDB Dataset.csv')
        
        # Ensure sentiment column is numeric (0 = negative, 1 = positive)
        df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

        pos_reviews = ' '.join(df[df['sentiment'] == 1]['review'])
        neg_reviews = ' '.join(df[df['sentiment'] == 0]['review'])

        wc_pos = WordCloud(width=400, height=200, background_color='white').generate(pos_reviews)
        wc_neg = WordCloud(width=400, height=200, background_color='white').generate(neg_reviews)

        col1, col2 = st.columns(2)
        with col1:
            st.image(wc_pos.to_array(), caption="Positive Reviews", use_column_width=True)
        with col2:
            st.image(wc_neg.to_array(), caption="Negative Reviews", use_column_width=True)
    except Exception as e:
        st.error(f"Error generating word clouds: {e}")

