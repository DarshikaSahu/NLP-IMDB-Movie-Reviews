import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np

# Load dataset
df = pd.read_csv("IMDB Dataset.csv")

# Text preprocessing
df["review"] = df["review"].str.lower().str.replace(r"[^\w\s]", "", regex=True)

# Encode labels
df["sentiment"] = df["sentiment"].map({"positive": 1, "negative": 0})

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(df["review"], df["sentiment"], test_size=0.2, random_state=42)

# Train TF-IDF vectorizer
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train model
model = LogisticRegression(max_iter=500)
model.fit(X_train_tfidf, y_train)

# Save as .npy files
np.save("X_test.npy", X_test)
np.save("y_test.npy", y_test)

# Save vectorizer and model
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

with open("sentiment.pkl", "wb") as f:
    pickle.dump(model, f)

print("Training complete! Model and vectorizer saved.")
