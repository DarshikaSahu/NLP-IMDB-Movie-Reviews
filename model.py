import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re
import string
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
from zipfile import ZipFile
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Extract dataset
with ZipFile("IMDB Dataset.csv.zip", "r") as zip_ref:
    zip_ref.extractall()

# Load dataset
df = pd.read_csv("IMDB Dataset.csv")

df['review'] = df['review'].str.lower()
df['review'] = df['review'].str.replace('[^\w\s]', '', regex=True)
df['review'] = df['review'].str.replace('<br />', '', regex=False)

# Sentiment distribution visualization
plt.pie(df['sentiment'].value_counts(), labels=df['sentiment'].value_counts().index,
        autopct='%1.1f%%', colors=['skyblue', 'orange'], startangle=90)
plt.title('Sentiment Distribution')
plt.show()

# Generate word clouds
pos_mask = df[df['sentiment'] == 'positive']['review']
neg_mask = df[df['sentiment'] == 'negative']['review']

wc = WordCloud(width=800, height=400, background_color='white').generate(' '.join(pos_mask))
plt.figure()
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()

wc_neg = WordCloud(width=800, height=400, background_color='white').generate(' '.join(neg_mask))
plt.figure()
plt.imshow(wc_neg, interpolation='bilinear')
plt.axis('off')
plt.show()

# Remove stopwords
stop = stopwords.words('english')
pattern = r'\b(?:' + r'|'.join(re.escape(word) for word in stop) + r')\b'
df['review'] = df['review'].str.replace(pattern, '', regex=True)

# Tokenization
tokens = [word_tokenize(review) for review in df['review']]

# Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized_tokens = [[lemmatizer.lemmatize(token) for token in review] for review in tokens]

# Remove punctuation and short words
cleaned_tokens = [[token for token in review if token not in string.punctuation and len(token) > 1] for review in lemmatized_tokens]

# Convert tokens back to sentences
cleaned_reviews = [' '.join(tokens) for tokens in cleaned_tokens]

# Encode labels
le = LabelEncoder()
df['sentiment'] = le.fit_transform(df['sentiment'])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], test_size=0.2, random_state=42, stratify=df['sentiment'])

# Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Train Logistic Regression model
model = LogisticRegression(max_iter=500, random_state=42)
model.fit(X_train, y_train)

# Save as .npy files
np.save("X_test.npy", X_test)
np.save("y_test.npy", y_test)

# Save the vectorizer
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

# Save the trained model
with open("sentiment.pkl", "wb") as f:
    pickle.dump(model, f)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print(classification_report(y_test, y_pred))

# Confusion matrix visualization
cfm = confusion_matrix(y_test, y_pred)
sns.heatmap(cfm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
