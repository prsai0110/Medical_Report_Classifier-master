import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Load Kaggle dataset
df = pd.read_csv("data/data.csv")

# Drop missing values
df = df.dropna(subset=["transcription", "medical_specialty"])

X = df["transcription"]
y = df["medical_specialty"]

# Vectorization
vectorizer = TfidfVectorizer(stop_words="english")
X_vectorized = vectorizer.fit_transform(X)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_vectorized, y)

# Save model
joblib.dump(model, "models/report_classifier.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")

print("Model trained and saved successfully!")
