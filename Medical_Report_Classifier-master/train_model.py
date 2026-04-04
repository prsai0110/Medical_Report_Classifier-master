import pandas as pd
import re
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

# ✅ Load dataset from CSV
df = pd.read_csv("data/dataa.csv")
df.columns = df.columns.str.strip()  # removes hidden spaces
df = df.dropna()  # removes empty rows
# ✅ Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

df['text'] = df['text'].apply(clean_text)

# ✅ Vectorization (better)
vectorizer = TfidfVectorizer(ngram_range=(1,2))
X = vectorizer.fit_transform(df['text'])

y = df['label']

# ✅ Split (for better learning)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# ✅ Better model
model = LinearSVC()
model.fit(X_train, y_train)

# ✅ Save
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("✅ Model trained successfully")
