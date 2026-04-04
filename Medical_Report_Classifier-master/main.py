import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from nlp.preprocessing import clean_text
from models.ml_model import train_model
from models.bert_model import BERTClassifier

# =========================
# LOAD DATASET
# =========================
df = pd.read_csv("data/data.csv")

# Keep needed columns
df = df[['description', 'medical_specialty']]
df.dropna(inplace=True)

# Rename columns for simplicity
df.columns = ["text", "label"]

# =========================
# CLEAN TEXT
# =========================
df["text"] = df["text"].apply(clean_text)

# =========================
# TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    df["text"],
    df["label"],
    test_size=0.2,
    random_state=42
)

# =========================
# TRAIN BASELINE MODEL
# =========================
print("\nTraining TF-IDF + Logistic Regression Model...\n")

model, vectorizer = train_model(X_train, y_train)

# =========================
# EVALUATE BASELINE MODEL
# =========================
X_test_vec = vectorizer.transform(X_test)
y_pred = model.predict(X_test_vec)

print("Baseline Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, zero_division=0))


# =========================
# BERT ZERO-SHOT MODEL
# =========================
print("\nSwitching to BERT Zero-Shot Model...\n")

bert = BERTClassifier()

# All possible labels
labels = list(set(df["label"]))


# =========================
# USER INPUT LOOP
# =========================
while True:
    text = input("\nEnter medical report (type 'exit' to stop): ")

    if text.lower() == "exit":
        break

    # Clean input
    text = clean_text(text)

    # Predict using BERT
    prediction = bert.predict(text, labels)

    print("Predicted Category:", prediction)
