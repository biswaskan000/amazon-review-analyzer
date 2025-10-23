# train_model.py
import pandas as pd
import numpy as np
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import os

# Step 1: Load processed dataset
print("Loading processed dataset...")
df = pd.read_csv("processed-dataset.csv")
print(f"Dataset loaded with {len(df)} reviews")
print("Columns:", list(df.columns))

# Step 2: Define features and labels
X_text = df["cleaned_text"]
y = df["label"]

# Step 3: Split dataset
print("Splitting dataset into training and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X_text, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# Step 4: Text vectorization (TF-IDF)
print("Vectorizing text data...")
vectorizer = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
print("TF-IDF vectorization complete!")

# Step 5: Model training
print("Training Random Forest model...")
model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(X_train_tfidf, y_train)
print("Model training complete!")

# Step 6: Evaluate performance
print("Evaluating model...")
y_pred = model.predict(X_test_tfidf)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

try:
    auc = roc_auc_score(y_test, model.predict_proba(X_test_tfidf)[:, 1])
    print(f"\nAUC Score: {auc:.4f}")
except:
    print("Skipping AUC calculation (model has no predict_proba method).")
    auc = None

# Step 7: Save model and metadata
print("\nSaving model and metadata...")
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/review_classifier.pkl")
joblib.dump(vectorizer, "model/tfidf_vectorizer.pkl")

metadata = {
    "num_samples": len(df),
    "training_samples": len(X_train),
    "test_samples": len(X_test),
    "auc_score": auc
}
with open("model/model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)

print("✅ Model and metadata saved successfully in the 'model' folder.")
