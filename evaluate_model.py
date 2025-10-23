# evaluate_model.py
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

print("\n📂 Loading processed dataset...")
df = pd.read_csv("data/processed-dataset.csv")

# Drop all object-type (text) columns before model prediction
text_cols = df.select_dtypes(include=["object"]).columns.tolist()
if text_cols:
    print(f"⚠️ Dropping non-numeric columns: {text_cols}")
    df = df.drop(columns=text_cols)

# Separate features and target
if "category" in df.columns:
    X = df.drop(columns=["category"])
    y = df["category"].astype("category").cat.codes
else:
    X = df.drop(columns=["label"]) if "label" in df.columns else df
    y = None

# Train/test split (we don’t really need y here except for metrics)
if y is not None:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
else:
    print("⚠️ No target column found, skipping accuracy metrics.")
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
    y_test = None

print("✅ Dataset ready. Loading model...")
model = joblib.load("models/fine_tuned_xgb.pkl")

print("🚀 Making predictions...")
y_pred = model.predict(X_test)

# -------------------------------
# Evaluation Metrics
# -------------------------------
if y_test is not None:
    acc = accuracy_score(y_test, y_pred)
    print(f"\n✅ Test Accuracy: {acc:.4f}")
    print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
else:
    print("⚠️ Skipped evaluation since y_test is None.")

# -------------------------------
# Feature Importances
# -------------------------------
importances = model.feature_importances_
features = X.columns
sorted_idx = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
sns.barplot(x=importances[sorted_idx], y=features[sorted_idx])
plt.title("Feature Importances (XGBoost)")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()
