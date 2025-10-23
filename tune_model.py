"""
Amazon Review Analyzer - Week 4
Fine-tune XGBoost using GridSearchCV.
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier

# --------------------------------------
# 📂 Load dataset
# --------------------------------------
print("\n📂 Loading processed dataset...")

data_path = "data/processed-dataset.csv"
if not os.path.exists(data_path):
    raise FileNotFoundError(f"❌ Dataset not found at {data_path}. Make sure it's in the 'data' folder.")

df = pd.read_csv(data_path)
print(f"✅ Dataset loaded. Shape: {df.shape}")

# --------------------------------------
# 🧹 Clean data for XGBoost
# --------------------------------------
# Drop text columns (XGBoost can’t handle raw text)
X = df.drop(columns=["category", "text_", "cleaned_text"])
y = df["category"].astype("category").cat.codes  # Encode labels

print(f"🔢 Labels encoded: {df['category'].unique().tolist()} → {list(y.unique())}")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"✅ Training size: {X_train.shape}, Test size: {X_test.shape}")

# --------------------------------------
# 🔄 Convert all object columns to numeric
# --------------------------------------
X_train = X_train.apply(lambda col: col.astype('category').cat.codes if col.dtype == 'object' else col)
X_test = X_test.apply(lambda col: col.astype('category').cat.codes if col.dtype == 'object' else col)

# --------------------------------------
# ⚙️ Define XGBoost model + parameter grid
# --------------------------------------
model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    use_label_encoder=False,
    random_state=42,
    tree_method="hist",  # faster on CPU
)

param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.05, 0.1],
    "subsample": [0.7, 1.0],
    "colsample_bytree": [0.7, 1.0],
}

print("\n⚙️ Starting hyperparameter tuning (this may take a while)...")

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring="accuracy",
    cv=5,
    verbose=1,
    n_jobs=-1,
)

# --------------------------------------
# 🚀 Run tuning
# --------------------------------------
grid_search.fit(X_train, y_train)

# --------------------------------------
# 🏆 Results
# --------------------------------------
print("\n✅ Best Parameters Found:")
print(grid_search.best_params_)
print(f"✅ Best Cross-Validation Accuracy: {grid_search.best_score_:.4f}")

# --------------------------------------
# 💾 Save tuned model
# --------------------------------------
os.makedirs("models", exist_ok=True)
model_path = "models/fine_tuned_xgb.pkl"
joblib.dump(grid_search.best_estimator_, model_path)
print(f"\n💾 Fine-tuned model saved to {model_path}")
