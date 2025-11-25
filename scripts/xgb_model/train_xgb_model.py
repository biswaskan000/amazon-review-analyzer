import json
import pickle
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ----------------------------------------------------
# 1. Load your dataset (replace with your dataset)
# ----------------------------------------------------
# You must provide two lists: texts and labels
# Example placeholder data:
texts = [
    "This product is amazing and worked perfectly!",
    "Terrible quality, broke after one day.",
    "Good value for the price.",
    "Worst purchase ever, do not buy.",
]

labels = [1, 0, 1, 0]   # Example (1=positive, 0=negative)

# ----------------------------------------------------
# 2. Vectorize text
# ----------------------------------------------------
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(texts)

# ----------------------------------------------------
# 3. Train/test split
# ----------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.2, random_state=42
)

# ----------------------------------------------------
# 4. Train XGB model
# ----------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.9,
    colsample_bytree=0.9,
    eval_metric="logloss"
)

model.fit(X_train, y_train)

# ----------------------------------------------------
# 5. Evaluate (optional)
# ----------------------------------------------------
preds = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, preds))

# ----------------------------------------------------
# 6. Save model and assets
# ----------------------------------------------------
model.save_model("scripts/xgb_model/xgb_model.json")

with open("scripts/xgb_model/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

feature_names = vectorizer.get_feature_names_out().tolist()
with open("scripts/xgb_model/feature_names.json", "w") as f:
    json.dump(feature_names, f)

print("Saved:")
print("- scripts/xgb_model/xgb_model.json")
print("- scripts/xgb_model/vectorizer.pkl")
print("- scripts/xgb_model/feature_names.json")