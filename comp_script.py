# comp_script.py
import sys
import os
from pathlib import Path
import joblib
import json
import pandas as pd

# allow importing from src folder
sys.path.append(str(Path(__file__).resolve().parent / "src"))

# model & feature names paths (update if different)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "fine_tuned_xgb.pkl")
FEATURE_NAMES_PATH = os.path.join(os.path.dirname(__file__), "models", "feature_names.json")

# constants used by your feature prep (update import if in different location)
try:
    from webapp.utils.constants import CATEGORY_MAPPING
except Exception:
    # fallback if you put constants elsewhere
    CATEGORY_MAPPING = {}

# import your preprocessing and feature extraction utilities
# path.sane: adjust module names to match your repo
from preprocess import preprocess_text           # if preprocess.py at src/root
# or: from feature_extraction import extract_features

def load_model(path):
    print("Loading model...", path)
    model = joblib.load(path)
    print("Model loaded.")
    return model

def load_feature_names(path):
    with open(path, "r") as f:
        return json.load(f)

def prepare_one_sample(text, category="Unknown", rating=5.0, feature_names=None):
    # replicate the feature extraction you used for training
    cleaned = preprocess_text(text)
    # basic features — modify to match training
    d = {
        "rating": rating,
        "char_length": len(cleaned),
        "word_count": len(cleaned.split()),
        "punctuation_ct": sum(1 for c in cleaned if c in '.,;:!?()[]{}"\''),
        "is_extreme_star": int(rating in [1.0, 5.0]),
        # add other features you used, e.g., POS counts
    }
    # initialize feature vector in right order
    features = {fn: 0.0 for fn in feature_names}
    for k, v in d.items():
        if k in features:
            features[k] = float(v)
    # set category one-hot columns if present (example suffix pattern)
    # find category column name in feature_names that matches mapping
    cat_col = None
    for fn in feature_names:
        if isinstance(fn, str) and fn.startswith("category_"):
            # decide matching approach; here we just try mapping names
            if CATEGORY_MAPPING.get(category, category) in fn:
                cat_col = fn
                break
    if cat_col:
        features[cat_col] = 1.0
    return pd.DataFrame([features])

def main():
    model = load_model(MODEL_PATH)
    feature_names = load_feature_names(FEATURE_NAMES_PATH)
    # Example test review
    sample_text = "This product exceeded my expectations! Quality is outstanding."
    X = prepare_one_sample(sample_text, category="Books", rating=5.0, feature_names=feature_names)
    print("Prepared features (first 10 cols):", X.iloc[0].to_dict())
    # predict
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    print("Prediction:", pred, "Probabilities:", proba)

if __name__ == "__main__":
    main()
