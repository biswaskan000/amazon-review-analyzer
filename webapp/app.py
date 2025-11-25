import streamlit as st
import pandas as pd
import joblib
import os
import string
import json
import nltk
import spacy
import requests
import shutil
import tempfile
from pathlib import Path
from nltk.sentiment import SentimentIntensityAnalyzer

# Attempt to import CATEGORY_MAPPING from utils.constants; provide a safe fallback if not available.
try:
    import importlib
    mod = importlib.import_module("utils.constants")
    CATEGORY_MAPPING = getattr(mod, "CATEGORY_MAPPING")
except Exception:
    # Fallback mapping used when utils.constants cannot be resolved.
    # Update these keys/values to match your dataset if you have a specific mapping file.
    CATEGORY_MAPPING = {
        "Unknown": "unknown"
    }

# ----------------------------
# 1. Cached resource loaders
# ----------------------------

@st.cache_resource
def get_nlp_models():
    """Load spaCy and VADER once for the app."""
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)

    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    analyzer = SentimentIntensityAnalyzer()
    return nlp, analyzer


@st.cache_resource
def get_xgb_model():
    """Load the trained XGBoost model."""
    # Resolve model path relative to this file so it works both locally and when
    # Streamlit changes the working directory in deployment environments.
    model_path = Path(__file__).resolve().parent / "models" / "fine_tuned_xgb.pkl"

    # If the model file is missing, try to download it from an external URL
    # provided through the MODEL_URL environment variable. This is useful for
    # deployment platforms that don't fetch Git LFS objects automatically.
    if not model_path.exists():
        model_url = os.environ.get("MODEL_URL")
        if not model_url:
            st.error(f"Model file not found at {model_path}")
            st.error("Set the MODEL_URL environment variable to an externally hosted model to enable automatic download.")
            return None

        # Ensure models directory exists
        model_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with st.spinner("Downloading model from MODEL_URL..."):
                resp = requests.get(model_url, stream=True, timeout=60)
                resp.raise_for_status()

                total = resp.headers.get("content-length")
                if total is not None:
                    total = int(total)

                # Write to a temporary file then move into place
                with tempfile.NamedTemporaryFile(delete=False) as tmp:
                    downloaded = 0
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            tmp.write(chunk)
                            downloaded += len(chunk)
                shutil.move(tmp.name, str(model_path))
                st.success(f"Downloaded model to {model_path}")
        except Exception as e:
            st.error(f"Failed to download model from MODEL_URL: {e}")
            return None

    try:
        model = joblib.load(model_path)
        return {"best_model": model}
    except Exception as e:
        st.error(f"Failed to load model at {model_path}: {e}")
        return None

# ----------------------------
# 2. Feature Extraction
# ----------------------------

def preprocess_text(text: str) -> str:
    """Clean text: lower, strip, remove punctuation."""
    return text.lower().translate(str.maketrans("", "", string.punctuation)).strip()


def extract_features(text, rating=5.0):
    """Simple feature extraction (expand later with POS features)."""
    cleaned_text = preprocess_text(text)
    nlp, analyzer = get_nlp_models()

    df = pd.DataFrame([{
        "rating": rating,
        "char_length": len(cleaned_text),
        "word_count": len(cleaned_text.split()),
        "punctuation_ct": sum(1 for c in text if c in string.punctuation),
        "is_extreme_star": rating in [1.0, 5.0],
        "sentiment_score": analyzer.polarity_scores(cleaned_text)["compound"],
    }])

    df["cleaned_text"] = cleaned_text
    return df


def prepare_features_for_prediction(text, category="unknown", rating=5.0, feature_names=None):
    """Prepare features and align with training data columns.

    If `feature_names` is provided (list of expected columns), use that. If not,
    fall back to the scripts/xgb_model/feature_names.json file. Any missing
    columns will be created with default 0.0 values.
    """
    df = extract_features(text, rating)

    # Determine expected feature names
    if feature_names is not None:
        feature_data = list(feature_names)
    else:
        repo_root = Path(__file__).resolve().parent.parent
        feature_file = repo_root / "scripts" / "xgb_model" / "feature_names.json"
        if not feature_file.exists():
            st.error(f"Feature names file not found at {feature_file}")
            feature_data = list(df.columns)
        else:
            with open(feature_file, "r") as f:
                feature_data = json.load(f)

    # Add category columns
    for cat in CATEGORY_MAPPING.values():
        df[cat] = 1 if cat == category else 0

    # Ensure all expected columns exist
    for feat in feature_data:
        if feat not in df.columns:
            # For text token features, derive count from cleaned text if possible
            if isinstance(feat, str) and feat.isalpha():
                # simple token count (case-insensitive)
                df[feat] = df["cleaned_text"].apply(lambda s: s.split().count(feat))
            else:
                df[feat] = 0.0

    # Return columns in expected order
    return df[feature_data]


def xgb_predict(text, model_dict, category="unknown", rating=5.0):
    """Make prediction using XGBoost model."""
    model = model_dict["best_model"]

    # Try to get the feature names from the model (preferred). Fall back to
    # the scripts file within prepare_features_for_prediction if not available.
    feature_names = None
    try:
        booster = model.get_booster()
        if hasattr(booster, "feature_names") and booster.feature_names is not None:
            feature_names = booster.feature_names
    except Exception:
        feature_names = None

    features = prepare_features_for_prediction(text, category, rating, feature_names)

    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    confidence = probabilities[prediction]
    label = "Human" if prediction == 1 else "AI"
    return label, confidence, probabilities.tolist()


# ----------------------------
# 3. Streamlit UI
# ----------------------------

def main():
    st.set_page_config(page_title="Amazon Review Analyzer", page_icon="🤖")

    st.title("🤖 Amazon Review Analyzer")
    st.write("Analyze reviews to determine if they are **AI-generated or human-written**.")

    with st.spinner("Loading model..."):
        model_dict = get_xgb_model()
    if model_dict is None:
        st.error("Model not loaded. Please check your model path.")
        return
    st.success("✅ Model loaded successfully!")

    # Layout: 2 columns
    col1, col2 = st.columns(2)

    with col1:
        input_review = st.text_area("Enter Review Text:")
        category = st.selectbox("Select Category:", list(CATEGORY_MAPPING.keys()))
        rating = st.number_input("Star Rating (1–5):", min_value=1.0, max_value=5.0, value=5.0)
        analyze_button = st.button("Analyze Review")

    with col2:
        st.subheader("Results")
        if analyze_button:
            if input_review.strip():
                dataset_category = CATEGORY_MAPPING[category]

                with st.spinner("Analyzing..."):
                    try:
                        label, confidence, _ = xgb_predict(input_review, model_dict, dataset_category, rating)
                        if label == "Human":
                            st.success(f"🧍 This review appears **Human-written** (Confidence: {confidence:.2f})")
                        else:
                            st.error(f"🤖 This review appears **AI-generated** (Confidence: {confidence:.2f})")
                    except Exception as e:
                        st.error(f"Error during analysis: {str(e)}")
                        st.exception(e)
            else:
                st.warning("Please enter some review text!")


if __name__ == "__main__":
    main()

