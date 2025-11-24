import streamlit as st
import pandas as pd
import joblib
import string
import json
import nltk
import spacy
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
    model_path = Path("../models/fine_tuned_xgb.pkl")

    if not model_path.exists():
        st.error(f"Model file not found at {model_path}")
        return None
    
    model = joblib.load(model_path)
    return {"best_model": model}

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


def prepare_features_for_prediction(text, category="unknown", rating=5.0):
    """Prepare features and align with training data columns."""
    df = extract_features(text, rating)

    # Load expected feature names
    with open("../scripts/xgb_model/feature_names.json", "r") as f:
        feature_data = json.load(f)

    # Add category columns
    for cat in CATEGORY_MAPPING.values():
        df[cat] = 1 if cat == category else 0

    # Ensure all expected columns exist
    for feat in feature_data:
        if feat not in df.columns:
            df[feat] = 0.0

    df = df[feature_data]  # Match training order
    return df


def xgb_predict(text, model_dict, category="unknown", rating=5.0):
    """Make prediction using XGBoost model."""
    features = prepare_features_for_prediction(text, category, rating)
    model = model_dict["best_model"]
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

