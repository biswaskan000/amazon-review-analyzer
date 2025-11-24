import json
import joblib

MODEL_PATH = "models/fine_tuned_xgb.pkl"
OUTPUT_PATH = "models/feature_names.json"

# Load trained model
print("Loading model...")
model = joblib.load(MODEL_PATH)

# Extract feature names
print("Extracting feature names...")
try:
    feature_names = model.get_booster().feature_names
except:
    feature_names = model.feature_names_in_

# Save to JSON
print("Saving feature_names.json...")
with open(OUTPUT_PATH, "w") as f:
    json.dump(feature_names, f, indent=4)

print("Done! Saved to:", OUTPUT_PATH)
