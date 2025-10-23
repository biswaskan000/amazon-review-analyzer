import pandas as pd
from pathlib import Path
import sys

# Add src folder to the system path
sys.path.append(str(Path(__file__).resolve().parent / "src"))

# Import the functions you built in Week 2
from preprocess import preprocess_text
from feature_extraction import extract_features

# Step 1: Load your raw dataset
data_path = Path(__file__).resolve().parent / "fake-reviews.csv"
df = pd.read_csv(data_path)

print(f"✅ Dataset loaded with {len(df)} reviews")
print(f"Columns in dataset: {list(df.columns)}")

# Step 2: Clean the review text
df["cleaned_text"] = df["text_"].apply(preprocess_text)
print("✅ Text cleaned")

# Step 3: Extract features (including part-of-speech features)
df = extract_features(df, include_pos=True)
print("✅ Features extracted")

# Step 4: Save processed dataset
output_path = Path(__file__).resolve().parent / "processed-dataset.csv"
df.to_csv(output_path, index=False)

print(f"🎉 Processed dataset saved to: {output_path}")