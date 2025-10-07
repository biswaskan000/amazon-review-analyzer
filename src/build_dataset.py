import pandas as pd
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent / "src"))

from preprocess import preprocess_text
from feature_extraction import extract_features

dataset_path = Path(__file__).resolve().parent / "fake reviews dataset.csv"
processed_path = Path(__file__).resolve().parent / "processed-dataset.csv"

df = pd.read_csv(dataset_path)
df["cleaned_text"] = df["text_"].apply(preprocess_text)
df = extract_features(df, include_pos=True)
df.to_csv(processed_path, index=False)

print("Processed dataset saved to:", processed_path)