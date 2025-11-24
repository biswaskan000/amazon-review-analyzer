import sys
from pathlib import Path
import os

# allow importing from src folder
sys.path.append(str(Path(__file__).resolve().parent / "src"))

# adjust these to actual names/locations in your repo:
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "fine_tuned_xgb.pkl")
FEATURE_NAMES_PATH = os.path.join(os.path.dirname(__file__), "models", "feature_names.json")
# if your constants are in webapp/utils/constants.py:
from webapp.utils.constants import CATEGORY_MAPPING

# import preprocessing function from src/preprocess.py
from preprocess import preprocess_text       # adjust if file name differs
