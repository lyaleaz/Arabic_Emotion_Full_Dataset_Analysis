
from pathlib import Path
import pandas as pd

# ==========================================
# 1. Configuration
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = PROJECT_DIR / "Arabic_Emotion_Plutchik8_datasets.xlsx"

SHEET_NAME = "Unified_Data"

# ==========================================
# 2. Load Full Dataset
# ==========================================

print("=" * 60)
print("FULL DATASET INSPECTION")
print("=" * 60)

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Dataset not found: {INPUT_FILE}")

df = pd.read_excel(
    INPUT_FILE,
    sheet_name=SHEET_NAME
)

print("\nTotal records:", len(df))

# ==========================================
# 3. Validate Required Columns
# ==========================================

required_columns = [
    "source_dataset",
    "original_id",
    "text",
    "emotion"
]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(f"Missing column: {column}")

print("\nRequired columns: OK")

# ==========================================
# 4. Check Missing Values
# ==========================================

print("\nMissing values:")

for column in required_columns:
    print(column, ":", df[column].isna().sum())

# ==========================================
# 5. Emotion Distribution
# ==========================================

print("\nEmotion Distribution:")
print(df["emotion"].value_counts().sort_index())

# ==========================================
# 6. Dataset Distribution
# ==========================================

print("\nDataset Distribution:")
print(df["source_dataset"].value_counts())

# ==========================================
# 7. Data Validation
# ==========================================

print("\nUnique emotions:", df["emotion"].nunique())

print("Unique datasets:", df["source_dataset"].nunique())

print("Duplicate text records:", df["text"].duplicated().sum())

print("Duplicate IDs:", df["original_id"].duplicated().sum())

# ==========================================
# 8. Final Verification
# ==========================================

print("\n" + "=" * 60)
print("DATASET INSPECTION COMPLETED")
print("=" * 60)

print("Total records:", len(df))
print("Total columns:", len(df.columns))
print("All required columns exist: True")

print("\nOriginal dataset was not modified.")
