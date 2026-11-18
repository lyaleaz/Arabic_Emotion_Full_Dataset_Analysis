
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path("ARABIC_EMOTION_DEDUPLICATED_32469")
INPUT = ROOT / "Arabic_Emotion_Clean_32469.xlsx"
OUTPUT = ROOT / "classification_data"
OUTPUT.mkdir(parents=True, exist_ok=True)

SEED = 42

# Load cleaned dataset
df = pd.read_excel(INPUT)

assert len(df) == 32469
assert df["text"].notna().all()
assert df["emotion"].notna().all()
assert df["text"].duplicated().sum() == 0
assert df["emotion"].nunique() == 8

# 80% development, 20% held-out test
dev, test = train_test_split(
    df,
    test_size=0.20,
    stratify=df["emotion"],
    random_state=SEED
)

# 70% train, 10% validation, 20% test overall
train, val = train_test_split(
    dev,
    test_size=0.125,
    stratify=dev["emotion"],
    random_state=SEED
)

# Create nested stratified training subsets
train_shuffled = train.sample(frac=1, random_state=SEED)
train_20k, _ = train_test_split(
    train_shuffled,
    train_size=20000,
    stratify=train_shuffled["emotion"],
    random_state=SEED
)
train_10k, _ = train_test_split(
    train_20k,
    train_size=10000,
    stratify=train_20k["emotion"],
    random_state=SEED
)
train_5k, _ = train_test_split(
    train_10k,
    train_size=5000,
    stratify=train_10k["emotion"],
    random_state=SEED
)

splits = {
    "train_full": train,
    "train_20k": train_20k,
    "train_10k": train_10k,
    "train_5k": train_5k,
    "validation": val,
    "test": test,
}

# Check no leakage between splits
assert set(train.index).isdisjoint(val.index)
assert set(train.index).isdisjoint(test.index)
assert set(val.index).isdisjoint(test.index)

for name, subset in splits.items():
    subset.to_csv(
        OUTPUT / f"{name}.csv",
        index=False,
        encoding="utf-8-sig"
    )
    print(f"{name}: {len(subset)} records")

print("\nClassification datasets prepared successfully!")
