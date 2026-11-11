
import pandas as pd
import numpy as np
from pathlib import Path

project = Path(__file__).resolve().parent.parent

excel = project / "Arabic_Emotion_Plutchik8_datasets.xlsx"
csv = project / "results" / "full_dataset" / "metadata.csv"

df = pd.read_excel(excel, sheet_name="Unified_Data")

current = df.copy()
current.insert(0, "row_index", np.arange(len(df)))

saved = pd.read_csv(csv, dtype=str, keep_default_na=False)
current = current.astype(str)

print("Excel rows:", len(current))
print("CSV rows:", len(saved))

print("\nExcel columns:")
print(current.columns.tolist())

print("\nCSV columns:")
print(saved.columns.tolist())

if list(current.columns) != list(saved.columns):
    print("\nCOLUMN MISMATCH")

else:
    print("\nColumn comparison:")

    for col in current.columns:
        equal = current[col].equals(saved[col])
        print(f"{col}: {'OK' if equal else 'DIFFERENT'}")

        if not equal:
            differences = np.where(
                current[col].values != saved[col].values
            )[0]

            print("Different rows:", len(differences))

            for i in differences[:5]:
                print(
                    f"Row {i}: "
                    f"Excel={current.iloc[i][col]!r} | "
                    f"CSV={saved.iloc[i][col]!r}"
                )
