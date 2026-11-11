"""
Re-check the first 3,600 AraBERT v2 embeddings without changing originals.

This script:
1. Reads the same Unified_Data sheet used for the original extraction.
2. Repeats external Farasa segmentation in chunks of 256.
3. Applies AraBERTPreprocessor with IdentitySegmenter.
4. Repeats tokenizer settings, mean pooling, and L2 normalization.
5. Saves the re-generated vectors to a NEW file.
6. Compares them with rows 0:3600 of the existing full embedding matrix.

Original embedding and checkpoint files are opened read-only and never modified.
"""

import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from arabert.preprocess import ArabertPreprocessor


# ------------------------------ Configuration -----------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = PROJECT_DIR / "Arabic_Emotion_Plutchik8_datasets.xlsx"
SHEET_NAME = "Unified_Data"

DATA_DIR = PROJECT_DIR / "results" / "full_dataset"
ORIGINAL_EMBEDDINGS = DATA_DIR / "AraBERT_v2_embeddings.npy"

CHECK_DIR = DATA_DIR / "first_3600_recheck"
CHECK_DIR.mkdir(parents=True, exist_ok=True)

RECHECK_EMBEDDINGS = CHECK_DIR / "AraBERT_v2_first_3600_recheck.npy"
COMPARISON_CSV = CHECK_DIR / "first_3600_comparison.csv"
REPORT_TXT = CHECK_DIR / "first_3600_comparison_report.txt"

MODEL_NAME = "aubmindlab/bert-base-arabertv2"
COUNT = 3600
BATCH_SIZE = 16
FARASA_CHUNK_SIZE = 256
MAX_LENGTH = 128
DIMENSION = 768

FARASA_JAR = Path(r"C:\Farasa\Farasa.jar")
FARASA_DATA = Path(r"C:\Farasa\FarasaData")
FARASA_LIB = Path(r"C:\Farasa\lib")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------ Farasa setup -------------------------------

required_paths = [
    INPUT_FILE,
    ORIGINAL_EMBEDDINGS,
    FARASA_JAR,
    FARASA_DATA,
    FARASA_LIB / "commons-lang3-3.1.jar",
    FARASA_LIB / "mapdb-2.0.0-SNAPSHOT.jar",
]
for required_path in required_paths:
    if not required_path.exists():
        raise FileNotFoundError(f"Required path not found: {required_path}")

os.environ["FarasaDataDir"] = str(FARASA_DATA.resolve()) + os.sep


def farasa_segment_texts(batch_texts):
    clean_texts = [
        str(text).replace("\r", " ").replace("\n", " ").strip()
        for text in batch_texts
    ]

    if not clean_texts:
        return []

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        input_file = temp_dir / "input.txt"
        output_file = temp_dir / "output.txt"

        input_file.write_text(
            "\n".join(clean_texts) + "\n",
            encoding="utf-8"
        )

        classpath = (
            f"{FARASA_JAR};"
            f"{FARASA_LIB / 'commons-lang3-3.1.jar'};"
            f"{FARASA_LIB / 'mapdb-2.0.0-SNAPSHOT.jar'}"
        )

        command = [
            "java",
            "-Dfile.encoding=UTF-8",
            "-cp",
            classpath,
            "farasa.TestCase",
            "-i",
            str(input_file),
            "-o",
            str(output_file),
        ]

        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )

        if not output_file.exists():
            raise RuntimeError("Farasa output file was not created.")

        segmented = output_file.read_text(encoding="utf-8").splitlines()

    if len(segmented) != len(clean_texts):
        raise ValueError(
            f"Farasa input/output mismatch: {len(clean_texts)} inputs, "
            f"{len(segmented)} outputs."
        )

    if any(not text.strip() for text in segmented):
        raise ValueError("Farasa produced an empty output.")

    return segmented


class IdentitySegmenter:
    def segment(self, text):
        return text


# ------------------------------- Mean pooling ------------------------------

def mean_pooling(hidden_states, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    summed = (hidden_states * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


# ---------------------------------- Main -----------------------------------

def main():
    print("=" * 68)
    print("ARABERT V2 FIRST 3,600 EMBEDDINGS RE-CHECK")
    print("=" * 68)
    print("Device:", DEVICE)
    print("Records to re-check:", f"{COUNT:,}")
    print("Farasa chunk size:", FARASA_CHUNK_SIZE)
    print("Embedding batch size:", BATCH_SIZE)
    print("Maximum sequence length:", MAX_LENGTH)

    if RECHECK_EMBEDDINGS.exists():
        raise FileExistsError(
            f"Re-check output already exists: {RECHECK_EMBEDDINGS}\n"
            "Rename or move that re-check output before running again. "
            "The original embeddings are not touched."
        )

    print("\nLoading dataset...")
    df = pd.read_excel(INPUT_FILE, sheet_name=SHEET_NAME)

    if len(df) < COUNT:
        raise ValueError(f"Dataset has fewer than {COUNT} rows.")

    texts = df["text"].astype(str).tolist()[:COUNT]

    original = np.load(ORIGINAL_EMBEDDINGS, mmap_mode="r")
    if original.shape != (len(df), DIMENSION):
        raise ValueError(
            f"Original embedding shape is {original.shape}; "
            f"expected {(len(df), DIMENSION)}."
        )

    original_first = np.asarray(original[:COUNT], dtype=np.float32)
    if not np.isfinite(original_first).all():
        raise ValueError("Original first 3,600 vectors contain NaN or Inf.")

    print("Dataset rows:", f"{len(df):,}")
    print("Original embedding shape:", original.shape)
    print("Original first 3,600 rows loaded read-only.")

    print("\nInitializing AraBERT preprocessing...")
    preprocessor = ArabertPreprocessor(
        model_name="bert-base-arabertv2",
        apply_farasa_segmentation=False,
    )
    preprocessor.farasa_segmenter = IdentitySegmenter()

    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()

    if model.config.hidden_size != DIMENSION:
        raise ValueError(
            f"Expected hidden size {DIMENSION}, got {model.config.hidden_size}."
        )

    # New output only; original embeddings remain untouched.
    regenerated = np.lib.format.open_memmap(
        RECHECK_EMBEDDINGS,
        mode="w+",
        dtype=np.float32,
        shape=(COUNT, DIMENSION),
    )

    print("\nRe-generating first 3,600 vectors...")
    for chunk_start in range(0, COUNT, FARASA_CHUNK_SIZE):
        chunk_end = min(chunk_start + FARASA_CHUNK_SIZE, COUNT)
        print(f"\nFarasa segmentation: {chunk_start:,} - {chunk_end:,}")

        segmented = farasa_segment_texts(texts[chunk_start:chunk_end])
        processed = [preprocessor.preprocess(text) for text in segmented]

        for offset in range(0, len(processed), BATCH_SIZE):
            batch_start = chunk_start + offset
            batch_end = min(batch_start + BATCH_SIZE, chunk_end)
            batch_texts = processed[offset:offset + BATCH_SIZE]

            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            ).to(DEVICE)

            with torch.inference_mode():
                output = model(**encoded)
                vectors = mean_pooling(
                    output.last_hidden_state,
                    encoded["attention_mask"],
                )
                vectors = F.normalize(vectors, p=2, dim=1)

            batch_vectors = vectors.cpu().numpy().astype(np.float32)

            if batch_vectors.shape != (batch_end - batch_start, DIMENSION):
                raise ValueError(
                    f"Unexpected batch shape at rows {batch_start}:{batch_end}: "
                    f"{batch_vectors.shape}"
                )

            if not np.isfinite(batch_vectors).all():
                raise ValueError(
                    f"Non-finite values at rows {batch_start}:{batch_end}."
                )

            regenerated[batch_start:batch_end] = batch_vectors

            print(
                f"Re-generated: {batch_end:,}/{COUNT:,} "
                f"({100 * batch_end / COUNT:.1f}%)"
            )

    regenerated.flush()
    new_vectors = np.asarray(regenerated, dtype=np.float32)

    print("\nComparing regenerated vectors with original vectors...")
    differences = new_vectors.astype(np.float64) - original_first.astype(np.float64)

    abs_diff = np.abs(differences)
    max_abs_diff = float(abs_diff.max())
    mean_abs_diff = float(abs_diff.mean())
    rmse = float(np.sqrt(np.mean(differences ** 2)))

    old_norms = np.linalg.norm(original_first, axis=1)
    new_norms = np.linalg.norm(new_vectors, axis=1)

    old_norms_safe = np.maximum(old_norms, 1e-12)
    new_norms_safe = np.maximum(new_norms, 1e-12)
    row_cosines = np.sum(
        original_first * new_vectors, axis=1
    ) / (old_norms_safe * new_norms_safe)

    exact_equal = bool(np.array_equal(original_first, new_vectors))
    within_1e_5 = bool(np.allclose(
        original_first, new_vectors, atol=1e-5, rtol=1e-5
    ))
    within_1e_4 = bool(np.allclose(
        original_first, new_vectors, atol=1e-4, rtol=1e-4
    ))

    comparison = pd.DataFrame({
        "row_index": np.arange(COUNT),
        "cosine_similarity_old_vs_recheck": row_cosines,
        "max_abs_difference_per_row": abs_diff.max(axis=1),
        "mean_abs_difference_per_row": abs_diff.mean(axis=1),
    })
    comparison.to_csv(COMPARISON_CSV, index=False, encoding="utf-8-sig")

    report_lines = [
        "ARABERT V2 FIRST 3,600 EMBEDDINGS RE-CHECK",
        "=" * 68,
        f"Rows checked: {COUNT}",
        f"Original shape: {original.shape}",
        f"Re-check shape: {new_vectors.shape}",
        f"Exact array equality: {exact_equal}",
        f"Allclose atol=1e-5, rtol=1e-5: {within_1e_5}",
        f"Allclose atol=1e-4, rtol=1e-4: {within_1e_4}",
        f"Maximum absolute difference: {max_abs_diff:.12g}",
        f"Mean absolute difference: {mean_abs_diff:.12g}",
        f"RMSE: {rmse:.12g}",
        f"Minimum row cosine similarity: {float(row_cosines.min()):.12g}",
        f"Mean row cosine similarity: {float(row_cosines.mean()):.12g}",
        f"Maximum row cosine similarity: {float(row_cosines.max()):.12g}",
        f"Original mean norm: {float(old_norms.mean()):.12g}",
        f"Re-check mean norm: {float(new_norms.mean()):.12g}",
        "",
        "IMPORTANT: This is a numerical reproducibility check using the",
        "current preprocessing pipeline. It cannot by itself prove which",
        "historical code version generated the original vectors.",
        "Original embedding file was opened read-only and not modified.",
    ]
    REPORT_TXT.write_text("\n".join(report_lines), encoding="utf-8")

    print("\n" + "=" * 68)
    print("RE-CHECK COMPARISON RESULTS")
    print("=" * 68)
    for line in report_lines[3:17]:
        print(line)

    print("\nSaved re-generated vectors:", RECHECK_EMBEDDINGS)
    print("Saved row comparison:", COMPARISON_CSV)
    print("Saved report:", REPORT_TXT)
    print("\nOriginal embedding file was not modified.")


if __name__ == "__main__":
    main()
