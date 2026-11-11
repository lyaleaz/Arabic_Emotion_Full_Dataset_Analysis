"""
Full-dataset embedding geometry analysis.

Inputs:
    results/full_dataset/AraBERT_v2_embeddings.npy
    results/full_dataset/MARBERT_embeddings.npy
    results/full_dataset/metadata.csv

Outputs:
    results/full_dataset/embedding_geometry/geometry_results.csv
    results/full_dataset/embedding_geometry/geometry_summary.txt

Method:
- Mean pairwise cosine similarity is computed exactly for all unordered pairs
  using the identity for L2-normalized vectors. This avoids constructing an
  N x N cosine-similarity matrix.
- Effective rank and principal-component variance are computed from the
  eigenvalues of the centered 768 x 768 feature covariance matrix.
- The covariance matrix is accumulated in batches to limit memory use.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd


# ----------------------------- Configuration -----------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "results" / "full_dataset"
OUTPUT_DIR = DATA_DIR / "embedding_geometry"

METADATA_PATH = DATA_DIR / "metadata.csv"

MODEL_FILES = {
    "AraBERT_v2": DATA_DIR / "AraBERT_v2_embeddings.npy",
    "MARBERT": DATA_DIR / "MARBERT_embeddings.npy",
}

EXPECTED_RECORDS = 41202
EXPECTED_DIMENSIONS = 768
BATCH_SIZE = 2048

# Tolerance for checking that every embedding has unit L2 norm.
L2_ATOL = 1e-3


# ------------------------------- Utilities --------------------------------

def fail(message: str) -> None:
    print(f"\nERROR: {message}")
    sys.exit(1)


def validate_metadata() -> pd.DataFrame:
    print("\nChecking metadata...")
    if not METADATA_PATH.exists():
        fail(f"Metadata file not found: {METADATA_PATH}")

    metadata = pd.read_csv(METADATA_PATH)

    required_columns = [
        "source_dataset",
        "original_id",
        "text",
        "emotion",
    ]
    missing_columns = [c for c in required_columns if c not in metadata.columns]
    if missing_columns:
        fail(f"Metadata is missing required columns: {missing_columns}")

    if len(metadata) != EXPECTED_RECORDS:
        fail(
            f"Metadata contains {len(metadata):,} rows; "
            f"expected {EXPECTED_RECORDS:,}."
        )

    print(f"Metadata rows: {len(metadata):,}")
    print("Required metadata columns: OK")
    return metadata


def analyze_one_model(model_name: str, embedding_path: Path) -> dict:
    print("\n" + "=" * 68)
    print(f"ANALYZING {model_name}")
    print("=" * 68)

    if not embedding_path.exists():
        fail(f"Embedding file not found: {embedding_path}")

    # Memory-map the file so the entire array is not copied into RAM.
    embeddings = np.load(embedding_path, mmap_mode="r")

    if embeddings.ndim != 2:
        fail(f"{model_name}: expected a 2D array, got shape {embeddings.shape}")

    n_records, dimensions = embeddings.shape
    print(f"Shape: {embeddings.shape}")

    if n_records != EXPECTED_RECORDS:
        fail(
            f"{model_name}: expected {EXPECTED_RECORDS:,} records, "
            f"found {n_records:,}."
        )

    if dimensions != EXPECTED_DIMENSIONS:
        fail(
            f"{model_name}: expected {EXPECTED_DIMENSIONS} dimensions, "
            f"found {dimensions}."
        )

    # Accumulate in float64 batches for numerical stability and lower memory use.
    sum_vectors = np.zeros(dimensions, dtype=np.float64)
    sum_outer_products = np.zeros((dimensions, dimensions), dtype=np.float64)

    norm_sum = 0.0
    norm_min = np.inf
    norm_max = -np.inf
    all_finite = True

    print(f"Accumulating statistics in batches of {BATCH_SIZE:,}...")
    for start in range(0, n_records, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n_records)
        batch = np.asarray(embeddings[start:end], dtype=np.float64)

        if not np.isfinite(batch).all():
            all_finite = False
            fail(f"{model_name}: NaN or Inf found in rows {start}:{end}.")

        norms = np.linalg.norm(batch, axis=1)
        norm_sum += float(norms.sum())
        norm_min = min(norm_min, float(norms.min()))
        norm_max = max(norm_max, float(norms.max()))

        if not np.allclose(norms, 1.0, atol=L2_ATOL, rtol=0.0):
            bad_count = int(np.count_nonzero(np.abs(norms - 1.0) > L2_ATOL))
            fail(
                f"{model_name}: {bad_count} vectors in rows {start}:{end} "
                f"are not L2-normalized within tolerance {L2_ATOL}."
            )

        sum_vectors += batch.sum(axis=0)
        sum_outer_products += batch.T @ batch

        print(f"{model_name}: processed {end:,}/{n_records:,} rows")

    if not all_finite:
        fail(f"{model_name}: finite-value validation failed.")

    # Exact mean cosine over all unordered pairs, assuming unit-normalized rows:
    # sum_{i != j} (x_i dot x_j) / (n * (n - 1))
    # = (||sum_i x_i||^2 - n) / (n * (n - 1)).
    mean_pairwise_cosine = (
        float(np.dot(sum_vectors, sum_vectors)) - n_records
    ) / (n_records * (n_records - 1))

    mean_vector = sum_vectors / n_records
    mean_vector_norm = float(np.linalg.norm(mean_vector))

    # Centered scatter matrix:
    # sum_i (x_i - mean)(x_i - mean)^T
    centered_scatter = (
        sum_outer_products
        - np.outer(sum_vectors, sum_vectors) / n_records
    )
    centered_scatter = (centered_scatter + centered_scatter.T) / 2.0

    # Eigenvalues of the centered covariance matrix. Dividing by n or n-1
    # does not change the explained-variance ratios or effective rank.
    eigenvalues = np.linalg.eigvalsh(centered_scatter)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    eigenvalues = np.sort(eigenvalues)[::-1]

    total_variance = float(eigenvalues.sum())
    if not np.isfinite(total_variance) or total_variance <= 0:
        fail(f"{model_name}: invalid total centered variance.")

    probabilities = eigenvalues / total_variance
    positive = probabilities > 0
    entropy = -float(np.sum(probabilities[positive] * np.log(probabilities[positive])))
    effective_rank = float(np.exp(entropy))
    effective_rank_ratio = effective_rank / dimensions

    first_component_variance = float(eigenvalues[:1].sum() / total_variance)
    top_5_components_variance = float(eigenvalues[:5].sum() / total_variance)
    top_10_components_variance = float(eigenvalues[:10].sum() / total_variance)

    results = {
        "model": model_name,
        "records": n_records,
        "dimensions": dimensions,
        "mean_vector_norm": mean_vector_norm,
        "mean_pairwise_cosine": float(mean_pairwise_cosine),
        "effective_rank": effective_rank,
        "effective_rank_ratio": effective_rank_ratio,
        "first_component_variance": first_component_variance,
        "top_5_components_variance": top_5_components_variance,
        "top_10_components_variance": top_10_components_variance,
        "mean_individual_norm": norm_sum / n_records,
        "minimum_individual_norm": norm_min,
        "maximum_individual_norm": norm_max,
        "all_values_finite": all_finite,
        "all_vectors_l2_normalized": True,
    }

    print("\nGeometry results:")
    for key, value in results.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")

    return results


# --------------------------------- Main ------------------------------------

def main() -> None:
    print("=" * 68)
    print("FULL DATASET EMBEDDING GEOMETRY ANALYSIS")
    print("=" * 68)
    print(f"Project directory: {PROJECT_ROOT}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Batch size: {BATCH_SIZE:,}")

    validate_metadata()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []
    for model_name, embedding_path in MODEL_FILES.items():
        result = analyze_one_model(model_name, embedding_path)
        all_results.append(result)

    results_df = pd.DataFrame(all_results)

    csv_path = OUTPUT_DIR / "full_dataset_geometry_results.csv"
    txt_path = OUTPUT_DIR / "full_dataset_geometry_summary.txt"

    results_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    with txt_path.open("w", encoding="utf-8") as f:
        f.write("FULL DATASET EMBEDDING GEOMETRY ANALYSIS\n")
        f.write("=" * 68 + "\n\n")
        f.write(f"Records per model: {EXPECTED_RECORDS:,}\n")
        f.write(f"Embedding dimensions: {EXPECTED_DIMENSIONS}\n")
        f.write(f"Batch size: {BATCH_SIZE:,}\n\n")
        f.write(results_df.to_string(index=False))
        f.write("\n\n")
        f.write("Methodological notes:\n")
        f.write(
            "- Mean pairwise cosine is computed exactly over all unordered "
            "pairs using the unit-vector identity; no N x N matrix is created.\n"
        )
        f.write(
            "- Effective rank and component variance are computed from "
            "eigenvalues of the centered feature covariance/scatter matrix.\n"
        )
        f.write(
            "- The covariance/scatter matrix is accumulated in batches.\n"
        )
        f.write(
            "- These results describe embedding geometry; they do not alone "
            "establish emotion-class separability or model superiority.\n"
        )

    print("\n" + "=" * 68)
    print("FULL DATASET GEOMETRY ANALYSIS COMPLETED")
    print("=" * 68)
    print("\nComparison:")
    print(results_df.to_string(index=False))
    print(f"\nCSV saved: {csv_path}")
    print(f"Summary saved: {txt_path}")


if __name__ == "__main__":
    main()
