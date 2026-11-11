"""
Emotion separability analysis for the full Arabic dataset.

Reads existing AraBERT v2 and MARBERT embeddings and metadata.
Does NOT modify or regenerate the original embeddings.

Exact full-dataset metrics (computed from class vector sums):
- Mean within-class pairwise cosine similarity (intra-class)
- Mean between-class pairwise cosine similarity (inter-class)
- Separation gap = intra - inter
- Inter/Intra ratio (reported only when intra is non-zero)
- Cosine similarity between emotion centroids

Approximate-by-design metric:
- Silhouette score with cosine distance on a fixed, stratified sample.
  Computing a full 41,202 x 41,202 distance matrix is intentionally avoided.

IMPORTANT:
This evaluates geometry relative to the existing emotion labels. It is
descriptive analysis, not a supervised classification performance test.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score


# ----------------------------- Configuration -----------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "results" / "full_dataset"
OUTPUT_DIR = DATA_DIR / "emotion_separability"

METADATA_PATH = DATA_DIR / "metadata.csv"

MODEL_FILES = {
    "AraBERT_v2": DATA_DIR / "AraBERT_v2_embeddings.npy",
    "MARBERT": DATA_DIR / "MARBERT_embeddings.npy",
}

EXPECTED_RECORDS = 41202
EXPECTED_DIMENSIONS = 768

# Stratified silhouette sample: up to this many rows per emotion.
# Same row indices are used for both models.
SILHOUETTE_SAMPLES_PER_CLASS = 400
RANDOM_SEED = 42
BATCH_SIZE = 2048
L2_ATOL = 1e-3


# -------------------------------- Helpers ---------------------------------

def fail(message):
    print(f"\nERROR: {message}")
    sys.exit(1)


def load_and_validate_metadata():
    if not METADATA_PATH.exists():
        fail(f"Metadata not found: {METADATA_PATH}")

    metadata = pd.read_csv(METADATA_PATH)

    required = ["source_dataset", "original_id", "text", "emotion"]
    missing = [column for column in required if column not in metadata.columns]
    if missing:
        fail(f"Missing metadata columns: {missing}")

    if len(metadata) != EXPECTED_RECORDS:
        fail(
            f"Expected {EXPECTED_RECORDS:,} metadata rows; "
            f"found {len(metadata):,}."
        )

    if metadata[required].isna().any().any():
        fail("Missing values found in required metadata columns.")

    metadata["emotion"] = metadata["emotion"].astype(str).str.strip().str.lower()
    if metadata["emotion"].eq("").any():
        fail("Empty emotion labels found.")

    classes = sorted(metadata["emotion"].unique().tolist())
    if len(classes) != 8:
        fail(f"Expected 8 emotion classes; found {len(classes)}: {classes}")

    print(f"Metadata records: {len(metadata):,}")
    print(f"Emotion classes ({len(classes)}): {classes}")
    print("\nEmotion distribution:")
    print(metadata["emotion"].value_counts().sort_index().to_string())

    return metadata, classes


def make_stratified_sample(labels):
    rng = np.random.default_rng(RANDOM_SEED)
    selected = []

    for emotion in sorted(np.unique(labels)):
        class_indices = np.flatnonzero(labels == emotion)
        n_take = min(SILHOUETTE_SAMPLES_PER_CLASS, len(class_indices))
        chosen = rng.choice(class_indices, size=n_take, replace=False)
        selected.extend(chosen.tolist())

    return np.array(sorted(selected), dtype=np.int64)


def validate_embeddings(path, n_records):
    if not path.exists():
        fail(f"Embedding file not found: {path}")

    vectors = np.load(path, mmap_mode="r")

    if vectors.shape != (n_records, EXPECTED_DIMENSIONS):
        fail(
            f"{path.name}: expected shape "
            f"({n_records}, {EXPECTED_DIMENSIONS}), got {vectors.shape}."
        )

    return vectors


def calculate_exact_pair_metrics(vectors, labels, classes):
    """
    Exact means over all unordered pairs, using unit-normalized vectors.

    For a class with n vectors and vector sum s:
      mean within-class cosine =
        (||s||^2 - n) / (n * (n - 1))

    For classes A and B with sums sA and sB:
      mean between-class cosine =
        (sA dot sB) / (nA * nB)
    """
    dimensions = vectors.shape[1]
    class_sums = {}
    class_counts = {}
    norm_min = np.inf
    norm_max = -np.inf
    norm_sum = 0.0

    print("\nAccumulating class vector sums...")
    for start in range(0, len(labels), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(labels))
        batch = np.asarray(vectors[start:end], dtype=np.float64)

        if not np.isfinite(batch).all():
            fail(f"NaN or Inf found in embeddings rows {start}:{end}.")

        norms = np.linalg.norm(batch, axis=1)
        norm_min = min(norm_min, float(norms.min()))
        norm_max = max(norm_max, float(norms.max()))
        norm_sum += float(norms.sum())

        if not np.allclose(norms, 1.0, atol=L2_ATOL, rtol=0.0):
            fail(
                f"Embeddings rows {start}:{end} are not L2-normalized "
                f"within tolerance {L2_ATOL}."
            )

        batch_labels = labels[start:end]
        for emotion in classes:
            mask = batch_labels == emotion
            if np.any(mask):
                if emotion not in class_sums:
                    class_sums[emotion] = np.zeros(dimensions, dtype=np.float64)
                    class_counts[emotion] = 0
                class_sums[emotion] += batch[mask].sum(axis=0)
                class_counts[emotion] += int(mask.sum())

        print(f"Processed {end:,}/{len(labels):,}")

    intra_by_class = {}
    for emotion in classes:
        n = class_counts[emotion]
        if n < 2:
            intra_by_class[emotion] = np.nan
        else:
            s = class_sums[emotion]
            intra_by_class[emotion] = (
                float(np.dot(s, s)) - n
            ) / (n * (n - 1))

    between_by_pair = {}
    between_sum_weighted = 0.0
    between_pair_count = 0

    for i, emotion_a in enumerate(classes):
        for emotion_b in classes[i + 1:]:
            n_a = class_counts[emotion_a]
            n_b = class_counts[emotion_b]
            value = float(
                np.dot(class_sums[emotion_a], class_sums[emotion_b])
                / (n_a * n_b)
            )
            between_by_pair[(emotion_a, emotion_b)] = value
            between_sum_weighted += value * n_a * n_b
            between_pair_count += n_a * n_b

    total_intra_sum = 0.0
    total_intra_pairs = 0
    for emotion in classes:
        n = class_counts[emotion]
        if n >= 2:
            total_intra_sum += intra_by_class[emotion] * n * (n - 1) / 2
            total_intra_pairs += n * (n - 1) // 2

    # Both are means over unordered pairs; the factor of 2 cancels.
    mean_intra = (
        total_intra_sum / total_intra_pairs if total_intra_pairs else np.nan
    )
    mean_inter = (
        between_sum_weighted / between_pair_count
        if between_pair_count else np.nan
    )

    centroid_vectors = {}
    for emotion in classes:
        centroid = class_sums[emotion] / class_counts[emotion]
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm == 0:
            centroid_vectors[emotion] = np.zeros_like(centroid)
        else:
            centroid_vectors[emotion] = centroid / centroid_norm

    centroid_pair_rows = []
    for i, emotion_a in enumerate(classes):
        for emotion_b in classes[i + 1:]:
            similarity = float(np.dot(
                centroid_vectors[emotion_a], centroid_vectors[emotion_b]
            ))
            centroid_pair_rows.append({
                "emotion_a": emotion_a,
                "emotion_b": emotion_b,
                "centroid_cosine_similarity": similarity,
            })

    per_emotion_rows = []
    for emotion in classes:
        intra = intra_by_class[emotion]
        other_values = [
            value for (a, b), value in between_by_pair.items()
            if a == emotion or b == emotion
        ]
        per_emotion_rows.append({
            "emotion": emotion,
            "records": class_counts[emotion],
            "mean_intra_class_pairwise_cosine": intra,
            "mean_inter_class_cosine_to_other_classes": (
                float(np.mean(other_values)) if other_values else np.nan
            ),
            "centroid_norm_before_unit_normalization": float(
                np.linalg.norm(class_sums[emotion] / class_counts[emotion])
            ),
        })

    return {
        "mean_intra_class_pairwise_cosine": float(mean_intra),
        "mean_inter_class_pairwise_cosine": float(mean_inter),
        "intra_minus_inter_gap": float(mean_intra - mean_inter),
        "inter_over_intra_ratio": (
            float(mean_inter / mean_intra)
            if np.isfinite(mean_intra) and abs(mean_intra) > 1e-12
            else np.nan
        ),
        "mean_individual_vector_norm": norm_sum / len(labels),
        "minimum_individual_vector_norm": norm_min,
        "maximum_individual_vector_norm": norm_max,
        "all_values_finite": True,
        "all_vectors_l2_normalized": True,
    }, per_emotion_rows, centroid_pair_rows


def calculate_silhouette(vectors, labels, sample_indices):
    sample_vectors = np.asarray(vectors[sample_indices], dtype=np.float32)
    sample_labels = labels[sample_indices]

    if len(np.unique(sample_labels)) < 2:
        fail("Silhouette sample contains fewer than two emotion classes.")

    print(
        f"\nCalculating cosine silhouette on stratified sample "
        f"({len(sample_indices):,} rows)..."
    )
    return float(silhouette_score(
        sample_vectors,
        sample_labels,
        metric="cosine",
    ))


# ---------------------------------- Main -----------------------------------

def main():
    print("=" * 70)
    print("FULL DATASET EMOTION SEPARABILITY ANALYSIS")
    print("=" * 70)
    print("This script does not modify original embeddings.")

    metadata, classes = load_and_validate_metadata()
    labels = metadata["emotion"].to_numpy()
    sample_indices = make_stratified_sample(labels)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sample_path = OUTPUT_DIR / "silhouette_sample_row_indices.csv"
    pd.DataFrame({
        "row_index": sample_indices,
        "emotion": labels[sample_indices],
    }).to_csv(sample_path, index=False, encoding="utf-8-sig")

    model_summary_rows = []
    all_per_emotion = []
    all_centroid_pairs = []

    for model_name, embedding_path in MODEL_FILES.items():
        print("\n" + "=" * 70)
        print(f"MODEL: {model_name}")
        print("=" * 70)

        vectors = validate_embeddings(embedding_path, len(metadata))

        exact_metrics, per_emotion_rows, centroid_pair_rows = (
            calculate_exact_pair_metrics(vectors, labels, classes)
        )

        silhouette = calculate_silhouette(
            vectors, labels, sample_indices
        )

        summary = {
            "model": model_name,
            "records": len(metadata),
            "dimensions": EXPECTED_DIMENSIONS,
            "silhouette_cosine_stratified_sample": silhouette,
            **exact_metrics,
        }
        model_summary_rows.append(summary)

        for row in per_emotion_rows:
            all_per_emotion.append({"model": model_name, **row})

        for row in centroid_pair_rows:
            all_centroid_pairs.append({"model": model_name, **row})

        print("\nModel results:")
        for key, value in summary.items():
            if isinstance(value, float):
                print(f"{key}: {value:.8f}")
            else:
                print(f"{key}: {value}")

        del vectors

    summary_df = pd.DataFrame(model_summary_rows)
    per_emotion_df = pd.DataFrame(all_per_emotion)
    centroid_df = pd.DataFrame(all_centroid_pairs)

    summary_path = OUTPUT_DIR / "emotion_separability_summary.csv"
    per_emotion_path = OUTPUT_DIR / "emotion_separability_by_emotion.csv"
    centroid_path = OUTPUT_DIR / "emotion_centroid_similarities.csv"
    report_path = OUTPUT_DIR / "emotion_separability_report.txt"

    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    per_emotion_df.to_csv(per_emotion_path, index=False, encoding="utf-8-sig")
    centroid_df.to_csv(centroid_path, index=False, encoding="utf-8-sig")

    with report_path.open("w", encoding="utf-8") as f:
        f.write("FULL DATASET EMOTION SEPARABILITY ANALYSIS\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Records: {len(metadata):,}\n")
        f.write(f"Dimensions: {EXPECTED_DIMENSIONS}\n")
        f.write(f"Emotion classes: {', '.join(classes)}\n")
        f.write(f"Silhouette sample size: {len(sample_indices):,}\n")
        f.write(f"Silhouette samples per class: up to {SILHOUETTE_SAMPLES_PER_CLASS}\n")
        f.write(f"Random seed: {RANDOM_SEED}\n\n")
        f.write("OVERALL MODEL COMPARISON\n")
        f.write(summary_df.to_string(index=False))
        f.write("\n\nPER-EMOTION RESULTS\n")
        f.write(per_emotion_df.to_string(index=False))
        f.write("\n\nEMOTION CENTROID PAIR SIMILARITIES\n")
        f.write(centroid_df.to_string(index=False))
        f.write("\n\nMETHODOLOGICAL NOTES\n")
        f.write(
            "- Intra-class and inter-class pairwise cosine means are exact "
            "over the full dataset and are calculated from class vector sums.\n"
        )
        f.write(
            "- Silhouette uses cosine distance on the same fixed stratified "
            "sample for both models; it is not computed on all 41,202 rows.\n"
        )
        f.write(
            "- Centroid cosine similarities compare normalized emotion "
            "centroid directions.\n"
        )
        f.write(
            "- These are label-conditioned geometric diagnostics, not "
            "classification accuracy or proof of causal model superiority.\n"
        )
        f.write(
            "- The overall intra/inter means are pair-weighted; per-emotion "
            "values are also provided to expose class imbalance.\n"
        )

    print("\n" + "=" * 70)
    print("EMOTION SEPARABILITY ANALYSIS COMPLETED")
    print("=" * 70)
    print("\nOverall comparison:")
    print(summary_df.to_string(index=False))
    print("\nSaved files:")
    print(summary_path)
    print(per_emotion_path)
    print(centroid_path)
    print(sample_path)
    print(report_path)
    print("\nOriginal embeddings and metadata were not modified.")


if __name__ == "__main__":
    main()
