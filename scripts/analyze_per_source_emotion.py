"""
Per-source emotion distribution and embedding analysis.
Uses ALL 41,202 records for both embedding models. No sampling.

Outputs:
1) source_emotion_counts.csv
2) source_emotion_row_percentages.csv
3) emotion_source_column_percentages.csv
4) source_distribution_summary.csv
5) per_source_emotion_embedding_similarity.csv
6) per_model_source_summary.csv
7) methodology_report.txt

The embedding analysis uses exact group sufficient statistics on L2-normalized
vectors to calculate average pairwise cosine similarity within and between
emotion groups inside each source. It does not calculate a sampled silhouette.
"""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "results" / "full_dataset"
OUT_DIR = DATA_DIR / "per_source_emotion_analysis"
META_PATH = DATA_DIR / "metadata.csv"

MODEL_FILES = {
    "AraBERT_v2": DATA_DIR / "AraBERT_v2_embeddings.npy",
    "MARBERT": DATA_DIR / "MARBERT_embeddings.npy",
}

EXPECTED_ROWS = 41202
EXPECTED_DIMS = 768
BATCH_SIZE = 4096
EXPECTED_EMOTIONS = [
    "anger", "anticipation", "disgust", "fear",
    "joy", "sadness", "surprise", "trust"
]


def fail(message):
    raise RuntimeError(message)


def load_and_validate_metadata():
    if not META_PATH.exists():
        fail(f"Missing metadata file: {META_PATH}")

    df = pd.read_csv(META_PATH, keep_default_na=False)
    required = {"row_index", "source_dataset", "emotion"}
    missing = required - set(df.columns)
    if missing:
        fail(f"Missing metadata columns: {sorted(missing)}")
    if len(df) != EXPECTED_ROWS:
        fail(f"Expected {EXPECTED_ROWS:,} rows, found {len(df):,}")

    row_index = pd.to_numeric(df["row_index"], errors="raise").to_numpy()
    if not np.array_equal(row_index, np.arange(EXPECTED_ROWS)):
        fail("row_index does not match embedding row order.")

    for col in ["source_dataset", "emotion"]:
        df[col] = df[col].astype(str).str.strip()
        if (df[col] == "").any():
            fail(f"Empty value found in {col}.")

    df["emotion"] = df["emotion"].str.lower()
    observed = set(df["emotion"].unique())
    if observed != set(EXPECTED_EMOTIONS):
        fail(f"Unexpected emotion labels: {sorted(observed)}")
    if df["source_dataset"].nunique() != 16:
        fail(f"Expected 16 sources; found {df['source_dataset'].nunique()}")

    return df


def distribution_analysis(df):
    counts = pd.crosstab(df["source_dataset"], df["emotion"])
    counts = counts.reindex(columns=EXPECTED_EMOTIONS, fill_value=0)

    row_pct = counts.div(counts.sum(axis=1), axis=0) * 100
    col_pct = counts.div(counts.sum(axis=0), axis=1) * 100

    rows = []
    for source in counts.index:
        vals = counts.loc[source].to_numpy(dtype=float)
        total = vals.sum()
        p = vals / total
        nonzero = p[p > 0]
        entropy = float(-(nonzero * np.log(nonzero)).sum())
        normalized_entropy = (
            entropy / np.log(len(EXPECTED_EMOTIONS))
            if len(EXPECTED_EMOTIONS) > 1 else 0.0
        )
        dominant = counts.loc[source].idxmax()
        rows.append({
            "source_dataset": source,
            "records": int(total),
            "percentage_of_full_dataset": 100 * total / len(df),
            "number_of_emotions_present": int((vals > 0).sum()),
            "dominant_emotion": dominant,
            "dominant_emotion_count": int(counts.loc[source, dominant]),
            "dominant_emotion_percentage_within_source":
                float(row_pct.loc[source, dominant]),
            "emotion_distribution_entropy": entropy,
            "normalized_emotion_entropy_0_to_1": normalized_entropy,
        })

    summary = pd.DataFrame(rows).sort_values(
        "records", ascending=False
    )

    counts.to_csv(OUT_DIR / "source_emotion_counts.csv", encoding="utf-8-sig")
    row_pct.to_csv(
        OUT_DIR / "source_emotion_row_percentages.csv",
        float_format="%.6f", encoding="utf-8-sig"
    )
    col_pct.to_csv(
        OUT_DIR / "emotion_source_column_percentages.csv",
        float_format="%.6f", encoding="utf-8-sig"
    )
    summary.to_csv(
        OUT_DIR / "source_distribution_summary.csv",
        index=False, float_format="%.6f", encoding="utf-8-sig"
    )

    return counts, row_pct, col_pct, summary


def cosine_group_statistics(df, embedding_path, model_name):
    if not embedding_path.exists():
        fail(f"Missing embeddings: {embedding_path}")

    X = np.load(embedding_path, mmap_mode="r")
    if X.shape != (EXPECTED_ROWS, EXPECTED_DIMS):
        fail(
            f"{embedding_path.name}: expected "
            f"{(EXPECTED_ROWS, EXPECTED_DIMS)}, got {X.shape}"
        )

    sources = df["source_dataset"].astype(str).to_numpy()
    emotions = df["emotion"].astype(str).to_numpy()
    source_names = sorted(df["source_dataset"].unique())

    # Sum vectors per observed source-emotion group.
    sums = {}
    counts = {}
    norm_min = float("inf")
    norm_max = 0.0
    all_finite = True

    for start in range(0, EXPECTED_ROWS, BATCH_SIZE):
        end = min(start + BATCH_SIZE, EXPECTED_ROWS)
        batch = np.asarray(X[start:end], dtype=np.float64)

        if not np.isfinite(batch).all():
            all_finite = False
            fail(f"NaN/Inf in {model_name} embeddings at rows {start}:{end}")

        norms = np.linalg.norm(batch, axis=1)
        norm_min = min(norm_min, float(norms.min()))
        norm_max = max(norm_max, float(norms.max()))

        # Embeddings are expected to be L2-normalized; verify, do not silently
        # normalize a different representation.
        if not np.allclose(norms, 1.0, atol=1e-4, rtol=1e-4):
            fail(
                f"{model_name} embeddings are not L2-normalized "
                f"(norm range {norm_min:.6f}..{norm_max:.6f})."
            )

        batch_sources = sources[start:end]
        batch_emotions = emotions[start:end]
        keys = pd.MultiIndex.from_arrays([batch_sources, batch_emotions])
        codes, unique_keys = pd.factorize(keys, sort=False)

        for code, key in enumerate(unique_keys):
            mask = codes == code
            source = str(key[0])
            emotion = str(key[1])
            group_key = (source, emotion)
            group_sum = batch[mask].sum(axis=0)
            group_count = int(mask.sum())

            if group_key not in sums:
                sums[group_key] = group_sum
                counts[group_key] = group_count
            else:
                sums[group_key] += group_sum
                counts[group_key] += group_count

        print(f"{model_name}: accumulated {end:,}/{EXPECTED_ROWS:,}")

    result_rows = []
    source_rows = []

    for source in source_names:
        present = []
        for emotion in EXPECTED_EMOTIONS:
            key = (source, emotion)
            n = counts.get(key, 0)
            if n > 0:
                present.append((emotion, n, sums[key]))

        within_weighted_sum = 0.0
        within_pair_count = 0
        within_group_rows = []

        for emotion, n, vector_sum in present:
            # For unit vectors: sum_{i != j} cos(x_i,x_j)
            # = ||sum(x)||^2 - n. Divide by n(n-1) for mean unordered pair.
            if n >= 2:
                pair_count = n * (n - 1) // 2
                pair_cos_sum = (float(np.dot(vector_sum, vector_sum)) - n) / 2.0
                mean_within = pair_cos_sum / pair_count
                within_weighted_sum += pair_cos_sum
                within_pair_count += pair_count
            else:
                pair_count = 0
                mean_within = np.nan

            within_group_rows.append({
                "model": model_name,
                "source_dataset": source,
                "emotion": emotion,
                "records": n,
                "mean_within_emotion_pairwise_cosine": mean_within,
                "within_emotion_pair_count": pair_count,
            })

        # Between-emotion pair similarities, weighted by number of text pairs.
        between_weighted_sum = 0.0
        between_pair_count = 0
        for i in range(len(present)):
            emotion_a, n_a, sum_a = present[i]
            for j in range(i + 1, len(present)):
                emotion_b, n_b, sum_b = present[j]
                pair_count = n_a * n_b
                pair_cos_sum = float(np.dot(sum_a, sum_b))
                between_weighted_sum += pair_cos_sum
                between_pair_count += pair_count

        mean_within_source = (
            within_weighted_sum / within_pair_count
            if within_pair_count else np.nan
        )
        mean_between_source = (
            between_weighted_sum / between_pair_count
            if between_pair_count else np.nan
        )
        gap = (
            mean_within_source - mean_between_source
            if np.isfinite(mean_within_source)
            and np.isfinite(mean_between_source)
            else np.nan
        )

        source_rows.append({
            "model": model_name,
            "source_dataset": source,
            "records": int(sum(n for _, n, _ in present)),
            "emotions_present": len(present),
            "mean_within_emotion_pairwise_cosine": mean_within_source,
            "mean_between_emotion_pairwise_cosine": mean_between_source,
            "within_minus_between_cosine_gap": gap,
            "within_emotion_pairs": within_pair_count,
            "between_emotion_pairs": between_pair_count,
            "embedding_norm_min": norm_min,
            "embedding_norm_max": norm_max,
            "all_values_finite": all_finite,
        })
        result_rows.extend(within_group_rows)

    return pd.DataFrame(result_rows), pd.DataFrame(source_rows)


def main():
    print("=" * 72)
    print("PER-SOURCE EMOTION DISTRIBUTION AND EMBEDDING ANALYSIS")
    print("FULL DATASET — NO SAMPLING")
    print("=" * 72)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_and_validate_metadata()
    print(f"Records: {len(df):,}")
    print(f"Sources: {df['source_dataset'].nunique()}")
    print(f"Emotions: {df['emotion'].nunique()}")

    counts, row_pct, col_pct, distribution_summary = distribution_analysis(df)

    all_emotion_rows = []
    all_source_rows = []

    for model_name, embedding_path in MODEL_FILES.items():
        print(f"\nAnalyzing {model_name}...")
        emotion_df, source_df = cosine_group_statistics(
            df, embedding_path, model_name
        )
        all_emotion_rows.append(emotion_df)
        all_source_rows.append(source_df)

    per_emotion = pd.concat(all_emotion_rows, ignore_index=True)
    per_source = pd.concat(all_source_rows, ignore_index=True)

    per_emotion.to_csv(
        OUT_DIR / "per_source_per_emotion_embedding_similarity.csv",
        index=False, float_format="%.8f", encoding="utf-8-sig"
    )
    per_source.to_csv(
        OUT_DIR / "per_source_embedding_summary.csv",
        index=False, float_format="%.8f", encoding="utf-8-sig"
    )

    report = OUT_DIR / "methodology_report.txt"
    with report.open("w", encoding="utf-8") as f:
        f.write("PER-SOURCE EMOTION DISTRIBUTION AND EMBEDDING ANALYSIS\n")
        f.write("=" * 72 + "\n")
        f.write(f"Records: {len(df):,}\n")
        f.write(f"Source datasets: {df['source_dataset'].nunique()}\n")
        f.write(f"Emotion classes: {df['emotion'].nunique()}\n")
        f.write("Sampling: none; all records included\n")
        f.write("Embedding dimensions: 768 per model\n\n")
        f.write("SOURCE DISTRIBUTION SUMMARY\n")
        f.write(distribution_summary.to_string(index=False))
        f.write("\n\nPER-SOURCE EMBEDDING SUMMARY\n")
        f.write(per_source.to_string(index=False))
        f.write("\n\nMethod notes:\n")
        f.write("- Row percentages describe emotion composition within each source.\n")
        f.write("- Column percentages describe each emotion's distribution across sources.\n")
        f.write("- Normalized entropy ranges from 0 to 1; higher values indicate a more even distribution across the eight emotions.\n")
        f.write("- Embedding cosine statistics use every vector and exact pair-count-weighted group sums; no pair sampling is used.\n")
        f.write("- Within-minus-between cosine gap is descriptive, not a classification score or causal effect.\n")
        f.write("- Sources with very few records or missing emotion classes require cautious interpretation.\n")
        f.write("- No embeddings, metadata, or prior results were modified.\n")

    print("\n" + "=" * 72)
    print("ANALYSIS COMPLETED")
    print("=" * 72)
    print("\nSource distribution summary:")
    print(distribution_summary.to_string(index=False))
    print("\nPer-source embedding summary:")
    print(per_source.to_string(index=False))
    print("\nOutput folder:", OUT_DIR)
    print("Original embeddings and metadata were not modified.")


if __name__ == "__main__":
    main()
