"""
Full-dataset variance analysis: source_dataset vs emotion.
Uses all 41,202 embeddings for each model. No sampling.

For each model, computes multivariate (all 768 dimensions together) sums of
squares using categorical linear models:
  1) intercept only
  2) emotion only
  3) source dataset only
  4) additive emotion + source dataset
  5) source_dataset x emotion cell-means model (interaction)

Reports:
- Raw R² for emotion-only and source-only models.
- Combined additive R².
- Unique partial R² for source given emotion and emotion given source.
- Additional interaction R² after additive main effects.
- Number of observed source-emotion cells and design rank.

This is an association/decomposition analysis, not a causal test.
It does not prove that source or emotion causes embedding variation.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "results" / "full_dataset"
OUT_DIR = DATA_DIR / "source_emotion_variance"
META_PATH = DATA_DIR / "metadata.csv"

MODEL_FILES = {
    "AraBERT_v2": DATA_DIR / "AraBERT_v2_embeddings.npy",
    "MARBERT": DATA_DIR / "MARBERT_embeddings.npy",
}
N_EXPECTED = 41202
D_EXPECTED = 768
BATCH_SIZE = 4096
EMOTIONS_EXPECTED = {
    "anger", "anticipation", "disgust", "fear",
    "joy", "sadness", "surprise", "trust"
}


def stop(message):
    raise RuntimeError(message)


def load_metadata():
    if not META_PATH.exists():
        stop(f"Metadata file not found: {META_PATH}")
    df = pd.read_csv(META_PATH, keep_default_na=False)
    required = {"row_index", "source_dataset", "emotion"}
    missing = required - set(df.columns)
    if missing:
        stop(f"Missing metadata columns: {sorted(missing)}")
    if len(df) != N_EXPECTED:
        stop(f"Expected {N_EXPECTED:,} rows, found {len(df):,}")
    idx = pd.to_numeric(df["row_index"], errors="raise").to_numpy()
    if not np.array_equal(idx, np.arange(N_EXPECTED)):
        stop("Metadata row_index is not in exact embedding row order.")
    for col in ["source_dataset", "emotion"]:
        df[col] = df[col].astype(str).str.strip()
        if (df[col] == "").any():
            stop(f"Empty values found in {col}.")
    df["emotion"] = df["emotion"].str.lower()
    observed_emotions = set(df["emotion"].unique())
    if observed_emotions != EMOTIONS_EXPECTED:
        stop(f"Unexpected emotion labels: {sorted(observed_emotions)}")
    if df["source_dataset"].nunique() != 16:
        stop(f"Expected 16 source datasets; found {df['source_dataset'].nunique()}")
    return df


def read_embedding_sufficient_statistics(path):
    if not path.exists():
        stop(f"Embedding file not found: {path}")
    X = np.load(path, mmap_mode="r")
    if X.shape != (N_EXPECTED, D_EXPECTED):
        stop(f"{path.name}: expected {(N_EXPECTED, D_EXPECTED)}, got {X.shape}")

    # Total sum of squares around the grand mean, trace across all 768 dims.
    total_yty = 0.0
    total_sum = np.zeros(D_EXPECTED, dtype=np.float64)
    for start in range(0, N_EXPECTED, BATCH_SIZE):
        end = min(start + BATCH_SIZE, N_EXPECTED)
        batch = np.asarray(X[start:end], dtype=np.float64)
        if not np.isfinite(batch).all():
            stop(f"NaN/Inf in embeddings rows {start}:{end}")
        total_yty += float(np.sum(batch * batch))
        total_sum += batch.sum(axis=0)
        print(f"  Read embeddings: {end:,}/{N_EXPECTED:,}")

    grand_mean = total_sum / N_EXPECTED
    total_ss = total_yty - N_EXPECTED * float(np.dot(grand_mean, grand_mean))
    if total_ss <= 0:
        stop("Total centered sum of squares is not positive.")
    return X, total_yty, total_sum, total_ss


def categorical_design(df, include_source, include_emotion, interaction=False):
    """
    Treatment-coded sparse design.
    With interaction=True, use one indicator per observed source-emotion cell;
    this is a saturated cell-means model (no separate intercept needed).
    """
    n = len(df)
    src = pd.Categorical(df["source_dataset"])
    emo = pd.Categorical(df["emotion"])

    if interaction:
        pairs = pd.MultiIndex.from_arrays([src.codes, emo.codes])
        pair_codes, unique_pairs = pd.factorize(pairs, sort=True)
        rows = np.arange(n)
        cols = pair_codes
        design = sparse.csr_matrix(
            (np.ones(n, dtype=np.float64), (rows, cols)),
            shape=(n, len(unique_pairs)),
        )
        return design, {
            "design": "source_emotion_cell_means",
            "columns": int(design.shape[1]),
            "observed_cells": int(len(unique_pairs)),
        }

    columns = [sparse.csr_matrix(np.ones((n, 1), dtype=np.float64))]
    names = ["intercept"]

    if include_source:
        src_dummy = pd.get_dummies(src, drop_first=True, dtype=np.float64)
        if src_dummy.shape[1]:
            columns.append(sparse.csr_matrix(src_dummy.to_numpy()))
            names.extend([f"source_{x}" for x in src.categories[1:]])

    if include_emotion:
        emo_dummy = pd.get_dummies(emo, drop_first=True, dtype=np.float64)
        if emo_dummy.shape[1]:
            columns.append(sparse.csr_matrix(emo_dummy.to_numpy()))
            names.extend([f"emotion_{x}" for x in emo.categories[1:]])

    design = sparse.hstack(columns, format="csr")
    return design, {"design": "+".join(names), "columns": int(design.shape[1])}


def model_sse(design, X, total_yty, total_sum, n):
    """
    SSE = y'y - beta'(X'y), with beta solving (X'X) beta = X'y.
    X is sparse; only small design cross-products are materialized.
    """
    xtx = (design.T @ design).toarray()
    xty = np.zeros((design.shape[1], D_EXPECTED), dtype=np.float64)

    for start in range(0, n, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n)
        x_batch = design[start:end]
        y_batch = np.asarray(X[start:end], dtype=np.float64)
        xty += x_batch.T @ y_batch

    rank = int(np.linalg.matrix_rank(xtx))
    beta = np.linalg.pinv(xtx, rcond=1e-12) @ xty
    explained_uncentered = float(np.sum(beta * xty))
    sse = total_yty - explained_uncentered

    # Numerical guard: nested model residual sums should not be negative.
    if sse < 0 and abs(sse) < 1e-8 * total_yty:
        sse = 0.0
    if sse < 0:
        stop(f"Computed negative residual sum of squares: {sse}")

    return sse, rank


def main():
    print("=" * 72)
    print("SOURCE DATASET VS EMOTION — FULL DATASET VARIANCE ANALYSIS")
    print("=" * 72)
    print("No sampling. All records and all 768 embedding dimensions are used.")

    df = load_metadata()
    print(f"Records: {len(df):,}")
    print(f"Sources: {df['source_dataset'].nunique()}")
    print(f"Emotions: {df['emotion'].nunique()}")
    print(f"Observed source-emotion combinations: {df.groupby(['source_dataset','emotion']).ngroups}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for model_name, path in MODEL_FILES.items():
        print("\n" + "=" * 72)
        print("MODEL:", model_name)
        print("=" * 72)

        X, total_yty, total_sum, total_ss = read_embedding_sufficient_statistics(path)

        designs = {
            "intercept": categorical_design(
                df, include_source=False, include_emotion=False
            )[0],
            "emotion_only": categorical_design(
                df, include_source=False, include_emotion=True
            )[0],
            "source_only": categorical_design(
                df, include_source=True, include_emotion=False
            )[0],
            "additive_source_emotion": categorical_design(
                df, include_source=True, include_emotion=True
            )[0],
            "source_emotion_interaction": categorical_design(
                df, include_source=False, include_emotion=False,
                interaction=True
            )[0],
        }

        results = {}
        for label, design in designs.items():
            print(f"\nFitting {label} ({design.shape[1]} design columns)...")
            sse, rank = model_sse(
                design, X, total_yty, total_sum, len(df)
            )
            results[label] = {
                "sse": sse,
                "rank": rank,
                "r2": float(1.0 - sse / total_ss),
            }
            print(
                f"  rank={rank}, residual SS={sse:.6f}, "
                f"R2={results[label]['r2']:.8f}"
            )

        r2_emotion = results["emotion_only"]["r2"]
        r2_source = results["source_only"]["r2"]
        r2_additive = results["additive_source_emotion"]["r2"]
        r2_interaction = results["source_emotion_interaction"]["r2"]

        sse_emotion = results["emotion_only"]["sse"]
        sse_source = results["source_only"]["sse"]
        sse_additive = results["additive_source_emotion"]["sse"]
        sse_cell = results["source_emotion_interaction"]["sse"]

        # Partial R²: incremental improvement relative to the reduced model's
        # remaining unexplained variation.
        partial_source_given_emotion = (
            (sse_emotion - sse_additive) / sse_emotion
            if sse_emotion > 0 else np.nan
        )
        partial_emotion_given_source = (
            (sse_source - sse_additive) / sse_source
            if sse_source > 0 else np.nan
        )
        interaction_incremental_r2 = (
            (sse_additive - sse_cell) / total_ss
        )

        row = {
            "model": model_name,
            "records": len(df),
            "dimensions": D_EXPECTED,
            "total_centered_sum_squares": total_ss,
            "emotion_only_R2": r2_emotion,
            "source_only_R2": r2_source,
            "additive_source_plus_emotion_R2": r2_additive,
            "source_unique_partial_R2_given_emotion": partial_source_given_emotion,
            "emotion_unique_partial_R2_given_source": partial_emotion_given_source,
            "additional_source_emotion_interaction_R2": interaction_incremental_r2,
            "cell_means_model_R2": r2_interaction,
            "observed_source_emotion_cells": int(
                df.groupby(["source_dataset", "emotion"]).ngroups
            ),
        }
        summary_rows.append(row)

        print("\nKey results:")
        for key, value in row.items():
            if isinstance(value, float):
                print(f"{key}: {value:.8f}")
            else:
                print(f"{key}: {value}")

        # Save nested-model fit details for auditability.
        fit_rows = []
        for name, values in results.items():
            fit_rows.append({
                "model": model_name,
                "categorical_model": name,
                "design_rank": values["rank"],
                "residual_sum_squares": values["sse"],
                "R2": values["r2"],
            })
        pd.DataFrame(fit_rows).to_csv(
            OUT_DIR / f"{model_name}_model_fits.csv",
            index=False, encoding="utf-8-sig"
        )
        del X

    summary_df = pd.DataFrame(summary_rows)
    summary_file = OUT_DIR / "source_emotion_variance_summary.csv"
    summary_df.to_csv(summary_file, index=False, encoding="utf-8-sig")

    report_file = OUT_DIR / "source_emotion_variance_report.txt"
    with report_file.open("w", encoding="utf-8") as f:
        f.write("SOURCE DATASET VS EMOTION — FULL DATASET VARIANCE ANALYSIS\n")
        f.write("=" * 72 + "\n")
        f.write(f"Records: {len(df):,}\n")
        f.write(f"Embedding dimensions: {D_EXPECTED}\n")
        f.write(f"Source datasets: {df['source_dataset'].nunique()}\n")
        f.write(f"Emotion classes: {df['emotion'].nunique()}\n")
        f.write("Sampling: none; all records included\n\n")
        f.write(summary_df.to_string(index=False))
        f.write("\n\nInterpretation notes:\n")
        f.write("- R2 values are multivariate trace-based proportions of total centered variation across all 768 dimensions.\n")
        f.write("- source_unique_partial_R2_given_emotion measures the additional proportion of residual variation explained by source after emotion is included.\n")
        f.write("- emotion_unique_partial_R2_given_source measures the additional proportion of residual variation explained by emotion after source is included.\n")
        f.write("- The interaction model uses observed source-emotion cells and captures non-additive cell-specific structure.\n")
        f.write("- These are descriptive associations, not causal effects or classification accuracy.\n")
        f.write("- Unbalanced source/emotion counts affect precision and interpretation; inspect source-emotion counts.\n")

    # Save contingency table, crucial for identifying confounding / empty cells.
    contingency = pd.crosstab(df["source_dataset"], df["emotion"])
    contingency.to_csv(
        OUT_DIR / "source_emotion_contingency.csv",
        encoding="utf-8-sig"
    )

    print("\n" + "=" * 72)
    print("ANALYSIS COMPLETED")
    print("=" * 72)
    print(summary_df.to_string(index=False))
    print("\nSaved:")
    print(summary_file)
    print(report_file)
    print(OUT_DIR / "source_emotion_contingency.csv")
    print("\nOriginal embeddings and metadata were not modified.")


if __name__ == "__main__":
    main()
