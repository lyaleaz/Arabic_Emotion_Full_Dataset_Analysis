"""
Full Dataset PCA Visualization — AraBERT v2 vs MARBERT
Uses ALL 41,202 texts, no sampling and no exclusions.

- Fits exact centered PCA from the full 768-dimensional embedding matrices.
- Uses a memory-efficient 768x768 covariance matrix (no 41,202x41,202 matrix).
- Projects ALL rows onto each model's first two principal components.
- Saves every projected row, model metrics, and full-resolution PNG/PDF charts.
- Never modifies the original embeddings or metadata.

Requirements: numpy, pandas, matplotlib
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# --------------------- Configuration ---------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "results" / "full_dataset"
OUT_DIR = DATA_DIR / "pca_full_dataset"
OUT_DIR.mkdir(parents=True, exist_ok=True)

METADATA_FILE = DATA_DIR / "metadata.csv"
MODEL_FILES = {
    "AraBERT_v2": DATA_DIR / "AraBERT_v2_embeddings.npy",
    "MARBERT": DATA_DIR / "MARBERT_embeddings.npy",
}

EXPECTED_N = 41202
EXPECTED_D = 768
BATCH_SIZE = 2048

EMOTIONS = [
    "anger", "anticipation", "disgust", "fear",
    "joy", "sadness", "surprise", "trust",
]

# Stable palette and same emotion colors across models
COLORS = {
    "anger": "#D73027",
    "anticipation": "#E69F00",
    "disgust": "#7B3294",
    "fear": "#2166AC",
    "joy": "#1A9850",
    "sadness": "#4D4D4D",
    "surprise": "#00A6CA",
    "trust": "#B2A400",
}

# --------------------- Load metadata ---------------------------------------

print("=" * 70)
print("FULL-DATASET PCA VISUALIZATION: NO SAMPLING")
print("=" * 70)

if not METADATA_FILE.exists():
    raise FileNotFoundError(METADATA_FILE)

meta = pd.read_csv(METADATA_FILE, keep_default_na=False)

if len(meta) != EXPECTED_N:
    raise ValueError(f"Expected {EXPECTED_N} metadata rows, got {len(meta)}")

for required in ("source_dataset", "original_id", "text", "emotion"):
    if required not in meta.columns:
        raise ValueError(f"Missing metadata column: {required}")

if "row_index" in meta.columns:
    row_idx = pd.to_numeric(meta["row_index"], errors="raise").to_numpy()
    if not np.array_equal(row_idx, np.arange(EXPECTED_N)):
        raise ValueError("Metadata row_index does not match embedding row order.")

meta["emotion"] = meta["emotion"].astype(str).str.strip().str.lower()
labels = meta["emotion"].to_numpy()

if set(labels) != set(EMOTIONS):
    raise ValueError(f"Expected Plutchik 8 labels, found {sorted(set(labels))}")

print(f"Metadata verified: {len(meta):,} records")
print(f"Source datasets: {meta['source_dataset'].nunique()}")
print("Emotion distribution:")
print(meta["emotion"].value_counts().sort_index().to_string())


def analyze_model(model_name, file_path):
    print("\n" + "=" * 70)
    print("PCA MODEL:", model_name)
    print("=" * 70)

    if not file_path.exists():
        raise FileNotFoundError(file_path)

    X = np.load(file_path, mmap_mode="r")
    if X.shape != (EXPECTED_N, EXPECTED_D):
        raise ValueError(
            f"Wrong shape for {model_name}: {X.shape}; "
            f"expected {(EXPECTED_N, EXPECTED_D)}"
        )

    # Numerically stable float64 sums over the ENTIRE dataset.
    vector_sum = np.zeros(EXPECTED_D, dtype=np.float64)
    gram = np.zeros((EXPECTED_D, EXPECTED_D), dtype=np.float64)
    print("Pass 1/2: fitting centered PCA from all records")

    for start in range(0, EXPECTED_N, BATCH_SIZE):
        end = min(start + BATCH_SIZE, EXPECTED_N)
        batch = np.asarray(X[start:end], dtype=np.float64)
        if not np.isfinite(batch).all():
            raise ValueError(f"Non-finite embeddings at rows {start}:{end}")

        vector_sum += batch.sum(axis=0)
        gram += batch.T @ batch
        print(f"  PCA fitting: {end:,}/{EXPECTED_N:,}")

    mean = vector_sum / EXPECTED_N
    scatter = gram - np.outer(vector_sum, vector_sum) / EXPECTED_N
    scatter = 0.5 * (scatter + scatter.T)

    eigenvalues, eigenvectors = np.linalg.eigh(scatter)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    total = eigenvalues.sum()

    if total <= 0:
        raise RuntimeError(f"{model_name} has zero or invalid PCA variance")

    variance_ratios = eigenvalues / total
    pc1 = float(variance_ratios[0])
    pc2 = float(variance_ratios[1])
    pc12 = pc1 + pc2
    pc5 = float(variance_ratios[:5].sum())
    pc10 = float(variance_ratios[:10].sum())

    # Fix arbitrary eigenvector signs for reproducible visualization.
    basis = eigenvectors[:, :2].copy()
    for k in range(2):
        pivot = np.argmax(np.abs(basis[:, k]))
        if basis[pivot, k] < 0:
            basis[:, k] *= -1

    # Project EVERY row, keeping same input row order.
    coords = np.empty((EXPECTED_N, 2), dtype=np.float32)
    print("\nPass 2/2: projecting every record")
    for start in range(0, EXPECTED_N, BATCH_SIZE):
        end = min(start + BATCH_SIZE, EXPECTED_N)
        batch = np.asarray(X[start:end], dtype=np.float64)
        coords[start:end] = ((batch - mean) @ basis).astype(np.float32)
        print(f"  PCA projection: {end:,}/{EXPECTED_N:,}")

    result = meta[["row_index", "source_dataset", "original_id", "emotion"]].copy()
    result["PC1"] = coords[:, 0]
    result["PC2"] = coords[:, 1]
    coords_path = OUT_DIR / f"{model_name}_pca_all_41202.csv"
    result.to_csv(coords_path, index=False, encoding="utf-8-sig")

    # Full variance profile for later scientific use.
    variance_path = OUT_DIR / f"{model_name}_pca_explained_variance.csv"
    pd.DataFrame({
        "component": np.arange(1, EXPECTED_D + 1),
        "explained_variance_ratio": variance_ratios,
        "cumulative_explained_variance_ratio": np.cumsum(variance_ratios),
    }).to_csv(variance_path, index=False)

    np.save(OUT_DIR / f"{model_name}_pca_first_2_components.npy", basis)
    np.save(OUT_DIR / f"{model_name}_pca_mean_vector.npy", mean)

    print(f"\n{model_name} explained variance:")
    print(f"PC1: {pc1:.6f} ({pc1*100:.2f}%)")
    print(f"PC2: {pc2:.6f} ({pc2*100:.2f}%)")
    print(f"PC1 + PC2: {pc12:.6f} ({pc12*100:.2f}%)")
    print(f"Top 5: {pc5:.6f} ({pc5*100:.2f}%)")
    print(f"Top 10: {pc10:.6f} ({pc10*100:.2f}%)")
    print("Coordinates:", coords_path)

    return {
        "model": model_name,
        "records": EXPECTED_N,
        "dimensions": EXPECTED_D,
        "pc1_variance": pc1,
        "pc2_variance": pc2,
        "pc1_pc2_variance": pc12,
        "top_5_variance": pc5,
        "top_10_variance": pc10,
    }, coords


def draw_scatter(ax, coords, title, pc1, pc2):
    # Every record is plotted; none are sampled or discarded.
    for emotion in EMOTIONS:
        mask = labels == emotion
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=4.0,
            alpha=0.22,
            c=COLORS[emotion],
            linewidths=0,
            rasterized=True,
        )

    ax.set_xlabel(f"PC1 ({pc1 * 100:.2f}% variance)")
    ax.set_ylabel(f"PC2 ({pc2 * 100:.2f}% variance)")
    ax.set_title(f"{title}\nAll {EXPECTED_N:,} records")
    ax.grid(alpha=0.12)
    ax.axhline(0, color="gray", linewidth=0.5, alpha=0.35)
    ax.axvline(0, color="gray", linewidth=0.5, alpha=0.35)


def save_figure(fig, filename):
    png = OUT_DIR / f"{filename}.png"
    pdf = OUT_DIR / f"{filename}.pdf"
    fig.savefig(png, dpi=240, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", png)
    print("Saved:", pdf)


metrics = []
all_coords = {}

for model_name, path in MODEL_FILES.items():
    metric, coords = analyze_model(model_name, path)
    metrics.append(metric)
    all_coords[model_name] = coords

    fig, ax = plt.subplots(figsize=(11, 8))
    draw_scatter(ax, coords, model_name, metric["pc1_variance"], metric["pc2_variance"])
    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="", label=e,
               markerfacecolor=COLORS[e], markeredgecolor="none", markersize=7)
        for e in EMOTIONS
    ]
    ax.legend(handles=legend_handles, bbox_to_anchor=(1.02, 1),
              loc="upper left", frameon=False, title="Emotion")
    fig.suptitle("Full Dataset PCA by Emotion — no sampling", fontsize=14)
    save_figure(fig, f"{model_name}_PCA_by_emotion_ALL")

# Side-by-side comparison. Each model has its OWN PCA axes and scale.
fig, axes = plt.subplots(1, 2, figsize=(17, 7))
for ax, metric in zip(axes, metrics):
    name = metric["model"]
    draw_scatter(
        ax, all_coords[name], name,
        metric["pc1_variance"], metric["pc2_variance"]
    )

handles = [
    Line2D([0], [0], marker="o", linestyle="", label=e,
           markerfacecolor=COLORS[e], markeredgecolor="none", markersize=7)
    for e in EMOTIONS
]
fig.legend(handles=handles, bbox_to_anchor=(0.5, 0.01),
           loc="lower center", ncol=8, frameon=False)
fig.suptitle("PCA — full dataset (41,202 texts per model)", fontsize=15)
fig.tight_layout(rect=(0, 0.09, 1, 0.95))
save_figure(fig, "AraBERT_vs_MARBERT_PCA_by_emotion_ALL")

summary_path = OUT_DIR / "pca_full_dataset_summary.csv"
pd.DataFrame(metrics).to_csv(summary_path, index=False, encoding="utf-8-sig")

with (OUT_DIR / "pca_methodology.txt").open("w", encoding="utf-8") as f:
    f.write(
        "PCA FULL DATASET — METHOD\n"
        "Every record was used in PCA fitting AND projection; no sampling.\n"
        "PCA is fit separately per model on mean-centered 768-D embeddings.\n"
        "Centered 768x768 scatter matrix built in float64 batches; "
        "eigendecomposition yields exact principal components up to floating-point error.\n"
        "Input embeddings and metadata are read only.\n"
        "Scatter plots contain all 41,202 points per model; overlapping "
        "points and transparency can visually hide observations.\n"
        "PCA axes are model-specific; positions and directions are not "
        "directly aligned between the two subplots.\n"
        "PCA is unsupervised and does not establish classification accuracy.\n"
    )

print("\n" + "=" * 70)
print("FULL DATASET PCA COMPLETE — NO SAMPLING")
print("=" * 70)
print(pd.DataFrame(metrics).to_string(index=False))
print("Output folder:", OUT_DIR)
print("Original input files unchanged.")
