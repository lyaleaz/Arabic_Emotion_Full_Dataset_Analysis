"""
Full-dataset PCA visualization by source dataset.
Uses existing full PCA coordinates, no sampling, no PCA refit.
All 41,202 records are included in every model plot.

Output: results/full_dataset/pca_by_source/
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
PCA_DIR = ROOT / "results" / "full_dataset" / "pca_full_dataset"
OUT_DIR = ROOT / "results" / "full_dataset" / "pca_by_source"
META_PATH = ROOT / "results" / "full_dataset" / "metadata.csv"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["AraBERT_v2", "MARBERT"]
EXPECTED_N = 41202

def require(path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return path

print("=" * 70)
print("PCA BY SOURCE DATASET — ALL RECORDS, NO SAMPLING")
print("=" * 70)

meta = pd.read_csv(require(META_PATH), keep_default_na=False)
if len(meta) != EXPECTED_N:
    raise ValueError(f"Expected {EXPECTED_N} metadata rows; found {len(meta)}.")
if "source_dataset" not in meta.columns:
    raise ValueError("Missing source_dataset in metadata.csv")
if "row_index" not in meta.columns:
    raise ValueError("Missing row_index in metadata.csv")
expected_indices = np.arange(EXPECTED_N)
if not np.array_equal(
    pd.to_numeric(meta["row_index"], errors="raise").to_numpy(),
    expected_indices
):
    raise ValueError("Metadata row_index must be 0..41201 in order.")

sources = sorted(meta["source_dataset"].astype(str).unique())
n_sources = len(sources)
if n_sources != 16:
    raise ValueError(f"Expected 16 source dataset labels; found {n_sources}.")
print(f"Metadata: {len(meta):,} rows, {n_sources} source labels")

# Consistent categorical colors between the two independently fitted PCAs.
palette = plt.get_cmap("tab20")
source_colors = {name: palette(i) for i, name in enumerate(sources)}

summary_path = PCA_DIR / "pca_full_dataset_summary.csv"
summary = pd.read_csv(require(summary_path)).set_index("model")

def load_coordinates(model_name):
    coord_path = require(PCA_DIR / f"{model_name}_pca_all_41202.csv")
    coords = pd.read_csv(coord_path, keep_default_na=False)
    for col in ["row_index", "PC1", "PC2", "source_dataset"]:
        if col not in coords.columns:
            raise ValueError(f"{coord_path.name}: missing {col}")
    if len(coords) != EXPECTED_N:
        raise ValueError(f"{coord_path.name}: unexpected row count.")

    idx = pd.to_numeric(coords["row_index"], errors="raise").to_numpy()
    if not np.array_equal(idx, expected_indices):
        raise ValueError(f"{coord_path.name}: row indices out of order.")

    if not np.array_equal(
        coords["source_dataset"].astype(str).to_numpy(),
        meta["source_dataset"].astype(str).to_numpy()
    ):
        raise ValueError(f"{coord_path.name}: source labels differ from metadata.")

    xy = coords[["PC1", "PC2"]].apply(pd.to_numeric, errors="raise").to_numpy()
    if not np.isfinite(xy).all():
        raise ValueError(f"{coord_path.name}: invalid PCA coordinates.")
    return xy

def draw_model(ax, xy, model_name):
    source_array = meta["source_dataset"].astype(str).to_numpy()
    # All points are drawn; smaller sources are drawn last to improve visibility.
    for source in sorted(sources, key=lambda s: -int(np.sum(source_array == s))):
        mask = source_array == source
        ax.scatter(
            xy[mask, 0], xy[mask, 1],
            color=source_colors[source], s=4, alpha=0.24,
            linewidths=0, rasterized=True
        )

    pc1 = float(summary.loc[model_name, "pc1_variance"])
    pc2 = float(summary.loc[model_name, "pc2_variance"])
    ax.set_title(f"{model_name}\nAll {EXPECTED_N:,} texts")
    ax.set_xlabel(f"PC1 ({pc1 * 100:.2f}%)")
    ax.set_ylabel(f"PC2 ({pc2 * 100:.2f}%)")
    ax.grid(alpha=0.12)
    ax.axhline(0, c="gray", lw=0.5, alpha=0.3)
    ax.axvline(0, c="gray", lw=0.5, alpha=0.3)

legend_handles = [
    Line2D([0], [0], linestyle="", marker="o",
           markerfacecolor=source_colors[s], markeredgecolor="none",
           markersize=7, label=s)
    for s in sources
]

def save_figure(fig, name):
    for ext in ("png", "pdf"):
        path = OUT_DIR / f"{name}.{ext}"
        fig.savefig(path, dpi=240, bbox_inches="tight")
        print("Saved:", path)
    plt.close(fig)

all_xy = {}
for model in MODELS:
    print(f"\nLoading existing PCA coordinates for {model}...")
    xy = load_coordinates(model)
    all_xy[model] = xy
    fig, ax = plt.subplots(figsize=(11, 8))
    draw_model(ax, xy, model)
    ax.legend(handles=legend_handles, loc="upper left",
              bbox_to_anchor=(1.01, 1), frameon=False,
              title="Source dataset", fontsize=8)
    fig.suptitle("Full-dataset PCA by source — no sampling", fontsize=14)
    save_figure(fig, f"{model}_PCA_by_source_ALL")
    print(f"All {len(xy):,} records included.")

fig, axes = plt.subplots(1, 2, figsize=(17, 7))
for model, ax in zip(MODELS, axes):
    draw_model(ax, all_xy[model], model)
fig.legend(handles=legend_handles, loc="center left",
           bbox_to_anchor=(0.81, 0.5), fontsize=8,
           title="Source dataset", frameon=False)
fig.suptitle("PCA by source dataset (41,202 texts per model)", fontsize=15)
fig.tight_layout(rect=(0, 0, 0.8, 0.94))
save_figure(fig, "AraBERT_vs_MARBERT_PCA_by_source_ALL")

counts = meta["source_dataset"].value_counts().rename_axis("source_dataset").reset_index(name="records")
counts["percentage"] = 100 * counts["records"] / EXPECTED_N
counts.to_csv(OUT_DIR / "source_dataset_counts.csv", index=False, encoding="utf-8-sig")

color_key = pd.DataFrame({
    "source_dataset": sources,
    "color_hex": [
        matplotlib.colors.to_hex(source_colors[s])
        for s in sources
    ]
})
color_key.to_csv(OUT_DIR / "source_color_legend.csv", index=False, encoding="utf-8-sig")

with (OUT_DIR / "pca_by_source_methodology.txt").open("w", encoding="utf-8") as f:
    f.write(
        "PCA BY SOURCE DATASET — FULL DATASET\n"
        "All 41,202 coordinates per model are plotted without sampling.\n"
        "Coordinates are loaded from earlier PCA files and are NOT recomputed.\n"
        "Source colors are identical for the two models.\n"
        "PCA axes are independently fitted for each model; do not directly\n"
        "compare coordinates or point distances between model panels.\n"
        "Overplotting can obscure points, particularly in large source groups.\n"
        "The visualization alone does not prove whether source or emotion\n"
        "explains more variance; further quantitative analysis is needed.\n"
        "No original input files were modified.\n"
    )

print("\n" + "=" * 70)
print("PCA BY SOURCE DATASET COMPLETED")
print("=" * 70)
print(counts.to_string(index=False))
print("\nOutput folder:", OUT_DIR)
print("All original PCA coordinates and embeddings remain unchanged.")
