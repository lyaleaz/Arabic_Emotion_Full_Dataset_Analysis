import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# ============================================================
# 1. Configuration
# ============================================================

PROJECT_DIR = os.getcwd()

DATA_DIR = os.path.join(
    PROJECT_DIR,
    "results",
    "full_dataset"
)

OUTPUT_DIR = os.path.join(
    DATA_DIR,
    "tsne_full_dataset"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

METADATA_FILE = os.path.join(DATA_DIR, "metadata.csv")

MODELS = {
    "AraBERT_v2": os.path.join(DATA_DIR, "AraBERT_v2_embeddings.npy"),
    "MARBERT": os.path.join(DATA_DIR, "MARBERT_embeddings.npy")
}

PCA_COMPONENTS = 50
TSNE_COMPONENTS = 2
TSNE_PERPLEXITY = 30
TSNE_RANDOM_STATE = 42
TSNE_INIT = "pca"
TSNE_LEARNING_RATE = "auto"

print("=" * 70)
print("FULL-DATASET t-SNE VISUALIZATION")
print("=" * 70)
print("Project directory:", PROJECT_DIR)
print("Data directory:", DATA_DIR)
print("Output directory:", OUTPUT_DIR)


# ============================================================
# 2. Load Metadata
# ============================================================

if not os.path.exists(METADATA_FILE):
    raise FileNotFoundError(f"Metadata file not found: {METADATA_FILE}")

metadata = pd.read_csv(METADATA_FILE)

required_columns = ["row_index", "source_dataset", "text", "emotion"]

for col in required_columns:
    if col not in metadata.columns:
        raise ValueError(f"Missing required metadata column: {col}")

print("\nMetadata verified:", len(metadata), "records")
print("Source datasets:", metadata["source_dataset"].nunique())
print("Emotion distribution:")
print(metadata["emotion"].value_counts().sort_index())


# ============================================================
# 3. Plot function
# ============================================================

def plot_tsne(df, model_label, output_dir):
    emotions = sorted(df["emotion"].unique())

    plt.figure(figsize=(12, 10))

    for emotion in emotions:
        subset = df[df["emotion"] == emotion]
        plt.scatter(
            subset["tSNE_1"],
            subset["tSNE_2"],
            s=8,
            alpha=0.65,
            label=emotion
        )

    plt.title(f"{model_label} t-SNE by Emotion (All Records)")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(title="Emotion", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()

    png_path = os.path.join(
        output_dir,
        f"{model_label}_tSNE_by_emotion_ALL.png"
    )
    pdf_path = os.path.join(
        output_dir,
        f"{model_label}_tSNE_by_emotion_ALL.pdf"
    )

    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()

    print("Saved:", png_path)
    print("Saved:", pdf_path)


# ============================================================
# 4. Process each model
# ============================================================

summary_rows = []

for model_label, emb_path in MODELS.items():

    print("\n" + "=" * 70)
    print("t-SNE MODEL:", model_label)
    print("=" * 70)

    if not os.path.exists(emb_path):
        raise FileNotFoundError(f"Embedding file not found: {emb_path}")

    embeddings = np.load(emb_path)

    if len(embeddings) != len(metadata):
        raise ValueError(
            f"Mismatch between metadata rows ({len(metadata)}) "
            f"and embeddings ({len(embeddings)}) for {model_label}"
        )

    if not np.isfinite(embeddings).all():
        raise ValueError(f"Embeddings contain non-finite values for {model_label}")

    print("Embedding shape:", embeddings.shape)

    norms = np.linalg.norm(embeddings, axis=1)
    print("All vectors L2-normalized:",
          np.allclose(norms, 1.0, atol=1e-5))

    # --------------------------------------------------------
    # PCA reduction before t-SNE
    # --------------------------------------------------------
    print("\nStep 1/2: PCA reduction to", PCA_COMPONENTS, "dimensions...")
    pca = PCA(
        n_components=PCA_COMPONENTS,
        random_state=TSNE_RANDOM_STATE
    )
    reduced = pca.fit_transform(embeddings)

    explained_50 = pca.explained_variance_ratio_.sum()
    print(f"PCA-{PCA_COMPONENTS} total explained variance: {explained_50:.6f}")

    # --------------------------------------------------------
    # t-SNE
    # --------------------------------------------------------
    print("Step 2/2: Running t-SNE...")
    tsne = TSNE(
        n_components=TSNE_COMPONENTS,
        perplexity=TSNE_PERPLEXITY,
        init=TSNE_INIT,
        learning_rate=TSNE_LEARNING_RATE,
        random_state=TSNE_RANDOM_STATE,
        verbose=1
    )

    tsne_coords = tsne.fit_transform(reduced)

    print("t-SNE completed.")
    print("Output shape:", tsne_coords.shape)

    # --------------------------------------------------------
    # Save coordinates
    # --------------------------------------------------------
    coords_df = metadata.copy()
    coords_df["tSNE_1"] = tsne_coords[:, 0]
    coords_df["tSNE_2"] = tsne_coords[:, 1]

    coords_csv = os.path.join(
        OUTPUT_DIR,
        f"{model_label}_tSNE_all_41202.csv"
    )
    coords_df.to_csv(coords_csv, index=False, encoding="utf-8-sig")
    print("Coordinates:", coords_csv)

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------
    plot_tsne(coords_df, model_label, OUTPUT_DIR)

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------
    summary_rows.append({
        "model": model_label,
        "records": len(metadata),
        "dimensions": embeddings.shape[1],
        "pca_components_before_tsne": PCA_COMPONENTS,
        "pca_explained_variance_sum": explained_50,
        "tsne_components": TSNE_COMPONENTS,
        "perplexity": TSNE_PERPLEXITY
    })


# ============================================================
# 5. Save summary
# ============================================================

summary_df = pd.DataFrame(summary_rows)

summary_csv = os.path.join(
    OUTPUT_DIR,
    "full_dataset_tsne_summary.csv"
)

summary_txt = os.path.join(
    OUTPUT_DIR,
    "full_dataset_tsne_summary.txt"
)

summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

with open(summary_txt, "w", encoding="utf-8") as f:
    f.write("FULL DATASET t-SNE SUMMARY\n")
    f.write("=" * 60 + "\n\n")
    f.write(summary_df.to_string(index=False))
    f.write("\n")

print("\n" + "=" * 70)
print("FULL DATASET t-SNE COMPLETED")
print("=" * 70)
print(summary_df.to_string(index=False))
print("Output folder:", OUTPUT_DIR)
print("Original input files unchanged.")