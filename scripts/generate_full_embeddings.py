
import os
import json
import tempfile
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel
from arabert.preprocess import ArabertPreprocessor


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = PROJECT_DIR / "Arabic_Emotion_Plutchik8_datasets.xlsx"
SHEET_NAME = "Unified_Data"

OUTPUT_DIR = PROJECT_DIR / "results" / "full_dataset"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# Model inference settings
BATCH_SIZE = 16
MAX_LENGTH = 128
EMBEDDING_DIMENSION = 768

# Farasa optimization
FARASA_CHUNK_SIZE = 256

# Farasa installation
FARASA_JAR = Path(r"C:\Farasa\Farasa.jar")
FARASA_DATA = Path(r"C:\Farasa\FarasaData")
FARASA_LIB = Path(r"C:\Farasa\lib")

# Models
MODELS = {
    "AraBERT_v2": "aubmindlab/bert-base-arabertv2",
    "MARBERT": "UBC-NLP/MARBERT",
}

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# 2. INITIALIZATION
# ============================================================

print("=" * 65)
print("FULL DATASET EMBEDDING EXTRACTION")
print("=" * 65)

print("Device:", DEVICE)
print("Embedding batch size:", BATCH_SIZE)
print("Farasa chunk size:", FARASA_CHUNK_SIZE)
print("Maximum sequence length:", MAX_LENGTH)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found: {INPUT_FILE}"
    )

required_farasa_paths = [
    FARASA_JAR,
    FARASA_DATA,
    FARASA_LIB / "commons-lang3-3.1.jar",
    FARASA_LIB / "mapdb-2.0.0-SNAPSHOT.jar",
]

for path in required_farasa_paths:
    if not path.exists():
        raise FileNotFoundError(
            f"Required Farasa path not found: {path}"
        )

os.environ["FarasaDataDir"] = (
    str(FARASA_DATA.resolve()) + os.sep
)


# ============================================================
# 3. LOAD DATASET
# ============================================================

print("\nLoading full dataset...")

df = pd.read_excel(
    INPUT_FILE,
    sheet_name=SHEET_NAME
)

REQUIRED_COLUMNS = [
    "source_dataset",
    "original_id",
    "text",
    "emotion"
]

for column in REQUIRED_COLUMNS:

    if column not in df.columns:
        raise ValueError(
            f"Missing required column: {column}"
        )

if df[REQUIRED_COLUMNS].isna().any().any():
    raise ValueError(
        "Missing values found in required columns."
    )

texts = df["text"].astype(str).tolist()

if any(not text.strip() for text in texts):
    raise ValueError("Empty text found in dataset.")

N = len(df)

print("Total records:", f"{N:,}")
print("Emotion classes:", df["emotion"].nunique())
print("Source datasets:", df["source_dataset"].nunique())


# ============================================================
# 4. METADATA MANAGEMENT
# ============================================================

metadata = df.copy()

metadata.insert(
    0,
    "row_index",
    np.arange(N, dtype=np.int64)
)

metadata_file = OUTPUT_DIR / "metadata.csv"


def normalize_metadata(frame):

    return (
        frame.copy()
        .fillna("")
        .astype(str)
    )


if metadata_file.exists():

    print("\nChecking existing metadata...")

    saved_metadata = pd.read_csv(
        metadata_file,
        dtype=str,
        keep_default_na=False
    )

    current_metadata = normalize_metadata(
        metadata
    )

    if list(saved_metadata.columns) != list(
        current_metadata.columns
    ):
        raise ValueError(
            "Metadata columns differ. "
            "Existing outputs preserved."
        )

    if len(saved_metadata) != len(current_metadata):

        raise ValueError(
            "Metadata row count differs. "
            "Existing outputs preserved."
        )

    if not saved_metadata.equals(current_metadata):

        different_columns = [
            column
            for column in current_metadata.columns
            if not saved_metadata[column].equals(
                current_metadata[column]
            )
        ]

        raise ValueError(
            f"Metadata differs in columns: "
            f"{different_columns}"
        )

    print(
        "Existing metadata verified successfully."
    )

else:

    metadata.to_csv(
        metadata_file,
        index=False,
        encoding="utf-8-sig"
    )

    print("Metadata saved:", metadata_file)


# ============================================================
# 5. FARASA SEGMENTATION
# ============================================================

def farasa_segment_texts(batch_texts):

    clean_texts = [
        str(text)
        .replace("\r", " ")
        .replace("\n", " ")
        .strip()
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
            str(output_file)
        ]

        try:

            subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8"
            )

        except subprocess.CalledProcessError as error:

            print("Farasa STDOUT:", error.stdout)
            print("Farasa STDERR:", error.stderr)

            raise

        if not output_file.exists():

            raise RuntimeError(
                "Farasa output file was not created."
            )

        segmented = output_file.read_text(
            encoding="utf-8"
        ).splitlines()

    if len(segmented) != len(clean_texts):

        raise ValueError(
            f"Farasa input/output mismatch: "
            f"{len(clean_texts)} vs {len(segmented)}"
        )

    if any(not text.strip() for text in segmented):

        raise ValueError(
            "Farasa produced an empty output line."
        )

    return segmented


# ============================================================
# 6. ARABERT PREPROCESSING
# ============================================================

# Farasa segmentation is performed explicitly above.
# Disable internal Farasa segmentation to avoid applying it twice.


class IdentitySegmenter:
    """
    Farasa has already segmented the texts.
    This prevents AraBERTPreprocessor from
    attempting a second Farasa segmentation.
    """

    def segment(self, text):
        return text


arabert_preprocessor = ArabertPreprocessor(
    model_name="bert-base-arabertv2",
    apply_farasa_segmentation=False
)

# Prevent a second Farasa segmentation.
# Farasa segmentation is performed externally.
arabert_preprocessor.farasa_segmenter = IdentitySegmenter()



# ============================================================
# 7. MEAN POOLING
# ============================================================

def mean_pooling(
    hidden_states,
    attention_mask
):

    mask = attention_mask.unsqueeze(-1).float()

    summed = (
        hidden_states * mask
    ).sum(dim=1)

    counts = mask.sum(dim=1).clamp(min=1e-9)

    return summed / counts


# ============================================================
# 8. CHECKPOINT MANAGEMENT
# ============================================================

def save_checkpoint(path, state):

    temp_path = path.with_suffix(".tmp")

    temp_path.write_text(
        json.dumps(state, indent=4),
        encoding="utf-8"
    )

    temp_path.replace(path)


def prepare_output(
    model_label,
    model_name
):

    embedding_file = OUTPUT_DIR / (
        f"{model_label}_embeddings.npy"
    )

    checkpoint_file = CHECKPOINT_DIR / (
        f"{model_label}_checkpoint.json"
    )

    # Existing checkpoint
    if checkpoint_file.exists():

        state = json.loads(
            checkpoint_file.read_text(
                encoding="utf-8"
            )
        )

        if state.get("records") != N:

            raise ValueError(
                "Checkpoint dataset size mismatch."
            )

        if state.get("model_name") != model_name:

            raise ValueError(
                "Checkpoint model name mismatch."
            )

        if not embedding_file.exists():

            raise FileNotFoundError(
                "Checkpoint exists but embedding file is missing."
            )

        mmap = np.lib.format.open_memmap(
            embedding_file,
            mode="r+"
        )

        if mmap.shape != (
            N,
            EMBEDDING_DIMENSION
        ):

            raise ValueError(
                "Existing embedding matrix has unexpected shape."
            )

        if mmap.dtype != np.float32:

            raise ValueError(
                "Existing embedding matrix has unexpected dtype."
            )

        start_index = int(
            state.get("next_index", 0)
        )

        if not 0 <= start_index <= N:

            raise ValueError(
                "Invalid checkpoint next_index."
            )

        print(
            f"Resuming {model_label} "
            f"from record index: {start_index}"
        )

        return (
            embedding_file,
            checkpoint_file,
            mmap,
            start_index,
            bool(state.get("completed", False))
        )

    # Do not overwrite an orphan embedding file
    if embedding_file.exists():

        raise FileExistsError(
            "Embedding file exists without checkpoint. "
            "Refusing to overwrite."
        )

    # Create new embedding matrix
    mmap = np.lib.format.open_memmap(
        embedding_file,
        mode="w+",
        dtype=np.float32,
        shape=(N, EMBEDDING_DIMENSION)
    )

    save_checkpoint(
        checkpoint_file,
        {
            "model_name": model_name,
            "records": N,
            "next_index": 0,
            "completed": False
        }
    )

    return (
        embedding_file,
        checkpoint_file,
        mmap,
        0,
        False
    )


# ============================================================
# 9. GENERATE EMBEDDINGS
# ============================================================

def generate_embeddings(
    model_label,
    model_name
):

    print("\n" + "=" * 65)
    print("Processing:", model_label)
    print("=" * 65)

    (
        embedding_file,
        checkpoint_file,
        mmap,
        start_index,
        completed
    ) = prepare_output(
        model_label,
        model_name
    )

    if completed:

        mmap.flush()
        del mmap

        print(
            model_label,
            "already completed. Skipping."
        )

        return

    print("Loading model:", model_name)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )

    model = AutoModel.from_pretrained(
        model_name
    ).to(DEVICE)

    model.eval()

    if model.config.hidden_size != EMBEDDING_DIMENSION:

        raise ValueError(
            f"Expected hidden size {EMBEDDING_DIMENSION}, "
            f"got {model.config.hidden_size}"
        )

    # --------------------------------------------------------
    # Process one embedding batch
    # --------------------------------------------------------

    def process_batch(
        batch_start,
        batch_texts
    ):

        batch_end = batch_start + len(batch_texts)

        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        ).to(DEVICE)

        with torch.inference_mode():

            output = model(**encoded)

            vectors = mean_pooling(
                output.last_hidden_state,
                encoded["attention_mask"]
            )

            vectors = F.normalize(
                vectors,
                p=2,
                dim=1
            )

        vectors = vectors.cpu().numpy()

        expected_shape = (
            len(batch_texts),
            EMBEDDING_DIMENSION
        )

        if vectors.shape != expected_shape:

            raise ValueError(
                f"Unexpected embedding shape: {vectors.shape}"
            )

        if not np.isfinite(vectors).all():

            raise ValueError(
                "Non-finite embedding values detected."
            )

        # Save embeddings
        mmap[batch_start:batch_end] = vectors

        mmap.flush()

        # Update checkpoint only after saving embeddings
        save_checkpoint(
            checkpoint_file,
            {
                "model_name": model_name,
                "records": N,
                "next_index": batch_end,
                "completed": False
            }
        )

        progress = 100.0 * batch_end / N

        print(
            f"{model_label}: "
            f"{batch_end:,}/{N:,} "
            f"({progress:.2f}%)"
        )

    # --------------------------------------------------------
    # AraBERT v2: Farasa + preprocessing
    # --------------------------------------------------------

    try:

        if model_label == "AraBERT_v2":

            for chunk_start in range(
                start_index,
                N,
                FARASA_CHUNK_SIZE
            ):

                chunk_end = min(
                    chunk_start + FARASA_CHUNK_SIZE,
                    N
                )

                print(
                    f"\nFarasa segmentation: "
                    f"{chunk_start:,} - {chunk_end:,}"
                )

                segmented = farasa_segment_texts(
                    texts[chunk_start:chunk_end]
                )

                processed = [
                    arabert_preprocessor.preprocess(text)
                    for text in segmented
                ]

                for offset in range(
                    0,
                    len(processed),
                    BATCH_SIZE
                ):

                    batch_start = chunk_start + offset

                    batch = processed[
                        offset:offset + BATCH_SIZE
                    ]

                    process_batch(
                        batch_start,
                        batch
                    )

        # ----------------------------------------------------
        # MARBERT: Original text
        # ----------------------------------------------------

        else:

            for batch_start in range(
                start_index,
                N,
                BATCH_SIZE
            ):

                batch = texts[
                    batch_start:batch_start + BATCH_SIZE
                ]

                process_batch(
                    batch_start,
                    batch
                )

    finally:

        mmap.flush()

        del mmap
        del model
        del tokenizer

        if DEVICE.type == "cuda":

            torch.cuda.empty_cache()

    # ========================================================
    # 10. FINAL VERIFICATION
    # ========================================================

    print("\nVerifying:", model_label)

    final_vectors = np.load(
        embedding_file,
        mmap_mode="r"
    )

    if final_vectors.shape != (
        N,
        EMBEDDING_DIMENSION
    ):

        raise ValueError(
            "Final embedding shape mismatch."
        )

    for block_start in range(0, N, 1000):

        block_end = min(
            block_start + 1000,
            N
        )

        block = final_vectors[
            block_start:block_end
        ]

        if not np.isfinite(block).all():

            raise ValueError(
                f"Invalid values in rows "
                f"{block_start}:{block_end}"
            )

        norms = np.linalg.norm(
            block,
            axis=1
        )

        if not np.allclose(
            norms,
            1.0,
            atol=1e-5
        ):

            raise ValueError(
                f"L2 normalization failed in rows "
                f"{block_start}:{block_end}"
            )

    # Mark model as completed
    save_checkpoint(
        checkpoint_file,
        {
            "model_name": model_name,
            "records": N,
            "next_index": N,
            "completed": True
        }
    )

    print("\nCompleted:", model_label)
    print("Shape:", final_vectors.shape)
    print("All values finite: True")
    print("All vectors L2-normalized: True")
    print("Saved:", embedding_file)


# ============================================================
# 11. RUN BOTH MODELS
# ============================================================

for label, model_name in MODELS.items():

    generate_embeddings(
        label,
        model_name
    )


# ============================================================
# 12. FINAL VERIFICATION
# ============================================================

print("\n" + "=" * 65)
print("FULL DATASET EMBEDDING EXTRACTION FINISHED")
print("=" * 65)

for label in MODELS:

    embedding_file = OUTPUT_DIR / (
        f"{label}_embeddings.npy"
    )

    checkpoint_file = CHECKPOINT_DIR / (
        f"{label}_checkpoint.json"
    )

    if not embedding_file.exists():

        print(label, ": Embedding file missing.")
        continue

    if not checkpoint_file.exists():

        print(label, ": Checkpoint missing.")
        continue

    state = json.loads(
        checkpoint_file.read_text(
            encoding="utf-8"
        )
    )

    vectors = np.load(
        embedding_file,
        mmap_mode="r"
    )

    print("\nModel:", label)
    print("Shape:", vectors.shape)
    print("Completed:", state.get("completed"))
    print("Next index:", state.get("next_index"))

print("\nMetadata:", metadata_file)
print("Output directory:", OUTPUT_DIR)
print("All available outputs checked.")
