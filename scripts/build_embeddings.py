"""Generate RoBERTa embeddings + FAISS index from the cleaned data.

This is the notebook (embeddings-1-1.ipynb) turned into a script that runs
locally instead of on Kaggle, reading and writing repo-relative paths.

Two changes from the notebook version:
  - Mileage is already numeric coming out of Clean_Data.py, so the notebook's
    `df['Mileage'].str.replace(' kmpl', '')` is gone (it would raise on a float
    column).
  - Missing feature columns degrade to "unknown" rather than raising KeyError,
    so a CarWale layout change cannot break embedding generation outright.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IN = ROOT / "Data cleaning" / "Cleaned_data.csv"
DEFAULT_OUT = ROOT / "Data cleaning" / "Cleaned_data_with_embeddings.csv"
DEFAULT_INDEX = ROOT / "Data cleaning" / "faiss_cosine_index.index"

# Fields fed to the encoder. Price and Mileage are excluded on purpose -- they
# are filtered numerically before the semantic search, not embedded.
TEXT_FIELDS = [
    "Brand", "Car", "Variant", "Fuel Type", "Transmission", "Engine", "Display",
    "Sunroof / Moonroof", "Dashcam", "Rear AC", "Central Locking",
    "Cruise Control", "Hill Hold Control", "Ventilated Seats",
    "Wireless Charger", "Instrument Cluster", "Adjustable ORVMs",
    "Integrated (in-dash) Music System", "Speakers",
]


def textual_representation(row):
    return " | ".join(f"{f}: {row.get(f, 'unknown')}" for f in TEXT_FIELDS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", default=str(DEFAULT_IN))
    ap.add_argument("--out", dest="dst", default=str(DEFAULT_OUT))
    ap.add_argument("--index", default=str(DEFAULT_INDEX))
    ap.add_argument("--model", default="all-roberta-large-v1")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    import torch
    from sentence_transformers import SentenceTransformer
    import faiss

    src = Path(args.src)
    if not src.exists():
        raise SystemExit(f"Not found: {src}\nRun Clean_Data.py first.")

    df = pd.read_csv(src)
    missing = [f for f in TEXT_FIELDS if f not in df.columns]
    if missing:
        print(f"WARNING: missing columns, embedded as 'unknown': {missing}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Encoding {len(df)} rows with {args.model} on {device}")
    model = SentenceTransformer(args.model, device=device)

    df["text_combined"] = df.apply(textual_representation, axis=1)
    embeddings = model.encode(df["text_combined"].tolist(),
                              convert_to_numpy=True,
                              batch_size=args.batch_size,
                              show_progress_bar=True)
    print(f"Embeddings shape: {embeddings.shape}")

    df["embeddings"] = embeddings.tolist()
    Path(args.dst).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.dst, index=False)
    print(f"Wrote {args.dst}")

    # Normalise so inner product equals cosine similarity.
    normalised = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    index = faiss.IndexFlatIP(normalised.shape[1])
    index.add(normalised)
    faiss.write_index(index, args.index)
    print(f"FAISS index: {index.ntotal} vectors -> {args.index}")


if __name__ == "__main__":
    main()
