"""Generate synthetic data for the Paladin workshop.

Creates a small synthetic dataset with fake tile embeddings, coordinate H5
files, and a parquet DataFrame suitable for classification, regression, and
survival tasks.

Usage:
    python scripts/synthesize_workshop_data.py
    python scripts/synthesize_workshop_data.py --output-dir /custom/path
    python scripts/synthesize_workshop_data.py --n-samples 200 --n-tiles 30
"""

import argparse
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic workshop data for Paladin")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: <project_root>/workshop_data)",
    )
    parser.add_argument("--n-samples", type=int, default=120, help="Number of samples to generate")
    parser.add_argument("--n-tiles", type=int, default=25, help="Number of tiles per sample")
    parser.add_argument("--tile-emb-dim", type=int, default=1536, help="Tile embedding dimension")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # Determine output directory
    if args.output_dir is None:
        project_root = Path(__file__).resolve().parent.parent
        output_dir = project_root / "workshop_data"
    else:
        output_dir = Path(args.output_dir)

    tensors_dir = output_dir / "tensors"
    h5_dir = output_dir / "h5"
    tensors_dir.mkdir(parents=True, exist_ok=True)
    h5_dir.mkdir(parents=True, exist_ok=True)

    n = args.n_samples
    n_tiles = args.n_tiles
    emb_dim = args.tile_emb_dim

    # --- Assign splits: 60% train, 20% val, 20% test ---
    splits = (["train"] * int(0.6 * n) + ["val"] * int(0.2 * n) + ["test"] * (n - int(0.6 * n) - int(0.2 * n)))
    rng.shuffle(splits)

    records = []
    for i in range(n):
        sample_id = f"SAMPLE_{i:04d}"
        patient_id = f"PATIENT_{i:04d}"
        image_id = str(1000 + i)

        # Generate fake tile embeddings and save as .pt
        tile_emb = torch.randn(n_tiles, emb_dim)
        pt_path = tensors_dir / f"{sample_id}.pt"
        torch.save(tile_emb, pt_path)

        # Generate fake coordinates and save as .h5
        coords = rng.integers(0, 10000, size=(n_tiles, 2))
        h5_path = h5_dir / f"{sample_id}.h5"
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("coords", data=coords.astype(np.int64))

        # Generate targets
        biomarker = float(rng.integers(0, 2))  # binary: 0.0 or 1.0
        expression_level = float(rng.standard_normal() * 2 + 5)  # continuous
        time_to_event = float(rng.exponential(scale=24.0))  # months
        event_occurred = float(rng.integers(0, 2))  # censoring indicator

        records.append(
            {
                "sample_id": sample_id,
                "patient_id": patient_id,
                "image_id": image_id,
                "oncotree_code": "SYNTH",
                "site": "Primary",
                "split": splits[i],
                "tile_tensor_path": str(pt_path.resolve()),
                "filtered_tiles_h5_path": str(h5_path.resolve()),
                "biomarker": biomarker,
                "expression_level": expression_level,
                "time_to_event": time_to_event,
                "event_occurred": event_occurred,
            }
        )

    df = pd.DataFrame(records)
    parquet_path = output_dir / "synthetic_data.parquet"
    df.to_parquet(parquet_path, index=False)

    print(f"Synthetic dataset written to {output_dir}")
    print(f"  Parquet : {parquet_path}")
    print(f"  Tensors : {tensors_dir} ({n} files)")
    print(f"  H5 files: {h5_dir} ({n} files)")
    print(f"  Samples : {n} ({dict(pd.Series(splits).value_counts())})")
    print(f"  Tiles/sample: {n_tiles}, embedding dim: {emb_dim}")


if __name__ == "__main__":
    main()
