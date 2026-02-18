"""
Create a minimal `preprocessed_chexpert.csv` for LLaVA-Rad evaluation.

This script reads CheXpert's `train.csv` and `valid.csv`, adds a `split`
column, keeps only `Path` and `split`, renames `Path` -> `path`, and writes:

    /lfs/skampere2/0/mahmedc/CUMC_Radiology/LLaVA-Rad/evaluation_results/preprocessed_chexpert.csv

Run (from anywhere) in your `mahmedc_env`:

    conda run -n mahmedc_env python /lfs/skampere2/0/mahmedc/CUMC_Radiology/LLaVA-Rad/evaluation_results/create_preprocessed_chexpert.py
"""

from pathlib import Path

import pandas as pd


def main() -> None:
    root = Path("/lfs/skampere2/0/mahmedc/CUMC_Radiology")
    chexpert_root = (
        root
        / "data/raw_data/chexpert/chexpertchestxrays-u20210408"
        / "CheXpert-v1.0"
    )

    train_csv = chexpert_root / "train.csv"
    valid_csv = chexpert_root / "valid.csv"

    if not train_csv.exists():
        raise FileNotFoundError(f"train.csv not found at {train_csv}")
    if not valid_csv.exists():
        raise FileNotFoundError(f"valid.csv not found at {valid_csv}")

    print(f"Reading train CSV: {train_csv}")
    train = pd.read_csv(train_csv)
    train["split"] = "train"

    print(f"Reading valid CSV: {valid_csv}")
    valid = pd.read_csv(valid_csv)
    valid["split"] = "valid"

    required_cols = ["Path", "split"]
    for df, name in ((train, "train"), (valid, "valid")):
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"{name}.csv missing required columns: {missing}")

    train = train[required_cols].rename(columns={"Path": "path"})
    valid = valid[required_cols].rename(columns={"Path": "path"})

    splits = pd.concat([train, valid], axis=0).reset_index(drop=True)

    print("Combined splits shape:", splits.shape)
    print(splits.head())
    print(splits["split"].value_counts())

    out_path = root / "LLaVA-Rad/evaluation_results/preprocessed_chexpert.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    splits.to_csv(out_path, index=False)

    print(f"\nWrote preprocessed_chexpert.csv to: {out_path}")


if __name__ == "__main__":
    main()

