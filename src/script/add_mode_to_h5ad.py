#!/usr/bin/env python3
"""Add an obs['mode'] train/test split to h5ad files by drug labels."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import scanpy as sc


def find_h5ad_files(data_dir: Path, recursive: bool) -> list[Path]:
    pattern = "**/*.h5ad" if recursive else "*.h5ad"
    return sorted(path for path in data_dir.glob(pattern) if path.is_file())


def choose_train_test_drugs(
    drug_values: pd.Series,
    control_value: str,
    train_frac: float,
    seed: int,
) -> tuple[set[str], set[str]]:
    non_control_drugs = np.array(
        sorted(drug_values.loc[drug_values != control_value].dropna().unique())
    )
    if len(non_control_drugs) == 0:
        return set(), set()

    rng = np.random.default_rng(seed)
    shuffled = non_control_drugs.copy()
    rng.shuffle(shuffled)

    n_train = int(round(len(shuffled) * train_frac))
    if len(shuffled) == 1:
        n_train = 1
    else:
        n_train = min(max(n_train, 1), len(shuffled) - 1)

    train_drugs = set(shuffled[:n_train])
    test_drugs = set(shuffled[n_train:])
    return train_drugs, test_drugs


def add_mode_column(
    h5ad_path: Path,
    drug_col: str,
    mode_col: str,
    control_value: str,
    train_frac: float,
    seed: int,
    inplace: bool,
    output_dir: Path | None,
    data_dir: Path,
    compression: str | None,
    compression_opts: int | None,
) -> dict[str, object]:
    adata = sc.read_h5ad(h5ad_path)
    if drug_col not in adata.obs:
        raise KeyError(f"{h5ad_path}: obs does not contain '{drug_col}'")

    drug_values = adata.obs[drug_col].astype(str)
    train_drugs, test_drugs = choose_train_test_drugs(
        drug_values=drug_values,
        control_value=control_value,
        train_frac=train_frac,
        seed=seed,
    )

    mode = pd.Series("train", index=adata.obs.index, dtype="object")
    test_mask = (drug_values != control_value) & drug_values.isin(test_drugs)
    mode.loc[test_mask] = "test"
    adata.obs[mode_col] = pd.Categorical(mode, categories=["train", "test"])

    if inplace:
        out_path = h5ad_path
    elif output_dir is not None:
        rel_path = h5ad_path.relative_to(data_dir)
        out_path = output_dir / rel_path
    else:
        raise ValueError("Use --inplace or --output-dir when not running --dry-run")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(out_path, compression=compression, compression_opts=compression_opts)

    mode_counts = adata.obs[mode_col].value_counts().to_dict()
    return {
        "path": str(h5ad_path),
        "output": str(out_path),
        "n_obs": int(adata.n_obs),
        "control_obs": int((drug_values == control_value).sum()),
        "non_control_unique": len(train_drugs) + len(test_drugs),
        "train_drug_unique": len(train_drugs),
        "test_drug_unique": len(test_drugs),
        "train_obs": int(mode_counts.get("train", 0)),
        "test_obs": int(mode_counts.get("test", 0)),
    }


def preview_split(
    h5ad_path: Path,
    drug_col: str,
    control_value: str,
    train_frac: float,
    seed: int,
) -> dict[str, object]:
    adata = sc.read_h5ad(h5ad_path, backed="r")
    if drug_col not in adata.obs:
        raise KeyError(f"{h5ad_path}: obs does not contain '{drug_col}'")

    drug_values = adata.obs[drug_col].astype(str)
    train_drugs, test_drugs = choose_train_test_drugs(
        drug_values=drug_values,
        control_value=control_value,
        train_frac=train_frac,
        seed=seed,
    )
    test_mask = (drug_values != control_value) & drug_values.isin(test_drugs)
    train_mask = ~test_mask

    return {
        "path": str(h5ad_path),
        "n_obs": int(adata.n_obs),
        "control_obs": int((drug_values == control_value).sum()),
        "non_control_unique": len(train_drugs) + len(test_drugs),
        "train_drug_unique": len(train_drugs),
        "test_drug_unique": len(test_drugs),
        "train_obs": int(train_mask.sum()),
        "test_obs": int(test_mask.sum()),
    }


def print_summary(rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        print("No h5ad files found.")
        return

    table = pd.DataFrame(rows)
    print(table.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Add obs['mode'] to h5ad files. Rows with drug == DMSO_TF are "
            "always train; other drug labels are split by unique drug value."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing h5ad files. Defaults to ./data.",
    )
    parser.add_argument(
        "--drug-col",
        default="drug",
        help="obs column used as the drug label. Defaults to drug.",
    )
    parser.add_argument(
        "--mode-col",
        default="mode",
        help="obs column to create or overwrite. Defaults to mode.",
    )
    parser.add_argument(
        "--control-value",
        default="DMSO_TF",
        help="Drug value that is always assigned train. Defaults to DMSO_TF.",
    )
    parser.add_argument(
        "--train-frac",
        type=float,
        default=0.8,
        help="Fraction of non-control unique drug values assigned train.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the drug-level split.",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Only process h5ad files directly under data-dir.",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Overwrite each input h5ad file in place.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Write updated files under this directory, preserving relative paths. "
            "Ignored when --inplace is set."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print the planned split; do not write files.",
    )
    parser.add_argument(
        "--compression",
        choices=["gzip", "lzf", "none"],
        default="gzip",
        help="Compression used when writing h5ad files. Defaults to gzip.",
    )
    parser.add_argument(
        "--compression-level",
        type=int,
        default=4,
        help="gzip compression level, 0-9. Ignored unless --compression=gzip.",
    )
    args = parser.parse_args()

    if not 0 <= args.train_frac <= 1:
        parser.error("--train-frac must be between 0 and 1")
    if not 0 <= args.compression_level <= 9:
        parser.error("--compression-level must be between 0 and 9")
    if args.inplace and args.output_dir is not None:
        parser.error("--inplace and --output-dir cannot be used together")
    if not args.dry_run and not args.inplace and args.output_dir is None:
        parser.error("use --inplace, --output-dir, or --dry-run")
    return args


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"{data_dir} does not exist")
    files = find_h5ad_files(data_dir, recursive=not args.no_recursive)
    compression = None if args.compression == "none" else args.compression
    compression_opts = args.compression_level if compression == "gzip" else None

    rows = []
    for h5ad_path in files:
        if args.dry_run:
            row = preview_split(
                h5ad_path=h5ad_path,
                drug_col=args.drug_col,
                control_value=args.control_value,
                train_frac=args.train_frac,
                seed=args.seed,
            )
        else:
            row = add_mode_column(
                h5ad_path=h5ad_path,
                drug_col=args.drug_col,
                mode_col=args.mode_col,
                control_value=args.control_value,
                train_frac=args.train_frac,
                seed=args.seed,
                inplace=args.inplace,
                output_dir=args.output_dir.resolve() if args.output_dir else None,
                data_dir=data_dir,
                compression=compression,
                compression_opts=compression_opts,
            )
        rows.append(row)

    print_summary(rows)


if __name__ == "__main__":
    main()
