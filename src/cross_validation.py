"""
RSNA Knee Abnormality Detection — Cross‑Validation Splitter
============================================================
Produces leak‑free, multilabel‑stratified K‑fold splits for multi‑label
knee‑abnormality classification.

Why this is non‑trivial
-----------------------
* ``MultilabelStratifiedKFold`` from *iterative‑stratification* does **not**
  accept a ``groups`` argument, so naively calling it may place slices from the
  same study (or the same patient) into both train and validation sets.
* Our solution: **deduplicate to one row per group** (``StudyInstanceUID`` or
  ``PatientID``), run the stratified split on that reduced frame, then
  broadcast the fold assignment back to every original row that shares the
  same group key.  This guarantees zero leakage while preserving label balance.

Usage
-----
.. code-block:: bash

    python -m src.cross_validation                        # defaults
    python -m src.cross_validation --csv data/train.csv --n_folds 5 --seed 42
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

# ---------------------------------------------------------------------------
# Module‑level constants & logger
# ---------------------------------------------------------------------------
logger: logging.Logger = logging.getLogger(__name__)

TARGET_COLUMNS: List[str] = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]


# ═══════════════════════════════════════════════════════════════════════════
# Core API
# ═══════════════════════════════════════════════════════════════════════════

def _resolve_group_column(df: pd.DataFrame) -> str:
    """Pick the best available group key to prevent data leakage.

    Prefers ``PatientID`` (coarser grouping — a single patient may have
    multiple studies) and falls back to ``StudyInstanceUID``.

    Parameters
    ----------
    df : pd.DataFrame
        Training metadata.

    Returns
    -------
    str
        Column name chosen for grouping.

    Raises
    ------
    ValueError
        If neither candidate column exists in *df*.
    """
    for candidate in ("PatientID", "StudyInstanceUID"):
        if candidate in df.columns:
            logger.info("Using '%s' as the group key for leak prevention.", candidate)
            return candidate
    raise ValueError(
        "DataFrame must contain 'PatientID' or 'StudyInstanceUID' to "
        "define group boundaries."
    )


def create_multilabel_stratified_folds(
    df: pd.DataFrame,
    target_columns: List[str] = TARGET_COLUMNS,
    n_folds: int = 5,
    seed: int = 42,
    group_column: Optional[str] = None,
) -> pd.DataFrame:
    """Assign a ``fold`` column (0 … *n_folds*‑1) to every row of *df*.

    The split is **group‑aware** (no study / patient leaks) *and*
    **multilabel‑stratified** (label proportions are as balanced as possible
    across folds).

    Parameters
    ----------
    df : pd.DataFrame
        Training metadata.  Must contain *group_column* and all
        *target_columns*.
    target_columns : List[str]
        The 12 competition target columns.
    n_folds : int
        Number of folds (default ``5``).
    seed : int
        Random seed for reproducibility.
    group_column : Optional[str]
        Column used for grouping.  Auto‑detected if *None*.

    Returns
    -------
    pd.DataFrame
        A copy of *df* with an additional integer ``fold`` column.
    """
    df = df.copy()

    # ── 1. Resolve group column ───────────────────────────────────────────
    if group_column is None:
        group_column = _resolve_group_column(df)

    # ── 2. Validate targets ───────────────────────────────────────────────
    missing_cols: List[str] = [c for c in target_columns if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"The following target columns are missing from the DataFrame: "
            f"{missing_cols}"
        )

    # ── 3. Deduplicate to one row per group ───────────────────────────────
    # For groups with multiple rows we aggregate labels via max (if *any*
    # slice is positive the group is positive).
    group_df: pd.DataFrame = (
        df.groupby(group_column, as_index=False)[target_columns]
        .max()
        .reset_index(drop=True)
    )

    logger.info(
        "Deduplication: %d rows → %d unique groups ('%s').",
        len(df),
        len(group_df),
        group_column,
    )

    # ── 4. Stratified split on the group‑level frame ─────────────────────
    X_dummy: np.ndarray = np.zeros((len(group_df), 1))  # placeholder features
    y_group: np.ndarray = group_df[target_columns].values.astype(np.int8)

    mskf = MultilabelStratifiedKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=seed,
    )

    group_df["fold"] = -1
    for fold_idx, (_train_idx, val_idx) in enumerate(mskf.split(X_dummy, y_group)):
        group_df.loc[val_idx, "fold"] = fold_idx

    assert (group_df["fold"] >= 0).all(), "Some groups were not assigned a fold."

    # ── 5. Map fold assignments back to the original (full) frame ─────────
    fold_map: Dict[str, int] = dict(
        zip(group_df[group_column], group_df["fold"])
    )
    df["fold"] = df[group_column].map(fold_map).astype(int)

    logger.info(
        "Fold assignment complete: %s",
        df["fold"].value_counts().sort_index().to_dict(),
    )

    return df


# ═══════════════════════════════════════════════════════════════════════════
# Reporting helpers
# ═══════════════════════════════════════════════════════════════════════════

def print_fold_distribution(
    df: pd.DataFrame,
    target_columns: List[str] = TARGET_COLUMNS,
) -> None:
    """Print a human‑readable summary of class distribution per fold.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a ``fold`` column and all *target_columns*.
    """
    n_folds: int = df["fold"].nunique()

    print("\n" + "=" * 72)
    print("  FOLD CLASS DISTRIBUTION SUMMARY")
    print("=" * 72)

    for fold_idx in range(n_folds):
        fold_mask: pd.Series = df["fold"] == fold_idx
        fold_size: int = int(fold_mask.sum())
        total: int = len(df)

        print(f"\n  Fold {fold_idx}  |  {fold_size:>6,} samples  "
              f"({fold_size / total * 100:.1f}%)")
        print("  " + "-" * 60)

        for col in target_columns:
            pos: int = int(df.loc[fold_mask, col].sum())
            neg: int = fold_size - pos
            ratio: float = pos / fold_size * 100 if fold_size > 0 else 0.0
            print(f"    {col:<22s}  pos={pos:>5,}  neg={neg:>5,}  "
                  f"({ratio:5.1f}% pos)")

    print("\n" + "=" * 72 + "\n")


# ═══════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Read ``train.csv``, create folds, print stats, and save."""
    # ── Logging setup ─────────────────────────────────────────────────────
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )

    # ── Argument parsing ──────────────────────────────────────────────────
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description=(
            "Generate multilabel‑stratified, group‑aware K‑fold splits "
            "for RSNA Knee Abnormality Detection."
        ),
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="train.csv",
        help="Path to training CSV (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="train_folds.csv",
        help="Output CSV path with fold column (default: %(default)s).",
    )
    parser.add_argument(
        "--n_folds",
        type=int,
        default=5,
        help="Number of folds (default: %(default)s).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: %(default)s).",
    )
    args: argparse.Namespace = parser.parse_args()

    # ── Load data ─────────────────────────────────────────────────────────
    csv_path: Path = Path(args.csv)
    if not csv_path.is_file():
        logger.warning(
            "'%s' not found — creating a synthetic dummy DataFrame for "
            "demonstration.",
            csv_path,
        )
        rng: np.random.Generator = np.random.default_rng(args.seed)
        n_samples: int = 200
        train_df: pd.DataFrame = pd.DataFrame({
            "StudyInstanceUID": [f"1.2.826.0.1.{i}" for i in range(n_samples)],
            "PatientID": [f"PAT_{i // 2}" for i in range(n_samples)],
            **{col: rng.integers(0, 2, size=n_samples) for col in TARGET_COLUMNS},
        })
    else:
        logger.info("Loading training metadata from '%s'.", csv_path)
        train_df = pd.read_csv(csv_path)

    logger.info("Training samples: %d", len(train_df))

    # ── Create folds ──────────────────────────────────────────────────────
    train_df = create_multilabel_stratified_folds(
        train_df,
        target_columns=TARGET_COLUMNS,
        n_folds=args.n_folds,
        seed=args.seed,
    )

    # ── Report & save ─────────────────────────────────────────────────────
    print_fold_distribution(train_df, TARGET_COLUMNS)

    output_path: Path = Path(args.output)
    train_df.to_csv(output_path, index=False)
    logger.info("Saved fold assignments to '%s'.", output_path.resolve())

    print(f"  Output shape : {train_df.shape}")
    print(f"  Columns      : {list(train_df.columns)}")
    print(f"\n  Head (5 rows):\n{train_df.head(5).to_string(index=False)}\n")


if __name__ == "__main__":
    main()
