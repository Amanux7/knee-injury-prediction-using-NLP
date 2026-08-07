"""
RSNA Knee Abnormality Detection — Zero‑Shot Baseline Submission
===============================================================
Generates a competition‑compliant ``submission.csv`` with every target column
set to **0.5** (random‑chance prior).

This script is intentionally self‑contained and offline‑compatible so it can
run inside a Kaggle notebook kernel with no GPU, no model weights, and no
internet access.

Usage
-----
.. code-block:: bash

    python submission_zero.py                     # uses test.csv if present
    python submission_zero.py --test_csv my.csv   # explicit path
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
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

DEFAULT_CONFIDENCE: float = 0.5
OUTPUT_FILENAME: str = "submission.csv"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger: logging.Logger = logging.getLogger(__name__)


def load_or_create_test_df(test_csv_path: str) -> pd.DataFrame:
    """Load ``test.csv`` or fabricate a dummy frame for local testing.

    Parameters
    ----------
    test_csv_path : str
        Path to the competition ``test.csv``.

    Returns
    -------
    pd.DataFrame
        DataFrame with at least a ``"StudyInstanceUID"`` column.
    """
    if os.path.isfile(test_csv_path):
        logger.info("Loading test metadata from '%s'.", test_csv_path)
        df: pd.DataFrame = pd.read_csv(test_csv_path)
        if "StudyInstanceUID" not in df.columns:
            raise ValueError(
                f"'{test_csv_path}' does not contain a 'StudyInstanceUID' column."
            )
        return df

    # ── Fallback: generate dummy data ─────────────────────────────────────
    logger.warning(
        "'%s' not found — generating a dummy test DataFrame with 5 samples "
        "for local development.",
        test_csv_path,
    )
    dummy_uids: List[str] = [f"1.2.826.0.1.{i}" for i in range(5)]
    return pd.DataFrame({"StudyInstanceUID": dummy_uids})


def build_submission(df: pd.DataFrame) -> pd.DataFrame:
    """Construct a submission frame with default confidence scores.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``"StudyInstanceUID"``.

    Returns
    -------
    pd.DataFrame
        Submission‑ready frame with UID + 12 target columns.
    """
    submission: pd.DataFrame = pd.DataFrame(
        {"StudyInstanceUID": df["StudyInstanceUID"]}
    )
    for col in TARGET_COLUMNS:
        submission[col] = DEFAULT_CONFIDENCE
    return submission


def main() -> None:
    """Entry point — parse arguments, build submission, and save to disk."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Generate a baseline 0.5‑confidence submission."
    )
    parser.add_argument(
        "--test_csv",
        type=str,
        default="test.csv",
        help="Path to test.csv (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=OUTPUT_FILENAME,
        help="Output CSV filename (default: %(default)s).",
    )
    args: argparse.Namespace = parser.parse_args()

    # ── 1. Load / create test frame ───────────────────────────────────────
    test_df: pd.DataFrame = load_or_create_test_df(args.test_csv)
    logger.info("Test samples: %d", len(test_df))

    # ── 2. Build submission ───────────────────────────────────────────────
    submission: pd.DataFrame = build_submission(test_df)

    # ── 3. Save ───────────────────────────────────────────────────────────
    output_path: Path = Path(args.output)
    submission.to_csv(output_path, index=False)
    logger.info("Submission saved to '%s'.", output_path.resolve())

    # ── 4. Verification logging ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUBMISSION VERIFICATION")
    print("=" * 60)
    print(f"  Shape   : {submission.shape}")
    print(f"  Columns : {list(submission.columns)}")
    print(f"\n  Head (3 rows):\n{submission.head(3).to_string(index=False)}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
