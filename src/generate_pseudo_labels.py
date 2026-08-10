"""
RSNA Knee Abnormality Detection -- Pseudo-Label Generator
==========================================================
Applies :class:`ReportLabeler` (regex backend, CPU-safe) to rows in a
competition CSV that have missing target labels, filling them with soft
pseudo-probabilities derived from the free-text radiology report column.

Usage
-----
    python src/generate_pseudo_labels.py \
        --input  data/train_unlabeled.csv \
        --output data/train_pseudo_labeled.csv

    # Use the LLM backend (requires transformers + GPU):
    python src/generate_pseudo_labels.py \
        --input  data/train_unlabeled.csv \
        --output data/train_pseudo_labeled.csv \
        --method llm

The script is deliberately conservative:
* Rows that *already* have non-NaN values for all 12 targets are left
  untouched -- we never overwrite human annotations.
* Rows where the report text is itself missing receive the baseline
  absent-confidence value (0.05) for every target.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import List

import pandas as pd

# ---------------------------------------------------------------------------
# Ensure the project root is importable regardless of where the script is
# invoked from (e.g.  `python src/generate_pseudo_labels.py`).
# ---------------------------------------------------------------------------
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.report_labeler import ReportLabeler, TARGET_COLUMNS  # noqa: E402

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# Candidate column names for the free-text report (checked in order).
_REPORT_COL_CANDIDATES: List[str] = [
    "report_text",
    "radiology_report",
    "report",
    "clinical_report",
    "text",
    "Report",
    "ReportText",
]


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def _detect_report_column(df: pd.DataFrame, override: str | None) -> str:
    """Resolve the report text column name.

    Parameters
    ----------
    df : pd.DataFrame
        The loaded dataframe.
    override : str or None
        If the user passed ``--report-col``, use that directly.

    Returns
    -------
    str
        The column name to use for report text.

    Raises
    ------
    SystemExit
        If no suitable column can be found.
    """
    if override:
        if override not in df.columns:
            logger.error(
                "Specified report column '%s' not found in CSV.  "
                "Available columns: %s",
                override,
                list(df.columns),
            )
            sys.exit(1)
        return override

    for candidate in _REPORT_COL_CANDIDATES:
        if candidate in df.columns:
            logger.info("Auto-detected report column: '%s'", candidate)
            return candidate

    logger.error(
        "Could not auto-detect a report text column.  "
        "Tried: %s.  Available columns: %s.  "
        "Use --report-col to specify it manually.",
        _REPORT_COL_CANDIDATES,
        list(df.columns),
    )
    sys.exit(1)


def _needs_pseudo_labels(row: pd.Series) -> bool:
    """Return True if *any* of the 12 target columns is NaN in this row."""
    for col in TARGET_COLUMNS:
        if col in row.index and pd.isna(row[col]):
            return True
    return False


def generate_pseudo_labels(
    input_path: str,
    output_path: str,
    method: str = "regex",
    report_col_override: str | None = None,
    model_name: str = "facebook/bart-large-mnli",
    device: int = -1,
) -> pd.DataFrame:
    """Load a CSV, pseudo-label missing rows, and save the result.

    Parameters
    ----------
    input_path : str
        Path to the input CSV file.
    output_path : str
        Path where the fully labeled CSV will be written.
    method : {"regex", "llm"}
        Labeler backend.
    report_col_override : str or None
        Explicit name of the report text column.
    model_name : str
        HuggingFace model (only for ``method="llm"``).
    device : int
        PyTorch device ordinal (only for ``method="llm"``).

    Returns
    -------
    pd.DataFrame
        The pseudo-labeled dataframe (also saved to ``output_path``).
    """
    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    logger.info("Loading input CSV: %s", input_path)
    df = pd.read_csv(input_path)
    logger.info("Loaded %d rows x %d columns", len(df), len(df.columns))

    # Ensure target columns exist (they may be entirely absent in
    # unlabeled CSVs).
    for col in TARGET_COLUMNS:
        if col not in df.columns:
            df[col] = float("nan")

    # ------------------------------------------------------------------
    # 2. Detect report column
    # ------------------------------------------------------------------
    report_col = _detect_report_column(df, report_col_override)

    # ------------------------------------------------------------------
    # 3. Identify rows that need pseudo-labels
    # ------------------------------------------------------------------
    mask = df.apply(_needs_pseudo_labels, axis=1)
    n_missing = mask.sum()
    n_existing = len(df) - n_missing
    logger.info(
        "Label status:  %d already labeled  |  %d need pseudo-labels",
        n_existing,
        n_missing,
    )

    if n_missing == 0:
        logger.info("Nothing to pseudo-label.  Saving unchanged.")
        df.to_csv(output_path, index=False)
        return df

    # ------------------------------------------------------------------
    # 4. Initialise labeler
    # ------------------------------------------------------------------
    labeler = ReportLabeler(
        method=method, model_name=model_name, device=device,
    )
    logger.info("Labeler ready  [backend=%s]", method)

    # ------------------------------------------------------------------
    # 5. Pseudo-label with progress bar
    # ------------------------------------------------------------------
    try:
        from tqdm import tqdm
        iterator = tqdm(
            df.loc[mask].iterrows(),
            total=n_missing,
            desc="Pseudo-labeling",
            unit="report",
            ncols=88,
            bar_format=(
                "{l_bar}{bar}| {n_fmt}/{total_fmt} "
                "[{elapsed}<{remaining}, {rate_fmt}]"
            ),
        )
    except ImportError:
        logger.warning(
            "tqdm not installed -- falling back to plain progress logging.  "
            "Install with: pip install tqdm"
        )
        iterator = df.loc[mask].iterrows()

    labeled_count = 0
    t0 = time.perf_counter()

    for idx, row in iterator:
        report_text = row.get(report_col, "")
        if pd.isna(report_text):
            report_text = ""

        scores = labeler(str(report_text))

        for col in TARGET_COLUMNS:
            if pd.isna(df.at[idx, col]):
                df.at[idx, col] = scores[col]

        labeled_count += 1

        # Fallback progress for non-tqdm environments
        if not _has_tqdm() and labeled_count % 500 == 0:
            elapsed = time.perf_counter() - t0
            rate = labeled_count / elapsed if elapsed > 0 else 0
            logger.info(
                "  Progress: %d / %d  (%.1f reports/s)",
                labeled_count,
                n_missing,
                rate,
            )

    elapsed = time.perf_counter() - t0
    rate = labeled_count / elapsed if elapsed > 0 else 0
    logger.info(
        "Pseudo-labeling complete: %d reports in %.1fs (%.1f reports/s)",
        labeled_count,
        elapsed,
        rate,
    )

    # ------------------------------------------------------------------
    # 6. Save output
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Saved pseudo-labeled CSV: %s  (%d rows)", output_path, len(df))

    # ------------------------------------------------------------------
    # 7. Print summary statistics
    # ------------------------------------------------------------------
    _print_summary(df, n_existing, labeled_count)

    return df


def _has_tqdm() -> bool:
    """Check whether tqdm is available."""
    try:
        import tqdm  # noqa: F401
        return True
    except ImportError:
        return False


def _print_summary(
    df: pd.DataFrame,
    n_existing: int,
    n_pseudo: int,
) -> None:
    """Print a clean summary table of pseudo-label statistics."""
    print("\n" + "=" * 72)
    print("  PSEUDO-LABELING SUMMARY")
    print("=" * 72)
    print(f"  Total rows          : {len(df):>8d}")
    print(f"  Already labeled     : {n_existing:>8d}")
    print(f"  Pseudo-labeled      : {n_pseudo:>8d}")
    print("-" * 72)
    print(f"  {'Target':<22s}  {'Mean':>8s}  {'Min':>8s}  {'Max':>8s}  {'Std':>8s}")
    print(f"  {'-' * 20}  {'-' * 8}  {'-' * 8}  {'-' * 8}  {'-' * 8}")

    for col in TARGET_COLUMNS:
        vals = df[col].astype(float)
        print(
            f"  {col:<22s}  {vals.mean():>8.4f}  {vals.min():>8.4f}  "
            f"{vals.max():>8.4f}  {vals.std():>8.4f}"
        )

    print("=" * 72 + "\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate pseudo-labels for unlabeled rows in a competition CSV "
            "using the ReportLabeler NLP pipeline."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python src/generate_pseudo_labels.py "
            "--input data/train.csv --output data/train_pseudo.csv\n"
            "  python src/generate_pseudo_labels.py "
            "--input data/train.csv --output data/train_pseudo.csv "
            "--method llm --device 0\n"
        ),
    )

    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to the input CSV containing a report text column.",
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Path where the pseudo-labeled CSV will be saved.",
    )
    parser.add_argument(
        "--method", "-m",
        choices=["regex", "llm"],
        default="regex",
        help="Labeling backend (default: regex -- fast, CPU-only, offline).",
    )
    parser.add_argument(
        "--report-col",
        default=None,
        help=(
            "Name of the column containing the free-text report.  "
            "Auto-detected if not specified (tries: report_text, "
            "radiology_report, report, text)."
        ),
    )
    parser.add_argument(
        "--model-name",
        default="facebook/bart-large-mnli",
        help="HuggingFace model for the LLM backend (default: bart-large-mnli).",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=-1,
        help="PyTorch device ordinal for LLM backend (-1=CPU, 0=GPU0).",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    generate_pseudo_labels(
        input_path=args.input,
        output_path=args.output,
        method=args.method,
        report_col_override=args.report_col,
        model_name=args.model_name,
        device=args.device,
    )


if __name__ == "__main__":
    main()
