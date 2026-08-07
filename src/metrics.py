"""
RSNA Knee Abnormality Detection — Evaluation Metrics
=====================================================
Competition‑aligned metric utilities.  The primary metric is **macro‑averaged
ROC‑AUC** across all 12 abnormality targets.

Design notes
------------
* Each per‑class AUC is computed independently so that a single degenerate
  column (all positives or all negatives in a fold) never kills the entire
  evaluation.  The fallback value is **0.5** (random‑chance baseline).
* All public functions carry strict type annotations and return structured
  results for downstream logging / tracking.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score

# ---------------------------------------------------------------------------
# Module‑level logger
# ---------------------------------------------------------------------------
logger: logging.Logger = logging.getLogger(__name__)


def compute_macro_auc(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_cols: List[str],
) -> Tuple[float, Dict[str, float]]:
    """Compute macro‑averaged ROC‑AUC over the 12 competition targets.

    Parameters
    ----------
    y_true : np.ndarray
        Ground‑truth binary labels of shape ``(N, num_classes)``.
    y_pred : np.ndarray
        Predicted probabilities of shape ``(N, num_classes)``.
    target_cols : List[str]
        Ordered list of target‑column names (length must equal the number of
        columns in *y_true* and *y_pred*).

    Returns
    -------
    macro_auc : float
        The arithmetic mean of all per‑class AUC scores.
    per_class : Dict[str, float]
        Mapping from target name → individual AUC score.

    Raises
    ------
    ValueError
        If the number of columns in *y_true* / *y_pred* does not match the
        length of *target_cols*.

    Examples
    --------
    >>> import numpy as np
    >>> y_true = np.array([[1, 0, 1], [0, 1, 0]])
    >>> y_pred = np.array([[0.9, 0.1, 0.8], [0.2, 0.7, 0.3]])
    >>> macro, per_cls = compute_macro_auc(y_true, y_pred, ["A", "B", "C"])
    >>> 0.0 <= macro <= 1.0
    True
    """
    # ── Input validation ──────────────────────────────────────────────────
    if y_true.shape[1] != len(target_cols):
        raise ValueError(
            f"Column count mismatch: y_true has {y_true.shape[1]} columns but "
            f"{len(target_cols)} target names were provided."
        )
    if y_pred.shape[1] != len(target_cols):
        raise ValueError(
            f"Column count mismatch: y_pred has {y_pred.shape[1]} columns but "
            f"{len(target_cols)} target names were provided."
        )

    per_class: Dict[str, float] = {}

    for idx, col_name in enumerate(target_cols):
        try:
            col_true: np.ndarray = y_true[:, idx]
            col_pred: np.ndarray = y_pred[:, idx]

            # Guard: AUC is undefined when only one class is present
            unique_labels: np.ndarray = np.unique(col_true)
            if len(unique_labels) < 2:
                logger.warning(
                    "Target '%s' has only one class present in y_true "
                    "(unique=%s). Defaulting AUC to 0.5.",
                    col_name,
                    unique_labels.tolist(),
                )
                per_class[col_name] = 0.5
                continue

            auc: float = float(roc_auc_score(col_true, col_pred))
            per_class[col_name] = auc

        except Exception as exc:
            # Catch‑all safety net — log and default to random‑chance AUC
            logger.error(
                "Failed to compute AUC for target '%s': %s. "
                "Defaulting to 0.5.",
                col_name,
                exc,
            )
            per_class[col_name] = 0.5

    macro_auc: float = float(np.mean(list(per_class.values())))

    logger.info("Macro AUC = %.4f | Per‑class: %s", macro_auc, per_class)
    return macro_auc, per_class
