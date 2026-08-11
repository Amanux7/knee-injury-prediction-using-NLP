"""
RSNA Knee Abnormality Detection -- Inference & Submission Generator
====================================================================
Loads a trained RSNADINOv2Model checkpoint, runs efficient GPU inference
on the test set, and writes a competition-ready ``submission.csv``.

Usage
-----
.. code-block:: bash

    # Standard inference on Kaggle GPU
    python src/inference.py \
        --weights checkpoints/best_model_fold_0.pth \
        --data-dir data/test_images \
        --csv data/test.csv \
        --output submission.csv

    # Override backbone / batch-size / num-workers
    python src/inference.py \
        --weights checkpoints/best_model_fold_0.pth \
        --data-dir data/test_images \
        --csv data/test.csv \
        --backbone dinov2_vitb14 \
        --batch-size 4 \
        --num-workers 4

    # Multi-fold TTA ensemble (average predictions from N checkpoints)
    python src/inference.py \
        --weights ckpt/fold0.pth ckpt/fold1.pth ckpt/fold2.pth \
        --data-dir data/test_images \
        --csv data/test.csv
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Ensure the project root is importable
# ---------------------------------------------------------------------------
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.dataset import RSNAKneeDataset, DEFAULT_TARGET_COLUMNS  # noqa: E402
from src.models import RSNADINOv2Model                           # noqa: E402

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

TARGET_COLUMNS: List[str] = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture",
]


# =========================================================================
# 1.  Device Selection
# =========================================================================

def _get_device() -> torch.device:
    """Auto-detect the best available accelerator."""
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        logger.info("Using CUDA: %s", torch.cuda.get_device_name(0))
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        dev = torch.device("mps")
        logger.info("Using Apple MPS backend.")
    else:
        dev = torch.device("cpu")
        logger.info("No GPU detected -- falling back to CPU.")
    return dev


# =========================================================================
# 2.  Model Loading
# =========================================================================

def load_model_from_checkpoint(
    checkpoint_path: str,
    backbone: str = "dinov2_vits14",
    num_classes: int = 12,
    num_slices: int = 32,
    device: torch.device = torch.device("cpu"),
) -> RSNADINOv2Model:
    """Load a trained RSNADINOv2Model from a ``.pth`` checkpoint.

    The checkpoint is expected to contain a ``model_state_dict`` key
    (matching the format produced by ``train.py``).  If the checkpoint
    stores the backbone name in a ``backbone`` key, that value is used
    automatically unless overridden.

    Parameters
    ----------
    checkpoint_path : str
        Path to the ``.pth`` checkpoint file.
    backbone : str
        DINOv2 variant (``dinov2_vits14`` or ``dinov2_vitb14``).
    num_classes : int
        Number of target logits.
    num_slices : int
        Number of 2D slices per volume.
    device : torch.device
        Target device for the model.

    Returns
    -------
    RSNADINOv2Model
        Model in ``eval()`` mode with loaded weights.
    """
    logger.info("Loading checkpoint: %s", checkpoint_path)
    ckpt: Dict[str, Any] = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False,
    )

    # Auto-detect backbone from checkpoint metadata
    if "backbone" in ckpt:
        saved_backbone = ckpt["backbone"]
        if saved_backbone != backbone:
            logger.info(
                "Checkpoint backbone '%s' differs from CLI '%s' -- "
                "using checkpoint value.",
                saved_backbone, backbone,
            )
            backbone = saved_backbone

    # Auto-detect config overrides
    if "config" in ckpt:
        cfg = ckpt["config"]
        num_classes = cfg.get("competition", {}).get("num_classes", num_classes)
        num_slices = cfg.get("data", {}).get("num_slices", num_slices)

    # Instantiate model (backbone weights don't matter -- we overwrite)
    model = RSNADINOv2Model(
        model_name=backbone,
        num_classes=num_classes,
        num_slices=num_slices,
        aggregator="attention",
        freeze_backbone=True,
        head_dropout=0.0,  # no dropout at inference time
    )

    # Load trained weights
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    epoch_info = ckpt.get("epoch", "?")
    auc_info = ckpt.get("macro_auc", "?")
    logger.info(
        "Model loaded | backbone=%s | epoch=%s | macro_auc=%s",
        backbone, epoch_info, auc_info,
    )
    return model


# =========================================================================
# 3.  Inference Loop
# =========================================================================

@torch.no_grad()
def run_inference(
    model: RSNADINOv2Model,
    loader: DataLoader,  # type: ignore[type-arg]
    device: torch.device,
    use_amp: bool = True,
) -> Tuple[List[str], np.ndarray]:
    """Run the inference loop with mixed precision and sigmoid activation.

    Parameters
    ----------
    model : RSNADINOv2Model
        Trained model in ``eval()`` mode.
    loader : DataLoader
        Test DataLoader yielding ``{"study_uid": ..., "image": ...}``
        batches.
    device : torch.device
        Execution device.
    use_amp : bool
        Whether to use automatic mixed precision.

    Returns
    -------
    study_uids : List[str]
        Ordered list of StudyInstanceUID strings.
    predictions : np.ndarray
        Sigmoid probabilities of shape ``(N, 12)`` in ``[0, 1]``.
    """
    model.eval()
    all_uids: List[str] = []
    all_preds: List[np.ndarray] = []

    # Optional tqdm progress bar
    try:
        from tqdm import tqdm
        iterator = tqdm(
            loader,
            desc="Inference",
            unit="batch",
            ncols=88,
            bar_format=(
                "{l_bar}{bar}| {n_fmt}/{total_fmt} "
                "[{elapsed}<{remaining}, {rate_fmt}]"
            ),
        )
    except ImportError:
        logger.info("tqdm not available -- using plain progress logging.")
        iterator = loader

    total_samples = 0
    t0 = time.perf_counter()

    for batch_idx, batch in enumerate(iterator):
        images: torch.Tensor = batch["image"].to(device, non_blocking=True)
        uids: List[str] = batch["study_uid"]

        # Handle 5D single-channel [B, 1, S, H, W] -> [B, S, H, W]
        if images.ndim == 5 and images.shape[1] == 1:
            images = images.squeeze(1)

        # Forward pass with mixed precision
        with torch.amp.autocast(
            device_type=device.type, enabled=use_amp,
        ):
            logits: torch.Tensor = model(images)

        # Sigmoid for probabilities, move to CPU immediately
        probs: np.ndarray = torch.sigmoid(logits).cpu().float().numpy()

        all_uids.extend(uids)
        all_preds.append(probs)
        total_samples += len(uids)

        # Clean up GPU memory
        del images, logits

    elapsed = time.perf_counter() - t0
    rate = total_samples / elapsed if elapsed > 0 else 0
    logger.info(
        "Inference complete: %d samples in %.1fs (%.1f samples/s)",
        total_samples, elapsed, rate,
    )

    predictions = np.concatenate(all_preds, axis=0)
    return all_uids, predictions


# =========================================================================
# 4.  Multi-Fold Ensemble
# =========================================================================

def run_ensemble_inference(
    checkpoint_paths: List[str],
    test_df: pd.DataFrame,
    data_dir: str,
    backbone: str,
    num_slices: int,
    image_size: Tuple[int, int],
    batch_size: int,
    num_workers: int,
    device: torch.device,
    use_amp: bool,
) -> Tuple[List[str], np.ndarray]:
    """Run inference with multiple checkpoints and average predictions.

    Parameters
    ----------
    checkpoint_paths : List[str]
        One or more ``.pth`` checkpoint file paths.

    Returns
    -------
    study_uids : List[str]
    avg_predictions : np.ndarray
        Ensemble-averaged sigmoid probabilities ``(N, 12)``.
    """
    n_folds = len(checkpoint_paths)
    logger.info(
        "Ensemble inference with %d checkpoint(s).", n_folds,
    )

    # Build test dataset once (shared across folds)
    test_ds = RSNAKneeDataset(
        df=test_df,
        image_dir=data_dir,
        target_columns=DEFAULT_TARGET_COLUMNS,
        image_size=image_size,
        num_slices=num_slices,
        is_train=False,  # test mode: no labels
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    accumulated_preds: Optional[np.ndarray] = None
    study_uids: List[str] = []

    for fold_idx, ckpt_path in enumerate(checkpoint_paths):
        print(f"\n  [{fold_idx + 1}/{n_folds}] Loading: {ckpt_path}")
        model = load_model_from_checkpoint(
            checkpoint_path=ckpt_path,
            backbone=backbone,
            num_classes=len(TARGET_COLUMNS),
            num_slices=num_slices,
            device=device,
        )

        uids, preds = run_inference(
            model=model,
            loader=test_loader,
            device=device,
            use_amp=use_amp,
        )

        if accumulated_preds is None:
            accumulated_preds = preds
            study_uids = uids
        else:
            accumulated_preds += preds

        # Free model memory between folds
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Average predictions across folds
    avg_preds = accumulated_preds / n_folds  # type: ignore[operator]

    return study_uids, avg_preds


# =========================================================================
# 5.  Submission CSV Writer
# =========================================================================

def write_submission(
    study_uids: List[str],
    predictions: np.ndarray,
    output_path: str,
) -> pd.DataFrame:
    """Write a competition-format submission CSV.

    Format
    ------
    ``StudyInstanceUID, ACL, MCL, ..., Fracture``

    Each target column contains the model's soft sigmoid probability
    in the range ``[0.0, 1.0]``.

    Parameters
    ----------
    study_uids : List[str]
        StudyInstanceUID identifiers.
    predictions : np.ndarray
        Probability array of shape ``(N, 12)``.
    output_path : str
        Destination file path.

    Returns
    -------
    pd.DataFrame
        The submission dataframe (also saved to disk).
    """
    assert predictions.shape[1] == len(TARGET_COLUMNS), (
        f"Expected {len(TARGET_COLUMNS)} columns, got {predictions.shape[1]}"
    )

    submission = pd.DataFrame(predictions, columns=TARGET_COLUMNS)
    submission.insert(0, "StudyInstanceUID", study_uids)

    # Clip to valid probability range
    for col in TARGET_COLUMNS:
        submission[col] = submission[col].clip(0.0, 1.0)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    submission.to_csv(output_path, index=False)

    logger.info("Submission saved: %s (%d rows)", output_path, len(submission))
    return submission


def _print_summary(submission: pd.DataFrame) -> None:
    """Print a clean summary of the submission predictions."""
    print("\n" + "=" * 72)
    print("  SUBMISSION SUMMARY")
    print("=" * 72)
    print(f"  Total predictions : {len(submission):>8d}")
    print("-" * 72)
    print(f"  {'Target':<22s}  {'Mean':>8s}  {'Min':>8s}  {'Max':>8s}  {'Std':>8s}")
    print(f"  {'-' * 20}  {'-' * 8}  {'-' * 8}  {'-' * 8}  {'-' * 8}")

    for col in TARGET_COLUMNS:
        vals = submission[col].astype(float)
        print(
            f"  {col:<22s}  {vals.mean():>8.4f}  {vals.min():>8.4f}  "
            f"{vals.max():>8.4f}  {vals.std():>8.4f}"
        )

    print("=" * 72 + "\n")


# =========================================================================
# 6.  CLI Entry Point
# =========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "RSNA Knee Abnormality Detection -- Inference & Submission Script.  "
            "Loads trained model checkpoint(s), runs inference on the test set, "
            "and writes a competition-format submission.csv."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python src/inference.py --weights best_model.pth "
            "--data-dir data/test_images --csv data/test.csv\n"
            "  python src/inference.py --weights fold0.pth fold1.pth fold2.pth "
            "--data-dir data/test_images --csv data/test.csv\n"
        ),
    )

    parser.add_argument(
        "--weights", "-w",
        nargs="+",
        required=True,
        help=(
            "Path(s) to trained .pth checkpoint file(s).  "
            "Multiple paths trigger ensemble averaging."
        ),
    )
    parser.add_argument(
        "--data-dir", "-d",
        required=True,
        help="Root directory containing test study folders (DICOMs or images).",
    )
    parser.add_argument(
        "--csv", "-c",
        required=True,
        help="Path to test.csv with StudyInstanceUID column.",
    )
    parser.add_argument(
        "--output", "-o",
        default="submission.csv",
        help="Output submission CSV path (default: submission.csv).",
    )
    parser.add_argument(
        "--backbone", "-b",
        default="dinov2_vits14",
        help=(
            "DINOv2 backbone variant (default: dinov2_vits14).  "
            "Automatically overridden by checkpoint metadata if available."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Inference batch size (default: 2 -- conservative for GPU memory).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="DataLoader workers (default: 2).",
    )
    parser.add_argument(
        "--num-slices",
        type=int,
        default=32,
        help="Number of slices per volume (default: 32).",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        default=[224, 224],
        help="Image spatial resolution H W (default: 224 224).",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable automatic mixed precision.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    # Validate inputs
    for w in args.weights:
        if not os.path.isfile(w):
            logger.error("Checkpoint not found: %s", w)
            sys.exit(1)

    if not os.path.isfile(args.csv):
        logger.error("Test CSV not found: %s", args.csv)
        sys.exit(1)

    # Device & AMP
    device = _get_device()
    use_amp = (device.type == "cuda") and (not args.no_amp)
    image_size = tuple(args.image_size)

    # Load test metadata
    logger.info("Loading test CSV: %s", args.csv)
    test_df = pd.read_csv(args.csv)
    logger.info("Test set: %d studies", len(test_df))

    print("\n" + "=" * 72)
    print("  RSNA KNEE ABNORMALITY DETECTION -- INFERENCE")
    print("=" * 72)
    print(f"  Checkpoints   : {args.weights}")
    print(f"  Test studies  : {len(test_df)}")
    print(f"  Backbone      : {args.backbone}")
    print(f"  Device        : {device}")
    print(f"  Mixed Prec.   : {use_amp}")
    print(f"  Batch size    : {args.batch_size}")
    print(f"  Output        : {args.output}")
    print("=" * 72)

    # Run inference (single or ensemble)
    study_uids, predictions = run_ensemble_inference(
        checkpoint_paths=args.weights,
        test_df=test_df,
        data_dir=args.data_dir,
        backbone=args.backbone,
        num_slices=args.num_slices,
        image_size=image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        use_amp=use_amp,
    )

    # Write submission
    submission = write_submission(study_uids, predictions, args.output)
    _print_summary(submission)

    print(f"  Submission saved to: {args.output}")
    print(f"  Shape: {submission.shape}")
    print("  Done!\n")


if __name__ == "__main__":
    main()
