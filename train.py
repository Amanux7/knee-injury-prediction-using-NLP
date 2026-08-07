"""
RSNA Knee Abnormality Detection -- Training Orchestrator
========================================================
End-to-end training pipeline: loads config, builds DataLoaders from
``train_folds.csv``, trains a ``timm`` multi-label classifier with mixed
precision, evaluates with macro-AUC, and checkpoints the best model per fold.

Usage
-----
.. code-block:: bash

    # Train fold 0 with default config
    python train.py --fold 0

    # Smoke-test: 1 mini-epoch on synthetic data (CPU-safe)
    python train.py --smoke-test

    # Custom config & output directory
    python train.py --fold 2 --config configs/my_config.yaml --output-dir runs/
"""

from __future__ import annotations

import argparse
import copy
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------------
from src.dataset import RSNAKneeDataset, DEFAULT_TARGET_COLUMNS
from src.metrics import compute_macro_auc

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger: logging.Logger = logging.getLogger(__name__)


# =========================================================================
# 1. Reproducibility
# =========================================================================

def seed_everything(seed: int = 42) -> None:
    """Pin every random source for deterministic training."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
        torch.backends.cudnn.benchmark = False      # type: ignore[attr-defined]
    logger.info("Random seed set to %d.", seed)


# =========================================================================
# 2. Device selection
# =========================================================================

def get_device() -> torch.device:
    """Auto-detect the best available accelerator (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        logger.info("Using CUDA: %s", torch.cuda.get_device_name(0))
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore[attr-defined]
        dev = torch.device("mps")
        logger.info("Using Apple MPS backend.")
    else:
        dev = torch.device("cpu")
        logger.info("No GPU detected -- falling back to CPU.")
    return dev


# =========================================================================
# 3. Model factory
# =========================================================================

def build_model(
    backbone: str,
    num_classes: int,
    in_chans: int,
    pretrained: bool = True,
) -> nn.Module:
    """Create a ``timm`` classifier adapted for multi-channel volumetric input.

    Strategy: MRI volumes have shape ``[B, 1, S, H, W]`` which we reshape to
    ``[B, S, H, W]`` and treat the *S* slices as input channels.  ``timm``
    handles arbitrary ``in_chans`` by adapting the first convolution layer
    automatically when ``in_chans != 3``.

    Parameters
    ----------
    backbone : str
        Any ``timm`` model name (e.g. ``"resnet34"``, ``"efficientnet_b0"``,
        ``"convnext_tiny"``).
    num_classes : int
        Number of output logits (12 for this competition).
    in_chans : int
        Number of input channels (equal to ``num_slices``).
    pretrained : bool
        Whether to load ImageNet pre-trained weights (the first conv is
        re-initialised by ``timm`` when ``in_chans != 3``).

    Returns
    -------
    nn.Module
        Ready-to-train classifier.
    """
    import timm  # lazy import to keep module-level fast

    model: nn.Module = timm.create_model(
        backbone,
        pretrained=pretrained,
        num_classes=num_classes,
        in_chans=in_chans,
    )
    total_params: int = sum(p.numel() for p in model.parameters())
    trainable: int = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "Model: %s | in_chans=%d | classes=%d | params=%s (trainable=%s)",
        backbone, in_chans, num_classes,
        f"{total_params:,}", f"{trainable:,}",
    )
    return model


# =========================================================================
# 4. Training & validation loops
# =========================================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,  # type: ignore[type-arg]
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[torch.amp.GradScaler],  # type: ignore[type-arg]
    epoch: int,
    use_amp: bool,
) -> float:
    """Run a single training epoch.

    Returns
    -------
    float
        Mean training loss for the epoch.
    """
    model.train()
    running_loss: float = 0.0
    num_batches: int = 0

    for batch_idx, batch in enumerate(loader):
        # -- Reshape volume: [B, 1, S, H, W] -> [B, S, H, W] -------------
        images: torch.Tensor = batch["image"].squeeze(1).to(device)
        labels: torch.Tensor = batch["label"].to(device)

        optimizer.zero_grad(set_to_none=True)

        # -- Forward (mixed precision when available) ----------------------
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):  # type: ignore[arg-type]
            logits: torch.Tensor = model(images)
            loss: torch.Tensor = criterion(logits, labels)

        # -- Backward ------------------------------------------------------
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        num_batches += 1

        if (batch_idx + 1) % max(1, len(loader) // 5) == 0:
            logger.info(
                "  Epoch %d | batch %d/%d | loss=%.4f",
                epoch, batch_idx + 1, len(loader), loss.item(),
            )

    avg_loss: float = running_loss / max(num_batches, 1)
    return avg_loss


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,  # type: ignore[type-arg]
    criterion: nn.Module,
    device: torch.device,
    target_columns: List[str],
    use_amp: bool,
) -> Tuple[float, float, Dict[str, float]]:
    """Run validation and compute macro-AUC.

    Returns
    -------
    val_loss : float
    macro_auc : float
    per_class_auc : Dict[str, float]
    """
    model.eval()
    running_loss: float = 0.0
    num_batches: int = 0
    all_preds: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    for batch in loader:
        images: torch.Tensor = batch["image"].squeeze(1).to(device)
        labels: torch.Tensor = batch["label"].to(device)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):  # type: ignore[arg-type]
            logits: torch.Tensor = model(images)
            loss: torch.Tensor = criterion(logits, labels)

        running_loss += loss.item()
        num_batches += 1

        probs: torch.Tensor = torch.sigmoid(logits).cpu().numpy()
        all_preds.append(probs)
        all_labels.append(labels.cpu().numpy())

    avg_loss: float = running_loss / max(num_batches, 1)

    y_pred: np.ndarray = np.concatenate(all_preds, axis=0)
    y_true: np.ndarray = np.concatenate(all_labels, axis=0)

    macro_auc, per_class = compute_macro_auc(y_true, y_pred, target_columns)
    return avg_loss, macro_auc, per_class


# =========================================================================
# 5. Synthetic data generator (smoke-test mode)
# =========================================================================

def create_smoke_test_df(
    n_samples: int = 32,
    target_columns: List[str] = DEFAULT_TARGET_COLUMNS,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a tiny synthetic DataFrame for pipeline verification.

    The generated data is *not* medically meaningful -- it exists solely to
    confirm that every code path (dataset, model, loss, metric) executes
    without errors.
    """
    rng: np.random.Generator = np.random.default_rng(seed)
    df = pd.DataFrame({
        "StudyInstanceUID": [f"SMOKE_{i:04d}" for i in range(n_samples)],
        **{col: rng.integers(0, 2, size=n_samples) for col in target_columns},
    })
    # Add a fold column: first half = fold 0, second half = fold 1
    df["fold"] = [0] * (n_samples // 2) + [1] * (n_samples - n_samples // 2)
    return df


# =========================================================================
# 6. Main training orchestration
# =========================================================================

def run_training(
    cfg: Dict[str, Any],
    fold: int,
    df: pd.DataFrame,
    output_dir: Path,
    smoke_test: bool = False,
) -> float:
    """Execute the full train/validate loop for one fold.

    Parameters
    ----------
    cfg : dict
        Parsed YAML configuration.
    fold : int
        Which fold to hold out for validation.
    df : pd.DataFrame
        Full training frame with a ``fold`` column.
    output_dir : Path
        Directory to save checkpoints.
    smoke_test : bool
        If ``True``, run only 1 epoch with a tiny batch.

    Returns
    -------
    float
        Best macro-AUC achieved on the validation set.
    """
    # -- Unpack config -----------------------------------------------------
    num_classes: int = cfg["competition"]["num_classes"]
    target_columns: List[str] = cfg["competition"]["target_columns"]
    image_size: Tuple[int, int] = tuple(cfg["data"]["image_size"])  # type: ignore[assignment]
    num_slices: int = cfg["data"]["num_slices"]
    batch_size: int = 4 if smoke_test else cfg["data"]["batch_size"]
    num_workers: int = 0 if smoke_test else cfg["data"]["num_workers"]
    backbone: str = cfg["model"]["backbone"]
    lr: float = cfg["training"]["learning_rate"]
    wd: float = cfg["training"]["weight_decay"]
    epochs: int = 1 if smoke_test else cfg["training"]["epochs"]
    seed: int = cfg["training"]["seed"]

    seed_everything(seed)
    device: torch.device = get_device()

    # -- Use AMP only on CUDA (MPS/CPU don't benefit or lack support) ------
    use_amp: bool = device.type == "cuda"

    # -- Split into train / val --------------------------------------------
    train_df: pd.DataFrame = df[df["fold"] != fold].reset_index(drop=True)
    val_df: pd.DataFrame = df[df["fold"] == fold].reset_index(drop=True)
    logger.info(
        "Fold %d | train=%d  val=%d", fold, len(train_df), len(val_df),
    )

    # -- Datasets & loaders ------------------------------------------------
    image_dir: str = "train_images"  # placeholder -- RSNAKneeDataset has fallback
    train_ds = RSNAKneeDataset(
        df=train_df,
        image_dir=image_dir,
        target_columns=target_columns,
        image_size=image_size,
        num_slices=num_slices,
        is_train=True,
    )
    val_ds = RSNAKneeDataset(
        df=val_df,
        image_dir=image_dir,
        target_columns=target_columns,
        image_size=image_size,
        num_slices=num_slices,
        is_train=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    # -- Model, loss, optimizer, scheduler ---------------------------------
    model: nn.Module = build_model(
        backbone=backbone,
        num_classes=num_classes,
        in_chans=num_slices,
        pretrained=(not smoke_test),  # skip download in smoke-test
    )
    model = model.to(device)

    criterion: nn.Module = nn.BCEWithLogitsLoss()
    optimizer: torch.optim.Optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=wd,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01,
    )

    scaler: Optional[torch.amp.GradScaler] = (  # type: ignore[type-arg]
        torch.amp.GradScaler("cuda") if use_amp else None  # type: ignore[arg-type]
    )

    # -- Training loop -----------------------------------------------------
    best_auc: float = 0.0
    best_epoch: int = -1
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path: Path = output_dir / f"best_model_fold_{fold}.pth"

    logger.info(
        "Starting training | backbone=%s  epochs=%d  lr=%.2e  bs=%d  AMP=%s",
        backbone, epochs, lr, batch_size, use_amp,
    )
    print("\n" + "=" * 70)
    print(f"  TRAINING FOLD {fold}")
    print("=" * 70)

    for epoch in range(1, epochs + 1):
        t0: float = time.time()

        # -- Train ---------------------------------------------------------
        train_loss: float = train_one_epoch(
            model, train_loader, criterion, optimizer,
            device, scaler, epoch, use_amp,
        )

        # -- Validate ------------------------------------------------------
        val_loss, macro_auc, per_class = validate(
            model, val_loader, criterion, device, target_columns, use_amp,
        )

        scheduler.step()
        elapsed: float = time.time() - t0
        current_lr: float = optimizer.param_groups[0]["lr"]

        # -- Logging -------------------------------------------------------
        print(
            f"  Epoch {epoch:>2d}/{epochs} | "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f} | "
            f"macro_AUC={macro_auc:.4f} | "
            f"lr={current_lr:.2e} | "
            f"{elapsed:.1f}s"
        )

        # -- Checkpoint best -----------------------------------------------
        if macro_auc > best_auc:
            best_auc = macro_auc
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "fold": fold,
                    "macro_auc": macro_auc,
                    "per_class_auc": per_class,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": cfg,
                },
                ckpt_path,
            )
            logger.info(
                "  >> New best AUC=%.4f at epoch %d -- saved to %s",
                macro_auc, epoch, ckpt_path,
            )

    print("-" * 70)
    print(f"  Fold {fold} complete | Best AUC={best_auc:.4f} @ epoch {best_epoch}")
    print("=" * 70 + "\n")

    return best_auc


# =========================================================================
# 7. CLI entry point
# =========================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="RSNA Knee Abnormality Detection -- Training Script",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline_config.yaml",
        help="Path to YAML config file (default: %(default)s).",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=0,
        help="Fold index to train (0-4, default: %(default)s).",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="train_folds.csv",
        help="Path to train_folds.csv (default: %(default)s).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints",
        help="Directory to save model checkpoints (default: %(default)s).",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run 1 mini-epoch on synthetic data to verify the pipeline.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point -- parse args, load config/data, and launch training."""
    args: argparse.Namespace = parse_args()

    # -- Load config -------------------------------------------------------
    config_path: Path = Path(args.config)
    if not config_path.is_file():
        logger.error("Config file '%s' not found.", config_path)
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)
    logger.info("Loaded config from '%s'.", config_path)

    # -- Load or generate data ---------------------------------------------
    if args.smoke_test:
        logger.info("=== SMOKE-TEST MODE === (synthetic data, 1 epoch)")
        target_columns: List[str] = cfg["competition"]["target_columns"]
        df: pd.DataFrame = create_smoke_test_df(
            n_samples=32,
            target_columns=target_columns,
            seed=cfg["training"]["seed"],
        )
        fold: int = 0
    else:
        csv_path: Path = Path(args.csv)
        if not csv_path.is_file():
            logger.error(
                "'%s' not found. Run `python -m src.cross_validation` first "
                "to generate fold assignments.",
                csv_path,
            )
            sys.exit(1)
        df = pd.read_csv(csv_path)
        fold = args.fold

    logger.info(
        "Data: %d rows | fold=%d | folds in data: %s",
        len(df), fold, sorted(df["fold"].unique().tolist()),
    )

    # -- Run training ------------------------------------------------------
    output_dir: Path = Path(args.output_dir)
    best_auc: float = run_training(
        cfg=cfg,
        fold=fold,
        df=df,
        output_dir=output_dir,
        smoke_test=args.smoke_test,
    )

    logger.info("Training finished. Best validation Macro AUC = %.4f", best_auc)


if __name__ == "__main__":
    main()
