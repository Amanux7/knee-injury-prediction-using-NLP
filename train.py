"""
RSNA Knee Abnormality Detection -- Training Orchestrator
========================================================
End-to-end training pipeline: loads config, builds DataLoaders from
``train_folds.csv``, trains DINOv2 / timm multi-label classifiers with mixed
precision, evaluates with macro-AUC, and checkpoints the best model per fold.

Usage
-----
.. code-block:: bash

    # Train fold 0 with DINOv2 Base backbone from YAML config
    python train.py --fold 0

    # Override backbone directly from CLI
    python train.py --fold 0 --backbone dinov2_vitb14

    # Train with ResNet34 timm backbone
    python train.py --fold 0 --backbone resnet34

    # Smoke-test: 1 mini-epoch on synthetic data (CPU-safe)
    python train.py --smoke-test
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
    weights_dir: Optional[str] = None,
) -> nn.Module:
    """Instantiate a DINOv2 or timm classification model dynamically.

    Parameters
    ----------
    backbone : str
        Backbone identifier (e.g. ``"dinov2_vitb14"``, ``"dinov2_vits14"``,
        ``"resnet34"``, ``"efficientnet_b0"``).
    num_classes : int
        Number of target logits (12 classes).
    in_chans : int
        Number of input 2D slices (e.g., 32).
    pretrained : bool
        Whether to load pre-trained weights.
    weights_dir : Optional[str]
        Directory path to unpacked local model weights (e.g. ``/content/models_cache/``).

    Returns
    -------
    nn.Module
        Initialized PyTorch model ready for training.
    """
    logger.info("Selected Backbone: %s", backbone)

    # ── Option A: DINOv2 Backbone (dinov2_vits14, dinov2_vitb14) ─────────
    if backbone.startswith("dinov2"):
        from src.models import RSNADINOv2Model

        # Check for local weights directory fallback
        if weights_dir is None:
            for candidate in ("./models_cache", "/content/models_cache", "models_cache"):
                if os.path.isdir(candidate):
                    weights_dir = candidate
                    break

        model: nn.Module = RSNADINOv2Model(
            model_name=backbone,
            num_classes=num_classes,
            num_slices=in_chans,
            aggregator="attention",
            weights_dir=weights_dir,
            freeze_backbone=False if pretrained else True,
        )

    # ── Option B: Standard timm 2D Backbone (ResNet, EfficientNet, ConvNeXt) ─
    else:
        import timm

        model = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=num_classes,
            in_chans=in_chans,
        )

    total_params: int = sum(p.numel() for p in model.parameters())
    trainable: int = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "Model created: %s | in_chans=%d | classes=%d | params=%s (trainable=%s)",
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
        images: torch.Tensor = batch["image"].to(device)
        labels: torch.Tensor = batch["label"].to(device)

        # Handle 5D single-channel tensor [B, 1, S, H, W] -> [B, S, H, W] for timm
        if images.ndim == 5 and images.shape[1] == 1:
            images = images.squeeze(1)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):  # type: ignore[arg-type]
            logits: torch.Tensor = model(images)
            loss: torch.Tensor = criterion(logits, labels)

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
    """Run validation and compute macro-AUC with memory leak protection.

    Returns
    -------
    val_loss : float
    macro_auc : float
    per_class_auc : Dict[str, float]
    """
    model.eval()

    # Clear GPU VRAM before starting validation loop
    if device.type == "cuda":
        torch.cuda.empty_cache()

    running_loss: float = 0.0
    num_batches: int = 0
    all_preds: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    for batch in loader:
        images: torch.Tensor = batch["image"].to(device)
        labels: torch.Tensor = batch["label"].to(device)

        if images.ndim == 5 and images.shape[1] == 1:
            images = images.squeeze(1)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):  # type: ignore[arg-type]
            logits: torch.Tensor = model(images)
            loss: torch.Tensor = criterion(logits, labels)

        running_loss += loss.item()
        num_batches += 1

        probs: np.ndarray = torch.sigmoid(logits.detach().float()).cpu().numpy()
        all_preds.append(probs)
        all_labels.append(labels.detach().cpu().numpy())

        # Immediate batch memory cleanup
        del images, labels, logits, loss

    if device.type == "cuda":
        torch.cuda.empty_cache()

    avg_loss: float = running_loss / max(num_batches, 1)
    y_pred: np.ndarray = np.concatenate(all_preds, axis=0)
    y_true: np.ndarray = np.concatenate(all_labels, axis=0)

    pred_std = np.std(y_pred, axis=0)
    logger.info(
        "Validation prediction spread | mean std=%.6f | min std=%.6f | "
        "max std=%.6f",
        float(pred_std.mean()), float(pred_std.min()), float(pred_std.max()),
    )
    if float(pred_std.max()) < 1e-4:
        logger.warning(
            "Predictions are effectively constant across validation studies. "
            "Verify image loading and optimizer settings before continuing."
        )

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
    """Generate a tiny synthetic DataFrame for pipeline verification."""
    rng: np.random.Generator = np.random.default_rng(seed)
    df = pd.DataFrame({
        "StudyInstanceUID": [f"SMOKE_{i:04d}" for i in range(n_samples)],
        **{col: rng.integers(0, 2, size=n_samples) for col in target_columns},
    })
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
    backbone_override: Optional[str] = None,
    smoke_test: bool = False,
    image_dir: str = "train_series",
    weights_dir: Optional[str] = None,
    series_df: Optional[pd.DataFrame] = None,
) -> float:
    """Execute the full train/validate loop for one fold."""
    # ── Dynamic Backbone Resolution ──────────────────────────────────────
    # Priority: CLI argument override > cfg['model']['backbone'] > cfg['backbone'] > default
    backbone: str = (
        backbone_override
        or cfg.get("model", {}).get("backbone")
        or cfg.get("backbone", "dinov2_vitb14")
    )

    num_classes: int = cfg["competition"]["num_classes"]
    target_columns: List[str] = cfg["competition"]["target_columns"]
    image_size: Tuple[int, int] = tuple(cfg["data"]["image_size"])  # type: ignore[assignment]
    num_slices: int = cfg["data"]["num_slices"]
    batch_size: int = 4 if smoke_test else cfg["data"]["batch_size"]
    num_workers: int = 0 if smoke_test else cfg["data"]["num_workers"]
    lr: float = cfg["training"]["learning_rate"]
    backbone_lr: float = cfg["training"].get("backbone_learning_rate", lr)
    wd: float = cfg["training"]["weight_decay"]
    epochs: int = 1 if smoke_test else cfg["training"]["epochs"]
    seed: int = cfg["training"]["seed"]

    seed_everything(seed)
    device: torch.device = get_device()
    use_amp: bool = device.type == "cuda"

    # ── Split into train / val ────────────────────────────────────────────
    train_df: pd.DataFrame = df[df["fold"] != fold].reset_index(drop=True)
    val_df: pd.DataFrame = df[df["fold"] == fold].reset_index(drop=True)
    logger.info("Fold %d | train=%d  val=%d", fold, len(train_df), len(val_df))

    # ── Datasets & Loaders ────────────────────────────────────────────────
    image_root = Path(image_dir).expanduser()
    if not smoke_test:
        if not image_root.is_dir():
            raise FileNotFoundError(
                f"Training image directory does not exist: '{image_root}'. "
                "On Kaggle, pass the mounted competition directory with "
                "--image-dir."
            )

        all_uids = df["StudyInstanceUID"].astype(str).unique().tolist()
        if series_df is None:
            raise ValueError(
                "Real competition training requires train_series.csv. Pass it "
                "with --series-csv so anatomical planes are not mixed."
            )
        series_studies = set(series_df["StudyInstanceUID"].astype(str))
        missing_metadata = [uid for uid in all_uids if uid not in series_studies]
        if missing_metadata:
            examples = ", ".join(missing_metadata[:5])
            raise ValueError(
                f"Series metadata is missing {len(missing_metadata)} studies. "
                f"Examples: {examples}."
            )

        missing_uids = []
        for uid in all_uids:
            candidate_roots = (
                image_root / uid,
                image_root / "train_series" / uid,
            )
            if not any(path.is_dir() for path in candidate_roots):
                missing_uids.append(uid)
        if missing_uids:
            examples = ", ".join(missing_uids[:5])
            raise FileNotFoundError(
                f"Image preflight failed: {len(missing_uids)}/{len(all_uids)} "
                f"study folders are missing below '{image_root}'. "
                f"Examples: {examples}. Check --image-dir and UID mapping."
            )
        logger.info(
            "Image preflight passed: %d/%d study folders found below '%s'.",
            len(all_uids), len(all_uids), image_root,
        )

    train_ds = RSNAKneeDataset(
        df=train_df,
        image_dir=image_root,
        target_columns=target_columns,
        image_size=image_size,
        num_slices=num_slices,
        is_train=True,
        allow_missing_images=smoke_test,
        series_df=series_df,
    )
    val_ds = RSNAKneeDataset(
        df=val_df,
        image_dir=image_root,
        target_columns=target_columns,
        image_size=image_size,
        num_slices=num_slices,
        is_train=True,
        allow_missing_images=smoke_test,
        series_df=series_df,
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
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    # ── Build Model ───────────────────────────────────────────────────────
    model: nn.Module = build_model(
        backbone=backbone,
        num_classes=num_classes,
        in_chans=num_slices,
        pretrained=(not smoke_test),
        weights_dir=weights_dir,
    )
    model = model.to(device)

    criterion: nn.Module = nn.BCEWithLogitsLoss()
    if hasattr(model, "backbone"):
        backbone_params = [
            param for param in model.backbone.parameters() if param.requires_grad
        ]
    else:
        backbone_params = []

    if backbone_params:
        backbone_param_ids = {id(param) for param in backbone_params}
        task_params = [
            param for param in model.parameters()
            if param.requires_grad and id(param) not in backbone_param_ids
        ]
        optimizer = torch.optim.AdamW(
            [
                {"params": task_params, "lr": lr},
                {"params": backbone_params, "lr": backbone_lr},
            ],
            weight_decay=wd,
        )
        logger.info(
            "Differential learning rates | head+aggregator=%.2e | backbone=%.2e",
            lr, backbone_lr,
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=wd,
        )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=min(lr, backbone_lr) * 0.01,
    )

    scaler: Optional[torch.amp.GradScaler] = (  # type: ignore[type-arg]
        torch.amp.GradScaler("cuda") if use_amp else None  # type: ignore[arg-type]
    )

    # ── Training Loop ─────────────────────────────────────────────────────
    best_auc: float = 0.0
    best_epoch: int = -1
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path: Path = output_dir / f"best_model_fold_{fold}.pth"

    logger.info(
        "Starting training | backbone=%s | epochs=%d | lr=%.2e | bs=%d | AMP=%s",
        backbone, epochs, lr, batch_size, use_amp,
    )
    print("\n" + "=" * 70)
    print(f"  TRAINING FOLD {fold} (Backbone: {backbone})")
    print("=" * 70)

    for epoch in range(1, epochs + 1):
        t0: float = time.time()

        train_loss: float = train_one_epoch(
            model, train_loader, criterion, optimizer,
            device, scaler, epoch, use_amp,
        )

        val_loss, macro_auc, per_class = validate(
            model, val_loader, criterion, device, target_columns, use_amp,
        )

        scheduler.step()
        elapsed: float = time.time() - t0
        current_lr: float = max(group["lr"] for group in optimizer.param_groups)

        print(
            f"  Epoch {epoch:>2d}/{epochs} | "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f} | "
            f"macro_AUC={macro_auc:.4f} | "
            f"lr={current_lr:.2e} | "
            f"{elapsed:.1f}s"
        )

        if macro_auc > best_auc:
            best_auc = macro_auc
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "fold": fold,
                    "backbone": backbone,
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
# 7. CLI Entry Point
# =========================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="RSNA Knee Abnormality Detection -- Training Script (Use --backbone to override vision backbone model name)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline_config.yaml",
        help="Path to YAML config file (default: %(default)s).",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default=None,
        help="Override vision backbone model name",
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
        "--image-dir",
        type=str,
        default=None,
        help=(
            "Root containing StudyInstanceUID folders. Overrides "
            "data.image_dir in the config."
        ),
    )
    parser.add_argument(
        "--weights-dir",
        type=str,
        default=None,
        help=(
            "Directory containing unpacked DINOv2 weights. Overrides "
            "model.weights_dir in the config."
        ),
    )
    parser.add_argument(
        "--series-csv",
        type=str,
        default=None,
        help=(
            "Path to train_series.csv. Overrides data.series_csv in the "
            "config and is required for real competition training."
        ),
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
    image_dir: str = (
        args.image_dir
        or cfg.get("data", {}).get("image_dir")
        or "train_series"
    )
    weights_dir: Optional[str] = (
        args.weights_dir
        or cfg.get("model", {}).get("weights_dir")
    )
    series_df: Optional[pd.DataFrame] = None
    if not args.smoke_test:
        series_csv = (
            args.series_csv
            or cfg.get("data", {}).get("series_csv")
            or "train_series.csv"
        )
        series_csv_path = Path(series_csv)
        if not series_csv_path.is_file():
            raise FileNotFoundError(
                f"Series metadata CSV not found: '{series_csv_path}'. Pass the "
                "Kaggle train_series.csv path with --series-csv."
            )
        series_df = pd.read_csv(series_csv_path)
        logger.info("Series metadata: %d rows from '%s'.", len(series_df), series_csv_path)
    best_auc: float = run_training(
        cfg=cfg,
        fold=fold,
        df=df,
        output_dir=output_dir,
        backbone_override=args.backbone,
        smoke_test=args.smoke_test,
        image_dir=image_dir,
        weights_dir=weights_dir,
        series_df=series_df,
    )

    logger.info("Training finished. Best validation Macro AUC = %.4f", best_auc)


if __name__ == "__main__":
    main()
