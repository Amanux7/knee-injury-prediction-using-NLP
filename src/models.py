"""
RSNA Knee Abnormality Detection -- DINOv2 Vision Backbone & 2.5D Model
======================================================================
Provides :class:`RSNADINOv2Model`, a volumetric 2.5D classifier that
processes multi-slice MRI volumes through Meta's DINOv2 vision transformer
(``dinov2_vits14`` or ``dinov2_vitb14``) and aggregates slice embeddings
with either temporal convolution or multi-head self-attention before a
12-class multi-label classification head.

Backbone Weight Loading Options
-------------------------------
1. **Unpacked Local Directory** (``weights_dir="/path/to/cache"``):
   Loads weights directly from a folder containing ``.pth``, ``.pt``, or ``.bin`` files
   (e.g., ``/content/models_cache/`` in Google Colab).
2. **Local Archive File** (``local_archive="weights.tar.gz"``):
   Extracts and loads weights from a compressed tarball archive.
3. **``torch.hub`` / ``timm``**:
   Downloads pre-trained DINOv2 weights online from GitHub/HuggingFace.

Supported Input Tensor Shapes
-----------------------------
- 5D Tensor: ``[Batch_Size, Slices=32, Channels=3, Height=224, Width=224]``
- 4D Tensor: ``[Batch_Size, Slices=32, Height=224, Width=224]`` (auto-broadcasts 1 -> 3 RGB channels)
- 5D Single-Channel: ``[Batch_Size, Slices=32, 1, Height=224, Width=224]`` (auto-broadcasts 1 -> 3 channels)
"""

from __future__ import annotations

import logging
import math
import os
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Module-level logger & constants
# ---------------------------------------------------------------------------
logger: logging.Logger = logging.getLogger(__name__)

TARGET_COLUMNS: List[str] = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture",
]

# DINOv2 model name --> CLS embedding dimension
_DINOV2_EMBED_DIMS: Dict[str, int] = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitg14": 1536,
}


# =========================================================================
# 1. DINOv2 Backbone Loaders
# =========================================================================

def _load_from_local_directory(
    weights_dir: Union[str, Path],
    model_name: str,
) -> nn.Module:
    """Load DINOv2 from an unpacked local directory containing model weights.

    Parameters
    ----------
    weights_dir : Union[str, Path]
        Path to directory containing weight files (e.g., ``/content/models_cache/``).
    model_name : str
        DINOv2 variant name (``dinov2_vits14`` or ``dinov2_vitb14``).

    Returns
    -------
    nn.Module
        Initialized DINOv2 model with loaded weights.
    """
    weights_path = Path(weights_dir)
    if not weights_path.is_dir():
        raise FileNotFoundError(f"Local weights directory not found: {weights_path}")

    logger.info("Scanning for DINOv2 weights in '%s'...", weights_path)

    # Search for weight files in directory tree
    weight_file: Optional[Path] = None
    for root, _dirs, files in os.walk(weights_path):
        for fname in files:
            if fname.endswith((".pth", ".pt", ".bin", ".safetensors")):
                if model_name in fname or "dinov2" in fname or weight_file is None:
                    weight_file = Path(root) / fname
                    if model_name in fname:
                        break

    if weight_file is None:
        raise FileNotFoundError(
            f"No .pth / .pt / .bin weight file found in '{weights_path}'."
        )

    logger.info("Loading weights from file: '%s'", weight_file)
    state_dict: Dict[str, Any] = torch.load(
        weight_file, map_location="cpu", weights_only=True,
    )

    # Unwrap nested state dict keys if needed
    if "model" in state_dict:
        state_dict = state_dict["model"]
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    # Instantiate model skeleton via torch.hub (offline mode if cached)
    model: nn.Module = torch.hub.load(
        "facebookresearch/dinov2", model_name, pretrained=False,
    )
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    logger.info("Successfully loaded DINOv2 '%s' from local directory.", model_name)
    return model


def _load_from_local_archive(
    archive_path: Union[str, Path],
    model_name: str,
) -> nn.Module:
    """Load DINOv2 from a compressed ``*.tar.gz`` weight archive."""
    archive_path = Path(archive_path)
    if not archive_path.is_file():
        raise FileNotFoundError(f"Archive not found: {archive_path}")

    logger.info("Extracting DINOv2 weights from archive '%s'...", archive_path)

    with tempfile.TemporaryDirectory() as tmp_dir:
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(tmp_dir)

        weight_file: Optional[Path] = None
        for root, _dirs, files in os.walk(tmp_dir):
            for fname in files:
                if fname.endswith((".pth", ".pt", ".bin")):
                    weight_file = Path(root) / fname
                    break
            if weight_file is not None:
                break

        if weight_file is None:
            raise FileNotFoundError(
                f"No .pth / .pt weight file found inside archive '{archive_path}'."
            )

        state_dict: Dict[str, Any] = torch.load(
            weight_file, map_location="cpu", weights_only=True,
        )

    model: nn.Module = torch.hub.load(
        "facebookresearch/dinov2", model_name, pretrained=False,
    )
    if "model" in state_dict:
        state_dict = state_dict["model"]
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    logger.info("Loaded DINOv2 '%s' from archive.", model_name)
    return model


def load_dinov2_backbone(
    model_name: str = "dinov2_vits14",
    weights_dir: Optional[Union[str, Path]] = None,
    local_archive: Optional[Union[str, Path]] = None,
    freeze: bool = True,
) -> Tuple[nn.Module, int]:
    """Load DINOv2 backbone with specified resolution and parameters.

    Priority Order:
    1. Unpacked directory (``weights_dir``)
    2. Compressed archive (``local_archive``)
    3. Online ``torch.hub.load``

    Parameters
    ----------
    model_name : str
        ``dinov2_vits14`` (384-dim) or ``dinov2_vitb14`` (768-dim).
    weights_dir : Optional[Union[str, Path]]
        Directory containing unpacked weights (e.g. ``/content/models_cache/``).
    local_archive : Optional[Union[str, Path]]
        Path to compressed ``.tar.gz`` archive.
    freeze : bool
        Whether to freeze backbone parameters.

    Returns
    -------
    (backbone, embed_dim) : Tuple[nn.Module, int]
    """
    if model_name not in _DINOV2_EMBED_DIMS:
        raise ValueError(
            f"Unknown DINOv2 model '{model_name}'. "
            f"Supported options: {list(_DINOV2_EMBED_DIMS.keys())}"
        )
    embed_dim: int = _DINOV2_EMBED_DIMS[model_name]

    backbone: nn.Module
    if weights_dir is not None and Path(weights_dir).is_dir():
        backbone = _load_from_local_directory(weights_dir, model_name)
    elif local_archive is not None and Path(local_archive).is_file():
        backbone = _load_from_local_archive(local_archive, model_name)
    else:
        logger.info("Loading DINOv2 '%s' via torch.hub...", model_name)
        backbone = torch.hub.load(
            "facebookresearch/dinov2", model_name, pretrained=True,
        )
        backbone.eval()

    if freeze:
        for param in backbone.parameters():
            param.requires_grad = False
        logger.info("DINOv2 backbone frozen (%s).", model_name)

    total_params: int = sum(p.numel() for p in backbone.parameters())
    trainable: int = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    logger.info(
        "DINOv2 '%s' ready | embed_dim=%d | params=%s (trainable=%s)",
        model_name, embed_dim, f"{total_params:,}", f"{trainable:,}",
    )
    return backbone, embed_dim


# =========================================================================
# 2. Temporal Aggregation Modules
# =========================================================================

class TemporalConv1DBlock(nn.Module):
    """1D temporal convolution for slice sequence aggregation."""

    def __init__(self, embed_dim: int, num_slices: int) -> None:
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Input ``[B, S, D]`` -> Output ``[B, D]``."""
        x = x.transpose(1, 2)  # [B, D, S]
        x = self.conv_block(x)
        x = self.pool(x).squeeze(-1)  # [B, D]
        return x


class TemporalMultiHeadAttention(nn.Module):
    """Multi-head self-attention pooling for slice sequence aggregation."""

    def __init__(
        self,
        embed_dim: int,
        num_slices: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.agg_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, num_slices + 1, embed_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Input ``[B, S, D]`` -> Output ``[B, D]``."""
        B: int = x.shape[0]
        agg_tokens: torch.Tensor = self.agg_token.expand(B, -1, -1)
        x = torch.cat([agg_tokens, x], dim=1)  # [B, S+1, D]
        x = x + self.pos_embed[:, : x.shape[1], :]
        x = self.transformer(x)
        x = self.norm(x[:, 0, :])  # [B, D]
        return x


# =========================================================================
# 3. Volumetric 2.5D DINOv2 Classifier Model
# =========================================================================

class RSNADINOv2Model(nn.Module):
    """Volumetric 2.5D multi-label classifier powered by Meta's DINOv2 backbone.

    Accepts 5D or 4D tensors:
    - ``[Batch_Size, Slices=32, Channels=3, Height=224, Width=224]``
    - ``[Batch_Size, Slices=32, Height=224, Width=224]`` (auto-broadcasts 1 -> 3 RGB channels)

    Parameters
    ----------
    model_name : str
        DINOv2 architecture variant (``dinov2_vits14`` or ``dinov2_vitb14``).
    num_classes : int
        Number of output binary target logits (default 12).
    num_slices : int
        Fixed number of 2D slices per 3D scan (default 32).
    aggregator : {"attention", "conv1d"}
        Temporal aggregation method across the slice dimension.
    weights_dir : Optional[str]
        Directory path to unpacked local model weights (e.g. ``/content/models_cache/``).
    local_archive : Optional[str]
        Path to local ``.tar.gz`` weights archive file.
    freeze_backbone : bool
        Whether to freeze DINOv2 backbone weights during training.
    head_dropout : float
        Dropout probability before the classification projection layer.
    """

    def __init__(
        self,
        model_name: str = "dinov2_vits14",
        num_classes: int = 12,
        num_slices: int = 32,
        aggregator: Literal["attention", "conv1d"] = "attention",
        weights_dir: Optional[str] = None,
        local_archive: Optional[str] = None,
        freeze_backbone: bool = True,
        head_dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.model_name: str = model_name
        self.num_classes: int = num_classes
        self.num_slices: int = num_slices

        # -- Load DINOv2 backbone ------------------------------------------
        self.backbone: nn.Module
        self.embed_dim: int
        self.backbone, self.embed_dim = load_dinov2_backbone(
            model_name=model_name,
            weights_dir=weights_dir,
            local_archive=local_archive,
            freeze=freeze_backbone,
        )

        # -- Temporal Sequence Aggregator ----------------------------------
        if aggregator == "attention":
            self.aggregator: nn.Module = TemporalMultiHeadAttention(
                embed_dim=self.embed_dim,
                num_slices=num_slices,
                num_heads=max(1, self.embed_dim // 96),
                dropout=0.1,
            )
        elif aggregator == "conv1d":
            self.aggregator = TemporalConv1DBlock(
                embed_dim=self.embed_dim,
                num_slices=num_slices,
            )
        else:
            raise ValueError(f"Unknown aggregator '{aggregator}'. Use 'attention' or 'conv1d'.")

        # -- Classification Projection Head --------------------------------
        self.head = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Dropout(head_dropout),
            nn.Linear(self.embed_dim, num_classes),
        )

        # Logging summary
        backbone_params: int = sum(p.numel() for p in self.backbone.parameters())
        head_params: int = (
            sum(p.numel() for p in self.aggregator.parameters())
            + sum(p.numel() for p in self.head.parameters())
        )
        trainable: int = sum(p.numel() for p in self.parameters() if p.requires_grad)

        logger.info(
            "RSNADINOv2Model | backbone=%s (%s params) | "
            "head+aggregator=%s params | trainable=%s",
            model_name,
            f"{backbone_params:,}",
            f"{head_params:,}",
            f"{trainable:,}",
        )

    def extract_slice_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract CLS token embeddings for a flat batch of 2D slices [N, 3, H, W]."""
        with torch.set_grad_enabled(any(p.requires_grad for p in self.backbone.parameters())):
            features: torch.Tensor = self.backbone(x)
        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass accepting 5D [B, S, 3, H, W] or 4D [B, S, H, W] inputs.

        Parameters
        ----------
        x : torch.Tensor
            Input volumetric tensor of shape:
            - ``[B, S, 3, H, W]``
            - ``[B, S, 1, H, W]``
            - ``[B, S, H, W]``

        Returns
        -------
        torch.Tensor
            Logits of shape ``[B, num_classes=12]``.
        """
        # Auto-reshape 4D [B, S, H, W] -> 5D [B, S, 3, H, W]
        if x.ndim == 4:
            x = x.unsqueeze(2).repeat(1, 1, 3, 1, 1)
        elif x.ndim == 5 and x.shape[2] == 1:
            x = x.repeat(1, 1, 3, 1, 1)

        B, S, C, H, W = x.shape

        # 1. Flatten slices into batch dimension: [B*S, 3, H, W]
        x_flat: torch.Tensor = x.reshape(B * S, C, H, W)

        # 2. Extract per-slice DINOv2 embeddings: [B*S, embed_dim]
        slice_features: torch.Tensor = self.extract_slice_features(x_flat)

        # 3. Reshape to sequence: [B, S, embed_dim]
        slice_features = slice_features.reshape(B, S, self.embed_dim)

        # 4. Temporal aggregation: [B, embed_dim]
        volume_repr: torch.Tensor = self.aggregator(slice_features)

        # 5. Classification head projection: [B, 12]
        logits: torch.Tensor = self.head(volume_repr)
        return logits


# Alias for backward compatibility
RSNADINOv2KneeModel = RSNADINOv2Model


# =========================================================================
# 4. Verification & Smoke Test
# =========================================================================

def main() -> None:
    """Verify RSNADINOv2Model on 5D and 4D dummy inputs."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")

    print("\n" + "=" * 70)
    print("  DINOv2 2.5D Knee Model -- Verification Test")
    print("=" * 70)

    model_name: str = "dinov2_vits14"
    num_classes: int = 12
    num_slices: int = 32
    batch_size: int = 2

    # Instantiate model
    print(f"\n  Instantiating RSNADINOv2Model (backbone={model_name})...")
    model = RSNADINOv2Model(
        model_name=model_name,
        num_classes=num_classes,
        num_slices=num_slices,
        aggregator="attention",
        freeze_backbone=True,
    )

    device: torch.device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = model.to(device)
    model.eval()
    print(f"  Execution Device: {device}")

    # Test 1: 5D Input Tensor [B, S, C=3, H=224, W=224]
    input_5d = torch.randn(batch_size, num_slices, 3, 224, 224, device=device)
    print(f"\n  [Test 1] 5D Input Shape: {list(input_5d.shape)}")
    with torch.no_grad():
        out_5d = model(input_5d)
    print(f"  [Test 1] Output Shape:  {list(out_5d.shape)}")
    assert out_5d.shape == (batch_size, num_classes), f"Test 1 failed! Got {out_5d.shape}"
    print("  [PASS] Test 1: 5D input shape verified!")

    # Test 2: 4D Input Tensor [B, S, H=224, W=224]
    input_4d = torch.randn(batch_size, num_slices, 224, 224, device=device)
    print(f"\n  [Test 2] 4D Input Shape: {list(input_4d.shape)}")
    with torch.no_grad():
        out_4d = model(input_4d)
    print(f"  [Test 2] Output Shape:  {list(out_4d.shape)}")
    assert out_4d.shape == (batch_size, num_classes), f"Test 2 failed! Got {out_4d.shape}"
    print("  [PASS] Test 2: 4D input shape verified!")

    print("\n" + "=" * 70)
    print("  ALL VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
