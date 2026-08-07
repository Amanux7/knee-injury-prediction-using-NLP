"""
RSNA Knee Abnormality Detection -- DINOv2 Vision Backbone & 2.5D Model
======================================================================
Provides :class:`RSNADINOv2KneeModel`, a volumetric 2.5D classifier that
processes multi-slice MRI volumes through Meta's DINOv2 vision transformer
and aggregates slice embeddings with either temporal convolution or
multi-head self-attention before a multi-label classification head.

Backbone loading priority
-------------------------
1. **Local offline archive** (``*.tar.gz``) -- for Kaggle kernels with no
   internet.  The archive is expected to contain a ``state_dict`` loadable
   by a ``torch.hub``-compatible DINOv2 model.
2. **``torch.hub.load``** -- pulls weights from the ``facebookresearch/dinov2``
   GitHub repository (requires internet on first run; cached afterwards).

Architecture overview
---------------------
::

    Input: [B, S=32, C=3, H=224, W=224]
           |
    (reshape to [B*S, C, H, W])
           |
    DINOv2 ViT backbone --> CLS token per slice: [B*S, D]
           |
    (reshape to [B, S, D])
           |
    Temporal Aggregator (1D Conv  OR  Multi-Head Attention)
           |
    Pooled representation: [B, D]
           |
    Classification Head --> 12 logits: [B, 12]
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
# 1. DINOv2 Backbone Loader
# =========================================================================

def _load_from_local_archive(
    archive_path: Union[str, Path],
    model_name: str,
) -> nn.Module:
    """Load a DINOv2 backbone from a local ``*.tar.gz`` weight archive.

    The archive is expected to contain either:
    - A single ``.pth`` / ``.pt`` file with the full state dict, or
    - A directory with a ``state_dict.pth`` inside.

    We first instantiate a *random-weight* model via ``torch.hub`` (which
    caches the architecture code locally after the first online call),
    then overwrite with the archived weights.

    Parameters
    ----------
    archive_path : Union[str, Path]
        Path to ``dinov2-pytorch-small-v1.tar.gz`` or similar.
    model_name : str
        One of ``dinov2_vits14``, ``dinov2_vitb14``, etc.

    Returns
    -------
    nn.Module
        DINOv2 backbone with loaded weights, set to eval mode.
    """
    archive_path = Path(archive_path)
    if not archive_path.is_file():
        raise FileNotFoundError(f"Archive not found: {archive_path}")

    logger.info("Extracting DINOv2 weights from '%s'...", archive_path)

    # -- Extract archive to a temp directory --------------------------------
    with tempfile.TemporaryDirectory() as tmp_dir:
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(tmp_dir)

        # Locate the .pth / .pt file inside the extracted tree
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
                f"No .pth/.pt/.bin weight file found inside '{archive_path}'."
            )

        logger.info("Found weight file: '%s'", weight_file.name)
        state_dict: Dict[str, Any] = torch.load(
            weight_file, map_location="cpu", weights_only=True,
        )

    # -- Build architecture skeleton and load weights ----------------------
    # torch.hub caches the repo code after first download; subsequent calls
    # are offline.  We use pretrained=False to skip weight download.
    model: nn.Module = torch.hub.load(
        "facebookresearch/dinov2", model_name, pretrained=False,
    )

    # Handle nested state dicts (some archives wrap in {"model": ...})
    if "model" in state_dict:
        state_dict = state_dict["model"]
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    logger.info(
        "Loaded DINOv2 '%s' from local archive (%d parameters).",
        model_name,
        sum(p.numel() for p in model.parameters()),
    )
    return model


def load_dinov2_backbone(
    model_name: str = "dinov2_vits14",
    local_archive: Optional[Union[str, Path]] = None,
    freeze: bool = True,
) -> Tuple[nn.Module, int]:
    """Load a DINOv2 backbone and return it with its embedding dimension.

    Parameters
    ----------
    model_name : str
        ``dinov2_vits14`` (384-d) or ``dinov2_vitb14`` (768-d).
    local_archive : Optional[Union[str, Path]]
        Path to a local ``.tar.gz`` weight file.  When provided, weights are
        loaded offline; otherwise ``torch.hub`` downloads them.
    freeze : bool
        If ``True``, all backbone parameters are frozen (no gradients).

    Returns
    -------
    (backbone, embed_dim) : Tuple[nn.Module, int]
    """
    if model_name not in _DINOV2_EMBED_DIMS:
        raise ValueError(
            f"Unknown DINOv2 model '{model_name}'. "
            f"Choose from: {list(_DINOV2_EMBED_DIMS.keys())}"
        )
    embed_dim: int = _DINOV2_EMBED_DIMS[model_name]

    # -- Load weights ------------------------------------------------------
    if local_archive is not None:
        backbone: nn.Module = _load_from_local_archive(local_archive, model_name)
    else:
        logger.info("Loading DINOv2 '%s' via torch.hub...", model_name)
        backbone = torch.hub.load(
            "facebookresearch/dinov2", model_name, pretrained=True,
        )
        backbone.eval()

    # -- Freeze if requested -----------------------------------------------
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
    """1-D temporal convolution over slice embeddings.

    Applies a stack of causal-style 1D convolutions across the slice
    dimension to capture local inter-slice dependencies, followed by
    global average pooling to produce a single [B, D] vector.
    """

    def __init__(self, embed_dim: int, num_slices: int) -> None:
        super().__init__()
        self.conv_block = nn.Sequential(
            # [B, D, S] --> [B, D, S]
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)  # [B, D, S] --> [B, D, 1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape ``[B, S, D]`` -- sequence of slice embeddings.

        Returns
        -------
        torch.Tensor
            Shape ``[B, D]`` -- pooled representation.
        """
        # Transpose to channels-first for Conv1d: [B, S, D] --> [B, D, S]
        x = x.transpose(1, 2)
        x = self.conv_block(x)
        x = self.pool(x).squeeze(-1)  # [B, D]
        return x


class TemporalMultiHeadAttention(nn.Module):
    """Multi-head self-attention over slice embeddings.

    Adds a learnable ``[AGG]`` token prepended to the slice sequence.
    After self-attention, the ``[AGG]`` token's output serves as the
    aggregated volumetric representation.
    """

    def __init__(
        self,
        embed_dim: int,
        num_slices: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Learnable aggregation token (like a CLS token for the slice sequence)
        self.agg_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # Positional embedding for (1 agg_token + num_slices) positions
        self.pos_embed = nn.Parameter(
            torch.randn(1, num_slices + 1, embed_dim) * 0.02
        )

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
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape ``[B, S, D]`` -- sequence of slice embeddings.

        Returns
        -------
        torch.Tensor
            Shape ``[B, D]`` -- aggregated representation.
        """
        B: int = x.shape[0]

        # Prepend learnable aggregation token
        agg_tokens: torch.Tensor = self.agg_token.expand(B, -1, -1)
        x = torch.cat([agg_tokens, x], dim=1)  # [B, S+1, D]

        # Add positional embeddings
        x = x + self.pos_embed[:, : x.shape[1], :]

        # Self-attention
        x = self.transformer(x)

        # Extract the aggregation token's output
        x = self.norm(x[:, 0, :])  # [B, D]
        return x


# =========================================================================
# 3. Full 2.5D DINOv2 Knee Model
# =========================================================================

class RSNADINOv2KneeModel(nn.Module):
    """Volumetric 2.5D multi-label classifier using DINOv2 backbone.

    Architecture
    ------------
    1. Each 2D slice is independently passed through a frozen (or fine-tuned)
       DINOv2 ViT to extract CLS-token embeddings.
    2. The resulting sequence of slice embeddings is aggregated via either
       temporal 1D convolution or multi-head self-attention.
    3. A linear classification head maps the aggregated vector to 12 logits.

    Parameters
    ----------
    model_name : str
        DINOv2 variant (``dinov2_vits14`` or ``dinov2_vitb14``).
    num_classes : int
        Number of output logits (default 12).
    num_slices : int
        Expected number of slices per volume (default 32).
    aggregator : {"attention", "conv1d"}
        Temporal aggregation strategy.
    local_archive : Optional[str]
        Path to offline weight archive, or ``None`` for hub download.
    freeze_backbone : bool
        Whether to freeze DINOv2 weights (recommended for small datasets).
    head_dropout : float
        Dropout probability before the final linear layer.
    """

    def __init__(
        self,
        model_name: str = "dinov2_vits14",
        num_classes: int = 12,
        num_slices: int = 32,
        aggregator: Literal["attention", "conv1d"] = "attention",
        local_archive: Optional[str] = None,
        freeze_backbone: bool = True,
        head_dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.model_name: str = model_name
        self.num_classes: int = num_classes
        self.num_slices: int = num_slices

        # -- DINOv2 backbone ------------------------------------------------
        self.backbone: nn.Module
        self.embed_dim: int
        self.backbone, self.embed_dim = load_dinov2_backbone(
            model_name=model_name,
            local_archive=local_archive,
            freeze=freeze_backbone,
        )

        # -- Temporal aggregator --------------------------------------------
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
            raise ValueError(
                f"Unknown aggregator '{aggregator}'. Use 'attention' or 'conv1d'."
            )
        logger.info("Temporal aggregator: %s", aggregator)

        # -- Classification head --------------------------------------------
        self.head = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Dropout(head_dropout),
            nn.Linear(self.embed_dim, num_classes),
        )

        # -- Log parameter counts ------------------------------------------
        backbone_params: int = sum(p.numel() for p in self.backbone.parameters())
        head_params: int = (
            sum(p.numel() for p in self.aggregator.parameters())
            + sum(p.numel() for p in self.head.parameters())
        )
        trainable: int = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        logger.info(
            "RSNADINOv2KneeModel | backbone=%s (%s params) | "
            "head+aggregator=%s params | total trainable=%s",
            model_name,
            f"{backbone_params:,}",
            f"{head_params:,}",
            f"{trainable:,}",
        )

    def extract_slice_features(self, x: torch.Tensor) -> torch.Tensor:
        """Pass a batch of 2D images through DINOv2 and return CLS tokens.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``[N, C=3, H, W]`` -- a flat batch of 2D slices.

        Returns
        -------
        torch.Tensor
            Shape ``[N, embed_dim]`` -- CLS token embeddings.
        """
        # DINOv2 forward returns the CLS token by default
        with torch.set_grad_enabled(
            any(p.requires_grad for p in self.backbone.parameters())
        ):
            features: torch.Tensor = self.backbone(x)
        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass: slices -> DINOv2 -> aggregate -> classify.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``[B, S, C, H, W]`` where S=num_slices, C=3, H=W=224.

        Returns
        -------
        torch.Tensor
            Shape ``[B, num_classes]`` -- raw logits (pre-sigmoid).
        """
        B, S, C, H, W = x.shape

        # -- 1. Flatten slices into a mega-batch for the backbone ----------
        x_flat: torch.Tensor = x.reshape(B * S, C, H, W)  # [B*S, 3, 224, 224]

        # -- 2. Extract per-slice CLS embeddings ---------------------------
        slice_features: torch.Tensor = self.extract_slice_features(x_flat)
        # [B*S, embed_dim]

        # -- 3. Reshape back to volumetric sequence ------------------------
        slice_features = slice_features.reshape(B, S, self.embed_dim)
        # [B, S, embed_dim]

        # -- 4. Temporal aggregation ---------------------------------------
        volume_repr: torch.Tensor = self.aggregator(slice_features)
        # [B, embed_dim]

        # -- 5. Classification head ----------------------------------------
        logits: torch.Tensor = self.head(volume_repr)
        # [B, num_classes]

        return logits


# =========================================================================
# 4. Smoke Test
# =========================================================================

def main() -> None:
    """Instantiate the model and verify output shape with a dummy tensor."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )

    # -- Configuration -----------------------------------------------------
    model_name: str = "dinov2_vits14"
    num_classes: int = 12
    num_slices: int = 32
    batch_size: int = 2

    print("\n" + "=" * 70)
    print("  DINOv2 2.5D Knee Model -- Smoke Test")
    print("=" * 70)

    # -- Build model -------------------------------------------------------
    print(f"\n  Building RSNADINOv2KneeModel (backbone={model_name})...")
    model = RSNADINOv2KneeModel(
        model_name=model_name,
        num_classes=num_classes,
        num_slices=num_slices,
        aggregator="attention",
        freeze_backbone=True,
    )

    # Move to best available device
    device: torch.device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model = model.to(device)
    print(f"  Device: {device}")

    # -- Dummy forward pass ------------------------------------------------
    dummy_input: torch.Tensor = torch.randn(
        batch_size, num_slices, 3, 224, 224, device=device,
    )
    print(f"  Input shape:  {list(dummy_input.shape)}")

    model.eval()
    with torch.no_grad():
        output: torch.Tensor = model(dummy_input)

    print(f"  Output shape: {list(output.shape)}")
    print(f"  Output dtype: {output.dtype}")

    # -- Verify ------------------------------------------------------------
    expected_shape: Tuple[int, int] = (batch_size, num_classes)
    assert output.shape == expected_shape, (
        f"Shape mismatch! Expected {expected_shape}, got {tuple(output.shape)}"
    )

    print(f"\n  [PASS] Output shape matches expected {list(expected_shape)}")
    print(f"  Sample logits (row 0): {output[0].cpu().tolist()}")

    # -- Parameter summary -------------------------------------------------
    total: int = sum(p.numel() for p in model.parameters())
    trainable: int = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen: int = total - trainable
    print(f"\n  Parameters:")
    print(f"    Total:     {total:>12,}")
    print(f"    Trainable: {trainable:>12,}")
    print(f"    Frozen:    {frozen:>12,}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
