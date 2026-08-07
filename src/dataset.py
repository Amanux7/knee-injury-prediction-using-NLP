"""
RSNA Knee Abnormality Detection — Dataset Module
=================================================
Provides :class:`RSNAKneeDataset`, a PyTorch Dataset that loads multi‑slice MRI
volumes, optional radiology‑report text, and multi‑label targets for the 12
abnormality classes defined in the competition.

Key design decisions
--------------------
* **Volumetric loading**: consecutive 2‑D DICOM/PNG slices are stacked into a
  single 4‑D tensor ``[C, S, H, W]`` (channels‑first).  If slices are missing
  or the run is in test mode without local images, synthetic random data is
  returned so that downstream code never crashes.
* **Text branch**: if a HuggingFace tokenizer *and* a ``"report"`` column are
  present, tokenised ``input_ids`` / ``attention_mask`` tensors are included in
  the returned dictionary.
* **Strict typing**: every public method and constructor carries full type
  annotations so that ``mypy --strict`` passes cleanly.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Module‑level logger
# ---------------------------------------------------------------------------
logger: logging.Logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default target columns (kept here for import convenience)
# ---------------------------------------------------------------------------
DEFAULT_TARGET_COLUMNS: List[str] = [
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


class RSNAKneeDataset(Dataset):  # type: ignore[type-arg]
    """Multi‑slice MRI dataset for knee‑abnormality classification.

    Parameters
    ----------
    df : pd.DataFrame
        Metadata frame.  Must contain ``"StudyInstanceUID"``; may also contain
        the 12 target columns and a ``"report"`` column.
    image_dir : Union[str, Path]
        Root directory where study images are stored (one sub‑folder per UID).
    target_columns : List[str]
        Ordered list of the 12 target column names.
    image_size : tuple[int, int]
        Spatial resolution ``(H, W)`` to which every slice is resized.
    num_slices : int
        Fixed number of slices per volume.  Volumes with fewer slices are
        zero‑padded; longer ones are centre‑cropped.
    tokenizer : Optional[Any]
        A HuggingFace ``PreTrainedTokenizerBase`` (or compatible) instance.
        When *None* no text features are generated.
    max_text_length : int
        Maximum token length for the tokenizer truncation / padding.
    is_train : bool
        Whether this dataset is used during training.  Controls label loading.
    transform : Optional[Callable]
        An image‑level transform (Albumentations ``Compose`` or
        ``torchvision.transforms``).  Applied independently to every 2‑D slice.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        image_dir: Union[str, Path],
        target_columns: List[str] = DEFAULT_TARGET_COLUMNS,
        image_size: tuple[int, int] = (224, 224),
        num_slices: int = 32,
        tokenizer: Optional[Any] = None,
        max_text_length: int = 256,
        is_train: bool = True,
        transform: Optional[Callable[..., Any]] = None,
    ) -> None:
        super().__init__()

        # ── Validation ────────────────────────────────────────────────────
        if "StudyInstanceUID" not in df.columns:
            raise ValueError(
                "DataFrame must contain a 'StudyInstanceUID' column."
            )

        self.df: pd.DataFrame = df.reset_index(drop=True)
        self.image_dir: Path = Path(image_dir)
        self.target_columns: List[str] = target_columns
        self.image_size: tuple[int, int] = image_size
        self.num_slices: int = num_slices
        self.tokenizer: Optional[Any] = tokenizer
        self.max_text_length: int = max_text_length
        self.is_train: bool = is_train
        self.transform: Optional[Callable[..., Any]] = transform

        logger.info(
            "RSNAKneeDataset initialised | samples=%d  is_train=%s  "
            "num_slices=%d  image_size=%s",
            len(self.df),
            self.is_train,
            self.num_slices,
            self.image_size,
        )

    # ------------------------------------------------------------------
    # Length
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.df)

    # ------------------------------------------------------------------
    # Item retrieval
    # ------------------------------------------------------------------
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Return a single sample as a dictionary.

        Returns
        -------
        dict
            Keys always present:
                ``"study_uid"`` — study identifier (str)
                ``"image"``     — ``[C, S, H, W]`` float tensor
            Conditional keys:
                ``"label"``          — ``[num_classes]`` float tensor (train)
                ``"input_ids"``      — ``[max_text_length]`` long tensor (text)
                ``"attention_mask"`` — ``[max_text_length]`` long tensor (text)
        """
        row: pd.Series = self.df.iloc[index]  # type: ignore[assignment]
        study_uid: str = str(row["StudyInstanceUID"])

        # ── 1. Volumetric image ───────────────────────────────────────────
        image: torch.Tensor = self._load_volume(study_uid)

        # ── 2. Build output dictionary ────────────────────────────────────
        sample: Dict[str, Any] = {
            "study_uid": study_uid,
            "image": image,
        }

        # ── 3. Labels (training mode only) ────────────────────────────────
        if self.is_train:
            sample["label"] = self._extract_labels(row)

        # ── 4. Text tokenisation (optional) ───────────────────────────────
        if self.tokenizer is not None and "report" in self.df.columns:
            text_features: Dict[str, torch.Tensor] = self._tokenize_report(
                str(row.get("report", ""))
            )
            sample.update(text_features)

        return sample

    # ==================================================================
    # Private helpers
    # ==================================================================

    def _load_volume(self, study_uid: str) -> torch.Tensor:
        """Load *num_slices* 2‑D images and stack into ``[1, S, H, W]``.

        Falls back to random noise when images are unavailable so that
        unit tests and offline development never crash.
        """
        study_path: Path = self.image_dir / study_uid
        slices: List[torch.Tensor] = []

        if study_path.is_dir():
            # Collect and sort slice file paths (PNG / JPG / DICOM)
            slice_paths: List[Path] = sorted(
                [
                    p
                    for p in study_path.iterdir()
                    if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".dcm"}
                ]
            )
            for sp in slice_paths[: self.num_slices]:
                try:
                    img: Image.Image = (
                        Image.open(sp).convert("L").resize(self.image_size)
                    )
                    arr: np.ndarray = np.array(img, dtype=np.float32) / 255.0

                    # Apply optional per‑slice transform
                    if self.transform is not None:
                        transformed: Any = self.transform(image=arr)
                        arr = transformed.get("image", arr)  # Albumentations
                        if isinstance(arr, torch.Tensor):
                            arr = arr.numpy()

                    slices.append(torch.from_numpy(arr))
                except Exception as exc:
                    logger.warning(
                        "Failed to load slice '%s': %s — using random fallback",
                        sp,
                        exc,
                    )
                    slices.append(
                        torch.randn(self.image_size[0], self.image_size[1])
                    )
        else:
            # ── Fallback: synthetic random volume ─────────────────────────
            logger.debug(
                "Study directory not found for '%s'; generating mock volume.",
                study_uid,
            )

        # Pad / trim to exactly *num_slices*
        while len(slices) < self.num_slices:
            slices.append(
                torch.randn(self.image_size[0], self.image_size[1])
            )
        slices = slices[: self.num_slices]

        # Stack → [S, H, W] then unsqueeze channel → [1, S, H, W]
        volume: torch.Tensor = torch.stack(slices, dim=0).unsqueeze(0)
        return volume

    def _extract_labels(self, row: pd.Series) -> torch.Tensor:  # type: ignore[type-arg]
        """Extract target columns into a ``[num_classes]`` float tensor.

        Missing columns are filled with ``0.0`` so the dataset never crashes
        when a target column is absent (useful during debugging).
        """
        labels: List[float] = []
        for col in self.target_columns:
            try:
                labels.append(float(row[col]))
            except (KeyError, ValueError, TypeError):
                labels.append(0.0)
        return torch.tensor(labels, dtype=torch.float32)

    def _tokenize_report(self, text: str) -> Dict[str, torch.Tensor]:
        """Tokenise a single radiology report string.

        Returns
        -------
        dict
            ``"input_ids"`` and ``"attention_mask"`` as ``torch.LongTensor``.
        """
        if not text or text.lower() in {"nan", "none", ""}:
            text = "[NO REPORT]"

        try:
            encoding: Dict[str, Any] = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.max_text_length,
                return_tensors="pt",
            )
            return {
                "input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0),
            }
        except Exception as exc:
            logger.error("Tokenisation failed for text='%s…': %s", text[:60], exc)
            return {
                "input_ids": torch.zeros(self.max_text_length, dtype=torch.long),
                "attention_mask": torch.zeros(
                    self.max_text_length, dtype=torch.long
                ),
            }
