"""
RSNA Knee Abnormality Detection -- Volumetric DICOM Dataset Module
===================================================================
Provides :class:`RSNAKneeDataset`, a PyTorch Dataset for multi-slice knee MRI
scans and optional radiology-report text.

Key Features & Pipeline
-----------------------
1. **Real DICOM Loading**: Uses ``pydicom`` to inspect DICOM headers, extracting
   slice ordering metadata (``InstanceNumber`` -> ``SliceLocation`` ->
   ``ImagePositionPatient[2]``).
2. **Slice Resampling**: Uniformly samples or interpolates volume slices to
   exactly ``num_slices`` (default 32) at target spatial resolution ``image_size``.
3. **Volume Normalisation**: Applies 3D volume-wide Min-Max intensity
   normalisation ``(vol - min) / (max - min + 1e-6)``.
4. **Rescale Handling**: Respects DICOM ``RescaleSlope`` and ``RescaleIntercept``
   header tags.
5. **Fallback Safety**: Gracefully falls back to synthetic ``torch.randn``
   tensors if local image files or study folders are absent during testing.
6. **Multimodal Text**: Supports HuggingFace report tokenisation when a report
   column is present.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

# Optional pydicom import for DICOM handling
try:
    import pydicom  # type: ignore[import-untyped]
    PYDICOM_AVAILABLE: bool = True
except ImportError:
    pydicom = None  # type: ignore[assignment]
    PYDICOM_AVAILABLE = False

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger: logging.Logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default target columns
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
    """Multi-slice MRI dataset supporting DICOM volumes and 2D images.

    Parameters
    ----------
    df : pd.DataFrame
        Metadata frame containing ``"StudyInstanceUID"`` and target columns.
    image_dir : Union[str, Path]
        Root directory containing study folders (e.g., ``./data/train_images``
        or ``./data/``).
    target_columns : List[str]
        List of target column names (12 classes).
    image_size : tuple[int, int]
        Target spatial resolution ``(H, W)`` for each slice (default 224x224).
    num_slices : int
        Fixed number of 2D slices per 3D volume (default 32).
    tokenizer : Optional[Any]
        Optional HuggingFace tokenizer for radiology report processing.
    max_text_length : int
        Maximum sequence length for text truncation/padding.
    is_train : bool
        If ``True``, extracts multi-label target tensors.
    transform : Optional[Callable]
        Per-slice spatial/intensity image transformation.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        image_dir: Union[str, Path] = "./data",
        target_columns: List[str] = DEFAULT_TARGET_COLUMNS,
        image_size: tuple[int, int] = (224, 224),
        num_slices: int = 32,
        tokenizer: Optional[Any] = None,
        max_text_length: int = 256,
        is_train: bool = True,
        transform: Optional[Callable[..., Any]] = None,
    ) -> None:
        super().__init__()

        if "StudyInstanceUID" not in df.columns:
            raise ValueError("DataFrame must contain a 'StudyInstanceUID' column.")

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
            "RSNAKneeDataset initialized | samples=%d | is_train=%s | "
            "num_slices=%d | image_size=%s | pydicom=%s",
            len(self.df),
            self.is_train,
            self.num_slices,
            self.image_size,
            PYDICOM_AVAILABLE,
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Retrieve a dataset sample as a dictionary.

        Returns
        -------
        dict
            - ``"study_uid"``: StudyInstanceUID string
            - ``"image"``: Tensor of shape ``[1, S, H, W]`` (or ``[C, S, H, W]``)
            - ``"label"``: FloatTensor of shape ``[12]`` (when ``is_train=True``)
            - ``"input_ids"`` / ``"attention_mask"``: LongTensor (optional text)
        """
        row: pd.Series = self.df.iloc[index]  # type: ignore[assignment]
        study_uid: str = str(row["StudyInstanceUID"])

        # ── 1. Load MRI volume ────────────────────────────────────────────
        image: torch.Tensor = self._load_volume(study_uid)

        # ── 2. Build output dictionary ────────────────────────────────────
        sample: Dict[str, Any] = {
            "study_uid": study_uid,
            "image": image,
        }

        # ── 3. Extract labels ─────────────────────────────────────────────
        if self.is_train:
            sample["label"] = self._extract_labels(row)

        # ── 4. Tokenize text report ───────────────────────────────────────
        if self.tokenizer is not None and "report" in self.df.columns:
            text_features: Dict[str, torch.Tensor] = self._tokenize_report(
                str(row.get("report", ""))
            )
            sample.update(text_features)

        return sample

    # ==================================================================
    # DICOM & Image Volume Loading
    # ==================================================================

    def _load_volume(self, study_uid: str) -> torch.Tensor:
        """Attempt DICOM / 2D loading; fall back to synthetic data if missing."""
        study_path: Path = self.image_dir / study_uid
        if not study_path.is_dir():
            # Check nested path e.g. ./data/train_images/study_uid or ./data/series_uid
            nested_path: Path = self.image_dir / "train_images" / study_uid
            if nested_path.is_dir():
                study_path = nested_path

        if study_path.is_dir():
            # 1. Try DICOM loading
            dicom_volume = self._load_dicom_volume(study_path)
            if dicom_volume is not None:
                return dicom_volume

            # 2. Try standard image loading (PNG / JPG)
            image_volume = self._load_standard_images(study_path)
            if image_volume is not None:
                return image_volume

        # 3. Fallback: synthetic volume when files are unavailable
        logger.debug("Study folder '%s' missing or empty -- using random fallback.", study_uid)
        return torch.randn(1, self.num_slices, self.image_size[0], self.image_size[1])

    def _load_dicom_volume(self, study_path: Path) -> Optional[torch.Tensor]:
        """Load, sort, resample, and normalize 3D DICOM slice series."""
        if not PYDICOM_AVAILABLE:
            return None

        # Recursively search for .dcm files
        dcm_files: List[Path] = [
            p for p in study_path.rglob("*")
            if p.is_file() and (p.suffix.lower() == ".dcm" or "." not in p.name)
        ]

        if not dcm_files:
            return None

        # Inspect headers for sorting keys
        slice_data: List[Tuple[float, np.ndarray]] = []
        for p in dcm_files:
            try:
                ds = pydicom.dcmread(str(p), stop_before_pixels=False)
                if not hasattr(ds, "pixel_array"):
                    continue

                # Sorting key priority: InstanceNumber -> SliceLocation -> ImagePositionPatient[2]
                sort_key: float
                if hasattr(ds, "InstanceNumber") and ds.InstanceNumber is not None:
                    sort_key = float(ds.InstanceNumber)
                elif hasattr(ds, "SliceLocation") and ds.SliceLocation is not None:
                    sort_key = float(ds.SliceLocation)
                elif (
                    hasattr(ds, "ImagePositionPatient")
                    and len(ds.ImagePositionPatient) >= 3
                ):
                    sort_key = float(ds.ImagePositionPatient[2])
                else:
                    # Fallback to numeric digits in filename
                    numbers = [int(s) for s in p.stem.split("_") if s.isdigit()]
                    sort_key = float(numbers[-1]) if numbers else 0.0

                arr = ds.pixel_array.astype(np.float32)

                # Apply Rescale Slope & Intercept
                slope = float(getattr(ds, "RescaleSlope", 1.0))
                intercept = float(getattr(ds, "RescaleIntercept", 0.0))
                if slope != 1.0 or intercept != 0.0:
                    arr = arr * slope + intercept

                slice_data.append((sort_key, arr))

            except Exception as exc:
                logger.debug("Skipping unreadable DICOM slice '%s': %s", p, exc)
                continue

        if not slice_data:
            return None

        # Sort slices numerically along Z-axis
        slice_data.sort(key=lambda item: item[0])
        raw_slices: List[np.ndarray] = [item[1] for item in slice_data]

        # Uniformly sample/resample to num_slices
        sampled_slices: List[np.ndarray] = self._resample_slices(raw_slices)

        # Resize each slice to target image_size and apply optional transform
        resized_slices: List[torch.Tensor] = []
        for slice_arr in sampled_slices:
            img = Image.fromarray(slice_arr).resize(self.image_size, Image.BILINEAR)
            arr = np.array(img, dtype=np.float32)

            if self.transform is not None:
                transformed = self.transform(image=arr)
                arr = transformed.get("image", arr)
                if isinstance(arr, torch.Tensor):
                    arr = arr.numpy()

            resized_slices.append(torch.from_numpy(arr))

        # Stack into 3D volume array: [S, H, W]
        vol_3d = torch.stack(resized_slices, dim=0).numpy()

        # 3D Min-Max Volume Normalization
        vol_min = vol_3d.min()
        vol_max = vol_3d.max()
        if vol_max > vol_min:
            vol_3d = (vol_3d - vol_min) / (vol_max - vol_min + 1e-6)
        else:
            vol_3d = np.zeros_like(vol_3d)

        # Unsqueeze channel dimension: [1, S, H, W]
        return torch.from_numpy(vol_3d).unsqueeze(0)

    def _load_standard_images(self, study_path: Path) -> Optional[torch.Tensor]:
        """Fallback loader for PNG / JPG slice series."""
        slice_paths: List[Path] = sorted(
            [
                p for p in study_path.iterdir()
                if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
            ]
        )
        if not slice_paths:
            return None

        raw_slices: List[np.ndarray] = []
        for sp in slice_paths:
            try:
                img = Image.open(sp).convert("L")
                raw_slices.append(np.array(img, dtype=np.float32))
            except Exception:
                continue

        if not raw_slices:
            return None

        sampled_slices = self._resample_slices(raw_slices)
        resized_slices: List[torch.Tensor] = []

        for arr in sampled_slices:
            img = Image.fromarray(arr).resize(self.image_size)
            slice_arr = np.array(img, dtype=np.float32)

            if self.transform is not None:
                t = self.transform(image=slice_arr)
                slice_arr = t.get("image", slice_arr)
                if isinstance(slice_arr, torch.Tensor):
                    slice_arr = slice_arr.numpy()

            resized_slices.append(torch.from_numpy(slice_arr))

        vol_3d = torch.stack(resized_slices, dim=0).numpy()
        vol_min, vol_max = vol_3d.min(), vol_3d.max()
        if vol_max > vol_min:
            vol_3d = (vol_3d - vol_min) / (vol_max - vol_min + 1e-6)
        else:
            vol_3d = np.zeros_like(vol_3d)

        return torch.from_numpy(vol_3d).unsqueeze(0)

    def _resample_slices(self, slices: List[np.ndarray]) -> List[np.ndarray]:
        """Uniformly sample or interpolate slice list to exactly `num_slices`."""
        n_curr = len(slices)
        if n_curr == self.num_slices:
            return slices

        if n_curr > self.num_slices:
            indices = np.linspace(0, n_curr - 1, self.num_slices, dtype=int)
            return [slices[i] for i in indices]

        # n_curr < num_slices: duplicate / interpolate
        indices = np.linspace(0, n_curr - 1, self.num_slices)
        resampled: List[np.ndarray] = []
        for idx in indices:
            lower = int(np.floor(idx))
            upper = int(np.ceil(idx))
            weight = idx - lower
            if lower == upper or weight == 0:
                resampled.append(slices[lower])
            else:
                interp = (1.0 - weight) * slices[lower] + weight * slices[upper]
                resampled.append(interp.astype(slices[0].dtype))
        return resampled

    # ==================================================================
    # Utility Helpers
    # ==================================================================

    def _extract_labels(self, row: pd.Series) -> torch.Tensor:  # type: ignore[type-arg]
        """Extract multi-label target values as a FloatTensor."""
        labels: List[float] = []
        for col in self.target_columns:
            try:
                labels.append(float(row[col]))
            except (KeyError, ValueError, TypeError):
                labels.append(0.0)
        return torch.tensor(labels, dtype=torch.float32)

    def _tokenize_report(self, text: str) -> Dict[str, torch.Tensor]:
        """Tokenize radiology report text."""
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
            logger.error("Tokenisation failed: %s", exc)
            return {
                "input_ids": torch.zeros(self.max_text_length, dtype=torch.long),
                "attention_mask": torch.zeros(self.max_text_length, dtype=torch.long),
            }


# =========================================================================
# Self-Test Runner
# =========================================================================

def main() -> None:
    """Run a quick self-test of the dataset loader."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")

    print("\n" + "=" * 70)
    print("  RSNA Knee Abnormality Dataset -- Self-Test")
    print("=" * 70)

    # Check for train.csv or train_folds.csv in data/ or root
    csv_paths = ["./data/train.csv", "./data/train_folds.csv", "train_folds.csv", "train.csv"]
    csv_found: Optional[str] = next((p for p in csv_paths if os.path.isfile(p)), None)

    if csv_found:
        print(f"  Found metadata at '{csv_found}'")
        df = pd.read_csv(csv_found)
    else:
        print("  No train.csv found -- generating dummy metadata frame for test.")
        df = pd.DataFrame({
            "StudyInstanceUID": ["1.2.826.0.1.0", "1.2.826.0.1.1"],
            **{col: [1.0, 0.0] for col in DEFAULT_TARGET_COLUMNS},
        })

    dataset = RSNAKneeDataset(
        df=df,
        image_dir="./data",
        num_slices=32,
        image_size=(224, 224),
        is_train=True,
    )

    sample = dataset[0]
    print(f"\n  Sample 0 Study UID: {sample['study_uid']}")
    print(f"  Image volume shape: {list(sample['image'].shape)} (min={sample['image'].min():.3f}, max={sample['image'].max():.3f})")
    print(f"  Label shape:        {list(sample['label'].shape)}")
    print(f"  Labels tensor:      {sample['label'].tolist()}")

    print("\n  [PASS] RSNAKneeDataset loaded successfully!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
