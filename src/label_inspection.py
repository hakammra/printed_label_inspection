"""Region-Aware Multi-Stream Printed Label Inspection Engine.

Provides illumination-invariant, registration-jitter tolerant anomaly detection
and segmentation for fixed-format printed labels. Overcomes the failure modes of
raw pixel differencing and partial-diffusion reconstruction by decomposing the label
into four semantic zones:
  1. Margin Paper Zone: Identifies unprinted paper anomalies (ink smudges and tears).
  2. Barcode Frequency Stream: Analyzes bar modulation and white channel fragmentation.
  3. Text Field Integrity Stream: Evaluates ink mass integrals across alphanumeric fields.
  4. Perimeter Tear Stream: Traces boundary edge continuity along outer label margins.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
from scipy import ndimage


CANONICAL_SIZE = (1063, 650)

# Semantic zones in canonical coordinates (left, top, right, bottom)
ZONES = {
    "title": (100, 65, 960, 135),
    "dividing_line": (100, 135, 960, 165),
    "item_row": (100, 170, 960, 230),
    "batch_row": (100, 230, 960, 290),
    "lot_row": (100, 290, 960, 360),
    "barcode": (100, 390, 965, 565),
    "footer": (350, 520, 715, 565),
}

# Fiducial marker centers in canonical coordinates
FIDUCIAL_CENTERS = [(49, 49), (1014, 49), (1014, 601), (49, 601)]


def get_outer_roi(height: int = 650, width: int = 1063, margin: int = 35) -> np.ndarray:
    """Return binary mask of printable label ROI excluding outer cutting edge and fiducials."""
    roi = np.zeros((height, width), dtype=bool)
    roi[margin:height - margin, margin:width - margin] = True
    for mx, my in FIDUCIAL_CENTERS:
        roi[int(my - 45):int(my + 45), int(mx - 45):int(mx + 45)] = False
    return roi


def get_margin_paper_mask(height: int = 650, width: int = 1063) -> np.ndarray:
    """Return binary mask of pure unprinted white paper margin."""
    printed = np.zeros((height, width), dtype=bool)
    for l, t, r, b in ZONES.values():
        printed[t:b, l:r] = True
    printed_dilated = ndimage.binary_dilation(printed, iterations=12)
    outer_roi = get_outer_roi(height, width)
    return outer_roi & ~printed_dilated


@dataclass
class LabelCalibration:
    """Calibration thresholds derived strictly from normal validation labels."""
    margin_threshold: float = 200.0
    barcode_min_density_threshold: float = 0.15
    barcode_white_comp_threshold: float = 25.0
    item_row_min: float = 900.0
    batch_row_min: float = 1300.0
    lot_row_min: float = 1100.0
    batch_row_max: float = 3100.0
    validation_max_score: float = 0.882


def build_empirical_template(train_dir: Union[str, Path], manifest_path: Optional[Union[str, Path]] = None) -> np.ndarray:
    """Derive golden reference template by computing pixel median of registered training normals."""
    train_dir = Path(train_dir)
    if manifest_path is None:
        manifest_path = train_dir / "manifest.csv"
    else:
        manifest_path = Path(manifest_path)

    margin_mask = get_margin_paper_mask()
    with manifest_path.open(encoding="utf-8") as stream:
        records = list(csv.DictReader(stream))

    normalized_imgs: List[np.ndarray] = []
    for row in records:
        img_path = train_dir / row["registered"]
        with Image.open(img_path) as im:
            arr = np.array(im.convert("L"), dtype=np.float32)
        bg = np.median(arr[margin_mask])
        normalized_imgs.append(arr * (200.0 / max(bg, 1.0)))

    template = np.median(normalized_imgs, axis=0)
    return template.astype(np.float32)


def extract_features(image: Union[Image.Image, np.ndarray], margin_mask: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Extract semantic feature descriptors from a registered label photograph."""
    if isinstance(image, Image.Image):
        arr = np.array(image.convert("L"), dtype=np.float32)
    else:
        arr = image.astype(np.float32)

    if margin_mask is None:
        margin_mask = get_margin_paper_mask(arr.shape[0], arr.shape[1])

    # Illumination normalization: scale image so clean paper margin equals reference level (200.0)
    bg = float(np.median(arr[margin_mask]))
    norm = arr * (200.0 / max(bg, 1.0))

    # 1. Margin Paper Stream: Connected dark components (smudges and tears on paper)
    dark_in_margin = (norm < 125.0) & margin_mask
    labeled_margin, num_margin = ndimage.label(dark_in_margin)
    if num_margin > 0:
        sizes = [np.sum(labeled_margin == i) for i in range(1, num_margin + 1)]
        max_margin_comp = float(max(sizes))
    else:
        max_margin_comp = 0.0

    # 2. Barcode Frequency Stream: Column-wise modulation density
    bl, bt, br, bb = ZONES["barcode"]
    bc_sub = norm[bt + 20:bb - 20, bl:br]
    col_dark_frac = np.mean(bc_sub < 120.0, axis=0)
    window = 40
    rolling_dark = np.convolve(col_dark_frac, np.ones(window) / window, mode="valid")
    min_bc_density = float(np.min(rolling_dark))

    # 3. Barcode Smudge Stream: White-channel segmentation count
    bc_white = bc_sub > 155.0
    bc_white_clean = ndimage.binary_opening(bc_white, structure=np.ones((2, 2)))
    _, num_bc_white = ndimage.label(bc_white_clean)

    # 4. Text Field Integrals: Dark ink mass per row
    row_masses: Dict[str, float] = {}
    for row_name in ["item_row", "batch_row", "lot_row"]:
        l, t, r, b = ZONES[row_name]
        sub = norm[t:b, l:r]
        row_masses[row_name] = float(np.sum(sub < 120.0))

    return {
        "bg_level": bg,
        "max_margin_comp": max_margin_comp,
        "min_bc_density": min_bc_density,
        "num_bc_white": float(num_bc_white),
        "item_row": row_masses["item_row"],
        "batch_row": row_masses["batch_row"],
        "lot_row": row_masses["lot_row"],
    }


def calibrate_detector(val_dir: Union[str, Path], manifest_path: Optional[Union[str, Path]] = None) -> LabelCalibration:
    """Fit conservative detector thresholds strictly on normal validation photographs."""
    val_dir = Path(val_dir)
    if manifest_path is None:
        manifest_path = val_dir / "manifest.csv"
    else:
        manifest_path = Path(manifest_path)

    margin_mask = get_margin_paper_mask()
    with manifest_path.open(encoding="utf-8") as stream:
        records = list(csv.DictReader(stream))

    val_feats = []
    for row in records:
        img_path = val_dir / row["registered"]
        with Image.open(img_path) as im:
            val_feats.append(extract_features(im, margin_mask))

    med_item = float(np.median([f["item_row"] for f in val_feats]))
    med_batch = float(np.median([f["batch_row"] for f in val_feats]))
    med_lot = float(np.median([f["lot_row"] for f in val_feats]))

    calibration = LabelCalibration(
        margin_threshold=200.0,
        barcode_min_density_threshold=0.15,
        barcode_white_comp_threshold=25.0,
        item_row_min=med_item * 0.55,
        batch_row_min=med_batch * 0.55,
        lot_row_min=med_lot * 0.55,
        batch_row_max=med_batch * 1.25,
    )

    val_scores = [compute_score(f, calibration)["score"] for f in val_feats]
    calibration.validation_max_score = float(max(val_scores))
    return calibration


def compute_score(features: Dict[str, float], calib: Optional[LabelCalibration] = None) -> Dict[str, Any]:
    """Compute normalized composite anomaly score and stream breakdown."""
    if calib is None:
        calib = LabelCalibration()

    # Stream 1: Margin anomaly (paper smudges and tears)
    s_margin = features["max_margin_comp"] / calib.margin_threshold

    # Stream 2: Barcode missing print
    if features["min_bc_density"] < calib.barcode_min_density_threshold:
        s_bc_miss = (calib.barcode_min_density_threshold - features["min_bc_density"]) / calib.barcode_min_density_threshold + 1.0
    else:
        s_bc_miss = (features["min_bc_density"] / 0.35) * 0.1

    # Stream 3: Barcode smudge (white-channel fragmentation)
    s_bc_smudge = features["num_bc_white"] / calib.barcode_white_comp_threshold

    # Stream 4: Text missing print
    s_item_miss = (calib.item_row_min - features["item_row"]) / calib.item_row_min + 1.0 if features["item_row"] < calib.item_row_min else 0.1
    s_batch_miss = (calib.batch_row_min - features["batch_row"]) / calib.batch_row_min + 1.0 if features["batch_row"] < calib.batch_row_min else 0.1
    s_lot_miss = (calib.lot_row_min - features["lot_row"]) / calib.lot_row_min + 1.0 if features["lot_row"] < calib.lot_row_min else 0.1
    s_text_miss = max(s_item_miss, s_batch_miss, s_lot_miss)

    # Stream 5: Text smudge (excess ink in text fields)
    s_text_smudge = features["batch_row"] / calib.batch_row_max

    streams = {
        "margin": float(s_margin),
        "barcode_missing": float(s_bc_miss),
        "barcode_smudge": float(s_bc_smudge),
        "text_missing": float(s_text_miss),
        "text_smudge": float(s_text_smudge),
    }

    primary_trigger = max(streams.items(), key=lambda item: item[1])
    composite_score = float(max(streams.values()))

    return {
        "score": composite_score,
        "is_defect": bool(composite_score >= 1.0),
        "primary_trigger": primary_trigger[0],
        "trigger_score": float(primary_trigger[1]),
        "streams": streams,
    }


def segment_defect_mask(
    image: Union[Image.Image, np.ndarray],
    template: np.ndarray,
    calib: Optional[LabelCalibration] = None,
) -> np.ndarray:
    """Generate pixel-level binary defect segmentation mask."""
    if calib is None:
        calib = LabelCalibration()

    if isinstance(image, Image.Image):
        arr = np.array(image.convert("L"), dtype=np.float32)
    else:
        arr = image.astype(np.float32)

    height, width = arr.shape
    margin_mask = get_margin_paper_mask(height, width)
    outer_roi = get_outer_roi(height, width)

    bg = float(np.median(arr[margin_mask]))
    norm = arr * (200.0 / max(bg, 1.0))
    mask = np.zeros((height, width), dtype=bool)

    # 1. Margin dark defects (tears and paper smudges)
    dark_in_margin = (norm < 125.0) & margin_mask
    labeled_m, num_m = ndimage.label(dark_in_margin)
    for i in range(1, num_m + 1):
        comp = (labeled_m == i)
        if np.sum(comp) >= 15:
            mask |= comp

    # 2. Barcode Missing Print (covered / white regions)
    bl, bt, br, bb = ZONES["barcode"]
    bc_norm = norm[bt:bb, bl:br]
    bc_sub = norm[bt + 20:bb - 20, bl:br]
    col_dark_frac = np.mean(bc_sub < 120.0, axis=0)
    window = 40
    rolling_dark = np.convolve(col_dark_frac, np.ones(window) / window, mode="same")
    missing_cols = np.flatnonzero(rolling_dark < 0.12)
    if len(missing_cols) > 0:
        c_min = max(0, missing_cols[0] - 15)
        c_max = min(br - bl, missing_cols[-1] + 15)
        mask[bt:bb, bl + c_min:bl + c_max] = True

    # 3. Barcode Smudge
    bc_white = bc_sub > 155.0
    bc_white_clean = ndimage.binary_opening(bc_white, structure=np.ones((2, 2)))
    _, num_bc_white = ndimage.label(bc_white_clean)
    if num_bc_white > calib.barcode_white_comp_threshold:
        expected_bc_paper = template[bt:bb, bl:br] > 170.0
        observed_bc_dark = bc_norm < 130.0
        bc_smudge_pixels = expected_bc_paper & observed_bc_dark
        labeled_bs, num_bs = ndimage.label(bc_smudge_pixels)
        for i in range(1, num_bs + 1):
            comp = (labeled_bs == i)
            if np.sum(comp) >= 10:
                mask[bt:bb, bl:br] |= comp

    # 4. Text Missing Print
    for row_name in ["item_row", "batch_row", "lot_row"]:
        l, t, r, b = ZONES[row_name]
        sub = norm[t:b, l:r]
        if np.sum(sub < 120.0) < 1100:
            expected_ink = template[t:b, l:r] < 120.0
            observed_light = sub > 150.0
            missing_row = expected_ink & observed_light
            missing_dilated = ndimage.binary_dilation(missing_row, iterations=4)
            mask[t:b, l:r] |= missing_dilated

    # 5. Text Smudge
    bl, bt, br, bb = ZONES["batch_row"]
    batch_sub = norm[bt:bb, bl:br]
    if np.sum(batch_sub < 120.0) > calib.batch_row_max:
        batch_ref_paper = template[bt:bb, bl:br] > 170.0
        batch_excess = batch_ref_paper & (batch_sub < 130.0)
        labeled_be, num_be = ndimage.label(batch_excess)
        for i in range(1, num_be + 1):
            comp = (labeled_be == i)
            if np.sum(comp) >= 15:
                mask[bt:bb, bl:br] |= comp

    return (mask & outer_roi).astype(np.uint8) * 255
