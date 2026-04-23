import json
import math
import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import binary_erosion, distance_transform_edt

LABEL_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
TOLERANCE_MM = 2.0  # FLARE2024 standard tolerance


def get_surface_mask(mask: np.ndarray) -> np.ndarray:
    """Returns the surface voxels of a binary mask via erosion."""
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    eroded = binary_erosion(mask, iterations=1, border_value=0)
    return mask & ~eroded


def compute_nsd(pred_mask: np.ndarray, ref_mask: np.ndarray,
                spacing_zyx: tuple, tolerance_mm: float) -> float:
    """
    Normalized Surface Distance at a given tolerance (mm).
    Returns NaN if both masks are empty (organ absent in both).
    Returns 0.0 if one mask is empty and the other is not.
    """
    pred_empty = not pred_mask.any()
    ref_empty = not ref_mask.any()

    if pred_empty and ref_empty:
        return float("nan")
    if pred_empty or ref_empty:
        return 0.0

    pred_surface = get_surface_mask(pred_mask)
    ref_surface = get_surface_mask(ref_mask)

    # Distance from every voxel to the nearest surface voxel of the OTHER mask
    # distance_transform_edt returns Euclidean distance in voxel units scaled by spacing
    dist_pred_to_ref = distance_transform_edt(~ref_surface, sampling=spacing_zyx)
    dist_ref_to_pred = distance_transform_edt(~pred_surface, sampling=spacing_zyx)

    # Surface voxels within tolerance
    pred_within = (dist_pred_to_ref[pred_surface] <= tolerance_mm).sum()
    ref_within = (dist_ref_to_pred[ref_surface] <= tolerance_mm).sum()

    nsd = (pred_within + ref_within) / (pred_surface.sum() + ref_surface.sum())
    return float(nsd)


def evaluate(summary_json_path: str, tolerance_mm: float = TOLERANCE_MM):
    with open(summary_json_path) as f:
        summary = json.load(f)

    nsd_per_case = []
    all_valid_nsd = []

    for case in summary["metric_per_case"]:
        pred_path = Path(case["prediction_file"])
        ref_path = Path(case["reference_file"])

        if not pred_path.exists():
            print(f"WARNING: prediction not found: {pred_path}, skipping.")
            continue
        if not ref_path.exists():
            print(f"WARNING: reference not found: {ref_path}, skipping.")
            continue

        pred_img = sitk.ReadImage(str(pred_path))
        ref_img = sitk.ReadImage(str(ref_path))

        # SimpleITK spacing is (x, y, z); scipy needs (z, y, x)
        spacing_xyz = ref_img.GetSpacing()
        spacing_zyx = (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])

        pred_arr = sitk.GetArrayFromImage(pred_img)  # (z, y, x)
        ref_arr = sitk.GetArrayFromImage(ref_img)

        case_nsd = {}
        for label in LABEL_IDS:
            pred_mask = (pred_arr == label)
            ref_mask = (ref_arr == label)
            nsd = compute_nsd(pred_mask, ref_mask, spacing_zyx, tolerance_mm)
            case_nsd[label] = nsd
            if not math.isnan(nsd):
                all_valid_nsd.append(nsd)

        mean_nsd = np.nanmean(list(case_nsd.values()))
        nsd_per_case.append({
            "file": str(pred_path),
            "NSD_per_label": case_nsd,
            "mean_NSD": mean_nsd,
        })
        print(f"{pred_path.name}: DSC foreground mean = {_get_foreground_dsc(case, summary):.4f} | mean NSD = {mean_nsd:.4f}")

    print(f"\n{'='*60}")
    print(f"Overall mean NSD ({tolerance_mm}mm tolerance): {np.mean(all_valid_nsd):.4f}")

    # Per-label NSD summary
    print(f"\nPer-label mean NSD:")
    for label in LABEL_IDS:
        label_nsds = [
            c["NSD_per_label"][label]
            for c in nsd_per_case
            if not math.isnan(c["NSD_per_label"][label])
        ]
        if label_nsds:
            print(f"  Label {label:2d}: {np.mean(label_nsds):.4f}")

    return nsd_per_case


def _get_foreground_dsc(case: dict, summary: dict) -> float:
    """Helper to get mean foreground DSC for a case from the summary."""
    dices = [
        v["Dice"]
        for v in case["metrics"].values()
        if not math.isnan(v["Dice"])
    ]
    return float(np.mean(dices)) if dices else float("nan")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute NSD from nnU-Net summary.json")
    parser.add_argument("summary_json", help="Path to summary.json")
    parser.add_argument("--tolerance", type=float, default=TOLERANCE_MM,
                        help=f"Surface tolerance in mm (default: {TOLERANCE_MM})")
    args = parser.parse_args()
    evaluate(args.summary_json, args.tolerance)