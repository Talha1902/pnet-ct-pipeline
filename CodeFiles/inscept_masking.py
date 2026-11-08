"""
Inspect one patient's CT + pancreas mask.

This script tells you:
- CT and mask shapes
- whether shape/affine match
- NIfTI orientation (e.g. RAS/LPS-style axis codes)
- which ARRAY axis corresponds to axial/coronal/sagittal slicing
- how many slices contain pancreas in each anatomical plane
- first and last masked slice in each plane
- slice with the largest pancreas area in each plane
- whether there are gaps in the mask slice range
- foreground voxel count and approximate mask volume
- saves an overlay figure for axial/coronal/sagittal views
- saves a CSV summary

Install:
    pip install nibabel numpy matplotlib

Edit only the paths in SECTION 1.
"""

from pathlib import Path
import csv

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# 1. EDIT THESE PATHS
# =========================================================

PATIENT_DIR = Path(r"D:/Thesis Folder/Training_dataset/Pat_08")

CT_PATH = PATIENT_DIR / "Origional_images.nii"
MASK_PATH = PATIENT_DIR / "masked_images.nii"

# If your actual files have different names, change them above.
# Examples:
# CT_PATH = PATIENT_DIR / "Origional_images.nii"
# MASK_PATH = PATIENT_DIR / "masked_images.nii"

OUTPUT_DIR = PATIENT_DIR / "Mask_Inspection"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# 2. LOAD FILES
# =========================================================

if not CT_PATH.exists():
    raise FileNotFoundError(f"CT file not found:\n{CT_PATH}")

if not MASK_PATH.exists():
    raise FileNotFoundError(f"Mask file not found:\n{MASK_PATH}")

ct_img = nib.load(str(CT_PATH))
mask_img = nib.load(str(MASK_PATH))

ct = ct_img.get_fdata(dtype=np.float32)
mask_raw = mask_img.get_fdata()

mask = (mask_raw > 0).astype(np.uint8)


# =========================================================
# 3. BASIC GEOMETRY CHECKS
# =========================================================

print("=" * 75)
print("SINGLE PATIENT MASK INSPECTION")
print("=" * 75)

print(f"Patient folder: {PATIENT_DIR.name}")
print(f"CT shape:       {ct.shape}")
print(f"Mask shape:     {mask.shape}")

shape_match = ct.shape == mask.shape
affine_match = np.allclose(ct_img.affine, mask_img.affine, atol=1e-4)

print(f"Shape match:    {shape_match}")
print(f"Affine match:   {affine_match}")

if not shape_match:
    raise ValueError("CT and mask shapes do not match.")

spacing = ct_img.header.get_zooms()[:3]
print(
    "Voxel spacing: "
    f"({spacing[0]:.4f}, {spacing[1]:.4f}, {spacing[2]:.4f}) mm"
)

unique_values = np.unique(mask_raw)
print(f"Original mask unique values: {unique_values[:20]}")

foreground_voxels = int(mask.sum())
voxel_volume_mm3 = float(np.prod(spacing))
mask_volume_ml = foreground_voxels * voxel_volume_mm3 / 1000.0

print(f"Foreground voxels: {foreground_voxels}")
print(f"Approx. mask volume: {mask_volume_ml:.2f} mL")


# =========================================================
# 4. DETERMINE NIFTI ORIENTATION
# =========================================================

axis_codes = nib.aff2axcodes(ct_img.affine)

print("\n" + "-" * 75)
print("NIFTI ORIENTATION")
print("-" * 75)
print(f"Axis codes: {axis_codes}")

# Each array axis has one anatomical direction code.
# R/L direction -> sagittal slice stack
# A/P direction -> coronal slice stack
# S/I direction -> axial slice stack

def plane_from_axis_code(code: str) -> str:
    if code in ("R", "L"):
        return "Sagittal"
    if code in ("A", "P"):
        return "Coronal"
    if code in ("S", "I"):
        return "Axial"
    return "Unknown"


axis_to_plane = {}

for axis_index, code in enumerate(axis_codes):
    plane = plane_from_axis_code(code)
    axis_to_plane[axis_index] = plane
    print(
        f"Array axis {axis_index} -> anatomical direction {code} "
        f"-> {plane} slice stack"
    )

print(
    "\nImportant: the segmentation is a 3D volume. "
    "It is not 'only axial' or 'only coronal'. "
    "The same mask can be viewed in all three planes."
)


# =========================================================
# 5. ANALYZE MASKED SLICES ALONG EACH AXIS
# =========================================================

def analyze_axis(binary_mask: np.ndarray, axis: int):
    """
    For a given array axis, determine which slices contain foreground.
    """
    other_axes = tuple(i for i in range(3) if i != axis)
    area_per_slice = np.sum(binary_mask, axis=other_axes)

    positive = np.where(area_per_slice > 0)[0]

    if len(positive) == 0:
        return {
            "positive_count": 0,
            "first": None,
            "last": None,
            "max_area_slice": None,
            "max_area_pixels": 0,
            "gaps": [],
        }

    first = int(positive.min())
    last = int(positive.max())
    max_area_slice = int(np.argmax(area_per_slice))
    max_area_pixels = int(area_per_slice[max_area_slice])

    full_range = np.arange(first, last + 1)
    gaps = np.setdiff1d(full_range, positive).astype(int).tolist()

    return {
        "positive_count": int(len(positive)),
        "first": first,
        "last": last,
        "max_area_slice": max_area_slice,
        "max_area_pixels": max_area_pixels,
        "gaps": gaps,
    }


results = {}

print("\n" + "=" * 75)
print("MASKED SLICE RANGES")
print("=" * 75)

for axis in range(3):
    plane = axis_to_plane[axis]
    info = analyze_axis(mask, axis)
    results[plane] = {
        "axis": axis,
        **info,
    }

    print(f"\n{plane.upper()} PLANE")
    print(f"  Array axis:                 {axis}")
    print(f"  Total number of slices:     {mask.shape[axis]}")
    print(f"  Pancreas-positive slices:   {info['positive_count']}")

    if info["positive_count"] > 0:
        print(
            f"  First -> last masked slice: "
            f"{info['first']} -> {info['last']}"
        )
        print(
            f"  Largest pancreas mask:      "
            f"slice {info['max_area_slice']}"
        )
        print(
            f"  Pixels on largest slice:    "
            f"{info['max_area_pixels']}"
        )

        if info["gaps"]:
            print(
                f"  Gaps inside masked range:   "
                f"{len(info['gaps'])} slices"
            )
            print(f"  Gap indices: {info['gaps'][:30]}")
        else:
            print("  Gaps inside masked range:   None")
    else:
        print("  No pancreas foreground found.")


# =========================================================
# 6. SAVE SUMMARY CSV
# =========================================================

csv_path = OUTPUT_DIR / "mask_plane_summary.csv"

with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(
        [
            "plane",
            "array_axis",
            "total_slices",
            "positive_slices",
            "first_positive_slice",
            "last_positive_slice",
            "max_area_slice",
            "max_area_pixels",
            "gap_count",
        ]
    )

    for plane in ("Axial", "Coronal", "Sagittal"):
        info = results.get(plane)

        if info is None:
            continue

        axis = info["axis"]

        writer.writerow(
            [
                plane,
                axis,
                mask.shape[axis],
                info["positive_count"],
                info["first"],
                info["last"],
                info["max_area_slice"],
                info["max_area_pixels"],
                len(info["gaps"]),
            ]
        )

print(f"\nSummary CSV saved to:\n{csv_path}")


# =========================================================
# 7. VISUALIZE LARGEST MASK SLICE IN EACH PLANE
# =========================================================

def get_slice(volume: np.ndarray, axis: int, index: int):
    return np.take(volume, index, axis=axis)


fig, axes = plt.subplots(3, 2, figsize=(11, 15))

plane_order = ["Axial", "Coronal", "Sagittal"]

for row, plane in enumerate(plane_order):
    info = results[plane]
    axis = info["axis"]

    if info["max_area_slice"] is None:
        axes[row, 0].text(0.5, 0.5, "No mask", ha="center", va="center")
        axes[row, 1].text(0.5, 0.5, "No mask", ha="center", va="center")
        axes[row, 0].axis("off")
        axes[row, 1].axis("off")
        continue

    index = info["max_area_slice"]

    ct_slice = get_slice(ct, axis, index)
    mask_slice = get_slice(mask, axis, index)

    mask_overlay = np.ma.masked_where(mask_slice == 0, mask_slice)

    axes[row, 0].imshow(ct_slice.T, cmap="gray", origin="lower")
    axes[row, 0].set_title(
        f"{plane} CT - slice {index}"
    )
    axes[row, 0].axis("off")

    axes[row, 1].imshow(ct_slice.T, cmap="gray", origin="lower")
    axes[row, 1].imshow(
        mask_overlay.T,
        alpha=0.55,
        origin="lower",
    )
    axes[row, 1].set_title(
        f"{plane} CT + pancreas mask"
    )
    axes[row, 1].axis("off")

plt.suptitle(
    f"{PATIENT_DIR.name}: Pancreas Mask Inspection",
    fontsize=14,
)
plt.tight_layout()

figure_path = OUTPUT_DIR / "axial_coronal_sagittal_mask_inspection.png"
plt.savefig(figure_path, dpi=250, bbox_inches="tight")
plt.show()

print(f"Inspection figure saved to:\n{figure_path}")

print("\n" + "=" * 75)
print("HOW TO INTERPRET THE RESULT")
print("=" * 75)
print(
    "Example:\n"
    "AXIAL PLANE\n"
    "  Pancreas-positive slices: 94\n"
    "  First -> last masked slice: 47 -> 140\n\n"
    "means that when scrolling through AXIAL slices, the pancreas mask appears "
    "on 94 slices, beginning at slice 47 and ending at slice 140.\n"
)
print(
    "The coronal and sagittal ranges describe the SAME 3D pancreas mask "
    "viewed from different anatomical directions."
)