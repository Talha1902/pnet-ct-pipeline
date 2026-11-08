from pathlib import Path
import argparse

import nibabel as nib
import numpy as np
from scipy import ndimage


parser = argparse.ArgumentParser()

parser.add_argument(
    "--patient",
    required=True,
    help="Example: PAN_12"
)

args = parser.parse_args()

patient = args.patient


ROOT = Path(
    r"C:\Users\Talha Riaz\Desktop\Thesis Folder"
)


CT_PATH = (
    ROOT
    / "Results"
    / "_Phase2_Experiment10D_Baseline_NIfTI"
    / patient
    / "baseline_ct.nii.gz"
)


PRED_PATH = (
    ROOT
    / "Results"
    / "_Phase2_Experiment11A_MONAI_DiNTS"
    / patient
    / "baseline_ct"
    / "baseline_ct_trans.nii.gz"
)


print("=" * 80)
print(f"{patient} — MONAI DiNTS AUDIT")
print("=" * 80)


if not CT_PATH.exists():
    raise FileNotFoundError(CT_PATH)

if not PRED_PATH.exists():
    raise FileNotFoundError(PRED_PATH)


ct_img = nib.load(str(CT_PATH))
pred_img = nib.load(str(PRED_PATH))

ct = ct_img.get_fdata()
pred = pred_img.get_fdata()


# ============================================================
# GEOMETRY
# ============================================================

print("\n--- GEOMETRY ---")

print("CT shape:", ct.shape)
print("Prediction shape:", pred.shape)

shape_match = ct.shape == pred.shape

affine_match = np.allclose(
    ct_img.affine,
    pred_img.affine,
    atol=1e-4
)

print("Shape match:", shape_match)
print("Affine match:", affine_match)

spacing = np.array(
    pred_img.header.get_zooms()[:3],
    dtype=float
)

print("Spacing:", spacing)


# ============================================================
# LABELS
# ============================================================

print("\n--- LABELS ---")

labels = np.unique(pred)

print("Unique labels:", labels)

pancreas = pred == 1
tumor = pred == 2

pancreas_voxels = int(pancreas.sum())
tumor_voxels = int(tumor.sum())

voxel_volume = float(np.prod(spacing))


print(
    "Pancreas voxels:",
    pancreas_voxels
)

print(
    "Tumor-candidate voxels:",
    tumor_voxels
)

print(
    "Pancreas volume (mL):",
    round(
        pancreas_voxels
        * voxel_volume
        / 1000,
        3
    )
)

print(
    "Tumor-candidate volume (mL):",
    round(
        tumor_voxels
        * voxel_volume
        / 1000,
        3
    )
)


# ============================================================
# SLICE COVERAGE
# ============================================================

def slice_report(mask):

    positive = np.where(
        np.any(
            mask,
            axis=(0, 1)
        )
    )[0]

    if len(positive) == 0:
        return 0, None, None

    return (
        len(positive),
        int(positive.min()),
        int(positive.max())
    )


p_count, p_first, p_last = slice_report(
    pancreas
)

t_count, t_first, t_last = slice_report(
    tumor
)


print("\n--- AXIAL COVERAGE ---")

print(
    "Pancreas:",
    p_count,
    "slices |",
    p_first,
    "to",
    p_last
)

print(
    "Tumor candidate:",
    t_count,
    "slices |",
    t_first,
    "to",
    t_last
)


# ============================================================
# CONNECTED COMPONENTS
# ============================================================

structure = np.ones(
    (3, 3, 3),
    dtype=np.uint8
)


def components(mask, name):

    labeled, number = ndimage.label(
        mask,
        structure=structure
    )

    counts = np.bincount(
        labeled.ravel()
    )[1:]

    print("\n---", name, "---")

    print(
        "Connected components:",
        number
    )

    if len(counts) == 0:
        return

    order = np.argsort(
        counts
    )[::-1]

    total = counts.sum()

    for rank, idx in enumerate(
        order[:5],
        start=1
    ):

        voxels = int(counts[idx])

        fraction = (
            voxels
            / total
            * 100
        )

        volume_ml = (
            voxels
            * voxel_volume
            / 1000
        )

        print(
            f"Component {rank}: "
            f"{voxels} voxels | "
            f"{volume_ml:.3f} mL | "
            f"{fraction:.2f}%"
        )


components(
    pancreas,
    "PANCREAS COMPONENTS"
)

components(
    tumor,
    "TUMOR-CANDIDATE COMPONENTS"
)


# ============================================================
# PANCREAS + TUMOR COMPLEX
# ============================================================

combined = pancreas | tumor

combined_labeled, combined_number = (
    ndimage.label(
        combined,
        structure=structure
    )
)

combined_counts = np.bincount(
    combined_labeled.ravel()
)[1:]


print(
    "\n--- PANCREAS + TUMOR COMPLEX ---"
)

print(
    "Connected components:",
    combined_number
)


if len(combined_counts) > 0:

    largest = int(
        combined_counts.max()
    )

    total = int(
        combined_counts.sum()
    )

    print(
        "Largest combined component:",
        largest,
        "voxels"
    )

    print(
        "Fraction in largest combined component:",
        round(
            largest
            / total
            * 100,
            2
        ),
        "%"
    )


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 80)

print(
    "GEOMETRY:",
    "PASS"
    if shape_match and affine_match
    else "REVIEW"
)

print(
    "PANCREAS:",
    "PRESENT"
    if pancreas_voxels > 0
    else "ABSENT"
)

print(
    "TUMOR CANDIDATE:",
    "PRESENT"
    if tumor_voxels > 0
    else "ABSENT"
)

print("=" * 80)