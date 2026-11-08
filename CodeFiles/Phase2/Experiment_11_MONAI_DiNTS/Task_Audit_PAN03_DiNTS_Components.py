from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage


PRED_PATH = Path(
    r"C:\Users\Talha Riaz\Desktop\Thesis Folder"
    r"\Results"
    r"\_Phase2_Experiment11A_MONAI_DiNTS"
    r"\PAN_03"
    r"\baseline_ct"
    r"\baseline_ct_trans.nii.gz"
)


print("=" * 80)
print("PAN_03 DiNTS CONNECTED-COMPONENT AUDIT")
print("=" * 80)


img = nib.load(str(PRED_PATH))
pred = img.get_fdata()

spacing = np.array(
    img.header.get_zooms()[:3],
    dtype=float
)

voxel_volume = float(
    np.prod(spacing)
)


# 26-connectivity in 3D
structure = np.ones(
    (3, 3, 3),
    dtype=np.uint8
)


def component_report(mask, name):

    labeled, number = ndimage.label(
        mask,
        structure=structure
    )

    counts = np.bincount(
        labeled.ravel()
    )

    # Remove background
    counts = counts[1:]

    print()
    print("-" * 80)
    print(name)
    print("-" * 80)

    print(
        "Number of connected components:",
        number
    )

    if number == 0:
        return

    order = np.argsort(
        counts
    )[::-1]

    total_voxels = int(
        counts.sum()
    )

    print(
        "Total foreground voxels:",
        total_voxels
    )

    print(
        "Total volume (mL):",
        round(
            total_voxels
            * voxel_volume
            / 1000,
            3
        )
    )

    print()
    print("Largest components:")

    for rank, idx in enumerate(
        order[:10],
        start=1
    ):

        component_id = idx + 1

        voxels = int(
            counts[idx]
        )

        volume_ml = (
            voxels
            * voxel_volume
            / 1000
        )

        coords = np.argwhere(
            labeled == component_id
        )

        minimum = coords.min(
            axis=0
        )

        maximum = coords.max(
            axis=0
        )

        centroid = coords.mean(
            axis=0
        )

        fraction = (
            voxels
            / total_voxels
            * 100
        )

        print()
        print(
            f"Component {rank}"
        )

        print(
            "  Voxels:",
            voxels
        )

        print(
            "  Volume (mL):",
            round(
                volume_ml,
                3
            )
        )

        print(
            "  Fraction of label:",
            round(
                fraction,
                2
            ),
            "%"
        )

        print(
            "  Bounding box:",
            minimum.tolist(),
            "to",
            maximum.tolist()
        )

        print(
            "  Centroid:",
            np.round(
                centroid,
                1
            ).tolist()
        )


pancreas = (
    pred == 1
)

tumor = (
    pred == 2
)


component_report(
    pancreas,
    "LABEL 1 — PANCREAS"
)

component_report(
    tumor,
    "LABEL 2 — TUMOR CANDIDATE"
)


# ============================================================
# ADJACENCY CHECK
# ============================================================

print()
print("=" * 80)
print("PANCREAS–TUMOR ADJACENCY")
print("=" * 80)


# Because labels 1 and 2 are mutually exclusive,
# they cannot overlap directly.
#
# Dilate the pancreas by one voxel and see how much tumor
# touches the pancreatic boundary.

dilated_pancreas = ndimage.binary_dilation(
    pancreas,
    structure=structure,
    iterations=1
)


touching_tumor = (
    tumor
    &
    dilated_pancreas
)


tumor_voxels = int(
    tumor.sum()
)

touching_voxels = int(
    touching_tumor.sum()
)


if tumor_voxels > 0:

    touching_percentage = (
        touching_voxels
        / tumor_voxels
        * 100
    )

else:

    touching_percentage = 0


print(
    "Tumor voxels:",
    tumor_voxels
)

print(
    "Tumor voxels touching/adjacent to pancreas:",
    touching_voxels
)

print(
    "Percentage adjacent to pancreas:",
    round(
        touching_percentage,
        2
    ),
    "%"
)


print()
print("=" * 80)
print("AUDIT COMPLETE")
print("=" * 80)