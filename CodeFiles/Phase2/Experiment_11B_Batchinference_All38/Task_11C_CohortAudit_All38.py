from pathlib import Path
import csv

import nibabel as nib
import numpy as np

from scipy import ndimage


ROOT = Path(
    r"C:\Users\Talha Riaz\Desktop\Thesis Folder"
)


CT_ROOT = (
    ROOT
    / "Results"
    / "_Phase2_Experiment10D_Baseline_NIfTI"
)


PRED_ROOT = (
    ROOT
    / "Results"
    / "_Phase2_Experiment11A_MONAI_DiNTS"
)


OUTPUT_ROOT = (
    ROOT
    / "Results"
    / "_Phase2_Experiment11C_CohortAudit"
)


PATIENTS = [
    f"PAN_{i:02d}"
    for i in range(
        1,
        39
    )
]


STRUCTURE = np.ones(
    (3, 3, 3),
    dtype=np.uint8
)


def get_ct(patient):

    return (

        CT_ROOT
        / patient
        / "baseline_ct.nii.gz"

    )


def get_prediction(patient):

    return (

        PRED_ROOT
        / patient
        / "baseline_ct"
        / "baseline_ct_trans.nii.gz"

    )


def slice_report(mask):

    positive = np.where(

        np.any(
            mask,
            axis=(0, 1)
        )

    )[0]


    if len(positive) == 0:

        return (
            0,
            "",
            ""
        )


    return (

        len(positive),

        int(
            positive.min()
        ),

        int(
            positive.max()
        )

    )


def component_report(
    mask,
    voxel_volume
):

    labeled, number = (
        ndimage.label(

            mask,

            structure=STRUCTURE

        )
    )


    if number == 0:

        return {

            "number": 0,

            "largest_voxels": 0,

            "largest_ml": 0,

            "largest_pct": 0,

            "second_pct": 0

        }


    counts = np.bincount(
        labeled.ravel()
    )[1:]


    counts = np.sort(
        counts
    )[::-1]


    total = counts.sum()


    largest = int(
        counts[0]
    )


    second = (

        int(counts[1])

        if len(counts) > 1

        else 0

    )


    return {

        "number":
        int(number),

        "largest_voxels":
        largest,

        "largest_ml":
        largest
        * voxel_volume
        / 1000,

        "largest_pct":
        largest
        / total
        * 100,

        "second_pct":
        second
        / total
        * 100

    }


def qc_priority(
    geometry,
    pancreas_ml,
    pancreas_slices,
    pancreas_pct,
    tumor_voxels,
    tumor_pct,
    combined_pct
):

    reasons = []

    priority = (
        "ROUTINE_REVIEW"
    )


    if not geometry:

        return (
            "HIGH_PRIORITY",
            "geometry mismatch"
        )


    # ---------------------------------
    # Obvious near-empty pancreas
    # ---------------------------------

    if (
        pancreas_ml < 5
        or
        pancreas_slices < 5
    ):

        reasons.append(
            "near-empty pancreas"
        )

        priority = (
            "HIGH_PRIORITY"
        )


    # ---------------------------------
    # Fragmented pancreas
    # ---------------------------------

    if pancreas_pct < 80:

        reasons.append(
            "fragmented pancreas"
        )

        if priority != (
            "HIGH_PRIORITY"
        ):

            priority = (
                "MEDIUM_PRIORITY"
            )


    # ---------------------------------
    # Tumor candidate
    # ---------------------------------

    if tumor_voxels == 0:

        reasons.append(
            "no tumor candidate"
        )

        if priority == (
            "ROUTINE_REVIEW"
        ):

            priority = (
                "MEDIUM_PRIORITY"
            )


    elif tumor_pct < 80:

        reasons.append(
            "fragmented tumor candidate"
        )

        if priority != (
            "HIGH_PRIORITY"
        ):

            priority = (
                "MEDIUM_PRIORITY"
            )


    # ---------------------------------
    # Combined structure
    # ---------------------------------

    if combined_pct < 90:

        reasons.append(
            "disconnected pancreas-tumor complex"
        )

        if priority != (
            "HIGH_PRIORITY"
        ):

            priority = (
                "MEDIUM_PRIORITY"
            )


    if not reasons:

        reasons.append(
            "no major automatic warning"
        )


    return (

        priority,

        "; ".join(
            reasons
        )

    )


def audit(patient):

    ct_path = get_ct(
        patient
    )

    pred_path = (
        get_prediction(
            patient
        )
    )


    if not ct_path.exists():

        return {

            "patient":
            patient,

            "status":
            "MISSING_CT",

            "auto_qc_priority":
            "HIGH_PRIORITY",

            "auto_qc_reason":
            "baseline CT missing"

        }


    if not pred_path.exists():

        return {

            "patient":
            patient,

            "status":
            "MISSING_PREDICTION",

            "auto_qc_priority":
            "HIGH_PRIORITY",

            "auto_qc_reason":
            "prediction missing"

        }


    ct_img = nib.load(
        str(ct_path)
    )


    pred_img = nib.load(
        str(pred_path)
    )


    ct_shape = ct_img.shape

    pred_shape = (
        pred_img.shape
    )


    shape_match = (
        ct_shape
        ==
        pred_shape
    )


    affine_match = bool(

        np.allclose(

            ct_img.affine,

            pred_img.affine,

            atol=1e-4

        )

    )


    geometry = (

        shape_match
        and
        affine_match

    )


    pred = np.asanyarray(

        pred_img.dataobj

    )


    spacing = np.array(

        pred_img.header
        .get_zooms()[:3],

        dtype=float

    )


    voxel_volume = float(

        np.prod(
            spacing
        )

    )


    pancreas = (
        pred == 1
    )


    tumor = (
        pred == 2
    )


    combined = (

        pancreas
        |
        tumor

    )


    pancreas_voxels = int(
        pancreas.sum()
    )


    tumor_voxels = int(
        tumor.sum()
    )


    pancreas_ml = (

        pancreas_voxels
        * voxel_volume
        / 1000

    )


    tumor_ml = (

        tumor_voxels
        * voxel_volume
        / 1000

    )


    p_slices, p_first, p_last = (
        slice_report(
            pancreas
        )
    )


    t_slices, t_first, t_last = (
        slice_report(
            tumor
        )
    )


    p_comp = component_report(

        pancreas,

        voxel_volume

    )


    t_comp = component_report(

        tumor,

        voxel_volume

    )


    c_comp = component_report(

        combined,

        voxel_volume

    )


    priority, reason = (
        qc_priority(

            geometry,

            pancreas_ml,

            p_slices,

            p_comp[
                "largest_pct"
            ],

            tumor_voxels,

            t_comp[
                "largest_pct"
            ],

            c_comp[
                "largest_pct"
            ]

        )
    )


    return {

        "patient":
        patient,

        "status":
        "AUDITED",

        "ct_shape":
        "x".join(
            map(
                str,
                ct_shape
            )
        ),

        "prediction_shape":
        "x".join(
            map(
                str,
                pred_shape
            )
        ),

        "shape_match":
        shape_match,

        "affine_match":
        affine_match,

        "geometry_pass":
        geometry,

        "spacing_x_mm":
        round(
            spacing[0],
            4
        ),

        "spacing_y_mm":
        round(
            spacing[1],
            4
        ),

        "spacing_z_mm":
        round(
            spacing[2],
            4
        ),

        "pancreas_voxels":
        pancreas_voxels,

        "pancreas_volume_ml":
        round(
            pancreas_ml,
            3
        ),

        "pancreas_slices":
        p_slices,

        "pancreas_first_slice":
        p_first,

        "pancreas_last_slice":
        p_last,

        "pancreas_components":
        p_comp["number"],

        "pancreas_largest_component_pct":
        round(
            p_comp[
                "largest_pct"
            ],
            2
        ),

        "pancreas_second_component_pct":
        round(
            p_comp[
                "second_pct"
            ],
            2
        ),

        "tumor_candidate_voxels":
        tumor_voxels,

        "tumor_candidate_volume_ml":
        round(
            tumor_ml,
            3
        ),

        "tumor_candidate_slices":
        t_slices,

        "tumor_candidate_first_slice":
        t_first,

        "tumor_candidate_last_slice":
        t_last,

        "tumor_candidate_components":
        t_comp["number"],

        "tumor_candidate_largest_component_pct":
        round(
            t_comp[
                "largest_pct"
            ],
            2
        ),

        "tumor_candidate_second_component_pct":
        round(
            t_comp[
                "second_pct"
            ],
            2
        ),

        "combined_components":
        c_comp["number"],

        "combined_largest_component_pct":
        round(
            c_comp[
                "largest_pct"
            ],
            2
        ),

        "auto_qc_priority":
        priority,

        "auto_qc_reason":
        reason

    }


def main():

    OUTPUT_ROOT.mkdir(

        parents=True,

        exist_ok=True

    )


    rows = []


    print(
        "=" * 80
    )

    print(
        "EXPERIMENT 11C"
    )

    print(
        "38-PATIENT DiNTS "
        "COHORT AUDIT"
    )

    print(
        "=" * 80
    )


    for number, patient in enumerate(

        PATIENTS,

        start=1

    ):


        result = audit(
            patient
        )


        rows.append(
            result
        )


        print(

            f"[{number:02d}/38] "
            f"{patient}: "

            f"{result.get('status')} | "

            f"{result.get('auto_qc_priority')} | "

            f"{result.get('auto_qc_reason')}"

        )


    # ----------------------------------
    # Full audit CSV
    # ----------------------------------

    keys = []

    for row in rows:

        for key in row:

            if key not in keys:

                keys.append(
                    key
                )


    full_csv = (

        OUTPUT_ROOT
        / "Phase2_Experiment11C_"
          "DiNTS_CohortAudit_All38.csv"

    )


    with open(

        full_csv,

        "w",

        newline="",

        encoding="utf-8-sig"

    ) as f:


        writer = csv.DictWriter(

            f,

            fieldnames=keys

        )


        writer.writeheader()


        writer.writerows(
            rows
        )


    # ----------------------------------
    # Visual review queue
    # ----------------------------------

    priority_order = {

        "HIGH_PRIORITY": 0,

        "MEDIUM_PRIORITY": 1,

        "ROUTINE_REVIEW": 2

    }


    ordered = sorted(

        rows,

        key=lambda x: (

            priority_order.get(

                x.get(
                    "auto_qc_priority"
                ),

                99

            ),

            x["patient"]

        )

    )


    queue_file = (

        OUTPUT_ROOT
        / "Phase2_Experiment11C_"
          "VisualReviewQueue.csv"

    )


    with open(

        queue_file,

        "w",

        newline="",

        encoding="utf-8-sig"

    ) as f:


        writer = csv.DictWriter(

            f,

            fieldnames=keys

        )


        writer.writeheader()


        writer.writerows(
            ordered
        )


    # ----------------------------------
    # Summary
    # ----------------------------------

    high = sum(

        x.get(
            "auto_qc_priority"
        )
        ==
        "HIGH_PRIORITY"

        for x in rows

    )


    medium = sum(

        x.get(
            "auto_qc_priority"
        )
        ==
        "MEDIUM_PRIORITY"

        for x in rows

    )


    routine = sum(

        x.get(
            "auto_qc_priority"
        )
        ==
        "ROUTINE_REVIEW"

        for x in rows

    )


    print(
        "\n" +
        "=" * 80
    )


    print(
        "AUDIT COMPLETE"
    )


    print(
        "=" * 80
    )


    print(
        "High priority:",
        high
    )


    print(
        "Medium priority:",
        medium
    )


    print(
        "Routine review:",
        routine
    )


    print(
        "\nMaster CSV:"
    )


    print(
        full_csv
    )


    print(
        "\nVisual review queue:"
    )


    print(
        queue_file
    )


    print(
        "\nIMPORTANT:"
        "\nAutomatic QC is only a "
        "review-prioritization system."
        "\nIt is NOT clinical validation "
        "and NOT ground truth."
    )


if __name__ == "__main__":

    main()