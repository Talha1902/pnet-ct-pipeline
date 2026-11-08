"""Experiment 12A - one row per patient: what was scanned, what DiNTS produced.

Joins:
    10D selected baseline series      -> which series is the baseline
    10B DICOM audit                   -> scanner / acquisition settings
    baseline NIfTI (10D)              -> real geometry + intensity sanity checks
    DiNTS prediction (11A) + 11C QC   -> mask measurements and QC priority
    clinical spreadsheet              -> grade, Ki-67

Empty columns are left for later steps:
    contrast_phase_image   -> Exp 12B (image-based phase estimate)
    visual_grade_A_to_E    -> manual 3D Slicer review
    visual_notes           -> manual 3D Slicer review
"""
from pathlib import Path
import csv

import nibabel as nib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]

RESULTS = ROOT / "Results"

BASELINE_CSV = (
    RESULTS
    / "_Phase2_Experiment10D_Baseline_NIfTI"
    / "Phase2_Task10D_Selected_Baseline_Series.csv"
)

DICOM_AUDIT_CSV = (
    RESULTS
    / "_Phase2_Experiment10B_Audit_DICOMDATASET"
    / "Phase2_DICOM_Audit_Series.csv"
)

QC_CSV = (
    RESULTS
    / "_Phase2_Experiment11C_CohortAudit"
    / "Phase2_Experiment11C_DiNTS_CohortAudit_All38.csv"
)

CLINICAL_XLSX = ROOT / "Clinical-Information-for-CTpred-Sunitinib-panNET.xlsx"

CT_ROOT = RESULTS / "_Phase2_Experiment10D_Baseline_NIfTI"

PRED_ROOT = RESULTS / "_Phase2_Experiment11A_MONAI_DiNTS"

OUTPUT_ROOT = RESULTS / "_Phase2_Experiment12_CaseTable"


# DiNTS ScaleIntensityRanged window (configs/inference.yaml)
DINTS_HU_MIN = -87
DINTS_HU_MAX = 199


def get_ct(patient):

    return CT_ROOT / patient / "baseline_ct.nii.gz"


def get_prediction(patient):

    return PRED_ROOT / patient / "baseline_ct" / "baseline_ct_trans.nii.gz"


def image_report(patient):
    """Geometry and intensity checks on the baseline CT and its prediction."""

    ct_img = nib.load(str(get_ct(patient)))

    ct = np.asanyarray(ct_img.dataobj).astype(np.float32)

    spacing = np.array(ct_img.header.get_zooms()[:3], dtype=float)

    # every 4th voxel in-plane is enough for percentiles
    sample = ct[::4, ::4, :].ravel()

    p01, p50, p99 = np.percentile(sample, [1, 50, 99])

    in_window = np.mean(
        (sample >= DINTS_HU_MIN) & (sample <= DINTS_HU_MAX)
    ) * 100

    row = {

        "nifti_axcodes": "".join(nib.aff2axcodes(ct_img.affine)),

        "n_slices": ct.shape[2],

        "z_spacing_mm": round(spacing[2], 3),

        "inplane_spacing_mm": round(spacing[0], 3),

        "fov_mm": round(ct.shape[0] * spacing[0], 1),

        "scan_length_mm": round(ct.shape[2] * spacing[2], 1),

        "hu_min": round(float(ct.min()), 1),

        "hu_p01": round(float(p01), 1),

        "hu_median": round(float(p50), 1),

        "hu_p99": round(float(p99), 1),

        "hu_max": round(float(ct.max()), 1),

        "pct_voxels_in_dints_window": round(float(in_window), 2),

    }

    pred_path = get_prediction(patient)

    if not pred_path.exists():

        return row

    pred = np.asanyarray(nib.load(str(pred_path)).dataobj)

    pancreas = pred == 1

    tumor = pred == 2

    row["pancreas_mean_hu"] = (
        round(float(ct[pancreas].mean()), 1) if pancreas.any() else ""
    )

    row["tumor_candidate_mean_hu"] = (
        round(float(ct[tumor].mean()), 1) if tumor.any() else ""
    )

    # a pancreas mask that reaches the first/last slice may be cut off by the scan range
    positive = np.where(np.any(pancreas | tumor, axis=(0, 1)))[0]

    row["mask_touches_scan_edge"] = (
        bool(positive.min() == 0 or positive.max() == ct.shape[2] - 1)
        if len(positive)
        else ""
    )

    return row


def load_tables():

    baseline = pd.read_csv(BASELINE_CSV)

    audit = pd.read_csv(DICOM_AUDIT_CSV)

    qc = pd.read_csv(QC_CSV)

    clinical = pd.read_excel(CLINICAL_XLSX)

    audit = audit.rename(columns={"series_instance_uid": "baseline_series_uid"})

    scan = baseline[["patient", "baseline_series_uid", "baseline_date_formatted"]].merge(
        audit[[
            "baseline_series_uid",
            "manufacturer",
            "manufacturer_model",
            "kvp",
            "reconstruction_kernel",
            "contrast_agent",
            "suggested_phase",
            "series_description",
            "protocol_name",
            "slice_thickness_mm",
            "spacing_between_slices_mm",
        ]],
        on="baseline_series_uid",
        how="left",
    )

    clinical = clinical.rename(columns={
        "Patient ID": "patient",
        "Dataset": "clinical_dataset_split",
        "Sex": "sex",
        "Age at diagnosis": "age",
        "Ki-67 index": "ki67_index",
        "Tumor grade": "tumor_grade",
    })[["patient", "clinical_dataset_split", "sex", "age", "ki67_index", "tumor_grade"]]

    clinical["grade_group"] = np.where(clinical.tumor_grade == 1, "G1", "G2/G3")

    qc = qc[[
        "patient",
        "pancreas_volume_ml",
        "pancreas_slices",
        "pancreas_components",
        "pancreas_largest_component_pct",
        "tumor_candidate_volume_ml",
        "tumor_candidate_slices",
        "tumor_candidate_components",
        "tumor_candidate_largest_component_pct",
        "combined_largest_component_pct",
        "auto_qc_priority",
        "auto_qc_reason",
    ]]

    return scan, clinical, qc


def failure_summary(table):
    """Plain-text cross-tabs of DiNTS outcome against acquisition settings."""

    t = table.copy()

    t["dints_outcome"] = np.select(
        [
            t.pancreas_volume_ml < 5,
            t.tumor_candidate_volume_ml == 0,
        ],
        [
            "PANCREAS_FAILURE",
            "NO_TUMOR_CANDIDATE",
        ],
        default="OUTPUT_PRESENT",
    )

    t["slice_thickness_group"] = pd.cut(
        t.slice_thickness_mm,
        bins=[0, 1.0, 2.0, 10],
        labels=["<=1 mm", "1-2 mm", ">2 mm"],
    )

    lines = [
        "EXPERIMENT 12A - DiNTS OUTCOME vs ACQUISITION SETTINGS",
        "=" * 80,
        "dints_outcome is automatic (pancreas < 5 mL / no tumor candidate).",
        "It is NOT the visual A-E grade and NOT ground truth.",
        f"n = {len(t)} patients; groups are small, so read these as descriptions, not tests.",
        "",
    ]

    for column in [
        "suggested_phase",
        "slice_thickness_group",
        "manufacturer",
        "kvp",
        "grade_group",
    ]:

        lines.append(f"--- {column}")

        lines.append(
            pd.crosstab(
                t[column].astype(str),
                t.dints_outcome,
                margins=True,
            ).to_string()
        )

        lines.append("")

    lines.append("--- cases without usable output")

    lines.append(
        t[t.dints_outcome != "OUTPUT_PRESENT"][[
            "patient",
            "dints_outcome",
            "suggested_phase",
            "slice_thickness_mm",
            "manufacturer",
            "manufacturer_model",
            "hu_median",
            "pct_voxels_in_dints_window",
            "mask_touches_scan_edge",
        ]].to_string(index=False)
    )

    return "\n".join(lines)


def main():

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("EXPERIMENT 12A")
    print("38-PATIENT CASE TABLE")
    print("=" * 80)

    scan, clinical, qc = load_tables()

    image_rows = []

    for number, patient in enumerate(scan.patient, start=1):

        row = {"patient": patient, **image_report(patient)}

        image_rows.append(row)

        print(
            f"[{number:02d}/{len(scan)}] {patient}: "
            f"z={row['z_spacing_mm']} mm | "
            f"HU median={row['hu_median']} | "
            f"in DiNTS window={row['pct_voxels_in_dints_window']}%"
        )

    table = (
        scan
        .merge(clinical, on="patient", how="left")
        .merge(pd.DataFrame(image_rows), on="patient", how="left")
        .merge(qc, on="patient", how="left")
    )

    table["contrast_phase_image"] = ""

    table["visual_grade_A_to_E"] = ""

    table["visual_notes"] = ""

    table_csv = OUTPUT_ROOT / "Phase2_Experiment12A_CaseTable_All38.csv"

    table.to_csv(table_csv, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)

    summary = failure_summary(table)

    summary_txt = OUTPUT_ROOT / "Phase2_Experiment12A_Outcome_vs_Settings.txt"

    summary_txt.write_text(summary, encoding="utf-8")

    print("\n" + summary)

    print("\nCase table:")
    print(table_csv)

    print("\nSummary:")
    print(summary_txt)

    print(
        "\nIMPORTANT:"
        "\nFill visual_grade_A_to_E after the 3D Slicer review."
        "\ncontrast_phase_image is filled by Experiment 12B."
    )


if __name__ == "__main__":

    main()
