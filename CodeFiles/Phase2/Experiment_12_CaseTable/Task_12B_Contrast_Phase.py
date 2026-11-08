"""Experiment 12B - estimate the contrast phase of each baseline CT from the image.

The DICOM tags give no usable phase for 22/38 baseline series, and DiNTS was
trained on portal-venous CT only, so the phase has to be measured.

For each patient:
    1. TotalSegmentator (task "total", roi_subset) segments the aorta,
       portal/splenic vein, IVC, kidneys, liver, spleen and pancreas.
       The pancreas mask is kept for Experiment 16 (second model).
    2. Median HU is measured inside each vessel (eroded masks, to avoid the
       vessel wall and partial-volume voxels).
    3. A transparent rule assigns the phase from aorta HU and the
       portal-vein / aorta ratio (see classify_phase).
    4. TotalSegmentator's own phase classifier (totalseg_get_phase) is run as a
       second opinion when available.

Run on the GPU node (SLURM): see run_12B.sbatch
"""
from pathlib import Path
import json
import subprocess
import sys

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage


ROOT = Path(__file__).resolve().parents[3]

CT_ROOT = ROOT / "Results" / "_Phase2_Experiment10D_Baseline_NIfTI"

OUTPUT_ROOT = ROOT / "Results" / "_Phase2_Experiment12B_ContrastPhase"

CASE_TABLE = (
    ROOT
    / "Results"
    / "_Phase2_Experiment12_CaseTable"
    / "Phase2_Experiment12A_CaseTable_All38.csv"
)

PATIENTS = [f"PAN_{i:02d}" for i in range(1, 39)]

ROI_SUBSET = [
    "aorta",
    "portal_vein_and_splenic_vein",
    "inferior_vena_cava",
    "kidney_left",
    "kidney_right",
    "liver",
    "spleen",
    "pancreas",
]

# minimum voxels left after erosion for a HU measurement to count
MIN_VOXELS = 200


def run_totalsegmentator(patient):

    from totalsegmentator.python_api import totalsegmentator

    out_dir = OUTPUT_ROOT / patient

    out_dir.mkdir(parents=True, exist_ok=True)

    seg_path = out_dir / "totalseg_roi.nii.gz"

    if not seg_path.exists():

        totalsegmentator(
            str(CT_ROOT / patient / "baseline_ct.nii.gz"),
            str(seg_path),
            ml=True,
            roi_subset=ROI_SUBSET,
            device="gpu",
            quiet=True,
        )

    return seg_path


def run_totalseg_phase(patient):
    """TotalSegmentator's built-in phase classifier; returns {} if unavailable."""

    out_json = OUTPUT_ROOT / patient / "totalseg_phase.json"

    if not out_json.exists():

        try:

            subprocess.run(
                [
                    str(Path(sys.executable).with_name("totalseg_get_phase")),
                    "-i", str(CT_ROOT / patient / "baseline_ct.nii.gz"),
                    "-o", str(out_json),
                    "-q",
                ],
                check=True,
                capture_output=True,
                timeout=1800,
            )

        except Exception as error:

            print(f"    totalseg_get_phase failed: {error}")

            return {}

    return json.loads(out_json.read_text())


def label_ids():
    """Map structure name -> label value for a TotalSegmentator multilabel file."""

    from totalsegmentator.map_to_binary import class_map

    return {name: value for value, name in class_map["total"].items()}


def median_hu(ct, seg, label, erode=2):

    mask = seg == label

    if erode:

        mask = ndimage.binary_erosion(mask, iterations=erode)

    if mask.sum() < MIN_VOXELS:

        mask = seg == label

    if mask.sum() < MIN_VOXELS:

        return np.nan, int(mask.sum())

    return float(np.median(ct[mask])), int(mask.sum())


def classify_phase(aorta, portal):
    """Rule-based phase from vessel HU (contrast-enhanced abdominal CT).

    native          : aorta < 100 HU (no intravascular contrast)
    arterial        : aorta clearly brighter than portal vein (PV/Ao < 0.70)
    portal-venous   : portal vein close to aorta (PV/Ao >= 0.70) with aorta >= 130 HU
    delayed/low     : PV/Ao >= 0.70 but aorta < 130 HU (contrast washed out)
    """

    if np.isnan(aorta):

        return "UNDETERMINED", np.nan

    if aorta < 100:

        return "NATIVE", np.nan

    if np.isnan(portal):

        return "UNDETERMINED", np.nan

    ratio = portal / aorta

    if ratio < 0.70:

        return "ARTERIAL", ratio

    if aorta >= 130:

        return "PORTAL_VENOUS", ratio

    return "DELAYED_OR_LOW_CONTRAST", ratio


def measure(patient):

    seg_path = run_totalsegmentator(patient)

    ct = np.asanyarray(nib.load(str(CT_ROOT / patient / "baseline_ct.nii.gz")).dataobj).astype(np.float32)

    seg_img = nib.load(str(seg_path))

    seg = np.asanyarray(seg_img.dataobj)

    ids = label_ids()

    row = {"patient": patient}

    for name in [
        "aorta",
        "portal_vein_and_splenic_vein",
        "inferior_vena_cava",
        "kidney_left",
        "liver",
        "pancreas",
    ]:

        hu, n = median_hu(ct, seg, ids[name])

        row[f"{name}_median_hu"] = round(hu, 1) if not np.isnan(hu) else np.nan

        row[f"{name}_voxels"] = n

    spacing = np.prod(seg_img.header.get_zooms()[:3])

    row["totalseg_pancreas_volume_ml"] = round(float((seg == ids["pancreas"]).sum() * spacing / 1000), 3)

    phase, ratio = classify_phase(
        row["aorta_median_hu"],
        row["portal_vein_and_splenic_vein_median_hu"],
    )

    row["portal_to_aorta_ratio"] = round(ratio, 3) if not np.isnan(ratio) else np.nan

    row["contrast_phase_image"] = phase

    ts_phase = run_totalseg_phase(patient)

    row["totalseg_phase"] = ts_phase.get("phase", "")

    row["totalseg_phase_probability"] = ts_phase.get("probability", "")

    return row


def main():

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    patients = sys.argv[1:] or PATIENTS

    print("=" * 80)
    print("EXPERIMENT 12B")
    print("IMAGE-BASED CONTRAST PHASE")
    print("=" * 80)

    rows = []

    for number, patient in enumerate(patients, start=1):

        try:

            row = measure(patient)

        except Exception as error:

            row = {"patient": patient, "contrast_phase_image": "ERROR", "error": str(error)}

        rows.append(row)

        print(
            f"[{number:02d}/{len(patients)}] {patient}: "
            f"aorta={row.get('aorta_median_hu')} HU | "
            f"portal={row.get('portal_vein_and_splenic_vein_median_hu')} HU | "
            f"ratio={row.get('portal_to_aorta_ratio')} | "
            f"{row.get('contrast_phase_image')} | "
            f"TotalSeg: {row.get('totalseg_phase')}",
            flush=True,
        )

    phase = pd.DataFrame(rows)

    phase_csv = OUTPUT_ROOT / "Phase2_Experiment12B_ContrastPhase.csv"

    phase.to_csv(phase_csv, index=False, encoding="utf-8-sig")

    print("\nPhase table:")
    print(phase_csv)

    if len(patients) == len(PATIENTS) and CASE_TABLE.exists():

        table = pd.read_csv(CASE_TABLE).drop(columns=["contrast_phase_image"], errors="ignore")

        table = table.merge(
            phase[["patient", "contrast_phase_image", "portal_to_aorta_ratio", "totalseg_phase"]],
            on="patient",
            how="left",
        )

        table.to_csv(CASE_TABLE, index=False, encoding="utf-8-sig")

        print("\nCase table updated:")
        print(CASE_TABLE)

        print("\nDICOM phase vs image phase:")
        print(pd.crosstab(table.suggested_phase, table.contrast_phase_image, margins=True).to_string())

    print(
        "\nIMPORTANT:"
        "\nThe phase rule is a heuristic on vessel HU, not a radiologist's reading."
    )


if __name__ == "__main__":

    main()
