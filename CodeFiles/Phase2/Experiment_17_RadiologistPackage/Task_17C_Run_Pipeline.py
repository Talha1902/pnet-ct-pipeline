"""Experiment 17C - the frozen pipeline for new CT scans (e.g. the hospital cohort).

Runs every step that Exp 10-17 validated on the 38-patient cohort, without changing any
model or rule, so new scans are processed exactly like the study cohort:

    1. input     : choose the DICOM series and convert it without resampling (10D),
                   or take a NIfTI as it is
    2. TotalSeg  : organ ROI (pancreas, liver, vessels, ...) and contrast phase (12B)
    3. checks    : slice spacing (Exp 15), HU range (10D), contrast phase (12B),
                   pancreas found and not cut by the scan edge (12A, PAN_14)
    4. DiNTS     : pretrained bundle on the whole scan (= 11A, reproduced by Exp 14 S0)
    5. nnU-Net   : v1 Task007 3d_fullres, 5 folds + TTA (Exp 16), in its own venv
    6. clean-up  : 13-B for both models, confidence level from their agreement (16)
    7. package   : start mask + reference layers per case for Slicer_Review.py (17A)

Input folder: one entry per case, either <case>.nii.gz or a folder <case>/ holding the
DICOM files (sub-folders allowed). Every step is cached, so a re-run continues where it
stopped.

    python Task_17C_Run_Pipeline.py --input /path/to/cases --run hospital_batch1
    python Task_17C_Run_Pipeline.py --validate PAN_09 PAN_14   -> cohort CTs; checks that
                                                                  the output equals Exp 13/16

Run on the GPU node (SLURM): see run_17C.sbatch
"""
from pathlib import Path
import argparse
import json
import os
import re
import shutil
import subprocess
import sys

import nibabel as nib
import numpy as np
import pandas as pd
import SimpleITK as sitk

ROOT = Path(__file__).resolve().parents[3]

HERE = Path(__file__).resolve().parent

for experiment in ("Experiment_12_CaseTable", "Experiment_13_MaskCleanup", "Experiment_14_RescueFailures", "Experiment_16_SecondModel"):

    sys.path.insert(0, str(ROOT / "CodeFiles" / "Phase2" / experiment))

sys.path.insert(0, str(HERE))

from Task_12B_Contrast_Phase import ROI_SUBSET, classify_phase, label_ids, median_hu  # noqa: E402
from Task_13_Mask_Cleanup import TOTALSEG_PANCREAS, dice, fill_holes, keep_anchored, remove_small_tumor  # noqa: E402
from Task_16_Compare_Models import compare_masks, load  # noqa: E402
from Task_17A_Build_Package import PRIORITY, save, write_review_sheet  # noqa: E402


OUTPUT_ROOT = ROOT / "Results" / "_Phase2_Experiment17C_Pipeline"

COHORT_CT_ROOT = ROOT / "Results" / "_Phase2_Experiment10D_Baseline_NIfTI"

NNUNET_RESULTS = ROOT / "Models" / "nnUNet_v1_results"

NNUNET_PREDICT = Path.home() / "venvs" / "nnunetv1" / "bin" / "nnUNet_predict"

PANCREAS, TUMOR = 1, 2

# a DICOM series with fewer images is a scout or a single slice, not a CT volume
MIN_IMAGES = 50

# Exp 15: tumor candidates change from 1.5 mm and are unreliable from 3 mm
THIN_MM, THICK_MM = 1.5, 3.0

# a missing pancreas or no contrast means the models were never meant for this scan
UNSUITABLE_FLAGS = {"NO_CONTRAST", "NO_PANCREAS_FOUND"}


def case_name(path):

    name = path.name.removesuffix(".gz").removesuffix(".nii")

    return re.sub(r"[^A-Za-z0-9_-]", "_", name)


def list_series(folder):
    """Every DICOM series under folder, with the tags used to choose one."""

    rows = []

    for directory, _, files in os.walk(folder):

        if not files:

            continue

        for uid in sitk.ImageSeriesReader.GetGDCMSeriesIDs(directory) or []:

            names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(directory, uid)

            reader = sitk.ImageFileReader()

            reader.SetFileName(names[0])

            reader.ReadImageInformation()

            def tag(key):

                return reader.GetMetaData(key).strip() if reader.HasMetaDataKey(key) else ""

            try:

                cosines = np.array(tag("0020|0037").split("\\"), dtype=float)

                axial = abs(np.cross(cosines[:3], cosines[3:])[2]) > 0.9

            except ValueError:

                axial = False

            try:

                thickness = float(tag("0018|0050"))

            except ValueError:

                thickness = np.nan

            rows.append({
                "directory": directory,
                "series_uid": uid,
                "study_uid": tag("0020|000d"),
                "study_date": tag("0008|0020"),
                "modality": tag("0008|0060"),
                "image_type": tag("0008|0008"),
                "description": tag("0008|103e"),
                "slice_thickness_mm": thickness,
                "images": len(names),
                "axial": axial,
            })

    return rows


def choose_series(series):
    """CT volume, ORIGINAL before DERIVED, axial, thinnest slices, then most images."""

    candidates = [s for s in series if s["modality"] == "CT" and s["images"] >= MIN_IMAGES]

    if not candidates:

        return None

    return sorted(
        candidates,
        key=lambda s: (
            "DERIVED" in s["image_type"].upper(),
            not s["axial"],
            np.nan_to_num(s["slice_thickness_mm"], nan=99.0),
            -s["images"],
        ),
    )[0]


def prepare(source, case_dir):
    """Write case_dir/ct.nii.gz; returns a row describing the input."""

    ct_path = case_dir / "ct.nii.gz"

    row = {"source": str(source)}

    if source.is_file():

        if not ct_path.exists():

            shutil.copy2(source, ct_path)

        row["input_type"] = "NIfTI"

        return row

    series = list_series(source)

    chosen = choose_series(series)

    if chosen is None:

        raise ValueError(f"no CT series with at least {MIN_IMAGES} images in {source}")

    pd.DataFrame(series).assign(chosen=lambda t: t.series_uid == chosen["series_uid"]).to_csv(
        case_dir / "dicom_series.csv", index=False, encoding="utf-8-sig"
    )

    if not ct_path.exists():

        reader = sitk.ImageSeriesReader()

        reader.SetFileNames(sitk.ImageSeriesReader.GetGDCMSeriesFileNames(chosen["directory"], chosen["series_uid"]))

        sitk.WriteImage(reader.Execute(), str(ct_path), True)

    row.update({
        "input_type": "DICOM",
        "series_found": len(series),
        "studies_found": len({s["study_uid"] for s in series}),
        "series_description": chosen["description"],
        "series_image_type": chosen["image_type"],
        "study_date": chosen["study_date"],
    })

    return row


def run_totalsegmentator(case_dir):

    from totalsegmentator.python_api import totalsegmentator

    seg_path = case_dir / "totalseg_roi.nii.gz"

    if not seg_path.exists():

        totalsegmentator(str(case_dir / "ct.nii.gz"), str(seg_path), ml=True, roi_subset=ROI_SUBSET, device="gpu", quiet=True)

    phase_path = case_dir / "totalseg_phase.json"

    if not phase_path.exists():

        try:

            subprocess.run(
                [str(Path(sys.executable).with_name("totalseg_get_phase")), "-i", str(case_dir / "ct.nii.gz"), "-o", str(phase_path), "-q"],
                check=True,
                capture_output=True,
                timeout=1800,
            )

        except Exception as error:

            print(f"    totalseg_get_phase failed: {error}", flush=True)

    return json.loads(phase_path.read_text()) if phase_path.exists() else {}


def input_checks(case_dir, ts_phase):
    """Flags for scans that differ from what the pipeline was validated on."""

    ct_img = nib.load(str(case_dir / "ct.nii.gz"))

    ct = np.asanyarray(ct_img.dataobj).astype(np.float32)

    seg = np.asanyarray(nib.load(str(case_dir / "totalseg_roi.nii.gz")).dataobj)

    spacing = np.array(ct_img.header.get_zooms()[:3], dtype=float)

    ids = label_ids()

    aorta = median_hu(ct, seg, ids["aorta"])[0]

    portal = median_hu(ct, seg, ids["portal_vein_and_splenic_vein"])[0]

    phase, ratio = classify_phase(aorta, portal)

    pancreas_slices = np.flatnonzero((seg == ids["pancreas"]).any(axis=(0, 1)))

    flags = []

    if spacing[2] >= THICK_MM:

        flags.append("SLICES_3MM_OR_MORE")

    elif spacing[2] > THIN_MM:

        flags.append("SLICES_OVER_1.5MM")

    if ct.min() > -500 or ct.max() < 200:

        flags.append("CHECK_HU")

    if phase == "NATIVE":

        flags.append("NO_CONTRAST")

    elif phase in ("UNDETERMINED", "DELAYED_OR_LOW_CONTRAST"):

        flags.append("PHASE_UNCLEAR")

    if len(pancreas_slices) == 0:

        flags.append("NO_PANCREAS_FOUND")

    elif pancreas_slices[0] == 0 or pancreas_slices[-1] == seg.shape[2] - 1:

        flags.append("PANCREAS_AT_SCAN_EDGE")

    status = "UNSUITABLE" if UNSUITABLE_FLAGS & set(flags) else "WARN" if flags else "OK"

    return {
        "size": "x".join(map(str, ct.shape)),
        "spacing_mm": "x".join(f"{s:.2f}" for s in spacing),
        "aorta_median_hu": round(aorta, 1) if not np.isnan(aorta) else np.nan,
        "portal_median_hu": round(portal, 1) if not np.isnan(portal) else np.nan,
        "portal_to_aorta_ratio": round(ratio, 3) if not np.isnan(ratio) else np.nan,
        "phase_hu_rule": phase,
        "phase_totalseg": ts_phase.get("phase", ""),
        "input_flags": " | ".join(flags),
        "input_status": status,
    }


def run_dints(network, case_dir):

    from Task_14_Rescue_Failures import PREPROCESSING, predict, to_original_space

    out_path = case_dir / "dints_raw.nii.gz"

    if out_path.exists():

        return

    data = PREPROCESSING({"image": str(case_dir / "ct.nii.gz"), "totalseg": str(case_dir / "totalseg_roi.nii.gz")})

    label = to_original_space(data, predict(network, data["image"]))

    save(label, nib.load(str(case_dir / "ct.nii.gz")), out_path)


def run_nnunet(run_root, cases):
    """One nnUNet_predict call for every case without a prediction yet."""

    input_dir, pred_dir = run_root / "nnunet_input", run_root / "nnunet_predictions"

    todo = [case for case in cases if not (pred_dir / f"{case}.nii.gz").exists()]

    if not todo:

        return

    input_dir.mkdir(parents=True, exist_ok=True)

    pred_dir.mkdir(parents=True, exist_ok=True)

    for link in input_dir.glob("*_0000.nii.gz"):

        link.unlink()

    for case in todo:

        (input_dir / f"{case}_0000.nii.gz").symlink_to(run_root / "cases" / case / "ct.nii.gz")

    env = dict(
        os.environ,
        RESULTS_FOLDER=str(NNUNET_RESULTS),
        nnUNet_raw_data_base=str(run_root / "nnunet_raw_unused"),
        nnUNet_preprocessed=str(run_root / "nnunet_preprocessed_unused"),
        # torch 2.6 defaults to weights_only loading, which the v1 checkpoints do not support
        TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1",
    )

    subprocess.run(
        [str(NNUNET_PREDICT), "-i", str(input_dir), "-o", str(pred_dir), "-t", "Task007_Pancreas", "-m", "3d_fullres",
         "--num_threads_preprocessing", "4", "--num_threads_nifti_save", "2"],
        check=True,
        env=env,
    )


def finish_case(run_root, case):
    """13-B clean-up of both models, confidence level and the review folder."""

    case_dir = run_root / "cases" / case

    dints_raw, img = load(case_dir / "dints_raw.nii.gz")

    nnunet_raw, nn_img = load(run_root / "nnunet_predictions" / f"{case}.nii.gz")

    if nnunet_raw.shape != dints_raw.shape or not np.allclose(nn_img.affine, img.affine, atol=1e-3):

        raise ValueError(f"nnU-Net grid {nnunet_raw.shape} does not match DiNTS grid {dints_raw.shape}")

    spacing = np.array(img.header.get_zooms()[:3], dtype=float)

    voxel_ml = float(np.prod(spacing) / 1000)

    ts_roi = np.asanyarray(nib.load(str(case_dir / "totalseg_roi.nii.gz")).dataobj).astype(np.int16)

    ts_pancreas = ts_roi == TOTALSEG_PANCREAS

    dints, anchor_used = keep_anchored(dints_raw, spacing, ts_pancreas)

    dints = fill_holes(remove_small_tumor(dints, voxel_ml)[0])[0]

    row, nnunet = compare_masks(dints, nnunet_raw, ts_roi, spacing)

    row["dints_anchor_used"] = anchor_used

    organ_failure = row["confidence"] == "LOW_ORGAN_DISAGREE"

    start = np.where(ts_pancreas, PANCREAS, 0).astype(np.uint8) if organ_failure else nnunet

    row["start_mask_source"] = "TotalSegmentator pancreas (no tumor)" if organ_failure else "nnU-Net (13-B clean-up)"

    row["start_pancreas_ml"] = round(float((start == PANCREAS).sum() * voxel_ml), 3)

    row["start_tumor_ml"] = round(float((start == TUMOR).sum() * voxel_ml), 3)

    save(dints, img, case_dir / "dints_clean.nii.gz")

    save(nnunet, img, case_dir / "nnunet_clean.nii.gz")

    review = run_root / "Package" / "Cases" / case

    review.mkdir(parents=True, exist_ok=True)

    shutil.copy2(case_dir / "ct.nii.gz", review / "ct.nii.gz")

    save(start, img, review / "start_mask.nii.gz")

    save(dints, img, review / "ref_dints.nii.gz")

    save(nnunet_raw, img, review / "ref_nnunet.nii.gz")

    save(ts_pancreas.astype(np.uint8), img, review / "ref_totalseg_pancreas.nii.gz")

    return row


def validate(run_root, table):
    """Cohort cases only: the frozen pipeline must reproduce Exp 12B, 13 and 16."""

    exp16 = pd.read_csv(ROOT / "Results" / "_Phase2_Experiment16_SecondModel" / "Phase2_Experiment16_SecondModel.csv").set_index("patient")

    rows = []

    for case in table.loc[table.confidence.notna(), "case"]:

        case_dir = run_root / "cases" / case

        def same(new, old, label=None):

            a, b = load(new)[0], load(old)[0]

            return round(dice(a == label, b == label) if label else dice(a > 0, b > 0), 4)

        rows.append({
            "case": case,
            "totalseg_pancreas_dice_vs_12B": same(case_dir / "totalseg_roi.nii.gz", ROOT / "Results" / "_Phase2_Experiment12B_ContrastPhase" / case / "totalseg_roi.nii.gz", TOTALSEG_PANCREAS),
            "dints_clean_dice_vs_13B": same(case_dir / "dints_clean.nii.gz", ROOT / "Results" / "_Phase2_Experiment13_MaskCleanup" / case / "dints_clean.nii.gz"),
            "nnunet_raw_dice_vs_16": same(run_root / "nnunet_predictions" / f"{case}.nii.gz", ROOT / "Results" / "_Phase2_Experiment16_SecondModel" / "nnunet_predictions" / f"{case}.nii.gz"),
            "confidence_17C": table.set_index("case").loc[case, "confidence"],
            "confidence_16": exp16.loc[case, "confidence"],
        })

    result = pd.DataFrame(rows)

    result["same_confidence"] = result.confidence_17C == result.confidence_16

    result.to_csv(run_root / "Phase2_Experiment17C_Validation.csv", index=False, encoding="utf-8-sig")

    return result


def main():

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, help="folder with one <case>.nii.gz or <case>/ DICOM folder per case")
    parser.add_argument("--run", default="run", help="name of the output folder")
    parser.add_argument("--validate", nargs="+", metavar="PAN_xx", help="run on cohort CTs and compare with Exp 12B/13/16")
    args = parser.parse_args()

    if args.validate:

        sources = {patient: COHORT_CT_ROOT / patient / "baseline_ct.nii.gz" for patient in args.validate}

        args.run = "validation"

    elif args.input:

        sources = {case_name(path): path for path in sorted(args.input.iterdir()) if path.is_dir() or path.name.endswith((".nii", ".nii.gz"))}

    else:

        parser.error("give --input or --validate")

    run_root = OUTPUT_ROOT / args.run

    print("=" * 80)
    print("EXPERIMENT 17C")
    print(f"FROZEN PIPELINE: {len(sources)} cases -> {run_root}")
    print("=" * 80)

    rows = {}

    # 1-3: input, TotalSegmentator and checks
    for number, (case, source) in enumerate(sources.items(), start=1):

        case_dir = run_root / "cases" / case

        case_dir.mkdir(parents=True, exist_ok=True)

        row = {"case": case}

        try:

            row.update(prepare(source, case_dir))

            row.update(input_checks(case_dir, run_totalsegmentator(case_dir)))

        except Exception as error:

            row.update({"input_status": "ERROR", "error": f"input: {error}"})

        rows[case] = row

        print(f"[{number:02d}/{len(sources)}] {case}: {row.get('spacing_mm')} mm | phase {row.get('phase_hu_rule')} / {row.get('phase_totalseg')} | "
              f"{row['input_status']} {row.get('input_flags', row.get('error', ''))}", flush=True)

    ready = [case for case, row in rows.items() if row["input_status"] != "ERROR"]

    # 4: DiNTS
    from Task_14_Rescue_Failures import load_network

    network = load_network()

    for case in ready:

        try:

            run_dints(network, run_root / "cases" / case)

        except Exception as error:

            rows[case]["error"] = f"DiNTS: {error}"

        print(f"    DiNTS {case}: {'ERROR ' + rows[case]['error'] if 'error' in rows[case] else 'done'}", flush=True)

    # free the GPU for nnU-Net
    del network

    import torch

    torch.cuda.empty_cache()

    ready = [case for case in ready if "error" not in rows[case]]

    # 5: nnU-Net (separate venv)
    run_nnunet(run_root, ready)

    # 6-7: clean-up, confidence, review folder
    for case in ready:

        try:

            rows[case].update(finish_case(run_root, case))

        except Exception as error:

            rows[case]["error"] = f"finish: {error}"

        row = rows[case]

        print(f"    {case}: {row.get('confidence', 'ERROR ' + row.get('error', ''))} | tumor DiNTS {row.get('dints_tumor_candidate_ml')} mL vs "
              f"nnU-Net {row.get('nnunet_tumor_candidate_ml')} mL ({row.get('tumor_agreement')})", flush=True)

    table = pd.DataFrame(rows.values())

    if "confidence" not in table:

        table["confidence"] = np.nan

    table["priority"] = table.confidence.map(lambda level: PRIORITY[level][0] if level in PRIORITY else np.nan)

    table["what_to_check"] = table.confidence.map(lambda level: PRIORITY[level][1] if level in PRIORITY else "")

    table = table.sort_values(["priority", "case"]).reset_index(drop=True)

    table.to_csv(run_root / "Phase2_Experiment17C_Pipeline.csv", index=False, encoding="utf-8-sig")

    package = run_root / "Package"

    done = table[table.confidence.notna()]

    if len(done):

        (package / "Returned").mkdir(parents=True, exist_ok=True)

        write_review_sheet(done.rename(columns={"case": "patient"}), package)

        for name in ("Slicer_Review.py", "README_Radiologist.txt"):

            shutil.copy2(HERE / name, package / name)

    lines = [
        "EXPERIMENT 17C - FROZEN PIPELINE",
        "=" * 80,
        f"Run: {run_root}",
        f"Cases: {len(table)}, finished: {len(done)}",
        "",
        "Input status (OK / WARN / UNSUITABLE / ERROR):",
        table.input_status.value_counts().to_string(),
        "",
        "Cases with input flags:",
        table.loc[table.input_flags.fillna("") != "", ["case", "spacing_mm", "phase_hu_rule", "phase_totalseg", "input_flags", "input_status"]].to_string(index=False)
        if "input_flags" in table else "none",
        "",
        "Confidence level:",
        done.confidence.value_counts().to_string() if len(done) else "none",
        "",
        "Review order (priority 1 = first):",
        *[f"  {priority:.0f} {level:20s} {', '.join(group.case)}" for (priority, level), group in done.groupby(["priority", "confidence"])],
    ]

    if "error" in table:

        lines += ["", "Errors:", table.loc[table.error.notna(), ["case", "error"]].to_string(index=False)]

    if args.validate:

        result = validate(run_root, done)

        lines += ["", "VALIDATION against Exp 12B / 13-B / 16 (Dice 1.0 = identical):", result.to_string(index=False)]

    lines += ["", "Agreement between the models is not accuracy; the radiologist review (17B) measures accuracy."]

    summary = "\n".join(lines)

    (run_root / "Phase2_Experiment17C_Summary.txt").write_text(summary, encoding="utf-8")

    print()
    print(summary)


if __name__ == "__main__":

    main()
