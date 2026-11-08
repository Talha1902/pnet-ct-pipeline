from pathlib import Path
import argparse
import json
import subprocess
import sys
import time
import csv


ROOT = Path(r"C:\Users\Talha Riaz\Desktop\Thesis Folder")

BASELINE_ROOT = (
    ROOT
    / "Results"
    / "_Phase2_Experiment10D_Baseline_NIfTI"
)

OUTPUT_ROOT = (
    ROOT
    / "Results"
    / "_Phase2_Experiment11A_MONAI_DiNTS"
)

CODE_ROOT = (
    ROOT
    / "CodeFiles"
    / "Phase2"
    / "Experiment_11_MONAI_DiNTS"
)

BUNDLE_ROOT = (
    ROOT
    / "Models"
    / "pancreas_ct_dints_segmentation"
)

CONFIG_FILE = (
    BUNDLE_ROOT
    / "configs"
    / "inference.yaml"
)

LOG_ROOT = (
    ROOT
    / "Results"
    / "_Phase2_Experiment11B_BatchInference_All38"
)

PATIENTS = [
    f"PAN_{i:02d}"
    for i in range(1, 39)
]


def ct_path(patient):

    return (
        BASELINE_ROOT
        / patient
        / "baseline_ct.nii.gz"
    )


def prediction_path(patient):

    return (
        OUTPUT_ROOT
        / patient
        / "baseline_ct"
        / "baseline_ct_trans.nii.gz"
    )


def make_json(patient):

    json_path = (
        CODE_ROOT
        / f"{patient}_testing.json"
    )

    data = {
        "testing": [
            {
                "image":
                f"{patient}/baseline_ct.nii.gz"
            }
        ]
    }

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            indent=2
        )

    return json_path


def run_patient(
    patient,
    force=False
):

    ct = ct_path(patient)

    pred = prediction_path(
        patient
    )

    output_dir = (
        OUTPUT_ROOT
        / patient
    )

    log_file = (
        LOG_ROOT
        / f"{patient}_inference.log"
    )

    # ---------------------------------------
    # Check CT
    # ---------------------------------------

    if not ct.exists():

        return {
            "patient": patient,
            "status":
            "MISSING_BASELINE_CT",
            "runtime_sec": "",
            "prediction_path":
            str(pred)
        }

    # ---------------------------------------
    # Skip completed patients
    # ---------------------------------------

    if (
        pred.exists()
        and not force
    ):

        print(
            f"{patient}: already exists "
            "→ SKIPPED"
        )

        return {
            "patient": patient,
            "status":
            "SKIPPED_EXISTING",
            "runtime_sec": "",
            "prediction_path":
            str(pred)
        }

    json_file = make_json(
        patient
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    LOG_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    cmd = [

        sys.executable,

        "-m",
        "monai.bundle",
        "run",

        "--config_file",
        str(CONFIG_FILE),

        "--bundle_root",
        str(BUNDLE_ROOT),

        "--dataset_dir",
        str(BASELINE_ROOT),

        "--data_list_file_path",
        str(json_file),

        "--output_dir",
        str(output_dir),

        "--inferer#sw_batch_size",
        "1",

        "--dataloader#num_workers",
        "0"
    ]


    print(
        "\n" +
        "=" * 80
    )

    print(
        f"RUNNING {patient}"
    )

    print(
        "=" * 80
    )


    start = time.perf_counter()


    with open(
        log_file,
        "w",
        encoding="utf-8"
    ) as log:

        process = subprocess.run(

            cmd,

            stdout=log,

            stderr=subprocess.STDOUT,

            text=True

        )


    runtime = (
        time.perf_counter()
        - start
    )


    if process.returncode != 0:

        status = (
            "FAILED_"
            f"{process.returncode}"
        )

    elif pred.exists():

        status = "SUCCESS"

    else:

        status = (
            "FINISHED_BUT_"
            "OUTPUT_MISSING"
        )


    print(
        patient,
        status,
        f"{runtime:.1f} sec"
    )


    return {

        "patient":
        patient,

        "status":
        status,

        "runtime_sec":
        round(
            runtime,
            2
        ),

        "prediction_path":
        str(pred)
    }


def main():

    parser = (
        argparse.ArgumentParser()
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Re-run existing patients"
        )
    )

    args = parser.parse_args()


    if not CONFIG_FILE.exists():

        raise FileNotFoundError(
            CONFIG_FILE
        )


    LOG_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )


    results = []


    print(
        "\nEXPERIMENT 11B"
    )

    print(
        "38-PATIENT DiNTS "
        "BATCH INFERENCE"
    )


    for index, patient in enumerate(
        PATIENTS,
        start=1
    ):

        print(
            f"\n[{index}/38] "
            f"{patient}"
        )

        result = run_patient(
            patient,
            force=args.force
        )

        results.append(
            result
        )


    summary_file = (

        LOG_ROOT
        / "Phase2_Experiment11B_"
          "BatchInference_Summary.csv"

    )


    with open(
        summary_file,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(

            f,

            fieldnames=[
                "patient",
                "status",
                "runtime_sec",
                "prediction_path"
            ]

        )

        writer.writeheader()

        writer.writerows(
            results
        )


    print(
        "\n" +
        "=" * 80
    )

    print(
        "BATCH INFERENCE COMPLETE"
    )

    print(
        "=" * 80
    )


    counts = {}

    for r in results:

        status = r["status"]

        counts[status] = (
            counts.get(
                status,
                0
            )
            + 1
        )


    for key, value in counts.items():

        print(
            key,
            "=",
            value
        )


    print(
        "\nSummary saved:"
    )

    print(
        summary_file
    )


if __name__ == "__main__":

    main()