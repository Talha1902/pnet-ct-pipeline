from pathlib import Path
from collections import defaultdict
import csv

import pydicom


ROOT = Path(
    r"C:\Users\Talha Riaz\Desktop\Thesis Folder\ctpred_sunitinib_pannet"
)

AUDIT_CSV = Path(
    r"C:\Users\Talha Riaz\Desktop\Thesis Folder\Results"
    r"\_Phase2_Experiment10B_Audit_DICOMDATASET"
    r"\Phase2_DICOM_Audit_Series.csv"
)

OUTPUT_DIR = Path(
    r"C:\Users\Talha Riaz\Desktop\Thesis Folder\Results"
    r"\_Phase2_Experiment10C_Baseline_VisualReview"
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "Phase2_Task10C_B_TimepointTags.csv"
)


def clean(value):

    if value is None:
        return ""

    return str(value).strip()


def first_dicom(folder):

    for file in folder.iterdir():

        if not file.is_file():
            continue

        try:

            ds = pydicom.dcmread(
                str(file),
                stop_before_pixels=True,
                force=True
            )

            if getattr(
                ds,
                "SeriesInstanceUID",
                None
            ):

                return ds

        except Exception:
            continue

    return None


print("=" * 80)
print("PHASE 2 — TASK 10C-B")
print("LONGITUDINAL / TIMEPOINT DICOM TAG AUDIT")
print("=" * 80)


with AUDIT_CSV.open(
    encoding="utf-8-sig"
) as file:

    audit = list(
        csv.DictReader(file)
    )


rows = []


for row in audit:

    relative_file = Path(
        row["example_file"]
    )

    folder = (
        ROOT
        / relative_file
    ).parent


    ds = first_dicom(
        folder
    )


    if ds is None:

        print(
            "Could not read:",
            folder
        )

        continue


    output = {

        "patient":
            row["patient_folder"],

        "study_date":
            row["study_date"],

        "series_number":
            row["series_number"],

        "series_instance_uid":
            row["series_instance_uid"],

        "series_description":
            row["series_description"],

        "protocol_name":
            row["protocol_name"],


        # ===========================================
        # LONGITUDINAL / CLINICAL TRIAL INFORMATION
        # ===========================================

        "ClinicalTrialTimePointID":
            clean(
                getattr(
                    ds,
                    "ClinicalTrialTimePointID",
                    ""
                )
            ),

        "ClinicalTrialTimePointDescription":
            clean(
                getattr(
                    ds,
                    "ClinicalTrialTimePointDescription",
                    ""
                )
            ),

        "LongitudinalTemporalOffsetFromEvent":
            clean(
                getattr(
                    ds,
                    "LongitudinalTemporalOffsetFromEvent",
                    ""
                )
            ),

        "LongitudinalTemporalEventType":
            clean(
                getattr(
                    ds,
                    "LongitudinalTemporalEventType",
                    ""
                )
            ),


        # ===========================================
        # ADDITIONAL DATE INFORMATION
        # ===========================================

        "StudyDate":
            clean(
                getattr(
                    ds,
                    "StudyDate",
                    ""
                )
            ),

        "SeriesDate":
            clean(
                getattr(
                    ds,
                    "SeriesDate",
                    ""
                )
            ),

        "AcquisitionDate":
            clean(
                getattr(
                    ds,
                    "AcquisitionDate",
                    ""
                )
            ),

        "ContentDate":
            clean(
                getattr(
                    ds,
                    "ContentDate",
                    ""
                )
            ),


        # ===========================================
        # ADDITIONAL DESCRIPTORS
        # ===========================================

        "BodyPartExamined":
            clean(
                getattr(
                    ds,
                    "BodyPartExamined",
                    ""
                )
            ),

        "StudyDescription_live":
            clean(
                getattr(
                    ds,
                    "StudyDescription",
                    ""
                )
            ),

        "SeriesDescription_live":
            clean(
                getattr(
                    ds,
                    "SeriesDescription",
                    ""
                )
            ),

        "ProtocolName_live":
            clean(
                getattr(
                    ds,
                    "ProtocolName",
                    ""
                )
            ),
    }


    rows.append(
        output
    )


with OUTPUT_CSV.open(
    "w",
    newline="",
    encoding="utf-8-sig"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=rows[0].keys()
    )

    writer.writeheader()

    writer.writerows(
        rows
    )


print()
print("Series checked:", len(rows))

print()
print("Created:")
print(OUTPUT_CSV)

print()
print("=" * 80)
print("TASK 10C-B COMPLETE")
print("=" * 80)