from pathlib import Path
from collections import defaultdict
import csv
import re

import pydicom


# ============================================================
# PHASE 2 — TASK 10B
# FULL DICOM DATASET AUDIT
# ============================================================

ROOT = Path(r"C:\Users\Talha Riaz\Desktop\Thesis Folder\ctpred_sunitinib_pannet")

OUTPUT_Address = Path(r"C:\Users\Talha Riaz\Desktop\Thesis Folder\Results")

OUTPUT_DIR = OUTPUT_Address / "_Phase2_Experiment10B_Audit_DICOMDATASET"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SERIES_CSV = OUTPUT_DIR / "Phase2_DICOM_Audit_Series.csv"
PATIENT_CSV = OUTPUT_DIR / "Phase2_DICOM_Audit_Patients.csv"
SUMMARY_TXT = OUTPUT_DIR / "Phase2_DICOM_Audit_Summary.txt"


def text(value):
    if value is None:
        return ""
    return str(value).strip()


def get(ds, field):
    return text(getattr(ds, field, ""))


def classify_orientation(iop):
    """
    ImageOrientationPatient -> approximate imaging plane.
    """

    if not iop:
        return "UNKNOWN"

    try:
        values = [float(x) for x in iop]

        if len(values) != 6:
            return "UNKNOWN"

        row = values[:3]
        col = values[3:]

        normal = [
            row[1] * col[2] - row[2] * col[1],
            row[2] * col[0] - row[0] * col[2],
            row[0] * col[1] - row[1] * col[0],
        ]

        normal = [abs(x) for x in normal]

        axis = normal.index(max(normal))

        if max(normal) < 0.90:
            return "OBLIQUE"

        if axis == 2:
            return "AXIAL"

        if axis == 1:
            return "CORONAL"

        if axis == 0:
            return "SAGITTAL"

    except Exception:
        pass

    return "UNKNOWN"


def suggest_phase(
    series_description,
    protocol_name,
    study_description,
    contrast_agent,
):
    """
    This is ONLY an automatic suggestion.

    We will manually verify phases later.
    """

    combined = (
        series_description
        + " "
        + protocol_name
        + " "
        + study_description
    ).lower()

    # --------------------------------------------------------
    # Non-diagnostic series
    # --------------------------------------------------------

    if any(
        word in combined
        for word in [
            "localizer",
            "scout",
            "topogram",
            "survey",
        ]
    ):
        return "LOCALIZER"

    if any(
        word in combined
        for word in [
            "mpr",
            "reformat",
            "coronal",
            "sagittal",
            "mip",
        ]
    ):
        return "DERIVED_REFORMAT"

    # --------------------------------------------------------
    # CT phases
    # --------------------------------------------------------

    if any(
        word in combined
        for word in [
            "arterial",
            "artery",
            "pancreatic phase",
            "pancreas phase",
        ]
    ):
        return "ARTERIAL"

    if any(
        word in combined
        for word in [
            "portal venous",
            "portalvenous",
            "portal phase",
            "venous phase",
            "venous",
        ]
    ):
        return "PORTAL_VENOUS"

    if any(
        word in combined
        for word in [
            "delayed",
            "delay phase",
            "equilibrium",
        ]
    ):
        return "DELAYED"

    if any(
        word in combined
        for word in [
            "noncontrast",
            "non contrast",
            "non-contrast",
            "unenhanced",
            "precontrast",
            "without contrast",
        ]
    ):
        return "NON_CONTRAST"

    if contrast_agent:
        return "CONTRAST_UNKNOWN_PHASE"

    return "UNKNOWN"


def candidate_status(modality, orientation, phase):

    if modality.upper() != "CT":
        return "NO_NON_CT"

    if phase == "LOCALIZER":
        return "NO_LOCALIZER"

    if phase == "DERIVED_REFORMAT":
        return "REVIEW_DERIVED"

    if orientation == "AXIAL":
        return "YES_CANDIDATE"

    return "REVIEW"


# ============================================================
# READ DATASET
# ============================================================

print("=" * 80)
print("PHASE 2 — TASK 10B")
print("FULL DICOM DATASET AUDIT")
print("=" * 80)

print("\nDataset:")
print(ROOT)

if not ROOT.exists():
    raise FileNotFoundError(ROOT)


# key = patient folder + study UID + series UID
series = defaultdict(list)

all_files = [
    p
    for p in ROOT.rglob("*")
    if p.is_file()
    and OUTPUT_DIR not in p.parents
]

print(f"\nFiles found: {len(all_files):,}")

dicom_count = 0


for number, file in enumerate(all_files, 1):

    if number % 500 == 0:
        print(
            f"Scanning {number:,} / "
            f"{len(all_files):,}"
        )

    try:

        ds = pydicom.dcmread(
            str(file),
            stop_before_pixels=True,
            force=True,
        )

    except Exception:
        continue

    series_uid = get(ds, "SeriesInstanceUID")

    # Ignore files that are not image series.
    if not series_uid:
        continue

    dicom_count += 1

    relative = file.relative_to(ROOT)

    # First folder inside the dataset.
    patient_folder = relative.parts[0]

    key = (
        patient_folder,
        get(ds, "StudyInstanceUID"),
        series_uid,
    )

    series[key].append(
        {
            "patient_folder": patient_folder,

            "patient_id":
                get(ds, "PatientID"),

            "study_uid":
                get(ds, "StudyInstanceUID"),

            "series_uid":
                series_uid,

            "modality":
                get(ds, "Modality"),

            "study_description":
                get(ds, "StudyDescription"),

            "series_description":
                get(ds, "SeriesDescription"),

            "protocol_name":
                get(ds, "ProtocolName"),

            "series_number":
                get(ds, "SeriesNumber"),

            "instance_number":
                get(ds, "InstanceNumber"),

            "rows":
                get(ds, "Rows"),

            "columns":
                get(ds, "Columns"),

            "pixel_spacing":
                get(ds, "PixelSpacing"),

            "slice_thickness":
                get(ds, "SliceThickness"),

            "spacing_between_slices":
                get(ds, "SpacingBetweenSlices"),

            "image_orientation":
                getattr(
                    ds,
                    "ImageOrientationPatient",
                    None,
                ),

            "image_type":
                get(ds, "ImageType"),

            "manufacturer":
                get(ds, "Manufacturer"),

            "manufacturer_model":
                get(
                    ds,
                    "ManufacturerModelName",
                ),

            "kvp":
                get(ds, "KVP"),

            "kernel":
                get(
                    ds,
                    "ConvolutionKernel",
                ),

            "contrast_agent":
                get(
                    ds,
                    "ContrastBolusAgent",
                ),

            "contrast_route":
                get(
                    ds,
                    "ContrastBolusRoute",
                ),

            "acquisition_time":
                get(ds, "AcquisitionTime"),

            "series_time":
                get(ds, "SeriesTime"),

            "study_date":
                get(ds, "StudyDate"),

            "example_file":
                str(relative),
        }
    )


print(f"\nDICOM images found: {dicom_count:,}")
print(f"Series found: {len(series):,}")


# ============================================================
# CREATE SERIES TABLE
# ============================================================

rows = []


for key, images in series.items():

    first = images[0]

    orientation = classify_orientation(
        first["image_orientation"]
    )

    phase = suggest_phase(
        first["series_description"],
        first["protocol_name"],
        first["study_description"],
        first["contrast_agent"],
    )

    keep = candidate_status(
        first["modality"],
        orientation,
        phase,
    )

    row = {

        "patient_folder":
            first["patient_folder"],

        "dicom_patient_id":
            first["patient_id"],

        "study_date":
            first["study_date"],

        "study_instance_uid":
            first["study_uid"],

        "series_instance_uid":
            first["series_uid"],

        "series_number":
            first["series_number"],

        "modality":
            first["modality"],

        "study_description":
            first["study_description"],

        "series_description":
            first["series_description"],

        "protocol_name":
            first["protocol_name"],

        "suggested_phase":
            phase,

        "orientation":
            orientation,

        "keep_recommendation":
            keep,

        "number_of_images":
            len(images),

        "rows":
            first["rows"],

        "columns":
            first["columns"],

        "pixel_spacing_mm":
            first["pixel_spacing"],

        "slice_thickness_mm":
            first["slice_thickness"],

        "spacing_between_slices_mm":
            first["spacing_between_slices"],

        "kvp":
            first["kvp"],

        "reconstruction_kernel":
            first["kernel"],

        "manufacturer":
            first["manufacturer"],

        "manufacturer_model":
            first["manufacturer_model"],

        "contrast_agent":
            first["contrast_agent"],

        "contrast_route":
            first["contrast_route"],

        "acquisition_time":
            first["acquisition_time"],

        "series_time":
            first["series_time"],

        "image_type":
            first["image_type"],

        "example_file":
            first["example_file"],

        # We will fill these after reviewing results.
        "manual_phase":
            "",

        "manual_keep":
            "",

        "manual_notes":
            "",
    }

    rows.append(row)


rows.sort(
    key=lambda x: (
        x["patient_folder"],
        x["series_number"],
    )
)


with SERIES_CSV.open(
    "w",
    newline="",
    encoding="utf-8-sig",
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=rows[0].keys(),
    )

    writer.writeheader()
    writer.writerows(rows)


# ============================================================
# PATIENT SUMMARY
# ============================================================

patients = defaultdict(list)

for row in rows:
    patients[row["patient_folder"]].append(row)


patient_rows = []


for patient in sorted(patients):

    patient_series = patients[patient]

    phases = sorted(
        set(
            x["suggested_phase"]
            for x in patient_series
        )
    )

    candidates = [
        x
        for x in patient_series
        if x["keep_recommendation"]
        == "YES_CANDIDATE"
    ]

    patient_rows.append(
        {
            "patient":
                patient,

            "number_of_series":
                len(patient_series),

            "axial_CT_candidates":
                len(candidates),

            "suggested_phases":
                " | ".join(phases),

            "arterial_candidate":
                int(
                    any(
                        x["suggested_phase"]
                        == "ARTERIAL"
                        and
                        x["keep_recommendation"]
                        == "YES_CANDIDATE"
                        for x in patient_series
                    )
                ),

            "portal_venous_candidate":
                int(
                    any(
                        x["suggested_phase"]
                        == "PORTAL_VENOUS"
                        and
                        x["keep_recommendation"]
                        == "YES_CANDIDATE"
                        for x in patient_series
                    )
                ),

            "manual_selected_series":
                "",

            "manual_selected_phase":
                "",

            "notes":
                "",
        }
    )


with PATIENT_CSV.open(
    "w",
    newline="",
    encoding="utf-8-sig",
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=patient_rows[0].keys(),
    )

    writer.writeheader()
    writer.writerows(patient_rows)


# ============================================================
# TEXT SUMMARY
# ============================================================

phase_counts = defaultdict(int)

for row in rows:
    phase_counts[
        row["suggested_phase"]
    ] += 1


summary = []

summary.append(
    "=" * 80
)

summary.append(
    "PHASE 2 — CTpred-Sunitinib-panNET "
    "DICOM AUDIT"
)

summary.append(
    "=" * 80
)

summary.append(
    f"\nPatients detected: {len(patients)}"
)

summary.append(
    f"Series detected: {len(rows)}"
)

summary.append(
    f"DICOM images: {dicom_count}"
)

summary.append(
    "\nSuggested phase counts:"
)

for phase in sorted(phase_counts):

    summary.append(
        f"  {phase}: "
        f"{phase_counts[phase]}"
    )


summary.append(
    "\nIMPORTANT:"
)

summary.append(
    "Suggested CT phases are metadata-based "
    "estimates only."
)

summary.append(
    "Do NOT delete or convert series yet."
)

summary.append(
    "We will manually review the audit first."
)


SUMMARY_TXT.write_text(
    "\n".join(summary),
    encoding="utf-8",
)


# ============================================================
# FINISH
# ============================================================

print("\n" + "=" * 80)
print("AUDIT COMPLETE")
print("=" * 80)

print(
    f"\nPatients detected: "
    f"{len(patients)}"
)

print(
    f"Series detected: "
    f"{len(rows)}"
)

print("\nCreated:")

print(SERIES_CSV)
print(PATIENT_CSV)
print(SUMMARY_TXT)