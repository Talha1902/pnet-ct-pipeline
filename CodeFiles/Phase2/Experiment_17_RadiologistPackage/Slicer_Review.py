"""3D Slicer helper for the pNET review package (Experiment 17).

In 3D Slicer open View > Python Console and paste once per session:

    exec(open(r"<package folder>/Slicer_Review.py").read())

A small "pNET review" window opens: choose a patient, click "Open", correct the
Pancreas and Tumor segments, click "Save" (or "Save and open next").

The same can be typed in the console:

    load_case("PAN_01")
    save_case("PAN_01")              # writes Returned/PAN_01_final.nii.gz

The editable segmentation is "<patient>_FINAL" with the segments "Pancreas" and
"Tumor". The model masks are in "<patient>_REFERENCE" (hidden; switch them on in the
Data module if useful). Only the FINAL segmentation is saved.
"""
import os

import numpy as np
import qt
import slicer


SEGMENTS = (("Pancreas", 1, (0.90, 0.75, 0.35)), ("Tumor", 2, (0.85, 0.20, 0.20)))

REFERENCES = (
    ("ref_dints.nii.gz", "DiNTS"),
    ("ref_nnunet.nii.gz", "nnU-Net"),
    ("ref_totalseg_pancreas.nii.gz", "TotalSegmentator"),
)

REFERENCE_COLORS = {"DiNTS": (0.30, 0.50, 0.90), "nnU-Net": (0.30, 0.75, 0.40), "TotalSegmentator": (0.60, 0.60, 0.60)}


def find_package():

    here = globals().get("__file__")

    if here and os.path.isdir(os.path.join(os.path.dirname(here), "Cases")):

        return os.path.dirname(here)

    return qt.QFileDialog.getExistingDirectory(None, "Select the review package folder (it contains Cases)")


PACKAGE = find_package()

CASES = os.path.join(PACKAGE, "Cases")

RETURNED = os.path.join(PACKAGE, "Returned")


def saved_path(patient):

    return os.path.join(RETURNED, f"{patient}_final.nii.gz")


def add_segment(segmentation, name, color, mask, ct):

    segment_id = segmentation.GetSegmentation().AddEmptySegment(name, name, color)

    if mask is not None and mask.any():

        slicer.util.updateSegmentBinaryLabelmapFromArray(mask.astype(np.uint8), segmentation, segment_id, ct)

    return segment_id


def read_label(path):

    node = slicer.util.loadLabelVolume(path, {"show": False})

    array = slicer.util.arrayFromVolume(node).copy()

    slicer.mrmlScene.RemoveNode(node)

    return array


def load_case(patient):

    slicer.mrmlScene.Clear(0)

    folder = os.path.join(CASES, patient)

    ct = slicer.util.loadVolume(os.path.join(folder, "ct.nii.gz"), {"name": f"{patient}_CT"})

    display = ct.GetDisplayNode()

    display.AutoWindowLevelOff()

    display.SetWindowLevel(400, 40)

    final = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", f"{patient}_FINAL")

    final.CreateDefaultDisplayNodes()

    final.SetReferenceImageGeometryParameterFromVolumeNode(ct)

    # a case saved before reopens with its correction, not with the start mask
    start = read_label(saved_path(patient) if os.path.exists(saved_path(patient)) else os.path.join(folder, "start_mask.nii.gz"))

    for name, value, color in SEGMENTS:

        add_segment(final, name, color, start == value, ct)

    reference = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", f"{patient}_REFERENCE")

    reference.CreateDefaultDisplayNodes()

    reference.SetReferenceImageGeometryParameterFromVolumeNode(ct)

    for filename, model in REFERENCES:

        label = read_label(os.path.join(folder, filename))

        for name, value, _ in SEGMENTS:

            if model == "TotalSegmentator" and value == 2:

                continue

            segment_id = add_segment(reference, f"{model} {name.lower()}", REFERENCE_COLORS[model], label == value, ct)

            reference.GetDisplayNode().SetSegmentVisibility(segment_id, False)

    slicer.util.setSliceViewerLayers(background=ct, fit=True)

    slicer.util.selectModule("SegmentEditor")

    editor = slicer.modules.segmenteditor.widgetRepresentation().self().editor

    editor.setSegmentationNode(final)

    if hasattr(editor, "setSourceVolumeNode"):

        editor.setSourceVolumeNode(ct)

    else:

        editor.setMasterVolumeNode(ct)

    print(f"{patient} loaded. Correct the Pancreas and Tumor segments, then save.")


def save_case(patient):

    ct = slicer.util.getNode(f"{patient}_CT")

    final = slicer.util.getNode(f"{patient}_FINAL")

    label = np.zeros(slicer.util.arrayFromVolume(ct).shape, dtype=np.uint8)

    # tumor is written last, so it wins where the two segments overlap
    for name, value, _ in SEGMENTS:

        segment_id = final.GetSegmentation().GetSegmentIdBySegmentName(name)

        if segment_id:

            label[slicer.util.arrayFromSegmentBinaryLabelmap(final, segment_id, ct) > 0] = value

    output = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", f"{patient}_saved")

    output.CopyOrientation(ct)

    slicer.util.updateVolumeFromArray(output, label)

    os.makedirs(RETURNED, exist_ok=True)

    slicer.util.saveNode(output, saved_path(patient))

    slicer.util.saveNode(final, saved_path(patient).replace(".nii.gz", ".seg.nrrd"))

    slicer.mrmlScene.RemoveNode(output)

    print(f"Saved {saved_path(patient)}  (pancreas {int((label == 1).sum())} voxels, tumor {int((label == 2).sum())} voxels)")


class ReviewPanel:
    """Small window: choose a patient, Open, Save, Save and open next."""

    def __init__(self):

        self.patients = sorted(name for name in os.listdir(CASES) if os.path.isdir(os.path.join(CASES, name)))

        # same order as Review_Sheet.xlsx (most uncertain cases first)
        order_file = os.path.join(PACKAGE, "review_order.txt")

        if os.path.exists(order_file):

            with open(order_file) as file:

                order = [line.strip() for line in file if line.strip() in self.patients]

            self.patients = order + [p for p in self.patients if p not in order]

        self.current = None

        self.window = qt.QWidget()

        self.window.setWindowTitle("pNET review")

        self.window.setWindowFlags(self.window.windowFlags() | qt.Qt.WindowStaysOnTopHint)

        layout = qt.QVBoxLayout(self.window)

        self.choice = qt.QComboBox()

        layout.addWidget(self.choice)

        buttons = qt.QHBoxLayout()

        for text, action in (("Open", self.open), ("Save", self.save), ("Save and open next", self.save_next)):

            button = qt.QPushButton(text)

            button.connect("clicked()", action)

            buttons.addWidget(button)

        layout.addLayout(buttons)

        self.status = qt.QLabel("Choose a patient and click Open.")

        self.status.setWordWrap(True)

        layout.addWidget(self.status)

        self.refresh()

        self.window.show()

    def refresh(self):

        index = max(self.choice.currentIndex, 0)

        self.choice.clear()

        for patient in self.patients:

            self.choice.addItem(f"{patient}   (saved)" if os.path.exists(saved_path(patient)) else patient)

        self.choice.setCurrentIndex(index)

        done = sum(os.path.exists(saved_path(p)) for p in self.patients)

        self.window.setWindowTitle(f"pNET review - {done} of {len(self.patients)} saved")

    def open(self):

        self.current = self.patients[self.choice.currentIndex]

        load_case(self.current)

        self.status.setText(f"{self.current} is open. Correct the Pancreas and Tumor segments, then click Save.")

    def save(self):

        if self.current is None:

            self.status.setText("Open a patient first.")

            return

        save_case(self.current)

        self.refresh()

        self.status.setText(f"{self.current} saved. Please fill in its row in Review_Sheet.xlsx.")

    def save_next(self):

        if self.current is None:

            self.status.setText("Open a patient first.")

            return

        self.save()

        position = self.patients.index(self.current) + 1

        if position < len(self.patients):

            self.choice.setCurrentIndex(position)

            self.open()


review_panel = ReviewPanel()
