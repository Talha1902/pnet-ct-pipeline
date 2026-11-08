pNET SEGMENTATION REVIEW - INSTRUCTIONS
=======================================

Thank you for helping with this study. A computer program has drawn the pancreas
and the tumor on 38 anonymised contrast-enhanced CT scans of patients with a
pancreatic neuroendocrine tumor (public TCIA dataset CTpred-Sunitinib-panNET).
Please check each drawing and correct it where needed. Your corrected masks are
the reference we compare the program with. You will not see grade, Ki-67 or any
clinical information.


WHAT IS IN THIS FOLDER
----------------------
Cases/PAN_xx/        one folder per patient (the CT and the computer's drawing)
Review_Sheet.xlsx    one row per patient, in the order to review
Slicer_Review.py     opens and saves the cases in 3D Slicer
Returned/            your corrected masks are saved here automatically


HOW TO DO IT (3D Slicer 5.x, free from slicer.org)
--------------------------------------------------
1. Open 3D Slicer, then View > Python Console. Paste this line once
   (change the path to where this folder is) and press Enter:

       exec(open(r"D:\pNET_Review\Package\Slicer_Review.py").read())

   A small window "pNET review" opens.

2. Choose a patient in the list and click "Open".
   The CT appears with the computer's drawing:
       Pancreas = yellow     Tumor = red
   The Segment Editor is open for editing.

3. Correct the drawing with the usual tools (Paint, Erase, Scissors, ...).

4. Click "Save" (or "Save and open next").

5. Fill in the patient's row in Review_Sheet.xlsx.

You can stop at any time. Patients already saved show "(saved)" in the list, and
open again with your correction.


WHAT TO SEGMENT
---------------
- Pancreas: the whole gland.
- Tumor: the primary pancreatic NET, including necrotic, cystic and calcified
  parts. If there are several pancreatic lesions, segment all of them and note
  this in the sheet. Do not segment liver metastases or lymph nodes (the
  computer sometimes marks them; please remove them).
- If you see no tumor, leave the Tumor segment empty and choose
  "no tumor present" in the sheet.


REVIEW SHEET COLUMNS
--------------------
Pancreas / Tumor : what you did (accepted, minor edit, major edit, redrawn, ...)
Number of lesions: pancreatic lesions you segmented
Your confidence  : 1 = very unsure, 5 = certain
Time (min)       : approximate time for the case
Notes            : anything unusual (artefacts, anatomy, image quality)


WHEN YOU ARE DONE
-----------------
Send back the Returned folder and Review_Sheet.xlsx.

Optional: the masks of the individual programs are loaded hidden in
"PAN_xx_REFERENCE" (Data module). Please judge the image, not the programs.

Questions: please contact me any time.
