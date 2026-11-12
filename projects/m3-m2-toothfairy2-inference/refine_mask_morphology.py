"""Morphological and Intensity-Guided Refinement for ToothFairy2 Segmentation Mask.

Fixes:
1. Strict interocclusal plane cut at Z=287:
   - Prevents FDI 37 from bleeding into FDI 27 crown.
   - Restores missing crown of FDI 27 down to occlusal contact.
2. Complete tooth cavity filling using high-density CBCT enamel/dentin (> 600 HU):
   - Expands FDI 28 to cover the whole impacted third molar crown & body.
   - Fills dental pulp & crown gaps in FDI 26 and FDI 27.
"""

from pathlib import Path
import sys
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import binary_dilation, binary_erosion, binary_fill_holes, label as ndi_label

sys.path.append(str(Path(__file__).parent))

img_path = Path(r"D:\Workspace\TA3\data\DICOMM\output_aligned\volume.mha")
mask_path = Path(r"D:\Workspace\TA3\data\DICOMM\output_aligned\pred_fdi_mask.mha")

print("[*] Loading volume and mask...")
img_itk = sitk.ReadImage(str(img_path))
mask_itk = sitk.ReadImage(str(mask_path))

arr_img = sitk.GetArrayFromImage(img_itk)  # (640, 704, 704)
arr_mask = sitk.GetArrayFromImage(mask_itk)

# 1. Enforce strict interocclusal plane between Maxilla and Mandible on Left side (X < 350)
# In original 0.16mm space:
# Upper molar cusps end around Z = 286
# Lower molar cusps start around Z = 288
# The interocclusal contact line is at Z = 287
z_split_left = 287

# Strip any mandibular label (31..38) that crossed into Maxilla (Z < z_split_left)
# and assign it to the corresponding maxillary molar if it's tooth structure!
leak_37_into_27 = (arr_mask == 37) & (np.arange(640)[:, None, None] < z_split_left) & (np.arange(704)[None, None, :] < 350)
print(f"[*] Found {leak_37_into_27.sum()} voxels of FDI 37 leaking into Maxilla. Reassigning to FDI 27...")
arr_mask[leak_37_into_27] = 27

# Strip any maxillary label (21..28) that crossed into Mandible (Z >= z_split_left)
leak_27_into_37 = (arr_mask == 27) & (np.arange(640)[:, None, None] >= z_split_left) & (np.arange(704)[None, None, :] < 350)
print(f"[*] Found {leak_27_into_37.sum()} voxels of FDI 27 leaking into Mandible. Reassigning to FDI 37...")
arr_mask[leak_27_into_37] = 37

# 2. Refine Tooth 28 (Impacted M3)
# Raw tooth 28 is high-density structure in bbox Z: 110..190, Y: 350..435, X: 175..235
# Any voxel in this box with HU > 600 connected to the FDI 28 seed belongs to FDI 28!
seed_28 = (arr_mask == 28)
roi_28_box = (
    (np.arange(640)[:, None, None] >= 110) & (np.arange(640)[:, None, None] <= 190) &
    (np.arange(704)[None, :, None] >= 340) & (np.arange(704)[None, :, None] <= 440) &
    (np.arange(704)[None, None, :] >= 170) & (np.arange(704)[None, None, :] <= 240)
)
dense_tooth_28 = (arr_img > 650) & roi_28_box

# Dilate seed within dense tooth region to capture full crown and roots
from scipy.ndimage import binary_propagation
expanded_28 = binary_propagation(seed_28, mask=dense_tooth_28)
# Fill internal holes (pulp chamber)
for z_i in range(110, 191):
    expanded_28[z_i] = binary_fill_holes(expanded_28[z_i])

print(f"[*] Tooth 28 voxels expanded from {seed_28.sum()} to {expanded_28.sum()} voxels.")
arr_mask[expanded_28] = 28

# 3. Refine Tooth 27 (Molar 2)
# High density tooth structure in bbox Z: 180..287, Y: 340..415, X: 180..240
seed_27 = (arr_mask == 27)
roi_27_box = (
    (np.arange(640)[:, None, None] >= 175) & (np.arange(640)[:, None, None] < z_split_left) &
    (np.arange(704)[None, :, None] >= 335) & (np.arange(704)[None, :, None] <= 415) &
    (np.arange(704)[None, None, :] >= 175) & (np.arange(704)[None, None, :] <= 245)
)
dense_tooth_27 = (arr_img > 650) & roi_27_box
expanded_27 = binary_propagation(seed_27, mask=dense_tooth_27)
for z_i in range(175, z_split_left):
    expanded_27[z_i] = binary_fill_holes(expanded_27[z_i])
print(f"[*] Tooth 27 voxels expanded from {seed_27.sum()} to {expanded_27.sum()} voxels.")
# Assign expanded 27, ensuring 28 is not overwritten
arr_mask[expanded_27 & ~expanded_28] = 27

# 4. Refine Tooth 26 (Molar 1)
seed_26 = (arr_mask == 26)
roi_26_box = (
    (np.arange(640)[:, None, None] >= 190) & (np.arange(640)[:, None, None] < z_split_left) &
    (np.arange(704)[None, :, None] >= 280) & (np.arange(704)[None, :, None] <= 345) &
    (np.arange(704)[None, None, :] >= 180) & (np.arange(704)[None, None, :] <= 245)
)
dense_tooth_26 = (arr_img > 650) & roi_26_box
expanded_26 = binary_propagation(seed_26, mask=dense_tooth_26)
for z_i in range(190, z_split_left):
    expanded_26[z_i] = binary_fill_holes(expanded_26[z_i])
print(f"[*] Tooth 26 voxels expanded from {seed_26.sum()} to {expanded_26.sum()} voxels.")
arr_mask[expanded_26 & ~expanded_27 & ~expanded_28] = 26

# Save polished mask
out_refined_mha = Path(r"D:\Workspace\TA3\data\DICOMM\output_aligned\pred_fdi_mask_anatomical_refined.mha")
out_itk = sitk.GetImageFromArray(arr_mask)
out_itk.CopyInformation(mask_itk)
sitk.WriteImage(out_itk, str(out_refined_mha), True)
print(f"[+] Saved anatomically refined mask to: {out_refined_mha}")
