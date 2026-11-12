"""Anatomical Tuning and Refinement for FDI 28 (Impacted Horizontal M3) & FDI 27 (M2).

Corrects:
1. False leakage of Tooth 27 into posterior tuberosity (Y > 385, Z < 200) under Tooth 28.
2. Complete, accurate segmentation of Tooth 28's horizontal crown (enamel) and roots (dentin/pulp)
   without spilling into maxillary cortical bone or sinus floor.
"""

from pathlib import Path
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    binary_fill_holes,
    binary_opening,
    binary_closing,
    generate_binary_structure,
    label as ndi_label,
)
from scipy.ndimage import distance_transform_edt


def tune_pratiwi_molars():
    vol_path = Path(r"D:\Workspace\TA3\data\DICOMM\output_aligned\volume.mha")
    raw_mask_path = Path(r"D:\Workspace\TA3\data\DICOMM\output_aligned\pred_fdi_mask.mha")
    
    print("[*] Loading volume and raw mask...")
    vol_itk = sitk.ReadImage(str(vol_path))
    raw_mask_itk = sitk.ReadImage(str(raw_mask_path))
    
    vol = sitk.GetArrayFromImage(vol_itk)  # (640, 704, 704)
    mask = sitk.GetArrayFromImage(raw_mask_itk)
    
    struct_3d = generate_binary_structure(3, 1)
    
    # -------------------------------------------------------------
    # Step 1: Interocclusal Plane Cut (Fix Maxilla vs Mandible)
    # -------------------------------------------------------------
    # Maxillary teeth cusps end at Z = 286, Mandibular at Z = 288
    z_split = 287
    leak_mandible = (mask >= 31) & (mask <= 38) & (np.arange(640)[:, None, None] < z_split) & (np.arange(704)[None, None, :] < 350)
    mask[leak_mandible] = 0
    
    leak_maxilla = (mask >= 21) & (mask <= 28) & (np.arange(640)[:, None, None] >= z_split) & (np.arange(704)[None, None, :] < 350)
    mask[leak_maxilla] = 0
    
    # -------------------------------------------------------------
    # Step 2: Accurate Segmentation of Tooth 28 (Horizontal Impacted M3)
    # -------------------------------------------------------------
    # True anatomical region of Tooth 28:
    # Z in [135, 192], Y in [345, 432], X in [172, 238]
    z_min_28, z_max_28 = 135, 192
    y_min_28, y_max_28 = 345, 432
    x_min_28, x_max_28 = 172, 238
    
    roi_28 = np.zeros_like(mask, dtype=bool)
    roi_28[z_min_28:z_max_28, y_min_28:y_max_28, x_min_28:x_max_28] = True
    
    # Tooth 28 Enamel core (HU > 1150)
    enamel_28 = (vol > 1150) & roi_28 & (np.arange(704)[None, :, None] >= 390)  # Crown is at posterior Y >= 390
    
    # Tooth 28 Dentin body (HU > 680)
    dentin_28 = (vol > 680) & roi_28
    
    # Use connected component from enamel core to grow through dentin
    labeled_dentin, num_features = ndi_label(dentin_28, structure=struct_3d)
    
    # Find which component overlaps the enamel core
    overlapping_labels = np.unique(labeled_dentin[enamel_28])
    overlapping_labels = overlapping_labels[overlapping_labels > 0]
    
    tooth_28_mask = np.zeros_like(mask, dtype=bool)
    for lbl in overlapping_labels:
        tooth_28_mask |= (labeled_dentin == lbl)
        
    # Fill internal pulp chamber holes slice-by-slice
    for z in range(z_min_28, z_max_28):
        tooth_28_mask[z] = binary_fill_holes(tooth_28_mask[z])
        
    # Morphological opening to detach thin cortical bone connections to the sinus wall
    tooth_28_clean = binary_opening(tooth_28_mask, structure=struct_3d, iterations=1)
    # Keep only the largest single component (the tooth itself)
    lbl_clean, n_c = ndi_label(tooth_28_clean, structure=struct_3d)
    if n_c > 0:
        sizes = [np.sum(lbl_clean == i) for i in range(1, n_c + 1)]
        main_lbl = np.argmax(sizes) + 1
        tooth_28_final = (lbl_clean == main_lbl)
    else:
        tooth_28_final = tooth_28_clean
        
    # Slight dilation (1 voxel) to recover outer enamel boundary
    tooth_28_final = binary_dilation(tooth_28_final, structure=struct_3d, iterations=1)
    
    print(f"[+] Tooth 28 tuned cleanly: {tooth_28_final.sum()} voxels.")
    
    # -------------------------------------------------------------
    # Step 3: Accurate Segmentation of Tooth 27 (Vertical M2)
    # -------------------------------------------------------------
    # Tooth 27 is an erupted vertical tooth:
    # Z in [195, 286], Y in [335, 388], X in [175, 235]
    z_min_27, z_max_27 = 195, 287
    y_min_27, y_max_27 = 335, 388
    x_min_27, x_max_27 = 175, 235
    
    roi_27 = np.zeros_like(mask, dtype=bool)
    roi_27[z_min_27:z_max_27, y_min_27:y_max_27, x_min_27:x_max_27] = True
    
    # Tooth 27 Enamel crown (HU > 1150 at Z > 240)
    enamel_27 = (vol > 1150) & roi_27 & (np.arange(640)[:, None, None] >= 240)
    dentin_27 = (vol > 650) & roi_27
    
    labeled_dentin_27, n_27 = ndi_label(dentin_27, structure=struct_3d)
    overlap_27 = np.unique(labeled_dentin_27[enamel_27])
    overlap_27 = overlap_27[overlap_27 > 0]
    
    tooth_27_mask = np.zeros_like(mask, dtype=bool)
    for lbl in overlap_27:
        tooth_27_mask |= (labeled_dentin_27 == lbl)
        
    for z in range(z_min_27, z_max_27):
        tooth_27_mask[z] = binary_fill_holes(tooth_27_mask[z])
        
    tooth_27_clean = binary_opening(tooth_27_mask, structure=struct_3d, iterations=1)
    lbl_clean_27, n_c27 = ndi_label(tooth_27_clean, structure=struct_3d)
    if n_c27 > 0:
        sizes27 = [np.sum(lbl_clean_27 == i) for i in range(1, n_c27 + 1)]
        main_lbl27 = np.argmax(sizes27) + 1
        tooth_27_final = (lbl_clean_27 == main_lbl27)
    else:
        tooth_27_final = tooth_27_clean
        
    tooth_27_final = binary_dilation(tooth_27_final, structure=struct_3d, iterations=1)
    print(f"[+] Tooth 27 tuned cleanly: {tooth_27_final.sum()} voxels.")
    
    # -------------------------------------------------------------
    # Step 4: Integrate Tuned Teeth into Master Mask
    # -------------------------------------------------------------
    tuned_mask = mask.copy()
    # Clear old 27 and 28 in Q2 molar area
    tuned_mask[(tuned_mask == 28) | (tuned_mask == 27)] = 0
    
    # Assign Tooth 27
    tuned_mask[tooth_27_final] = 27
    # Assign Tooth 28 (priority over 28's space, ensuring no overlap)
    tuned_mask[tooth_28_final & ~tooth_27_final] = 28
    
    # Verify no overlap
    assert np.logical_and(tuned_mask == 27, tuned_mask == 28).sum() == 0
    
    # Save tuned mask
    out_tuned_path = Path(r"D:\Workspace\TA3\data\DICOMM\output_aligned\pred_fdi_mask_clinically_tuned.mha")
    print(f"[*] Saving tuned mask to {out_tuned_path}...")
    tuned_itk = sitk.GetImageFromArray(tuned_mask)
    tuned_itk.SetSpacing(vol_itk.GetSpacing())
    tuned_itk.SetOrigin(vol_itk.GetOrigin())
    tuned_itk.SetDirection(vol_itk.GetDirection())
    
    sitk.WriteImage(tuned_itk, str(out_tuned_path), True)
    print("[+] Successfully saved tuned mask!")
    
    return vol, mask, tuned_mask


if __name__ == "__main__":
    tune_pratiwi_molars()
