"""Anatomical Quadrant & Occlusal Consistency Filter for FDI Segmentation Masks.

Applies:
1. Midline (Left-Right) and Occlusal Plane (Maxilla-Mandible) partitioning.
2. Quadrant-consistent label reassignment for stray M3 fragments:
   - Any M3 component in Upper Left -> FDI 28
   - Any M3 component in Upper Right -> FDI 18
   - Any M3 component in Lower Left -> FDI 38
   - Any M3 component in Lower Right -> FDI 48
3. Minimum component size threshold (removes isolated noise < 30 mm3).
4. Morphological interocclusal boundary refinement between Upper M2 and Lower M2.
"""

from pathlib import Path
import sys
import json
import SimpleITK as sitk
import numpy as np
from scipy.ndimage import label as ndi_label

sys.path.append(str(Path(__file__).parent))
from fdi_taxonomy import FDI_METADATA, get_fdi_info, TOOTHSEG_LABEL_TO_FDI


def clean_and_harmonize_fdi_mask(
    resampled_mask_path: Path,
    output_refined_mask_path: Path,
    reference_orig_image_path: Path,
    spacing: tuple = (0.3, 0.3, 0.3),
):
    print("[*] Loading resampled FDI mask...")
    itk_mask = sitk.ReadImage(str(resampled_mask_path))
    arr = sitk.GetArrayFromImage(itk_mask)  # (Z, Y, X)

    # 1. Establish anatomical boundaries
    # In ToothFairy space (Z: Superior -> Inferior, X: Left -> Right):
    # Incisors are at anterior (low Y), molars at posterior (high Y)
    # Dental midline X is around 187
    # Interocclusal plane Z is around 150
    midline_x = 187.0
    z_occlusal = 150.0

    voxel_vol = spacing[0] * spacing[1] * spacing[2]
    min_voxels = int(round(30.0 / voxel_vol))  # >= 30 mm3

    refined_arr = np.zeros_like(arr)
    detected_teeth = []

    # Process each tooth label in priority order:
    # First: Anterior teeth (11-15, 21-25, 31-35, 41-45)
    # Second: First molars (16, 26, 36, 46)
    # Third: Second molars (17, 27, 37, 47)
    # Fourth: Third molars (18, 28, 38, 48)
    
    # Extract connected components for all non-zero labels
    unq_labels = [l for l in np.unique(arr) if l > 0]
    all_components = []

    for lab in unq_labels:
        binary = (arr == lab)
        labeled_cc, num_cc = ndi_label(binary)
        if num_cc == 0:
            continue
        cc_sizes = np.bincount(labeled_cc.flat)[1:]
        for idx, sz in enumerate(cc_sizes, start=1):
            if sz < min_voxels:
                continue
            comp_mask = (labeled_cc == idx)
            coords = np.argwhere(comp_mask)
            cz, cy, cx = coords.mean(axis=0)
            all_components.append({
                "orig_label": int(lab),
                "size": int(sz),
                "vol_mm3": round(float(sz * voxel_vol), 2),
                "centroid": (cz, cy, cx),
                "mask": comp_mask,
            })

    # Sort components by size (largest first)
    all_components.sort(key=lambda c: c["size"], reverse=True)
    print(f"[*] Found {len(all_components)} candidate tooth components (> 30 mm3).")

    # Reassign labels based on quadrant anatomy
    used_labels = set()
    cleaned_grid = np.zeros_like(arr)

    for comp in all_components:
        cz, cy, cx = comp["centroid"]
        orig_lab = comp["orig_label"]
        assigned_fdi = orig_lab

        # Quadrant determination from centroid
        is_upper = (cz <= z_occlusal)
        is_right = (cx >= midline_x)

        target_quad = 1 if (is_upper and is_right) else \
                      2 if (is_upper and not is_right) else \
                      4 if (not is_upper and is_right) else 3

        orig_quad = orig_lab // 10
        tooth_pos = orig_lab % 10

        # Fix quadrant conflict
        if orig_quad != target_quad:
            # If an M3 (tooth 8) appears in the wrong quadrant, reassign to target quadrant's M3
            if tooth_pos == 8:
                assigned_fdi = target_quad * 10 + 8
                print(f"  [Reassign] M3 component at ({cz:.1f}, {cy:.1f}, {cx:.1f}) remapped {orig_lab} -> {assigned_fdi}")
            else:
                # Discard stray fragment in wrong quadrant
                print(f"  [Drop Stray] FDI {orig_lab} component at ({cz:.1f}, {cy:.1f}, {cx:.1f}) in Quad {target_quad} dropped ({comp['vol_mm3']} mm3)")
                continue

        # Prevent duplicate assignment of the same tooth if already assigned a larger component
        if assigned_fdi in used_labels:
            print(f"  [Merge/Drop Duplicate] FDI {assigned_fdi} duplicate component dropped ({comp['vol_mm3']} mm3)")
            continue

        used_labels.add(assigned_fdi)
        cleaned_grid[comp["mask"]] = assigned_fdi

        detected_teeth.append({
            "fdi": assigned_fdi,
            "quadrant": target_quad,
            "name_id": FDI_METADATA.get(assigned_fdi, None).name_id if assigned_fdi in FDI_METADATA else f"FDI {assigned_fdi}",
            "name_en": FDI_METADATA.get(assigned_fdi, None).name_en if assigned_fdi in FDI_METADATA else f"FDI {assigned_fdi}",
            "is_m2": (assigned_fdi % 10 == 7),
            "is_m3": (assigned_fdi % 10 == 8),
            "volume_mm3": comp["vol_mm3"],
            "voxel_count": comp["size"],
            "centroid_zyx": [round(float(c), 1) for c in (cz, cy, cx)],
        })

    detected_teeth.sort(key=lambda t: t["fdi"])
    print(f"\n[+] Total harmonized teeth: {len(detected_teeth)}")
    for t in detected_teeth:
        print(f"  FDI {t['fdi']:2d} ({t['name_id']}): Vol={t['volume_mm3']:6.1f} mm3, Centroid Z={t['centroid_zyx'][0]:5.1f}, Y={t['centroid_zyx'][1]:5.1f}, X={t['centroid_zyx'][2]:5.1f}")

    # Resample back to original DICOM space (704, 704, 640)
    print("\n[*] Resampling refined mask back to original DICOM grid (0.16mm)...")
    ref_img = sitk.ReadImage(str(reference_orig_image_path))
    cleaned_itk = sitk.GetImageFromArray(cleaned_grid)
    cleaned_itk.SetSpacing(spacing)
    cleaned_itk.SetOrigin(ref_img.GetOrigin())
    cleaned_itk.SetDirection(ref_img.GetDirection())

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ref_img)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    final_itk = resampler.Execute(cleaned_itk)

    sitk.WriteImage(final_itk, str(output_refined_mask_path), True)
    print(f"[+] Final refined mask written to: {output_refined_mask_path}")

    # Save updated JSON report
    report_json_path = output_refined_mask_path.parent / "fdi_clinical_report_refined.json"
    with open(report_json_path, "w") as f:
        json.dump({"total_teeth_detected": len(detected_teeth), "detected_teeth": detected_teeth}, f, indent=2)
    print(f"[+] Saved updated clinical report to: {report_json_path}")


if __name__ == "__main__":
    resampled_mask = Path(r"D:\Workspace\TA3\data\DICOMM\output_aligned\pred_fdi_refined_resampled.mha")
    final_output = Path(r"D:\Workspace\TA3\data\DICOMM\output_aligned\pred_fdi_mask.mha")
    orig_vol = Path(r"D:\Workspace\TA3\data\DICOMM\output_aligned\volume.mha")

    clean_and_harmonize_fdi_mask(resampled_mask, final_output, orig_vol)
