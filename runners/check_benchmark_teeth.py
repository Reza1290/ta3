import SimpleITK as sitk
import numpy as np
from pathlib import Path

vol_path = Path("data/DICOM_BENCHMARK/output/volume.mha")
pred_mask_path = Path("data/DICOM_BENCHMARK/output/pred_fdi_mask.mha")

vol = sitk.ReadImage(str(vol_path))
mask = sitk.ReadImage(str(pred_mask_path))

vol_arr = sitk.GetArrayFromImage(vol)
mask_arr = sitk.GetArrayFromImage(mask)

spacing = mask.GetSpacing()
print(f"Volume shape (Z, Y, X): {vol_arr.shape}")
print(f"Voxel spacing: {spacing}")

unique_labels = np.unique(mask_arr)
print(f"Unique labels in pred_fdi_mask: {unique_labels.tolist()}")

m2_list = [17, 27, 37, 47]
m3_list = [18, 28, 38, 48]

print("\n--- Molar Detection Status ---")
for q, (m2, m3) in enumerate(zip(m2_list, m3_list), 1):
    c2 = np.sum(mask_arr == m2)
    c3 = np.sum(mask_arr == m3)
    print(f"Q{q}: M2 ({m2}) voxels={c2}, M3 ({m3}) voxels={c3}")

# Let's also check if there is a ground truth file for ToothFairy2F_001
gt_candidates = [
    Path("data/toothfairy2/Dataset112_ToothFairy2/labelsTr/ToothFairy2F_001.nii.gz"),
    Path("data/toothfairy2/Dataset112_ToothFairy2/labelsTr/ToothFairy2F_001.mha"),
    Path("data/toothfairy2/ToothFairy2F_001.mha"),
    Path("data/DICOM_BENCHMARK/ToothFairy2F_001.mha")
]
for p in gt_candidates:
    if p.exists():
        print(f"Found GT file: {p}")
        gt = sitk.GetArrayFromImage(sitk.ReadImage(str(p)))
        u_gt = np.unique(gt)
        print(f"  GT unique labels: {u_gt.tolist()}")
        for q, (m2, m3) in enumerate(zip(m2_list, m3_list), 1):
            print(f"  GT Q{q}: M2 ({m2})={np.sum(gt==m2)}, M3 ({m3})={np.sum(gt==m3)}")
