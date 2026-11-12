import SimpleITK as sitk
import numpy as np
from pathlib import Path

vol_path = Path("data/DICOMM/output_aligned/volume.mha")
mask_path = Path("data/DICOMM/output_aligned/pred_fdi_mask.mha")

vol = sitk.GetArrayFromImage(sitk.ReadImage(str(vol_path)))
mask = sitk.GetArrayFromImage(sitk.ReadImage(str(mask_path)))

for fdi in [17, 18, 27, 28, 37, 38, 47]:
    pts = np.argwhere(mask == fdi)
    if len(pts):
        z_min, z_max = pts[:, 0].min(), pts[:, 0].max()
        z_indices, counts = np.unique(pts[:, 0], return_counts=True)
        # Find where maximum area is (typically crown)
        max_z = z_indices[np.argmax(counts)]
        max_area = np.max(counts)
        print(f"FDI {fdi}: Z range [{z_min:3d}, {z_max:3d}], max area at Z={max_z:3d} ({max_area:4d} px)")
