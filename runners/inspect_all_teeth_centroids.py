import SimpleITK as sitk
import numpy as np

vol_img = sitk.ReadImage("data/DICOMM/output_aligned/volume.mha")
mask_img = sitk.ReadImage("data/DICOMM/output_aligned/pred_fdi_mask.mha")
vol_arr = sitk.GetArrayFromImage(vol_img)
mask_arr = sitk.GetArrayFromImage(mask_img)

# Let's inspect all teeth centroids:
print("Centroids of all detected teeth in mask_arr:")
for fdi in sorted(np.unique(mask_arr)):
    if fdi == 0:
        continue
    pts = np.argwhere(mask_arr == fdi)
    cz, cy, cx = np.round(pts.mean(axis=0)).astype(int)
    print(f"FDI {fdi:2d}: Z={cz:3d}, Y={cy:3d}, X={cx:3d}")
