import SimpleITK as sitk
import numpy as np

vol_path = "data/DICOMM/output_aligned/volume.mha"
mask_path = "data/DICOMM/output_aligned/pred_fdi_mask.mha"

mask_img = sitk.ReadImage(mask_path)
mask_arr = sitk.GetArrayFromImage(mask_img)

# Let's inspect tooth 26, 27, 28 along sagittal slices X
print("Tooth voxel counts per sagittal slice X:")
for x in range(195, 225, 2):
    c26 = np.sum(mask_arr[:, :, x] == 26)
    c27 = np.sum(mask_arr[:, :, x] == 27)
    c28 = np.sum(mask_arr[:, :, x] == 28)
    print(f"Slice X={x:3d}: FDI 26={c26:4d}, FDI 27={c27:4d}, FDI 28={c28:4d}")
