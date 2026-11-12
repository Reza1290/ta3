import SimpleITK as sitk
import numpy as np
from PIL import Image

vol_img = sitk.ReadImage("data/DICOMM/output_aligned/volume.mha")
mask_img = sitk.ReadImage("data/DICOMM/output_aligned/pred_fdi_mask.mha")
vol_arr = sitk.GetArrayFromImage(vol_img)
mask_arr = sitk.GetArrayFromImage(mask_img)

center_z, center_y, center_x = 180, 360, 209
# Crop window
r_min, r_max = center_z - 112, center_z + 112
c_min, c_max = center_y - 112, center_y + 112

slice_vol = vol_arr[r_min:r_max, c_min:c_max, center_x]
slice_mask = mask_arr[r_min:r_max, c_min:c_max, center_x]

print("Slice vol range:", slice_vol.min(), slice_vol.max())
print("Mean:", slice_vol.mean(), "Median:", np.median(slice_vol))

thresh = np.percentile(slice_vol, 92)
print("92nd percentile threshold:", thresh)
bright_teeth = slice_vol > thresh

unsegmented_bright = bright_teeth & (slice_mask == 0)
print("Unsegmented bright pixels count:", np.sum(unsegmented_bright))
print("Segmented bright pixels count:", np.sum(bright_teeth & (slice_mask > 0)))
