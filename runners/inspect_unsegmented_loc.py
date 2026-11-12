import SimpleITK as sitk
import numpy as np
from PIL import Image, ImageDraw

vol_img = sitk.ReadImage("data/DICOMM/output_aligned/volume.mha")
mask_img = sitk.ReadImage("data/DICOMM/output_aligned/pred_fdi_mask.mha")
vol_arr = sitk.GetArrayFromImage(vol_img)
mask_arr = sitk.GetArrayFromImage(mask_img)

center_z, center_y, center_x = 180, 360, 209
r_min, r_max = center_z - 112, center_z + 112
c_min, c_max = center_y - 112, center_y + 112

slice_vol = vol_arr[r_min:r_max, c_min:c_max, center_x]
slice_mask = mask_arr[r_min:r_max, c_min:c_max, center_x]

# Find where unsegmented bright pixels are:
thresh = np.percentile(slice_vol, 92)
bright_teeth = slice_vol > thresh
unsegmented = bright_teeth & (slice_mask == 0)

# Let's inspect where unsegmented pixels are in row/col:
unseg_pts = np.argwhere(unsegmented)
print(f"Unsegmented bright points: {len(unseg_pts)}")
print(f"Rows (Z/vertical): [{unseg_pts[:,0].min()}, {unseg_pts[:,0].max()}], Cols (Y/anterior-posterior): [{unseg_pts[:,1].min()}, {unseg_pts[:,1].max()}]")
print(f"Centroid of unsegmented: {unseg_pts.mean(axis=0)}")

# Let's see what is nearby:
# Is it the alveolar bone, maxilla sinus floor, or another tooth crown/root?
