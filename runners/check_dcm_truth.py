import SimpleITK as sitk
import numpy as np
from PIL import Image

# Read directly from DICOM files
# Slice 265 vs Slice 374
dcm265 = sitk.GetArrayFromImage(sitk.ReadImage("data/DICOMM/265.dcm"))[0]
dcm374 = sitk.GetArrayFromImage(sitk.ReadImage("data/DICOMM/374.dcm"))[0]

print(f"Slice 265.dcm: min={dcm265.min()}, max={dcm265.max()}, mean={dcm265.mean():.1f}, bright (>800)={np.sum(dcm265 > 800)}")
print(f"Slice 374.dcm: min={dcm374.min()}, max={dcm374.max()}, mean={dcm374.mean():.1f}, bright (>800)={np.sum(dcm374 > 800)}")

# Check tooth 11 location in 265.dcm vs 374.dcm:
# In mask_aligned: tooth 11 has Y=123, X=381
print(f"Value at (Y=123, X=381) in 265.dcm: {dcm265[123, 381]}")
print(f"Value at (Y=123, X=381) in 374.dcm: {dcm374[123, 381]}")

# Save both slices as PNG to visually see which one is Maxilla and which one is Mandible!
def norm_save(arr, path):
    n = np.clip((arr - (-500)) / (1500 - (-500)) * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(n).save(path)

norm_save(dcm265, "runners/outputs/slice_265_raw.png")
norm_save(dcm374, "runners/outputs/slice_374_raw.png")
print("Saved slice_265_raw.png and slice_374_raw.png")
