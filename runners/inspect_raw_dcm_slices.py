import SimpleITK as sitk
import numpy as np

# Read slice 000.dcm and slice 639.dcm directly
dcm0 = sitk.ReadImage("data/DICOMM/000.dcm")
dcm639 = sitk.ReadImage("data/DICOMM/639.dcm")

arr0 = sitk.GetArrayFromImage(dcm0)
arr639 = sitk.GetArrayFromImage(dcm639)

print(f"Slice 000.dcm shape: {arr0.shape}, mean: {arr0.mean():.1f}, min: {arr0.min()}, max: {arr0.max()}")
print(f"Slice 639.dcm shape: {arr639.shape}, mean: {arr639.mean():.1f}, min: {arr639.min()}, max: {arr639.max()}")

# Let's inspect where teeth appear in raw DICOM slices
# Let's read slice 100, 200, 300, 400, 500
for s in [50, 150, 250, 350, 450, 550]:
    fn = f"data/DICOMM/{s:03d}.dcm"
    d = sitk.ReadImage(fn)
    a = sitk.GetArrayFromImage(d)
    # count bright bone/tooth voxels (> 800)
    cnt = np.sum(a > 800)
    print(f"Slice {s:03d}.dcm: mean={a.mean():.1f}, bright (>800)={cnt}")
