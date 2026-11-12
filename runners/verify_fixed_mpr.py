from PIL import Image
import numpy as np

img = Image.open("C:/Users/rezam/.gemini/antigravity/brain/54ccded8-7443-4f4d-b5c3-d6bbe9f123b5/pratiwi_radiological_mpr.png")
arr = np.array(img)
w, h = img.size
col_w = w // 3
row_h = h // 2

p1 = img.crop((0, 0, col_w, row_h))
p2 = img.crop((col_w, 0, col_w * 2, row_h))

p1.save("runners/outputs/panel1_fixed.png")
p2.save("runners/outputs/panel2_fixed.png")

# Inspect Cyan points in panel1_fixed
arr1 = np.array(p1)
cyan_pts = np.argwhere((arr1[:, :, 0] < 50) & (arr1[:, :, 1] > 200) & (arr1[:, :, 2] > 200))
print(f"Fixed Cyan pts (Q1): {len(cyan_pts)}, Y(row) range: [{cyan_pts[:,0].min()}, {cyan_pts[:,0].max()}], X(col) range: [{cyan_pts[:,1].min()}, {cyan_pts[:,1].max()}]")
