from PIL import Image
import numpy as np

img = Image.open("C:/Users/rezam/.gemini/antigravity/brain/54ccded8-7443-4f4d-b5c3-d6bbe9f123b5/reseg_27_28_Independent.png")
arr = np.array(img)
panel2 = arr[36:36+224, 244:244+224] # Mask panel (row: 0-223, col: 0-223)

cyan_pts = np.argwhere((panel2[:, :, 0] == 0) & (panel2[:, :, 1] == 220) & (panel2[:, :, 2] == 220))
orange_pts = np.argwhere((panel2[:, :, 0] == 255) & (panel2[:, :, 1] == 120) & (panel2[:, :, 2] == 0))
grey_pts = np.argwhere((panel2[:, :, 0] == 120) & (panel2[:, :, 1] == 120) & (panel2[:, :, 2] == 120))

print(f"Cyan (Tooth 27): {len(cyan_pts)} pts | Rows(Y/Z-slice): [{cyan_pts[:,0].min()}, {cyan_pts[:,0].max()}], Cols(X/Y-axial): [{cyan_pts[:,1].min()}, {cyan_pts[:,1].max()}] | Centroid: {cyan_pts.mean(axis=0)}")
print(f"Orange (Tooth 28): {len(orange_pts)} pts | Rows(Y/Z-slice): [{orange_pts[:,0].min()}, {orange_pts[:,0].max()}], Cols(X/Y-axial): [{orange_pts[:,1].min()}, {orange_pts[:,1].max()}] | Centroid: {orange_pts.mean(axis=0)}")
print(f"Grey (Tooth 26): {len(grey_pts)} pts | Rows(Y/Z-slice): [{grey_pts[:,0].min()}, {grey_pts[:,0].max()}], Cols(X/Y-axial): [{grey_pts[:,1].min()}, {grey_pts[:,1].max()}] | Centroid: {grey_pts.mean(axis=0)}")

# Check distance between Cyan and Orange in 2D patch:
dists_2d = []
for p_o in orange_pts:
    d = np.min(np.linalg.norm(cyan_pts - p_o, axis=1))
    dists_2d.append(d)
print(f"Min 2D pixel distance between Orange (28) and Cyan (27): {min(dists_2d):.2f} pixels ({min(dists_2d)*0.16:.2f} mm)")

dists_grey_orange = []
for p_o in orange_pts:
    d = np.min(np.linalg.norm(grey_pts - p_o, axis=1))
    dists_grey_orange.append(d)
print(f"Min 2D pixel distance between Orange (28) and Grey (26): {min(dists_grey_orange):.2f} pixels ({min(dists_grey_orange)*0.16:.2f} mm)")
