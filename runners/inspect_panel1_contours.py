from PIL import Image
import numpy as np

img = Image.open("runners/outputs/panel1_maxilla.png")
arr = np.array(img)
print("Panel 1 size:", arr.shape)

# Let's inspect where Cyan (Q1: #00E5FF = [0, 229, 255]) and Orange (Q2: #FF9100 = [255, 145, 0]) contours are:
# Cyan: r < 50, g > 200, b > 200
# Orange: r > 200, g between 100 and 180, b < 50

cyan_pts = np.argwhere((arr[:, :, 0] < 50) & (arr[:, :, 1] > 200) & (arr[:, :, 2] > 200))
orange_pts = np.argwhere((arr[:, :, 0] > 200) & (arr[:, :, 1] > 100) & (arr[:, :, 1] < 180) & (arr[:, :, 2] < 50))

print(f"Cyan pts (Q1): {len(cyan_pts)}, Y(row) range: [{cyan_pts[:,0].min()}, {cyan_pts[:,0].max()}], X(col) range: [{cyan_pts[:,1].min()}, {cyan_pts[:,1].max()}]")
print(f"Orange pts (Q2): {len(orange_pts)}, Y(row) range: [{orange_pts[:,0].min()}, {orange_pts[:,0].max()}], X(col) range: [{orange_pts[:,1].min()}, {orange_pts[:,1].max()}]")
