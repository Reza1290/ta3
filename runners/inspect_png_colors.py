from PIL import Image
import numpy as np

img = Image.open("C:/Users/rezam/.gemini/antigravity/brain/54ccded8-7443-4f4d-b5c3-d6bbe9f123b5/reseg_27_28_Independent.png")
arr = np.array(img)
print("Image dimensions:", arr.shape)

# Panel 1: CBCT (x: 10 to 234)
# Panel 2: Mask (x: 244 to 468)
# Panel 3: Overlay (x: 478 to 702)

panel2 = arr[36:36+224, 244:244+224] # Mask panel
# Unique colors in panel 2:
colors, counts = np.unique(panel2.reshape(-1, 3), axis=0, return_counts=True)
for c, cnt in zip(colors, counts):
    if cnt > 10:
        print(f"Color {c}: {cnt} pixels")
