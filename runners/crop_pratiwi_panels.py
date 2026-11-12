from PIL import Image
import numpy as np

img = Image.open("C:/Users/rezam/.gemini/antigravity/brain/54ccded8-7443-4f4d-b5c3-d6bbe9f123b5/pratiwi_radiological_mpr.png")
print("Image size:", img.size)

# The image is 3899 x 2573
# Let's crop Panel 1 (Maxilla) and Panel 2 (Mandible) and save them to runners/outputs/
# Let's check how the subplots were arranged in fig, axes = plt.subplots(2, 3, figsize=(20, 13))
# Panel 1: Top-Left (row 0, col 0)
# Panel 2: Top-Center (row 0, col 1)
# Panel 3: Top-Right (row 0, col 2)
# Panel 4: Bottom-Left (row 1, col 0)
# Panel 5: Bottom-Center (row 1, col 1)
# Panel 6: Bottom-Right (row 1, col 2)

w, h = img.size
col_w = w // 3
row_h = h // 2

p1 = img.crop((0, 0, col_w, row_h))
p2 = img.crop((col_w, 0, col_w * 2, row_h))

p1.save("runners/outputs/panel1_maxilla.png")
p2.save("runners/outputs/panel2_mandible.png")
print("Saved panel1_maxilla.png and panel2_mandible.png")
