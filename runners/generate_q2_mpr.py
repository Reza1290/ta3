import SimpleITK as sitk
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

vol_img = sitk.ReadImage("data/DICOMM/output_aligned/volume.mha")
mask_img = sitk.ReadImage("data/DICOMM/output_aligned/pred_fdi_mask.mha")
vol_arr = sitk.GetArrayFromImage(vol_img)
mask_arr = sitk.GetArrayFromImage(mask_img)

spacing = np.array(mask_img.GetSpacing()) # X, Y, Z
# Centroids:
c28 = np.round(np.argwhere(mask_arr == 28).mean(axis=0)).astype(int) # Z, Y, X
c27 = np.round(np.argwhere(mask_arr == 27).mean(axis=0)).astype(int) # Z, Y, X
c26 = np.round(np.argwhere(mask_arr == 26).mean(axis=0)).astype(int) # Z, Y, X

print(f"Centroid 28 (Z,Y,X): {c28}")
print(f"Centroid 27 (Z,Y,X): {c27}")
print(f"Centroid 26 (Z,Y,X): {c26}")

# Let's inspect the axial slice through 28 (Z = c28[0])
# coronal slice through 28 (Y = c28[1])
# sagittal slice through 28 (X = c28[2])

def make_mpr_panel(slice_vol, slice_mask, title):
    p_min, p_max = slice_vol.min(), slice_vol.max()
    norm = np.clip((slice_vol - p_min) / (p_max - p_min + 1e-7) * 255.0, 0, 255).astype(np.uint8)
    base_rgb = np.stack([norm]*3, axis=-1)
    
    color_map = {
        28: (255, 140, 0),   # 28: Orange
        27: (0, 220, 255),   # 27: Cyan
        26: (255, 255, 0),   # 26: Yellow
        37: (50, 255, 50),   # 37: Lime
        38: (255, 50, 255),  # 38: Magenta
    }
    
    ov = base_rgb.copy()
    for fdi, color in color_map.items():
        idx = (slice_mask == fdi)
        if np.any(idx):
            ov[idx] = (0.5 * base_rgb[idx] + 0.5 * np.array(color)).astype(np.uint8)
            
    img = Image.fromarray(ov)
    draw = ImageDraw.Draw(img)
    draw.rectangle([(0, 0), (img.width, 22)], fill=(0, 0, 0))
    draw.text((8, 4), title, fill=(255, 255, 255))
    
    # Draw tooth labels on centroids if visible
    for fdi in [26, 27, 28]:
        pts = np.argwhere(slice_mask == fdi)
        if len(pts) > 30:
            cr, cc = np.round(pts.mean(axis=0)).astype(int)
            draw.text((cc-10, cr-5), f"FDI {fdi}", fill=(0, 0, 0))
            draw.text((cc-11, cr-6), f"FDI {fdi}", fill=(255, 255, 255))
            
    return img

# Window around Q2:
# Z: 100 to 324 (height 224)
# Y: 260 to 484 (width 224)
# X: 100 to 324 (width 224)
z_center = (c28[0] + c27[0]) // 2
y_center = (c28[1] + c27[1]) // 2
x_center = (c28[2] + c27[2]) // 2

z_slice = vol_arr[z_center, y_center-112:y_center+112, x_center-112:x_center+112]
z_mask = mask_arr[z_center, y_center-112:y_center+112, x_center-112:x_center+112]

y_slice = vol_arr[z_center-112:z_center+112, y_center, x_center-112:x_center+112]
y_mask = mask_arr[z_center-112:z_center+112, y_center, x_center-112:x_center+112]

x_slice = vol_arr[z_center-112:z_center+112, y_center-112:y_center+112, x_center]
x_mask = mask_arr[z_center-112:z_center+112, y_center-112:y_center+112, x_center]

img_ax = make_mpr_panel(z_slice, z_mask, f"Axial (Z={z_center})")
img_cor = make_mpr_panel(y_slice, y_mask, f"Coronal (Y={y_center})")
img_sag = make_mpr_panel(x_slice, x_mask, f"Sagittal (X={x_center})")

mpr_w = 224 * 3 + 40
mpr_h = 224 + 50
canvas = Image.new("RGB", (mpr_w, mpr_h), (20, 20, 20))
draw = ImageDraw.Draw(canvas)
draw.text((10, 8), f"QUADRANT 2 3D MPR INSPECTION: M3 28 (Orange) | M2 27 (Cyan) | M1 26 (Yellow)", fill=(255, 255, 255))

canvas.paste(img_ax, (10, 36))
canvas.paste(img_cor, (224 + 20, 36))
canvas.paste(img_sag, (224 * 2 + 30, 36))

out_file = Path("runners/outputs/dicom_aligned_test/q2_mpr_validation.png")
brain_file = Path("C:/Users/rezam/.gemini/antigravity/brain/54ccded8-7443-4f4d-b5c3-d6bbe9f123b5/q2_mpr_validation.png")
canvas.save(out_file)
canvas.save(brain_file)
print(f"Saved MPR validation to {out_file} and {brain_file}")
