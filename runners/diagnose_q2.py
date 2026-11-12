import SimpleITK as sitk
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from scipy.spatial import cKDTree
from scipy import ndimage

def extract_surface_voxels(binary_mask: np.ndarray) -> np.ndarray:
    if not np.any(binary_mask):
        return np.empty((0, 3), dtype=np.int64)
    struct = ndimage.generate_binary_structure(3, 3)
    eroded = ndimage.binary_erosion(binary_mask, structure=struct)
    surface = binary_mask ^ eroded
    return np.argwhere(surface)

def crop_window_2d(slice_2d: np.ndarray, center_yx: tuple[int, int], crop_size: tuple[int, int] = (224, 224), pad_value: float = 0.0):
    target_h, target_w = crop_size
    src_h, src_w = slice_2d.shape
    cy, cx = center_yx
    half_h = target_h // 2
    half_w = target_w // 2

    r_min = cy - half_h
    r_max = r_min + target_h
    c_min = cx - half_w
    c_max = c_min + target_w

    patch = np.full((target_h, target_w), pad_value, dtype=slice_2d.dtype)
    src_r_start = max(0, r_min)
    src_r_end = min(src_h, r_max)
    src_c_start = max(0, c_min)
    src_c_end = min(src_w, c_max)

    dst_r_start = src_r_start - r_min
    dst_r_end = dst_r_start + (src_r_end - src_r_start)
    dst_c_start = src_c_start - c_min
    dst_c_end = dst_c_start + (src_c_end - src_c_start)

    if src_r_end > src_r_start and src_c_end > src_c_start:
        patch[dst_r_start:dst_r_end, dst_c_start:dst_c_end] = slice_2d[src_r_start:src_r_end, src_c_start:src_c_end]

    return patch, (r_min, r_max, c_min, c_max)

vol_path = Path("data/DICOMM/output_aligned/volume.mha")
mask_path = Path("data/DICOMM/output_aligned/pred_fdi_mask.mha")

vol_img = sitk.ReadImage(str(vol_path))
mask_img = sitk.ReadImage(str(mask_path))
vol_arr = sitk.GetArrayFromImage(vol_img)
mask_arr = sitk.GetArrayFromImage(mask_img)

spacing = np.array(mask_img.GetSpacing())
spacing_zyx = np.array([spacing[2], spacing[1], spacing[0]])
sp = spacing_zyx.astype(np.float64)

# In Q2: M2 = 27, M3 = 28
surf_27 = extract_surface_voxels(mask_arr == 27)
surf_28 = extract_surface_voxels(mask_arr == 28)

tree_27 = cKDTree(surf_27.astype(np.float64) * sp)
dists, nearest_m2_idxs = tree_27.query(surf_28.astype(np.float64) * sp)
best_m3_idx = int(np.argmin(dists))
dmin_mm = float(dists[best_m3_idx])

best_28 = surf_28[best_m3_idx]
best_27 = surf_27[nearest_m2_idxs[best_m3_idx]]

center_z = int(np.round((best_27[0] + best_28[0]) / 2.0))
center_y = int(np.round((best_27[1] + best_28[1]) / 2.0))
center_x = int(np.round((best_27[2] + best_28[2]) / 2.0))

print(f"Center: Z={center_z}, Y={center_y}, X={center_x}")
print(f"D_min: {dmin_mm:.3f} mm")
print(f"Best 28: {best_28}")
print(f"Best 27: {best_27}")

# Sagittal slice
slice_2d = vol_arr[:, :, center_x]
mask_slice_2d = mask_arr[:, :, center_x]

raw_patch, (r_min, r_max, c_min, c_max) = crop_window_2d(slice_2d, center_yx=(center_z, center_y), crop_size=(224, 224))
mask_patch, _ = crop_window_2d(mask_slice_2d, center_yx=(center_z, center_y), crop_size=(224, 224))

print(f"Crop window Z:[{r_min}, {r_max}], Y:[{c_min}, {c_max}]")
unique_in_patch, counts_in_patch = np.unique(mask_patch, return_counts=True)
print("\nUnique labels in the 224x224 patch:")
for u, cnt in zip(unique_in_patch, counts_in_patch):
    print(f"  FDI {u}: {cnt} voxels")

# Let's see what teeth are near this region in 3D:
print("\nCentroids of teeth in this region:")
for fdi in [26, 27, 28, 36, 37, 38]:
    if np.any(mask_arr == fdi):
        pts = np.argwhere(mask_arr == fdi)
        print(f"  FDI {fdi}: Centroid={np.round(pts.mean(axis=0)).astype(int)}, Z:[{pts[:,0].min()},{pts[:,0].max()}], Y:[{pts[:,1].min()},{pts[:,1].max()}], X:[{pts[:,2].min()},{pts[:,2].max()}]")

# Now let's generate a clear diagnostic visualization showing EXACTLY what each tooth label is!
p_min, p_max = raw_patch.min(), raw_patch.max()
roi_norm = (raw_patch - p_min) / (p_max - p_min + 1e-7)
cbct_u8 = np.clip(roi_norm * 255.0, 0, 255).astype(np.uint8)

# Color each tooth with a distinct vibrant color
# 28: Orange [255, 140, 0]
# 27: Cyan [0, 220, 255]
# 26: Yellow [255, 255, 0]
# 37: Lime [50, 255, 50]
# 38: Magenta [255, 50, 255]

color_map = {
    28: (255, 140, 0),    # M3 28 (Orange)
    27: (0, 220, 255),    # M2 27 (Cyan)
    26: (255, 255, 0),    # M1 26 (Yellow)
    37: (50, 255, 50),    # M2 37 (Lime)
    38: (255, 50, 255),   # M3 38 (Magenta)
}

mask_colored = np.zeros((224, 224, 3), dtype=np.uint8)
for u in unique_in_patch:
    if u == 0:
        continue
    c = color_map.get(u, (200, 200, 200))
    mask_colored[mask_patch == u] = c

# Overlay on CBCT
base_rgb = np.stack([cbct_u8]*3, axis=-1)
overlay_rgb = base_rgb.copy()
for u in unique_in_patch:
    if u == 0:
        continue
    c = color_map.get(u, (200, 200, 200))
    idx = (mask_patch == u)
    overlay_rgb[idx] = (0.55 * base_rgb[idx] + 0.45 * np.array(c)).astype(np.uint8)

# Mark the closest contact points
cp_28_r = best_28[0] - r_min
cp_28_c = best_28[1] - c_min
cp_27_r = best_27[0] - r_min
cp_27_c = best_27[1] - c_min

img_cbct = Image.fromarray(cbct_u8).convert("RGB")
img_mask = Image.fromarray(mask_colored)
img_overlay = Image.fromarray(overlay_rgb)

draw_ov = ImageDraw.Draw(img_overlay)
draw_mask = ImageDraw.Draw(img_mask)

# Draw line between closest points if inside crop
if 0 <= cp_28_r < 224 and 0 <= cp_28_c < 224 and 0 <= cp_27_r < 224 and 0 <= cp_27_c < 224:
    draw_ov.line([(cp_28_c, cp_28_r), (cp_27_c, cp_27_r)], fill=(255, 0, 0), width=2)
    draw_ov.ellipse([(cp_28_c-3, cp_28_r-3), (cp_28_c+3, cp_28_r+3)], fill=(255, 140, 0), outline=(255, 255, 255))
    draw_ov.ellipse([(cp_27_c-3, cp_27_r-3), (cp_27_c+3, cp_27_r+3)], fill=(0, 220, 255), outline=(255, 255, 255))

# Add text labels on the masks
for u in unique_in_patch:
    if u == 0:
        continue
    pts = np.argwhere(mask_patch == u)
    if len(pts) > 50:
        cr, cc = np.round(pts.mean(axis=0)).astype(int)
        draw_ov.text((cc-12, cr-6), f"FDI {u}", fill=(0, 0, 0))
        draw_ov.text((cc-13, cr-7), f"FDI {u}", fill=(255, 255, 255))
        draw_mask.text((cc-12, cr-6), f"FDI {u}", fill=(0, 0, 0))
        draw_mask.text((cc-13, cr-7), f"FDI {u}", fill=(255, 255, 255))

# Assemble 3-panel figure
canvas = Image.new("RGB", (224 * 3 + 40, 224 + 50), (30, 30, 30))
draw_canvas = ImageDraw.Draw(canvas)
draw_canvas.text((10, 8), f"QUADRANT 2 ANATOMY: M2 (27) vs M3 (28) | True 3D D_min = {dmin_mm:.2f} mm | Sagittal Slice X={center_x}", fill=(255, 255, 255))

canvas.paste(img_cbct, (10, 36))
canvas.paste(img_mask, (224 + 20, 36))
canvas.paste(img_overlay, (224 * 2 + 30, 36))

out_path = Path("runners/outputs/dicom_aligned_test/q2_detailed_anatomy.png")
canvas.save(out_path)
brain_path = Path("C:/Users/rezam/.gemini/antigravity/brain/54ccded8-7443-4f4d-b5c3-d6bbe9f123b5/q2_detailed_anatomy.png")
canvas.save(brain_path)
print(f"Saved diagnostic visualization: {out_path} and {brain_path}")
