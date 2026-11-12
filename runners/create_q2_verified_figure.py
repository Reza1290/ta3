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

vol_img = sitk.ReadImage("data/DICOMM/output_aligned/volume.mha")
mask_img = sitk.ReadImage("data/DICOMM/output_aligned/pred_fdi_mask.mha")
vol_arr = sitk.GetArrayFromImage(vol_img)
mask_arr = sitk.GetArrayFromImage(mask_img)

spacing = np.array(mask_img.GetSpacing())
spacing_zyx = np.array([spacing[2], spacing[1], spacing[0]])
sp = spacing_zyx.astype(np.float64)

# 1. Generate Q2 Panoramic / Arch View
# Panoramic MIP or curved slice or composite sagittal/axial
# Let's create an informative 4-panel figure:
# Panel A: Sagittal Slice at X=207 (Cutting through 27 & 28 closest contact)
# Panel B: Coronal Slice at Y=375 (Cutting through 27 & 28)
# Panel C: Axial Slice at Z=180 (Cutting right at the interface between 27 root and 28 crown)
# Panel D: 3D MIP / Anatomy explanation

s27 = extract_surface_voxels(mask_arr == 27)
s28 = extract_surface_voxels(mask_arr == 28)
tree = cKDTree(s27.astype(np.float64) * sp)
dists, idxs = tree.query(s28.astype(np.float64) * sp)
b28_idx = np.argmin(dists)
b28 = s28[b28_idx]
b27 = s27[idxs[b28_idx]]
dmin = dists[b28_idx]

contact_z = int((b28[0] + b27[0]) / 2)
contact_y = int((b28[1] + b27[1]) / 2)
contact_x = int((b28[2] + b27[2]) / 2)

print(f"Contact point: Z={contact_z}, Y={contact_y}, X={contact_x}, D_min={dmin:.3f} mm")

color_map = {
    28: (255, 140, 0),    # Orange
    27: (0, 220, 255),    # Cyan
    26: (255, 230, 0),    # Yellow
    24: (200, 100, 255),  # Violet
    22: (100, 255, 100),  # Green
    21: (255, 100, 100),  # Pink
}

def render_slice(vol_slice, mask_slice, title, mark_point=None, labels_to_show=None):
    p_min, p_max = np.percentile(vol_slice, 1), np.percentile(vol_slice, 99.5)
    norm = np.clip((vol_slice - p_min) / (p_max - p_min + 1e-7) * 255.0, 0, 255).astype(np.uint8)
    rgb = np.stack([norm]*3, axis=-1)
    
    for fdi, color in color_map.items():
        idx = (mask_slice == fdi)
        if np.any(idx):
            rgb[idx] = (0.5 * rgb[idx] + 0.5 * np.array(color)).astype(np.uint8)
            
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)
    draw.rectangle([(0, 0), (img.width, 24)], fill=(0, 0, 0))
    draw.text((6, 4), title, fill=(255, 255, 255))
    
    if labels_to_show:
        for fdi in labels_to_show:
            pts = np.argwhere(mask_slice == fdi)
            if len(pts) > 20:
                cr, cc = np.round(pts.mean(axis=0)).astype(int)
                name = f"FDI {fdi}"
                if fdi == 28: name += " (M3)"
                elif fdi == 27: name += " (M2)"
                elif fdi == 26: name += " (M1)"
                draw.text((cc-15, cr-6), name, fill=(0, 0, 0))
                draw.text((cc-16, cr-7), name, fill=(255, 255, 255))
                
    if mark_point and 0 <= mark_point[0] < img.height and 0 <= mark_point[1] < img.width:
        r, c = mark_point
        draw.ellipse([(c-4, r-4), (c+4, r+4)], fill=(255, 0, 0), outline=(255, 255, 255))
        draw.text((c+6, r-6), f"D_min={dmin:.2f}mm", fill=(255, 50, 50))
        
    return img

# Window sizes: 240 x 240
half = 120
z_rng = slice(max(0, contact_z-half), min(vol_arr.shape[0], contact_z+half))
y_rng = slice(max(0, contact_y-half), min(vol_arr.shape[1], contact_y+half))
x_rng = slice(max(0, contact_x-half), min(vol_arr.shape[2], contact_x+half))

# Sagittal view at contact_x: slice across Z (vertical) and Y (anterior-posterior)
sag_vol = vol_arr[z_rng, y_rng, contact_x]
sag_mask = mask_arr[z_rng, y_rng, contact_x]
sag_pt = (contact_z - (contact_z - half), contact_y - (contact_y - half))
img_sag = render_slice(sag_vol, sag_mask, f"Sagittal View (X={contact_x})", mark_point=sag_pt, labels_to_show=[26, 27, 28])

# Coronal view at contact_y: slice across Z (vertical) and X (medial-lateral)
cor_vol = vol_arr[z_rng, contact_y, x_rng]
cor_mask = mask_arr[z_rng, contact_y, x_rng]
cor_pt = (contact_z - (contact_z - half), contact_x - (contact_x - half))
img_cor = render_slice(cor_vol, cor_mask, f"Coronal View (Y={contact_y})", mark_point=cor_pt, labels_to_show=[27, 28])

# Axial view at contact_z: slice across Y (anterior-posterior) and X (medial-lateral)
ax_vol = vol_arr[contact_z, y_rng, x_rng]
ax_mask = mask_arr[contact_z, y_rng, x_rng]
ax_pt = (contact_y - (contact_y - half), contact_x - (contact_x - half))
img_ax = render_slice(ax_vol, ax_mask, f"Axial View (Z={contact_z} Interface)", mark_point=ax_pt, labels_to_show=[26, 27, 28])

# Assemble 3-panel figure: Sagittal, Coronal, Axial
panel_w = 240
panel_h = 240
total_w = panel_w * 3 + 40
total_h = panel_h + 80

composite = Image.new("RGB", (total_w, total_h), (25, 25, 25))
draw = ImageDraw.Draw(composite)
draw.text((15, 10), "VERIFIKASI ANATOMI KUADRAN 2: HUBUNGAN SPASIAL FDI 27 (M2) & FDI 28 (M3)", fill=(255, 255, 255))
draw.text((15, 28), f"Klasifikasi Benar: M3 28 Impaksi Tuber Maksila (Horizontal 89.8°) | Jarak 3D Asli D_min = {dmin:.2f} mm (Bukan 11.38 mm)", fill=(0, 220, 255))

composite.paste(img_sag, (10, 52))
composite.paste(img_cor, (panel_w + 20, 52))
composite.paste(img_ax, (panel_w * 2 + 30, 52))

out_path = Path("runners/outputs/dicom_aligned_test/q2_verified_anatomy.png")
brain_path = Path("C:/Users/rezam/.gemini/antigravity/brain/54ccded8-7443-4f4d-b5c3-d6bbe9f123b5/q2_verified_anatomy.png")
composite.save(out_path)
composite.save(brain_path)
print(f"Saved verified anatomy figure to {out_path} and {brain_path}")
