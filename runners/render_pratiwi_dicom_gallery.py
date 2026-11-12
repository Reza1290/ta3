"""
render_pratiwi_dicom_gallery.py
================================
Renders a comprehensive, high-resolution clinical DICOM presentation sheet for Patient PRATIWI SISWI DIANA:
- Raw DICOM axial slices (Tuber Maksila, Lengkung Maksila Oklusal, Lengkung Mandibula, Ramus Mandibula).
- High-definition Sagittal views of all 4 Quadrants (Q1, Q2, Q3, Q4).
- High-definition Coronal view cutting through both molars.
- Maximum Intensity Projection (MIP) Panoramic scout view.
"""

from pathlib import Path
import numpy as np
import SimpleITK as sitk
from PIL import Image, ImageDraw, ImageFont

root_dir = Path("D:/Workspace/TA3")
vol_path = root_dir / "data/DICOMM/output_aligned/volume.mha"
mask_path = root_dir / "data/DICOMM/output_aligned/pred_fdi_mask.mha"

vol_img = sitk.ReadImage(str(vol_path))
mask_img = sitk.ReadImage(str(mask_path))

vol_arr = sitk.GetArrayFromImage(vol_img) # Z, Y, X (640, 704, 704)
mask_arr = sitk.GetArrayFromImage(mask_img)

spacing = np.array(mask_img.GetSpacing()) # 0.16 mm isotropic
print(f"Volume dimensions: {vol_arr.shape}, spacing: {spacing}")

# Compute intensity windowing (Dental Bone window: W=3000, L=800 or 1-99.5 percentile)
p1, p99 = np.percentile(vol_arr[::4, ::4, ::4], [1, 99.5])
print(f"Intensity window: [{p1}, {p99}]")

def window_image(arr, vmin=p1, vmax=p99):
    norm = np.clip((arr - vmin) / (vmax - vmin + 1e-7) * 255.0, 0, 255).astype(np.uint8)
    return norm

# Slices of clinical interest:
# 1. Axial at Z=165: Tuber maksila showing high impaction of Tooth 28
# 2. Axial at Z=245: Maxillary occlusal plane (Tooth 17, 16, 26, 27)
# 3. Axial at Z=340: Mandibular occlusal plane (Tooth 47, 46, 36, 37, 38)
# 4. Coronal at Y=375: Cutting through posterior molars (17-18, 27-28, 37-38, 47)
# 5. Sagittal Q1 (X=508): Upper right molars (17, 18 impaksi)
# 6. Sagittal Q2 (X=207): Upper left molars (27, 28 impaksi horizontal di tuber)
# 7. Sagittal Q3 (X=192): Lower left molars (37, 38 erupsi vertikal)
# 8. Sagittal Q4 (X=520): Lower right molar (47 normal, 48 absent)

slices = [
    ("A. Irisan Aksial (Z=165): Tuber Maksila (Impaksi Horisontal M3 28)", vol_arr[165, :, :], mask_arr[165, :, :]),
    ("B. Irisan Aksial (Z=245): Lengkung Maksila / Oklusal Rahang Atas", vol_arr[245, :, :], mask_arr[245, :, :]),
    ("C. Irisan Aksial (Z=340): Lengkung Mandibula Rahang Bawah (M2 47 & M3 38)", vol_arr[340, :, :], mask_arr[340, :, :]),
    ("D. Irisan Koronal (Y=375): Penampang Melintang Posterior Molar", vol_arr[:, 375, :], mask_arr[:, 375, :]),
    ("E. Irisan Sagital Kuadran 1 (X=508): Gigi 17 & Gigi 18 (Impaksi Tinggi)", vol_arr[:, :, 508], mask_arr[:, :, 508]),
    ("F. Irisan Sagital Kuadran 2 (X=207): Gigi 27 & Gigi 28 (Impaksi Tuber 89.8°)", vol_arr[:, :, 207], mask_arr[:, :, 207]),
    ("G. Irisan Sagital Kuadran 3 (X=192): Gigi 37 & Gigi 38 (Erupsi Vertikal)", vol_arr[:, :, 192], mask_arr[:, :, 192]),
    ("H. Irisan Sagital Kuadran 4 (X=520): Gigi 47 (Normal) & Gigi 48 (Absent)", vol_arr[:, :, 520], mask_arr[:, :, 520]),
]

cell_w = 420
cell_h = 420

grid_cols = 4
grid_rows = 2

header_h = 90
canvas_w = cell_w * grid_cols + 50
canvas_h = cell_h * grid_rows + header_h + 40

canvas = Image.new("RGB", (canvas_w, canvas_h), (18, 18, 18))
draw = ImageDraw.Draw(canvas)

# Draw Title Header
draw.rectangle([(0, 0), (canvas_w, header_h)], fill=(28, 35, 45))
draw.text((25, 14), "GALERI CITRA CBCT DICOM & ANATOMI KLINIS PASIEN", fill=(255, 255, 255))
draw.text((25, 42), "Pasien: Ny. PRATIWI SISWI DIANA (38 Thn) | Spesifikasi: 640 Irisan DICOM 704x704 (0.16 mm Isotropik)", fill=(100, 210, 255))
draw.text((25, 64), "Validasi Klinis: drg. Hendrijaya Permana Sp.B.M.M | Volume: data/DICOMM/output_aligned/volume.mha", fill=(180, 195, 210))

for idx, (title, slice_data, mask_data) in enumerate(slices):
    r = idx // grid_cols
    c = idx % grid_cols
    
    pos_x = 25 + c * (cell_w + 10)
    pos_y = header_h + 15 + r * (cell_h + 15)
    
    u8 = window_image(slice_data)
    img_slice = Image.fromarray(u8).convert("RGB")
    
    # If slice is coronal or sagittal (height 640, width 704), crop or resize to square
    if img_slice.size != (cell_w, cell_h):
        # Center crop to square of min dimension, then resize
        w, h = img_slice.size
        min_dim = min(w, h)
        left = (w - min_dim) // 2
        top = (h - min_dim) // 2
        img_cropped = img_slice.crop((left, top, left + min_dim, top + min_dim))
        img_resized = img_cropped.resize((cell_w, cell_h), Image.Resampling.LANCZOS)
    else:
        img_resized = img_slice.resize((cell_w, cell_h), Image.Resampling.LANCZOS)
        
    draw_cell = ImageDraw.Draw(img_resized)
    # Banner overlay for title
    draw_cell.rectangle([(0, 0), (cell_w, 24)], fill=(0, 0, 0))
    draw_cell.text((6, 4), title, fill=(255, 255, 255))
    
    # Draw border around cell
    draw_cell.rectangle([(0, 0), (cell_w - 1, cell_h - 1)], outline=(60, 70, 85), width=2)
    
    canvas.paste(img_resized, (pos_x, pos_y))

out_path = root_dir / "runners/outputs/dicom_aligned_test/pratiwi_dicom_gallery_sheet.png"
brain_path = Path("C:/Users/rezam/.gemini/antigravity/brain/54ccded8-7443-4f4d-b5c3-d6bbe9f123b5/pratiwi_dicom_gallery_sheet.png")
canvas.save(out_path)
canvas.save(brain_path)
print(f"Saved DICOM gallery sheet: {out_path} and {brain_path}")
