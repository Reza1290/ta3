"""
render_orientation_comparison.py
================================
Renders a clear, educational side-by-side comparison explaining:
1. Array Matrix View (X=0 left, X=704 right)
2. Clinical Radiological View (Patient Right on Left, Patient Left on Right)
3. Dental Odontogram Standard (Q1 Top-Left, Q2 Top-Right, Q4 Bottom-Left, Q3 Bottom-Right)
"""

from pathlib import Path
import numpy as np
import SimpleITK as sitk
from PIL import Image, ImageDraw

vol_path = "data/DICOMM/output_aligned/volume.mha"
mask_path = "data/DICOMM/output_aligned/pred_fdi_mask.mha"

vol_img = sitk.ReadImage(vol_path)
mask_img = sitk.ReadImage(mask_path)
vol_arr = sitk.GetArrayFromImage(vol_img)
mask_arr = sitk.GetArrayFromImage(mask_img)

# Choose Axial slice Z=245 (Maxillary arch with teeth 17, 16, 11, 21, 26, 27)
z_idx = 245
slice_vol = vol_arr[z_idx, :, :]
slice_mask = mask_arr[z_idx, :, :]

p1, p99 = np.percentile(slice_vol, [1, 99.5])
norm = np.clip((slice_vol - p1) / (p99 - p1 + 1e-7) * 255.0, 0, 255).astype(np.uint8)

color_map = {
    17: (0, 200, 255), 16: (0, 255, 200), 11: (100, 255, 100),
    21: (255, 200, 0), 26: (255, 120, 0), 27: (255, 50, 50),
}

# 1. Matrix View (Raw Array)
rgb_raw = np.stack([norm]*3, axis=-1)
for fdi in range(11, 49):
    idx = (slice_mask == fdi)
    if np.any(idx):
        c = color_map.get(fdi, (220, 150, 255))
        rgb_raw[idx] = (0.5 * rgb_raw[idx] + 0.5 * np.array(c)).astype(np.uint8)

img_raw = Image.fromarray(rgb_raw).resize((460, 460), Image.Resampling.LANCZOS)
draw_raw = ImageDraw.Draw(img_raw)
draw_raw.rectangle([(0, 0), (460, 26)], fill=(0, 0, 0))
draw_raw.text((10, 5), "VIEW A: Matriks Array NumPy (X=0 di Kiri, X=704 di Kanan)", fill=(255, 255, 255))
draw_raw.text((20, 40), "<-- X=200: Kiri Pasien (Q2)", fill=(255, 100, 100))
draw_raw.text((260, 40), "X=500: Kanan Pasien (Q1) -->", fill=(100, 200, 255))
draw_raw.text((20, 430), "Tampak: Gigi Kiri di Kiri Layar (Non-Radiologis)", fill=(255, 220, 100))

# 2. Radiological View (Flipped horizontally so Patient Right is on Viewer's Left)
rgb_radio = np.fliplr(rgb_raw)
img_radio = Image.fromarray(rgb_radio).resize((460, 460), Image.Resampling.LANCZOS)
draw_radio = ImageDraw.Draw(img_radio)
draw_radio.rectangle([(0, 0), (460, 26)], fill=(0, 0, 0))
draw_radio.text((10, 5), "VIEW B: Standar Radiologi Medis (Kanan Pasien di Kiri Layar)", fill=(100, 255, 150))
draw_radio.text((20, 40), "<-- Kanan Pasien (Q1: 17, 16, 11)", fill=(100, 200, 255))
draw_radio.text((260, 40), "Kiri Pasien (Q2: 21, 26, 27) -->", fill=(255, 100, 100))
draw_radio.text((20, 430), "Tampak: Standar CT / CBCT Viewer (R di Kiri, L di Kanan)", fill=(100, 255, 150))

# 3. Canvas combined
canvas = Image.new("RGB", (460 * 2 + 30, 460 + 80), (22, 22, 22))
draw = ImageDraw.Draw(canvas)
draw.rectangle([(0, 0), (canvas.width, 60)], fill=(30, 40, 55))
draw.text((20, 12), "PENJELASAN ORIENTASI CITRA: MENGAPA MASK TAMPAK TERBALIK KIRI-KANAN?", fill=(255, 255, 255))
draw.text((20, 36), "Perbedaan antara Koordinat Indeks Matriks Pemrograman vs Standar Tampilan Radiologi / Odontogram", fill=(100, 220, 255))

canvas.paste(img_raw, (10, 70))
canvas.paste(img_radio, (460 + 20, 70))

out_path = Path("runners/outputs/dicom_aligned_test/orientation_explanation.png")
brain_path = Path("C:/Users/rezam/.gemini/antigravity/brain/54ccded8-7443-4f4d-b5c3-d6bbe9f123b5/orientation_explanation.png")
canvas.save(out_path)
canvas.save(brain_path)
print(f"Saved orientation explanation: {out_path} and {brain_path}")
