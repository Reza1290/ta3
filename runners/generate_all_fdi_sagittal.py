"""
generate_all_fdi_sagittal.py
=============================
Generates individual sagittal slice inspections for EVERY detected tooth (FDI) in patient Pratiwi.
Produces:
1. Individual tooth sagittal crops.
2. Quadrant gallery sheets (Q1, Q2, Q3, Q4).
3. A Master 29-Tooth Clinical Catalog sheet.
"""

from pathlib import Path
import numpy as np
import SimpleITK as sitk
from PIL import Image, ImageDraw, ImageFont

root_dir = Path("D:/Workspace/TA3")
vol_path = root_dir / "data/DICOMM/output_aligned/volume.mha"
mask_path = root_dir / "data/DICOMM/output_aligned/pred_fdi_mask.mha"

out_dir = root_dir / "runners/outputs/fdi_sagittal_gallery"
out_dir.mkdir(parents=True, exist_ok=True)
brain_dir = Path("C:/Users/rezam/.gemini/antigravity/brain/54ccded8-7443-4f4d-b5c3-d6bbe9f123b5")

vol_img = sitk.ReadImage(str(vol_path))
mask_img = sitk.ReadImage(str(mask_path))
vol_arr = sitk.GetArrayFromImage(vol_img)
mask_arr = sitk.GetArrayFromImage(mask_img)

spacing = np.array(mask_img.GetSpacing()) # 0.16 mm
voxel_vol_mm3 = spacing[0] * spacing[1] * spacing[2]

# Bone window: 200 - 2400 HU
vmin, vmax = 200, 2400
vol_norm = np.clip((vol_arr - vmin) / (vmax - vmin + 1e-7) * 255.0, 0, 255).astype(np.uint8)

FDI_NAMES = {
    11: "Gigi Seri 1 Kanan Atas",
    12: "Gigi Seri 2 Kanan Atas",
    13: "Gigi Taring Kanan Atas",
    14: "Premolar 1 Kanan Atas",
    15: "Premolar 2 Kanan Atas",
    16: "Molar 1 Kanan Atas",
    17: "Molar 2 Kanan Atas",
    18: "Molar 3 Kanan Atas (Impaksi)",
    21: "Gigi Seri 1 Kiri Atas",
    22: "Gigi Seri 2 Kiri Atas",
    23: "Gigi Taring Kiri Atas (Missing)",
    24: "Premolar 1 Kiri Atas",
    25: "Premolar 2 Kiri Atas (Missing)",
    26: "Molar 1 Kiri Atas",
    27: "Molar 2 Kiri Atas",
    28: "Molar 3 Kiri Atas (Impaksi Tuber)",
    31: "Gigi Seri 1 Kiri Bawah",
    32: "Gigi Seri 2 Kiri Bawah",
    33: "Gigi Taring Kiri Bawah",
    34: "Premolar 1 Kiri Bawah",
    35: "Premolar 2 Kiri Bawah",
    36: "Molar 1 Kiri Bawah",
    37: "Molar 2 Kiri Bawah",
    38: "Molar 3 Kiri Bawah (Erupsi Vertikal)",
    41: "Gigi Seri 1 Kanan Bawah",
    42: "Gigi Seri 2 Kanan Bawah",
    43: "Gigi Taring Kanan Bawah",
    44: "Premolar 1 Kanan Bawah",
    45: "Premolar 2 Kanan Bawah",
    46: "Molar 1 Kanan Bawah",
    47: "Molar 2 Kanan Bawah",
    48: "Molar 3 Kanan Bawah (Absent)",
}

QUAD_COLORS = {
    1: (0, 229, 255),    # Cyan (Q1)
    2: (255, 145, 0),    # Orange (Q2)
    3: (0, 230, 118),    # Green (Q3)
    4: (255, 51, 102),   # Hot Pink (Q4)
}

def crop_patch(slice_2d, center_yx, size=180):
    half = size // 2
    cy, cx = center_yx
    h, w = slice_2d.shape
    r_min = cy - half
    r_max = r_min + size
    c_min = cx - half
    c_max = c_min + size

    patch = np.zeros((size, size), dtype=slice_2d.dtype)
    sr_start = max(0, r_min)
    sr_end = min(h, r_max)
    sc_start = max(0, c_min)
    sc_end = min(w, c_max)

    dr_start = sr_start - r_min
    dr_end = dr_start + (sr_end - sr_start)
    dc_start = sc_start - c_min
    dc_end = dc_start + (sc_end - sc_start)

    if sr_end > sr_start and sc_end > sc_start:
        patch[dr_start:dr_end, dc_start:dc_end] = slice_2d[sr_start:sr_end, sc_start:sc_end]

    return patch, (r_min, c_min)

def render_tooth_tile(fdi, size=220):
    pts = np.argwhere(mask_arr == fdi)
    color = QUAD_COLORS.get(fdi // 10, (255, 255, 255))
    name = FDI_NAMES.get(fdi, f"FDI {fdi}")

    if len(pts) == 0:
        # Absent tooth card
        tile = Image.new("RGB", (size, size), (25, 25, 30))
        d = ImageDraw.Draw(tile)
        d.rectangle([(0, 0), (size-1, size-1)], outline=(80, 80, 80), width=2)
        d.text((12, 10), f"FDI {fdi}", fill=(180, 180, 180))
        d.text((12, 30), name, fill=(130, 130, 130))
        d.text((12, 90), "[ TIDAK ADA / ABSEN ]", fill=(255, 75, 75))
        d.text((12, 115), "Agenesis / Ekstraksi", fill=(160, 160, 160))
        return tile

    cz, cy, cx = np.round(pts.mean(axis=0)).astype(int)
    voxels = len(pts)
    vol_mm3 = voxels * voxel_vol_mm3

    # Sagittal slice at X = cx
    sag_vol = vol_norm[:, :, cx]
    sag_mask = mask_arr[:, :, cx]

    patch_vol, (r_min, c_min) = crop_patch(sag_vol, (cz, cy), size=160)
    patch_mask, _ = crop_patch(sag_mask, (cz, cy), size=160)

    # Base RGB image
    rgb = np.stack([patch_vol]*3, axis=-1)
    
    # Overlay other teeth with low opacity grey, target tooth with vibrant color
    other_idx = (patch_mask > 0) & (patch_mask != fdi)
    rgb[other_idx] = (0.7 * rgb[other_idx] + 0.3 * np.array([120, 120, 120])).astype(np.uint8)

    target_idx = (patch_mask == fdi)
    rgb[target_idx] = (0.45 * rgb[target_idx] + 0.55 * np.array(color)).astype(np.uint8)

    img_patch = Image.fromarray(rgb).resize((size, size - 46), Image.Resampling.LANCZOS)

    # Tile container with header info
    tile = Image.new("RGB", (size, size), (16, 18, 24))
    d = ImageDraw.Draw(tile)

    # Paste crop
    tile.paste(img_patch, (0, 46))

    # Top info banner
    d.rectangle([(0, 0), (size, 46)], fill=(24, 28, 38))
    d.text((8, 4), f"FDI {fdi}: {name}", fill=(255, 255, 255))
    d.text((8, 22), f"Vol: {vol_mm3:.1f} mm³ ({voxels:,} vx) | Sagittal X={cx}", fill=color)

    # Border
    d.rectangle([(0, 0), (size-1, size-1)], outline=color, width=2)
    return tile

# Generate 4 Quadrant Sheets
quadrants = [
    (1, "KUADRAN 1: MAKSILA KANAN (UPPER RIGHT)", [11, 12, 13, 14, 15, 16, 17, 18], 4, 2),
    (2, "KUADRAN 2: MAKSILA KIRI (UPPER LEFT)", [21, 22, 23, 24, 25, 26, 27, 28], 4, 2),
    (3, "KUADRAN 3: MANDIBULA KIRI (LOWER LEFT)", [31, 32, 33, 34, 35, 36, 37, 38], 4, 2),
    (4, "KUADRAN 4: MANDIBULA KANAN (LOWER RIGHT)", [41, 42, 43, 44, 45, 46, 47, 48], 4, 2),
]

tile_size = 230
pad = 12

for q_num, q_title, fdi_list, cols, rows in quadrants:
    sheet_w = cols * (tile_size + pad) + pad
    sheet_h = rows * (tile_size + pad) + pad + 60
    sheet = Image.new("RGB", (sheet_w, sheet_h), (12, 14, 18))
    d = ImageDraw.Draw(sheet)

    # Header
    d.rectangle([(0, 0), (sheet_w, 54)], fill=(22, 26, 36))
    d.text((20, 10), q_title, fill=(255, 255, 255))
    d.text((20, 30), f"Pasien: Ny. PRATIWI SISWI DIANA (38 Thn) | Irisan Sagital Rekonstruksi per Gigi", fill=QUAD_COLORS[q_num])

    for i, fdi in enumerate(fdi_list):
        r = i // cols
        c = i % cols
        pos_x = pad + c * (tile_size + pad)
        pos_y = 60 + pad + r * (tile_size + pad)

        t_img = render_tooth_tile(fdi, size=tile_size)
        sheet.paste(t_img, (pos_x, pos_y))

    q_name_code = f"Q{q_num}"
    fn = f"sagittal_fdi_{q_name_code}.png"
    sheet.save(out_dir / fn)
    sheet.save(brain_dir / fn)
    print(f"Saved {fn} to out_dir and brain_dir")

# Master Catalog (32 teeth in 8 columns x 4 rows)
m_cols = 8
m_rows = 4
m_tile = 180
m_pad = 8
m_w = m_cols * (m_tile + m_pad) + m_pad
m_h = m_rows * (m_tile + m_pad) + m_pad + 70

master_sheet = Image.new("RGB", (m_w, m_h), (10, 12, 16))
d_m = ImageDraw.Draw(master_sheet)
d_m.rectangle([(0, 0), (m_w, 60)], fill=(20, 24, 34))
d_m.text((20, 12), "KATALOG LENGKAP 32 GIGI CBCT (IRISAN SAGITAL PER NOMOR FDI)", fill=(255, 255, 255))
d_m.text((20, 34), "Pasien: Ny. PRATIWI SISWI DIANA | Terdeteksi 29 Gigi | Q1: Cyan, Q2: Orange, Q3: Green, Q4: Hot Pink", fill=(100, 220, 255))

all_teeth_ordered = [
    [18, 17, 16, 15, 14, 13, 12, 11],  # Row 0: Upper Right
    [21, 22, 23, 24, 25, 26, 27, 28],  # Row 1: Upper Left
    [48, 47, 46, 45, 44, 43, 42, 41],  # Row 2: Lower Right
    [31, 32, 33, 34, 35, 36, 37, 38],  # Row 3: Lower Left
]

for row_idx, teeth_row in enumerate(all_teeth_ordered):
    for col_idx, fdi in enumerate(teeth_row):
        pos_x = m_pad + col_idx * (m_tile + m_pad)
        pos_y = 66 + m_pad + row_idx * (m_tile + m_pad)
        t_img = render_tooth_tile(fdi, size=m_tile)
        master_sheet.paste(t_img, (pos_x, pos_y))

fn_master = "sagittal_all_fdi_master_catalog.png"
master_sheet.save(out_dir / fn_master)
master_sheet.save(brain_dir / fn_master)
print(f"Saved {fn_master} to out_dir and brain_dir")
