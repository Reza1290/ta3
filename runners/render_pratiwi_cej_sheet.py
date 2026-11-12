"""
render_pratiwi_cej_sheet.py
===========================
Generates a comprehensive, high-resolution CEJ (Cementoenamel Junction) Diagnostic Sheet for Patient Pratiwi:
- Sagittal cross-sections for all 4 Quadrants.
- Clear overlay of M2 and M3 segmentation masks.
- Exact CEJ horizontal reference lines (Z_cej).
- Crown Zone (Above CEJ) and Root Zone (Below CEJ) indicators.
- Exact 3D closest contact / proximity points with Euclidean measurement lines.
- Cross-sectional cervical constriction curve (Waistline profile) per M2.
"""

from pathlib import Path
import numpy as np
import SimpleITK as sitk
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

root_dir = Path("D:/Workspace/TA3")
vol_path = root_dir / "data/DICOMM/output_aligned/volume.mha"
mask_path = root_dir / "data/DICOMM/output_aligned/pred_fdi_mask.mha"

out_dir = root_dir / "runners/outputs/dicom_aligned_test"
out_dir.mkdir(parents=True, exist_ok=True)
brain_dir = Path("C:/Users/rezam/.gemini/antigravity/brain/54ccded8-7443-4f4d-b5c3-d6bbe9f123b5")

vol = sitk.GetArrayFromImage(sitk.ReadImage(str(vol_path)))
mask = sitk.GetArrayFromImage(sitk.ReadImage(str(mask_path)))

sz = 0.16 # mm

# Bone window
vmin, vmax = 200, 2400
vol_norm = np.clip((vol - vmin) / (vmax - vmin + 1e-7), 0.0, 1.0)

# Quadrant configurations based on test_pratiwi_cej.py
quad_data = [
    {
        "q_name": "Kuadran 1: Maksila Kanan (FDI 17 - 18)",
        "m2": 17, "m3": 18, "is_mand": False,
        "cej_z": 203, "ratio": 0.80,
        "cp_m2": [181, 374, 515], "cp_m3": [180, 376, 514],
        "dmin": 0.392, "delta_z": -3.52,
        "class_name": "Below_CEJ (Physical Contact)",
        "risk": "HIGH RISK: External Root Resorption (ERR) pada Akar M2",
        "color_m2": "#00E5FF", "color_m3": "#FF9100",
        "slice_x": 515,
        "center_z": 215, "center_y": 380,
    },
    {
        "q_name": "Kuadran 2: Maksila Kiri (FDI 27 - 28)",
        "m2": 27, "m3": 28, "is_mand": False,
        "cej_z": 216, "ratio": 0.77,
        "cp_m2": [181, 361, 207], "cp_m3": [179, 359, 211],
        "dmin": 0.784, "delta_z": -5.60,
        "class_name": "Independent (Celah PDL Utuh)",
        "risk": "LOW RISK: D_min = 0.78 mm > 0.50 mm (Septum Tulang Utuh)",
        "color_m2": "#00E5FF", "color_m3": "#FF9100",
        "slice_x": 207,
        "center_z": 210, "center_y": 370,
    },
    {
        "q_name": "Kuadran 3: Mandibula Kiri (FDI 37 - 38)",
        "m2": 37, "m3": 38, "is_mand": True,
        "cej_z": 361, "ratio": 0.98,
        "cp_m2": [326, 385, 196], "cp_m3": [326, 388, 194],
        "dmin": 0.577, "delta_z": +5.60,
        "class_name": "Independent (Celah PDL Utuh)",
        "risk": "LOW RISK: D_min = 0.58 mm > 0.50 mm (Aproksimasi Mahkota)",
        "color_m2": "#00E676", "color_m3": "#B2FF59",
        "slice_x": 195,
        "center_z": 345, "center_y": 390,
    },
    {
        "q_name": "Kuadran 4: Mandibula Kanan (FDI 47 - 48)",
        "m2": 47, "m3": 48, "is_mand": True,
        "cej_z": 339, "ratio": 1.12,
        "cp_m2": [339, 354, 520], "cp_m3": None,
        "dmin": float("inf"), "delta_z": 0.0,
        "class_name": "Independent (M3 Absent / Agenesis)",
        "risk": "ZERO RISK: Gigi 48 Tidak Ada (Tulang Mandibula Bebas)",
        "color_m2": "#FF3366", "color_m3": "#888888",
        "slice_x": 520,
        "center_z": 340, "center_y": 360,
    },
]

# Create figure: 2 rows x 4 columns:
# Row 0: Sagittal Cross-Section with CEJ Overlay for Q1, Q2, Q3, Q4
# Row 1: Cross-Sectional Area Waistline Curve for M2 17, 27, 37, 47
fig = plt.figure(figsize=(24, 13), facecolor="#0e1117")

# GridSpec: top row for sagittal images, bottom row for area curves
gs = fig.add_gridspec(2, 4, height_ratios=[1.6, 1.0], hspace=0.25, wspace=0.18)

crop_half = 90 # 180x180 crop

for col_idx, q in enumerate(quad_data):
    ax_img = fig.add_subplot(gs[0, col_idx])
    ax_curve = fig.add_subplot(gs[1, col_idx])
    
    # 1. Image Crop
    sx = q["slice_x"]
    cz = q["center_z"]
    cy = q["center_y"]
    
    z_start = max(0, cz - crop_half)
    z_end = min(vol.shape[0], cz + crop_half)
    y_start = max(0, cy - crop_half)
    y_end = min(vol.shape[1], cy + crop_half)
    
    crop_vol = vol_norm[z_start:z_end, y_start:y_end, sx]
    crop_mask = mask[z_start:z_end, y_start:y_end, sx]
    
    ax_img.imshow(crop_vol, cmap="gray", origin="upper", extent=[y_start, y_end, z_end, z_start])
    
    # Draw tooth masks
    m2_bin = (crop_mask == q["m2"])
    if np.any(m2_bin):
        ax_img.contour(m2_bin, levels=[0.5], colors=[q["color_m2"]], linewidths=2.0,
                      extent=[y_start, y_end, z_start, z_end])
        
    if q["cp_m3"] is not None:
        m3_bin = (crop_mask == q["m3"])
        if np.any(m3_bin):
            ax_img.contour(m3_bin, levels=[0.5], colors=[q["color_m3"]], linewidths=2.0,
                          extent=[y_start, y_end, z_start, z_end])
            
    # Draw CEJ reference line
    cej_z = q["cej_z"]
    ax_img.axhline(y=cej_z, color="#FFD600", linestyle="--", linewidth=2.2, label=f"Garis CEJ (Z={cej_z})")
    
    # Annotate Crown vs Root side
    is_mand = q["is_mand"]
    if is_mand:
        crown_txt = "Crown (Enamel / Above CEJ) ↑"
        root_txt = "↓ Root (Dentin / Below CEJ)"
        crown_y = cej_z - 35
        root_y = cej_z + 40
    else:
        root_txt = "↑ Root (Dentin / Below CEJ)"
        crown_txt = "Crown (Enamel / Above CEJ) ↓"
        root_y = cej_z - 35
        crown_y = cej_z + 40
        
    ax_img.text(y_start + 10, crown_y, crown_txt, color="#00E676", fontsize=8.5, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="#000000", ec="#00E676", alpha=0.7))
    ax_img.text(y_start + 10, root_y, root_txt, color="#FF5252", fontsize=8.5, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="#000000", ec="#FF5252", alpha=0.7))
    
    # Draw Contact Point & Measurement Arrow
    if q["cp_m3"] is not None:
        cp2 = q["cp_m2"]
        cp3 = q["cp_m3"]
        ax_img.plot(cp2[1], cp2[0], marker="o", markersize=7, color="#FF1744", markeredgecolor="#FFFFFF", markeredgewidth=1.2)
        ax_img.plot(cp3[1], cp3[0], marker="s", markersize=7, color="#FF9100", markeredgecolor="#FFFFFF", markeredgewidth=1.2)
        ax_img.plot([cp2[1], cp3[1]], [cp2[0], cp3[0]], color="#FF1744", linestyle="-", linewidth=2.0)
        
        # Annotation text for D_min & Delta_Z
        d_str = f"D_min: {q['dmin']:.2f} mm\nΔZ_CEJ: {q['delta_z']:+.2f} mm"
        ax_img.text(cp2[1] + 8, cp2[0] - 8, d_str, color="#FFFFFF", fontsize=8.5, fontweight="bold",
                    bbox=dict(boxstyle="square,pad=0.3", fc="#B71C1C", ec="#FF8A80", alpha=0.85))
        
    ax_img.set_title(f"{q['q_name']}\n{q['class_name']}", color="#F0F6FC", fontsize=10.5, fontweight="bold", pad=8)
    ax_img.axis("off")
    
    # 2. Area vs Z Waistline Curve
    ax_curve.set_facecolor("#161b22")
    pts_m2 = np.argwhere(mask == q["m2"])
    if len(pts_m2):
        z_unq, counts = np.unique(pts_m2[:, 0], return_counts=True)
        # Area in mm2
        area_mm2 = counts * (sz * sz)
        ax_curve.plot(z_unq, area_mm2, color=q["color_m2"], linewidth=2.0, label=f"M2 ({q['m2']}) Area")
        ax_curve.axvline(x=cej_z, color="#FFD600", linestyle="--", linewidth=1.8, label=f"CEJ Z={cej_z}")
        
        # Mark constriction point
        c_val = area_mm2[np.where(z_unq == cej_z)[0][0]] if cej_z in z_unq else np.min(area_mm2)
        ax_curve.plot(cej_z, c_val, marker="v", markersize=8, color="#FFD600")
        ax_curve.text(cej_z + 2, c_val, f"  Leher CEJ\n  (Ratio: {q['ratio']:.2f})", color="#FFD600", fontsize=8)
        
    ax_curve.set_xlabel("Sumbu Vertikal Z (Voxel Slice)", color="#8B949E", fontsize=9)
    ax_curve.set_ylabel("Luas Irisan (mm²)", color="#8B949E", fontsize=9)
    ax_curve.tick_params(colors="#8B949E", labelsize=8)
    ax_curve.grid(True, linestyle=":", alpha=0.3, color="#30363d")
    ax_curve.legend(loc="upper right", fontsize=8, facecolor="#1f242c", edgecolor="#30363d", labelcolor="#F0F6FC")
    ax_curve.set_title(f"Kurva Konstriksi Servikal (M2 FDI {q['m2']})", color="#E0E0E0", fontsize=9.5, fontweight="bold")

# Main Title & Clinical Takeaway
fig.suptitle(
    "ANALISIS LENGKAP TITIK KONTAK RELATIF TERHADAP CEJ (CEMENTOENAMEL JUNCTION)\n"
    "Pasien: Ny. PRATIWI SISWI DIANA (38 Thn) | Resolusi Spasial: 0.16 mm Isotropik | Evaluasi 4 Kuadran",
    color="#FFFFFF", fontsize=14, fontweight="bold", y=0.98
)

out_png = out_dir / "pratiwi_cej_analysis_sheet.png"
brain_png = brain_dir / "pratiwi_cej_analysis_sheet.png"
plt.savefig(out_png, dpi=180, facecolor=fig.get_facecolor(), bbox_inches="tight")
plt.savefig(brain_png, dpi=180, facecolor=fig.get_facecolor(), bbox_inches="tight")
plt.close(fig)

print(f"[+] Successfully saved CEJ analysis sheet to: {out_png}")
print(f"[+] Mirrored to artifact directory: {brain_png}")
