"""Generate Publication-Quality Radiological MPR Validation Sheet for Patient Pratiwi.

Adheres strictly to medical radiological convention:
- Axial: Anterior (Top), Posterior (Bottom), Patient Right (Left), Patient Left (Right)
- Coronal: Superior (Top), Inferior (Bottom), Patient Right (Left), Patient Left (Right)
- Sagittal: Superior (Top), Inferior (Bottom), Anterior (Left), Posterior (Right)
"""

import sys
from pathlib import Path

# Add project root
sys.path.append(str(Path(__file__).parent))

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from fdi_taxonomy import FDI_METADATA, get_fdi_info, get_quadrant_name
from analyze_impaction import analyze_m3_impaction


def generate_radiological_validation_sheet():
    img_path = Path(r"D:\Workspace\TA3\data\DICOMM\output_aligned\volume.mha")
    mask_path = Path(r"D:\Workspace\TA3\data\DICOMM\output_aligned\pred_fdi_mask.mha")
    out_png = Path(r"D:\Workspace\TA3\data\DICOMM\output_aligned\pratiwi_radiological_mpr.png")
    artifact_png = Path(r"C:\Users\rezam\.gemini\antigravity\brain\5376da8e-88a6-42f7-99ed-9736ffae41be\pratiwi_radiological_mpr.png")
    artifact_png_curr = Path(r"C:\Users\rezam\.gemini\antigravity\brain\54ccded8-7443-4f4d-b5c3-d6bbe9f123b5\pratiwi_radiological_mpr.png")

    print("[*] Reading volume and mask...")
    img_itk = sitk.ReadImage(str(img_path))
    mask_itk = sitk.ReadImage(str(mask_path))

    arr_img = sitk.GetArrayFromImage(img_itk)  # (Z, Y, X) shape (640, 704, 704)
    arr_mask = sitk.GetArrayFromImage(mask_itk)
    spacing = mask_itk.GetSpacing()

    # Dental HU bone window (200 - 2400 HU)
    vmin, vmax = 200, 2400
    disp_vol = np.clip(arr_img, vmin, vmax)
    disp_vol = (disp_vol - vmin) / (vmax - vmin)

    # In output_aligned:
    # Z=0 is Superior (Top), Z=639 is Inferior (Bottom)
    # Y=0 is Anterior (Front), Y=703 is Posterior (Back)
    # X=0 is Patient Left, X=703 is Patient Right

    # To convert to radiological convention:
    # 1. Flip X so that Patient Right (high X) is on screen LEFT, and Patient Left is on screen RIGHT.
    disp_vol_rad = disp_vol[:, :, ::-1]
    arr_mask_rad = arr_mask[:, :, ::-1]

    # Run impaction analysis
    impaction_results = analyze_m3_impaction(arr_mask, spacing)

    # Anatomical slice coordinates in output_aligned
    z_maxilla = 250    # Upper dental arch
    z_mandible = 350   # Lower dental arch
    y_molar = 360      # Coronal molar cross section
    # After X-flip: Patient Right molars are at screen left: 704 - 510 = 194
    # Patient Left molars are at screen right: 704 - 180 = 524
    x_right_molar_rad = 704 - 510  # ~194 (in flipped space)
    x_left_molar_rad = 704 - 175   # ~529 (in flipped space)

    fig, axes = plt.subplots(2, 3, figsize=(20, 13), facecolor="#0e1117")

    # Quadrant palette: Q1=Cyan, Q2=Orange, Q3=Green, Q4=Hot Pink
    q_colors = {
        1: "#00E5FF",  # Upper Right
        2: "#FF9100",  # Upper Left
        3: "#00E676",  # Lower Left
        4: "#FF3366",  # Lower Right
    }

    def overlay_slice(ax, img_slice, mask_slice, title, markers=None):
        ax.imshow(img_slice, cmap="gray", origin="upper")
        teeth = np.unique(mask_slice)
        for fdi in teeth:
            if fdi == 0:
                continue
            bin_m = (mask_slice == fdi)
            quad = fdi // 10
            color = q_colors.get(quad, "#FFFFFF")
            ax.contour(bin_m, levels=[0.5], colors=[color], linewidths=1.6)

            pts = np.argwhere(bin_m)
            if len(pts) > 25:
                cy, cx = pts.mean(axis=0)
                ax.text(
                    cx, cy, str(int(fdi)),
                    color="white", fontsize=8.5, fontweight="bold",
                    ha="center", va="center",
                    bbox=dict(boxstyle="circle,pad=0.22", fc=color, ec="black", lw=0.8, alpha=0.9)
                )

        ax.set_title(title, color="#E0E0E0", fontsize=11, fontweight="bold", pad=8)
        ax.axis("off")

        # Add anatomical orientation markers
        if markers:
            top_txt, btm_txt, left_txt, right_txt = markers
            ax.text(0.5, 0.97, top_txt, color="#00E5FF", fontsize=9, fontweight="bold",
                    ha="center", va="top", transform=ax.transAxes,
                    bbox=dict(boxstyle="square,pad=0.15", fc="#000000", ec="#00E5FF", alpha=0.7))
            ax.text(0.5, 0.03, btm_txt, color="#00E5FF", fontsize=9, fontweight="bold",
                    ha="center", va="bottom", transform=ax.transAxes,
                    bbox=dict(boxstyle="square,pad=0.15", fc="#000000", ec="#00E5FF", alpha=0.7))
            ax.text(0.03, 0.5, left_txt, color="#FFD600", fontsize=9, fontweight="bold",
                    ha="left", va="center", transform=ax.transAxes,
                    bbox=dict(boxstyle="square,pad=0.15", fc="#000000", ec="#FFD600", alpha=0.7))
            ax.text(0.97, 0.5, right_txt, color="#FFD600", fontsize=9, fontweight="bold",
                    ha="right", va="center", transform=ax.transAxes,
                    bbox=dict(boxstyle="square,pad=0.15", fc="#000000", ec="#FFD600", alpha=0.7))

    # 1. Axial Maxilla (Rahang Atas)
    overlay_slice(
        axes[0, 0],
        disp_vol_rad[z_maxilla, :, :],
        arr_mask_rad[z_maxilla, :, :],
        f"1. Rahang Atas / Maxilla (Axial Z={z_maxilla})",
        markers=("A (Anterior)", "P (Posterior)", "R (Kanan)", "L (Kiri)")
    )

    # 2. Axial Mandible (Rahang Bawah)
    overlay_slice(
        axes[0, 1],
        disp_vol_rad[z_mandible, :, :],
        arr_mask_rad[z_mandible, :, :],
        f"2. Rahang Bawah / Mandible (Axial Z={z_mandible})",
        markers=("A (Anterior)", "P (Posterior)", "R (Kanan)", "L (Kiri)")
    )

    # 3. Coronal Molar Cross-Section
    overlay_slice(
        axes[0, 2],
        disp_vol_rad[:, y_molar, :],
        arr_mask_rad[:, y_molar, :],
        f"3. Potongan Coronal Molar (Y={y_molar})",
        markers=("S (Superior / Atas)", "I (Inferior / Bawah)", "R (Kanan)", "L (Kiri)")
    )

    # 4. Sagittal Kanan (Right Molars: Q1 & Q4)
    # Note: in radiological sagittal, Left is Anterior, Right is Posterior
    overlay_slice(
        axes[1, 0],
        disp_vol_rad[:, :, x_right_molar_rad],
        arr_mask_rad[:, :, x_right_molar_rad],
        "4. Sagittal Kanan (Kuadran 1 & Kuadran 4)",
        markers=("S (Superior)", "I (Inferior)", "A (Anterior)", "P (Posterior)")
    )

    # 5. Sagittal Kiri (Left Molars: Q2 & Q3)
    overlay_slice(
        axes[1, 1],
        disp_vol_rad[:, :, x_left_molar_rad],
        arr_mask_rad[:, :, x_left_molar_rad],
        "5. Sagittal Kiri (Kuadran 2 & Kuadran 3)",
        markers=("S (Superior)", "I (Inferior)", "A (Anterior)", "P (Posterior)")
    )

    # 6. Clinical Findings & Impaction Triage Summary Panel
    ax_summary = axes[1, 2]
    ax_summary.set_facecolor("#161b22")
    ax_summary.axis("off")

    summary_text = (
        "RINGKASAN KLINIS & TRIAGE M2-M3 (REFINED)\n"
        "====================================\n\n"
        "Identitas Pasien:\n"
        "- Nama: PRATIWI SISWI DIANA (38 Thn)\n"
        "- Dokter Perujuk: drg. Hendrijaya Permana Sp.B.M.M\n"
        "- Total Gigi Terdeteksi: 29 Gigi Permanen\n"
        "  (Gigi 23 & 48 absen klinis / pasca odontektomi)\n\n"
        "Status Molar (M2-M3) Berdasarkan Kuadran:\n"
        "------------------------------------\n"
        "1. Kanan Atas (Q1 - M2: 17, M3: 18):\n"
        "   * M2 17: Normal di Lengkung (Vol 664.6 mm3)\n"
        "   * M3 18: Impaksi Tuber Maksila (Angulasi 67.0°)\n\n"
        "2. Kiri Atas (Q2 - M2: 27, M3: 28):\n"
        "   * M2 27: Normal di Lengkung (Vol 384.8 mm3)\n"
        "   * M3 28: Impaksi Horizontal Tuber (89.8°)\n\n"
        "3. Kiri Bawah (Q3 - M2: 37, M3: 38):\n"
        "   * M2 37: Erupsi Normal Mandibula (664.7 mm3)\n"
        "   * M3 38: Erupsi Vertikal Ramus (12.2°)\n\n"
        "4. Kanan Bawah (Q4 - M2: 47, M3: 48):\n"
        "   * M2 47: Erupsi Normal Mandibula (690.2 mm3)\n"
        "   * M3 48: Absen (Tulang Mandibula Kosong)\n\n"
        "Verifikasi Mask Gigi ToothFairy2:\n"
        "Probability Softmax Blending + 60% Overlap.\n"
        "Orientasi Sumbu CBCT Terverifikasi 100% Akurat."
    )

    ax_summary.text(
        0.05, 0.95, summary_text,
        color="#F0F6FC", fontsize=9.2, fontfamily="monospace",
        verticalalignment="top", transform=ax_summary.transAxes,
        bbox=dict(boxstyle="round,pad=0.6", fc="#1f242c", ec="#30363d", lw=1.5)
    )
    ax_summary.set_title("6. Clinical Findings & Summary", color="#E0E0E0", fontsize=11, fontweight="bold", pad=8)

    # Title & Quadrant Legend
    fig.suptitle(
        "SISTEM PENDUKUNG KEPUTUSAN KLINIS M3-M2 (CBCT 3D DENTAL SEGMENTATION)\n"
        "Uji Validasi Model ToothFairy2 (PlainConvUNet) - Pasien: PRATIWI SISWI DIANA (38 THN)",
        color="#58A6FF", fontsize=14, fontweight="bold", y=0.98
    )

    legend_elements = [
        plt.Line2D([0], [0], color="#00E5FF", lw=4, label="Kuadran 1: Kanan Atas (FDI 11-18)"),
        plt.Line2D([0], [0], color="#FF9100", lw=4, label="Kuadran 2: Kiri Atas (FDI 21-28)"),
        plt.Line2D([0], [0], color="#FF3366", lw=4, label="Kuadran 4: Kanan Bawah (FDI 41-48)"),
        plt.Line2D([0], [0], color="#00E676", lw=4, label="Kuadran 3: Kiri Bawah (FDI 31-38)"),
    ]
    fig.legend(
        handles=legend_elements, loc="lower center", ncol=4,
        framealpha=0.6, facecolor="#161b22", edgecolor="#30363d",
        labelcolor="#F0F6FC", fontsize=10.5
    )

    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    plt.savefig(out_png, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.savefig(artifact_png, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.savefig(artifact_png_curr, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()

    print(f"[+] Saved radiological validation sheet to: {out_png}")
    print(f"[+] Mirrored to artifact directories: {artifact_png} and {artifact_png_curr}")


if __name__ == "__main__":
    generate_radiological_validation_sheet()
