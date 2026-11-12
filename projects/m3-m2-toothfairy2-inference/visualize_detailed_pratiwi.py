"""High-Resolution 6-Panel Anatomical Validation Sheet for Patient Pratiwi.

Shows:
1. Maxillary Arch (Axial Upper Teeth 11-28)
2. Mandibular Arch (Axial Lower Teeth 31-48)
3. Right Molar Sagittal (M2 FDI 47 & M3 FDI 48)
4. Left Molar Sagittal (M2 FDI 37 & M3 FDI 38)
5. Anterior Coronal View (Incisors & Canines)
6. Posterior Coronal View (Molars & Premolars)
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from matplotlib.colors import ListedColormap

from fdi_taxonomy import FDI_METADATA


def render_pratiwi_validation_sheet():
    img_path = Path(r"D:\Workspace\TA3\data\DICOMM\output_aligned\volume.mha")
    mask_path = Path(r"D:\Workspace\TA3\data\DICOMM\output_aligned\pred_fdi_mask.mha")
    out_png = Path(r"D:\Workspace\TA3\data\DICOMM\output_aligned\pratiwi_verified_mpr.png")

    img_itk = sitk.ReadImage(str(img_path))
    mask_itk = sitk.ReadImage(str(mask_path))

    arr_img = sitk.GetArrayFromImage(img_itk)  # (Z, Y, X)
    arr_mask = sitk.GetArrayFromImage(mask_itk)

    # Dental HU bone window: 200 - 2400 HU
    vmin, vmax = 200, 2400
    vol_disp = np.clip(arr_img, vmin, vmax)
    vol_disp = (vol_disp - vmin) / (vmax - vmin)

    # Anatomical slice coordinates
    # Upper arch (Maxilla): around Z = 340
    # Lower arch (Mandible): around Z = 220
    # Midline incisors: X = 350, Y = 140
    # Right molars: X = 505
    # Left molars: X = 200
    z_upper = 345
    z_lower = 220
    x_right = 505
    x_left = 200
    y_ant = 140
    y_post = 360

    fig, axes = plt.subplots(2, 3, figsize=(18, 12), facecolor="#111111")
    cmap_base = plt.get_cmap("tab20")

    def draw_slice(ax, slice_img, slice_mask, title, xlabel="", ylabel=""):
        ax.imshow(slice_img, cmap="gray", origin="lower")
        teeth_in_slice = np.unique(slice_mask)
        for fdi in teeth_in_slice:
            if fdi == 0:
                continue
            binary = (slice_mask == fdi)
            quad = fdi // 10
            # Quad colors: Q1=cyan, Q2=orange, Q3=green, Q4=magenta
            q_colors = {1: "#00E5FF", 2: "#FF9100", 3: "#00E676", 4: "#FF1744"}
            color = q_colors.get(quad, "#FFFF00")
            ax.contour(binary, levels=[0.5], colors=[color], linewidths=1.8, origin="lower")

            pts = np.argwhere(binary)
            if len(pts) > 15:
                cy, cx = pts.mean(axis=0)
                ax.text(cx, cy, str(int(fdi)), color="white", fontsize=8, fontweight="bold",
                        ha="center", va="center",
                        bbox=dict(boxstyle="circle,pad=0.2", fc=color, ec="black", alpha=0.85))

        ax.set_title(title, color="white", fontsize=11, fontweight="bold", pad=8)
        ax.axis("off")

    # 1. Axial Upper Arch (Maxilla)
    draw_slice(axes[0, 0], vol_disp[z_upper, :, :], arr_mask[z_upper, :, :],
               f"1. Rahang Atas (Axial Z={z_upper}) - Kuadran 1 (Kanan) & 2 (Kiri)")

    # 2. Axial Lower Arch (Mandible)
    draw_slice(axes[0, 1], vol_disp[z_lower, :, :], arr_mask[z_lower, :, :],
               f"2. Rahang Bawah (Axial Z={z_lower}) - Kuadran 4 (Kanan) & 3 (Kiri)")

    # 3. Coronal Posterior (Molar Cross-Section)
    draw_slice(axes[0, 2], vol_disp[:, y_post, :], arr_mask[:, y_post, :],
               f"3. Potongan Coronal Molar (Y={y_post}) - Atas vs Bawah")

    # 4. Right Sagittal (Q1 & Q4 Molars: 17, 18, 47, 48)
    draw_slice(axes[1, 0], vol_disp[:, :, x_right], arr_mask[:, :, x_right],
               f"4. Sagittal Kanan (X={x_right}) - Molar Q1 (17, 18) & Q4 (47, 48)")

    # 5. Left Sagittal (Q2 & Q3 Molars: 27, 28, 37, 38)
    draw_slice(axes[1, 1], vol_disp[:, :, x_left], arr_mask[:, :, x_left],
               f"5. Sagittal Kiri (X={x_left}) - Molar Q2 (27, 28) & Q3 (37, 38)")

    # 6. Coronal Anterior (Incisors & Canines)
    draw_slice(axes[1, 2], vol_disp[:, y_ant, :], arr_mask[:, y_ant, :],
               f"6. Potongan Coronal Depan (Y={y_ant}) - Gigi Seri & Taring")

    fig.suptitle("PENGUJIAN VALIDASI ANATOMI GIGI FDI (TOOTHFAIRY2)\nPasien: PRATIWI SISWI DIANA (38 THN) - 31 Gigi Terklasifikasi",
                 color="#00E5FF", fontsize=15, fontweight="bold", y=0.98)

    # Legend for quadrants
    handles = [
        plt.Line2D([0], [0], color="#00E5FF", lw=4, label="Kuadran 1: Kanan Atas (FDI 11-18)"),
        plt.Line2D([0], [0], color="#FF9100", lw=4, label="Kuadran 2: Kiri Atas (FDI 21-28)"),
        plt.Line2D([0], [0], color="#00E676", lw=4, label="Kuadran 3: Kiri Bawah (FDI 31-38)"),
        plt.Line2D([0], [0], color="#FF1744", lw=4, label="Kuadran 4: Kanan Bawah (FDI 41-48)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, framealpha=0.3, facecolor="black",
               edgecolor="white", labelcolor="white", fontsize=10)

    plt.tight_layout(rect=[0, 0.05, 1, 0.94])
    plt.savefig(out_png, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    print(f"[+] Saved Pratiwi verified MPR sheet to: {out_png}")


if __name__ == "__main__":
    render_pratiwi_validation_sheet()
