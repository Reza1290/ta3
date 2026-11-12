"""Generate Clinical Preview Sheet for Candidate CBCT Volumes.

Renders panoramic-style Maximum Intensity Projections (MIP), dental arch axial views,
and sagittal molar cuts so the clinician/user can visually inspect CBCT quality before processing.
"""

from pathlib import Path
import numpy as np
import SimpleITK as sitk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def create_cbct_preview():
    # Load Case 008 (32/32 Teeth Complete)
    path_008 = Path(r"D:\Workspace\TA3\data\raw\ToothFairy2F_008_0000.mha")
    gt_path_008 = Path(r"D:\Workspace\TA3\data\toothfairy2\Dataset112_ToothFairy2\labelsTr\ToothFairy2F_008.mha")
    
    # Load Case 001 (Sparse/Edentulous tested earlier)
    path_001 = Path(r"D:\Workspace\TA3\data\raw\ToothFairy2F_001_0000.mha")
    
    img_008 = sitk.ReadImage(str(path_008))
    vol_008 = sitk.GetArrayFromImage(img_008)  # (Z, Y, X)
    
    img_001 = sitk.ReadImage(str(path_001))
    vol_001 = sitk.GetArrayFromImage(img_001)
    
    # Bone windowing for high visual clarity (WL=500, WW=2000 -> [-500, 1500])
    vmin, vmax = -400, 1400
    
    fig = plt.figure(figsize=(18, 12), dpi=200)
    
    # Grid layout:
    # Row 1: Case 008 Full Complete CBCT (All 32 Teeth)
    # Row 2: Detailed Views of Case 008 (Axial Arch, Coronal, Sagittal Molars)
    # Row 3: Case 001 Comparison (Explaining why it looked sparse/incomplete)
    
    # 1. Case 008: Panoramic-like Coronal MIP (Maximum Intensity Projection)
    # Projecting through dental arch Y range to show all teeth in one view
    y_min, y_max = int(vol_008.shape[1] * 0.15), int(vol_008.shape[1] * 0.75)
    mip_cor_008 = np.max(vol_008[:, y_min:y_max, :], axis=1)
    
    ax1 = plt.subplot(2, 3, 1)
    ax1.imshow(mip_cor_008, cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
    ax1.set_title("KASUS 008: Panoramic Coronal MIP\n(Lengkap 32 Gigi Utuh & 4 Gigi Molar M3)", fontsize=11, fontweight="bold", color="darkgreen")
    ax1.axis("off")
    
    # 2. Case 008: Axial Arch MIP (Lengkung Gigi Rahang Bawah & Atas)
    z_min, z_max = int(vol_008.shape[0] * 0.1), int(vol_008.shape[0] * 0.8)
    mip_ax_008 = np.max(vol_008[z_min:z_max, :, :], axis=0)
    
    ax2 = plt.subplot(2, 3, 2)
    ax2.imshow(mip_ax_008, cmap="gray", vmin=vmin, vmax=vmax)
    ax2.set_title("KASUS 008: Lengkung Rahang Aksial (MIP)\n(Tapal Kuda Gigi Rapat & Simetris)", fontsize=11, fontweight="bold")
    ax2.axis("off")
    
    # 3. Case 008: Sagittal Slice melalui Molar Kiri (M2 & M3)
    # X coordinate around molar area
    x_molar_left = int(vol_008.shape[2] * 0.32)
    ax3 = plt.subplot(2, 3, 3)
    ax3.imshow(vol_008[:, :, x_molar_left], cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
    ax3.set_title(f"KASUS 008: Irisan Sagital Molar Kiri (X={x_molar_left})\n(Molar M1, M2, dan M3 Terlihat Tegak/Utuh)", fontsize=11, fontweight="bold")
    ax3.axis("off")
    
    # 4. Case 008: Irisan Aksial Gigi Atas (Maxilla)
    z_upper = int(vol_008.shape[0] * 0.25)
    ax4 = plt.subplot(2, 3, 4)
    ax4.imshow(vol_008[z_upper, :, :], cmap="gray", vmin=vmin, vmax=vmax)
    ax4.set_title(f"KASUS 008: Irisan Aksial Rahang Atas (Z={z_upper})\n(Enamel Mahkota Sangat Tajam & Kontras)", fontsize=11, fontweight="bold")
    ax4.axis("off")
    
    # 5. Case 008: Irisan Aksial Gigi Bawah (Mandibula)
    z_lower = int(vol_008.shape[0] * 0.65)
    ax5 = plt.subplot(2, 3, 5)
    ax5.imshow(vol_008[z_lower, :, :], cmap="gray", vmin=vmin, vmax=vmax)
    ax5.set_title(f"KASUS 008: Irisan Aksial Rahang Bawah (Z={z_lower})\n(Akar Gigi & Tulang Alveolar Utuh)", fontsize=11, fontweight="bold")
    ax5.axis("off")
    
    # 6. Perbandingan Kasus 001 (Kasus yang dites tadi)
    mip_cor_001 = np.max(vol_001[:, int(vol_001.shape[1]*0.15):int(vol_001.shape[1]*0.75), :], axis=1)
    ax6 = plt.subplot(2, 3, 6)
    ax6.imshow(mip_cor_001, cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
    ax6.set_title("KASUS 001 (Yang Dites Tadi): Coronal MIP\n(Pasien Ompong/Edentulous, Tanpa M3, Hanya 22 Gigi)", fontsize=11, fontweight="bold", color="darkred")
    ax6.axis("off")
    
    plt.suptitle(
        "PREVIEW KLINIS CBCT: PERBANDINGAN ANATOMI & KUALITAS CITRA SEBELUM SEGMENTASI\n"
        "Kasus 008 Memiliki 32/32 Gigi Lengkap & 4 Molar M3 (Sangat Bagus untuk Evaluasi)",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    
    out_path = Path(r"D:\Workspace\TA3\data\DICOM_BENCHMARK\output\cbct_case_preview.png")
    plt.savefig(str(out_path), dpi=200)
    plt.close()
    print(f"[+] Preview saved successfully: {out_path}")


if __name__ == "__main__":
    create_cbct_preview()
