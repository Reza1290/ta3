import SimpleITK as sitk
import numpy as np
from pathlib import Path
from scipy import ndimage
from scipy.spatial import cKDTree

vol_path = Path("data/DICOMM/output_aligned/volume.mha")
mask_path = Path("data/DICOMM/output_aligned/pred_fdi_mask.mha")

vol = sitk.GetArrayFromImage(sitk.ReadImage(str(vol_path)))
mask = sitk.GetArrayFromImage(sitk.ReadImage(str(mask_path)))

spacing = (0.16, 0.16, 0.16)
sz = spacing[0]

def extract_surface_voxels(mask_3d):
    struct = ndimage.generate_binary_structure(3, 1)
    eroded = ndimage.binary_erosion(mask_3d, structure=struct)
    return np.argwhere(mask_3d & (~eroded))

def detect_cej_m2(mask_m2, is_mandibular):
    pts = np.argwhere(mask_m2)
    z_min, z_max = pts[:, 0].min(), pts[:, 0].max()
    z_span = z_max - z_min
    z_indices, counts = np.unique(pts[:, 0], return_counts=True)
    counts_dict = dict(zip(z_indices, counts))

    # In our volume (Z=0 is cranial, Z=640 is caudal):
    # Maxillary: Root is at lower Z (closer to 0), crown is at higher Z (closer to bite plane)
    # Mandibular: Crown is at lower Z (closer to bite plane), root is at higher Z (closer to 640)
    if is_mandibular:
        # Crown: z_min to z_min + 0.3*z_span
        # CEJ waistline: search in 25% - 60% of height from crown (z_min)
        s_min = int(z_min + 0.25 * z_span)
        s_max = int(z_min + 0.60 * z_span)
    else:
        # Maxillary: Root is at z_min. Crown is at z_max.
        # CEJ waistline: search in 35% - 75% of height from root (z_min)
        s_min = int(z_min + 0.35 * z_span)
        s_max = int(z_min + 0.70 * z_span)

    valid_z = [z for z in range(s_min, s_max + 1) if z in counts_dict]
    if not valid_z:
        cej_z = int(z_min + 0.5 * z_span)
        ratio = 1.0
    else:
        # smooth counts
        c_vals = [counts_dict[z] for z in valid_z]
        min_idx = np.argmin(c_vals)
        cej_z = valid_z[min_idx]
        mean_area = np.mean(counts)
        ratio = c_vals[min_idx] / (mean_area + 1e-6)

    return cej_z, ratio, z_min, z_max

pairs = [
    ("Kuadran 1 (Upper Right)", 17, 18, False),
    ("Kuadran 2 (Upper Left)", 27, 28, False),
    ("Kuadran 3 (Lower Left)", 37, 38, True),
    ("Kuadran 4 (Lower Right)", 47, 48, True),
]

print("=" * 70)
print("CEJ DETECTION & SPATIAL CONTACT ANALYSIS: PATIENT PRATIWI")
print("=" * 70)

for q_name, m2, m3, is_mand in pairs:
    m2_mask = (mask == m2)
    m3_mask = (mask == m3)
    cnt_m2 = np.sum(m2_mask)
    cnt_m3 = np.sum(m3_mask)

    print(f"\n--- {q_name}: FDI {m2} (M2) vs FDI {m3} (M3) ---")
    if cnt_m2 == 0:
        print("  M2 is absent.")
        continue

    cej_z, ratio, z_min, z_max = detect_cej_m2(m2_mask, is_mand)
    print(f"  M2 ({m2}) Z-range: [{z_min}, {z_max}] (span: {(z_max-z_min)*sz:.2f} mm)")
    print(f"  Detected CEJ plane: Z = {cej_z} (waist constriction ratio: {ratio:.2f})")

    if cnt_m3 == 0:
        print("  M3 is ABSENT (no contact).")
        print("  Clinical Status: Independent (No M3)")
        continue

    # Surface proximity
    surf_m2 = extract_surface_voxels(m2_mask)
    surf_m3 = extract_surface_voxels(m3_mask)

    tree = cKDTree(surf_m2 * sz)
    dists, nearest_m2 = tree.query(surf_m3 * sz)
    best_m3_idx = np.argmin(dists)
    dmin = dists[best_m3_idx]
    best_m3_vox = surf_m3[best_m3_idx]
    best_m2_vox = surf_m2[nearest_m2[best_m3_idx]]

    print(f"  Closest point on M2: {best_m2_vox}")
    print(f"  Closest point on M3: {best_m3_vox}")
    print(f"  Minimum distance D_min: {dmin:.3f} mm")

    # Elevation Delta_z relative to CEJ
    # In maxilla: higher Z is crown (coronal), lower Z is root (apical)
    # In mandible: lower Z is crown (coronal), higher Z is root (apical)
    cp_z = best_m2_vox[0]
    if is_mand:
        # Coronal is lower Z -> delta_z = (cej_z - cp_z) * sz
        # if cp_z < cej_z: positive (Above CEJ / Coronal)
        delta_z = (cej_z - cp_z) * sz
    else:
        # Coronal is higher Z -> delta_z = (cp_z - cej_z) * sz
        # if cp_z > cej_z: positive (Above CEJ / Coronal)
        delta_z = (cp_z - cej_z) * sz

    print(f"  Contact elevation Delta_z: {delta_z:+.3f} mm relative to CEJ")
    tau_cej = 1.0 # mm

    if dmin > 0.50:
        c_name = "Independent"
        c_status = f"NO CONTACT (D_min = {dmin:.2f} mm > 0.50 mm)"
    elif delta_z > tau_cej:
        c_name = "Above_CEJ"
        c_status = f"CORONAL CONTACT (+{delta_z:.2f} mm diatas CEJ, Enamel Zone)"
    elif delta_z < -tau_cej:
        c_name = "Below_CEJ"
        c_status = f"RADICULAR CONTACT ({delta_z:.2f} mm dibawah CEJ, Root Dentin / ERR Risk Zone!)"
    else:
        c_name = "At_CEJ"
        c_status = f"CERVICAL CONTACT ({delta_z:+.2f} mm pada margin leher CEJ, Plaque/Pocket Risk)"

    print(f"  CEJ Classification: {c_name} -> {c_status}")
