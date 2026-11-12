"""Clinical Tooth Impaction & Angulation Analyzer for CBCT Molar Pairs.

Computes Winter's Classification (angulation) and Pell & Gregory-style
depth metrics from 3D segmented FDI tooth masks.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import SimpleITK as sitk

from fdi_taxonomy import M2_M3_PAIRS, get_fdi_info


def analyze_m3_impaction(
    mask_zyx: np.ndarray,
    spacing_xyz: Tuple[float, float, float],
) -> Dict[str, Dict]:
    """Evaluates whether each M3 third molar is impacted, its angulation, and depth.

    Args:
        mask_zyx: 3D numpy array of segmented FDI labels.
        spacing_xyz: Physical spacing in mm (X, Y, Z).

    Returns:
        Dict mapping quadrant name to impaction assessment.
    """
    results = {}

    for q_name, (m2, m3) in M2_M3_PAIRS.items():
        pts_m2 = np.argwhere(mask_zyx == m2)
        pts_m3 = np.argwhere(mask_zyx == m3)

        if len(pts_m2) == 0 or len(pts_m3) == 0:
            results[q_name] = {
                "m2_fdi": m2,
                "m3_fdi": m3,
                "is_evaluable": False,
                "status": "Molar tidak lengkap (absen)",
                "impaction": "Unknown",
            }
            continue

        z_m2_min, z_m2_max = pts_m2[:, 0].min(), pts_m2[:, 0].max()
        z_m3_min, z_m3_max = pts_m3[:, 0].min(), pts_m3[:, 0].max()

        c_m2 = pts_m2.mean(axis=0)
        c_m3 = pts_m3.mean(axis=0)

        # Principal Component Analysis (PCA) for long axis of tooth
        cov_m3 = np.cov((pts_m3 - c_m3).T)
        evals_m3, evecs_m3 = np.linalg.eigh(cov_m3)
        long_axis_m3 = evecs_m3[:, -1]

        cov_m2 = np.cov((pts_m2 - c_m2).T)
        evals_m2, evecs_m2 = np.linalg.eigh(cov_m2)
        long_axis_m2 = evecs_m2[:, -1]

        # 3D Angle between tooth long axes
        cos_ang = np.abs(np.dot(long_axis_m2, long_axis_m3))
        angle_deg = float(np.degrees(np.arccos(np.clip(cos_ang, -1.0, 1.0))))

        is_mandible = m3 in [38, 48]
        # In HFS orientation:
        # Mandible: Crown is towards superior (higher Z)
        # Maxilla: Crown is towards inferior (lower Z)
        if is_mandible:
            occlusal_diff_mm = float((z_m2_max - z_m3_max) * spacing_xyz[2])
            is_impacted = (angle_deg > 25.0) or (occlusal_diff_mm > 2.0)
            if angle_deg > 65.0:
                winter_class = "Horizontal"
            elif angle_deg > 25.0:
                # Check anterior-posterior direction of tilt
                y_tilt = c_m3[1] - c_m2[1]
                winter_class = "Mesioangular" if y_tilt < 0 else "Distoangular"
            else:
                winter_class = "Vertical"

            # Pell & Gregory depth classification
            if occlusal_diff_mm <= 1.0:
                pg_level = "Level A (Sejajar bidang oklusal M2)"
            elif occlusal_diff_mm <= 4.0:
                pg_level = "Level B (Antara bidang oklusal dan garis servikal M2)"
            else:
                pg_level = "Level C (Di bawah garis servikal M2 / Impaksi Dalam)"
        else:
            # Maxilla
            occlusal_diff_mm = float((z_m3_min - z_m2_min) * spacing_xyz[2])
            is_impacted = (angle_deg > 25.0) or (occlusal_diff_mm > 2.5)
            winter_class = "Vertical / High" if angle_deg <= 25.0 else "Angulated High Maxillary"
            pg_level = "Level A" if occlusal_diff_mm <= 1.5 else "Level B/C (Tertanam Tinggi)"

        results[q_name] = {
            "m2_fdi": m2,
            "m3_fdi": m3,
            "is_evaluable": True,
            "is_impacted": is_impacted,
            "angulation_degrees": round(angle_deg, 1),
            "winters_classification": winter_class,
            "pell_gregory_depth": pg_level,
            "occlusal_difference_mm": round(occlusal_diff_mm, 2),
            "clinical_diagnosis": "Gigi Impaksi (Impacted Third Molar)" if is_impacted else "Erupsi Normal (Erupted)",
        }

    return results
