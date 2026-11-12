"""FDI World Dental Federation (ISO 3950) Tooth Notation & Taxonomy.

Maps ToothFairy2 multi-structure / tooth classes to standard FDI two-digit
numbering and anatomical descriptions.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class ToothInfo:
    fdi: int
    quadrant: int
    tooth_number: int  # 1 to 8
    name_id: str
    name_en: str
    is_molar: bool
    is_m2: bool
    is_m3: bool


# Mapping ToothSeg Dataset121 labels (1-32) -> FDI codes
# Reference: ToothFairy2 Challenge / ToothSeg (Zenodo 14893540)
TOOTHSEG_LABEL_TO_FDI: Dict[int, int] = {
    # Quadrant 1 (Upper Right / Maxillary Right)
    1: 11, 2: 12, 3: 13, 4: 14, 5: 15, 6: 16, 7: 17, 8: 18,
    # Quadrant 2 (Upper Left / Maxillary Left)
    9: 21, 10: 22, 11: 23, 12: 24, 13: 25, 14: 26, 15: 27, 16: 28,
    # Quadrant 3 (Lower Left / Mandibular Left)
    17: 31, 18: 32, 19: 33, 20: 34, 21: 35, 22: 36, 23: 37, 24: 38,
    # Quadrant 4 (Lower Right / Mandibular Right)
    25: 41, 26: 42, 27: 43, 28: 44, 29: 45, 30: 46, 31: 47, 32: 48,
}

# Reverse mapping: FDI -> ToothSeg label ID
FDI_TO_TOOTHSEG_LABEL: Dict[int, int] = {v: k for k, v in TOOTHSEG_LABEL_TO_FDI.items()}

# OralSeg (HKU AIADIR) Label Mapping:
# 0: Background, 1: Maxilla, 2: Mandible, 3-10: FDI 11-18, 11-18: FDI 21-28,
# 19-26: FDI 31-38, 27-34: FDI 41-48, 35: Mandibular Canal
ORALSEG_LABEL_TO_FDI: Dict[int, int] = {
    idx: fdi for idx, fdi in enumerate(
        [11, 12, 13, 14, 15, 16, 17, 18,
         21, 22, 23, 24, 25, 26, 27, 28,
         31, 32, 33, 34, 35, 36, 37, 38,
         41, 42, 43, 44, 45, 46, 47, 48],
        start=3
    )
}

# Anatomical metadata for all 32 adult permanent teeth
FDI_METADATA: Dict[int, ToothInfo] = {
    # Q1: Upper Right
    11: ToothInfo(11, 1, 1, "Gigi Seri Pertama Kanan Atas", "Upper Right Central Incisor", False, False, False),
    12: ToothInfo(12, 1, 2, "Gigi Seri Kedua Kanan Atas", "Upper Right Lateral Incisor", False, False, False),
    13: ToothInfo(13, 1, 3, "Gigi Taring Kanan Atas", "Upper Right Canine", False, False, False),
    14: ToothInfo(14, 1, 4, "Gigi Premolar Pertama Kanan Atas", "Upper Right First Premolar", False, False, False),
    15: ToothInfo(15, 1, 5, "Gigi Premolar Kedua Kanan Atas", "Upper Right Second Premolar", False, False, False),
    16: ToothInfo(16, 1, 6, "Gigi Molar Pertama Kanan Atas (M1)", "Upper Right First Molar", True, False, False),
    17: ToothInfo(17, 1, 7, "Gigi Molar Kedua Kanan Atas (M2)", "Upper Right Second Molar", True, True, False),
    18: ToothInfo(18, 1, 8, "Gigi Molar Ketiga Kanan Atas (M3/Bungsu)", "Upper Right Third Molar", True, False, True),

    # Q2: Upper Left
    21: ToothInfo(21, 2, 1, "Gigi Seri Pertama Kiri Atas", "Upper Left Central Incisor", False, False, False),
    22: ToothInfo(22, 2, 2, "Gigi Seri Kedua Kiri Atas", "Upper Left Lateral Incisor", False, False, False),
    23: ToothInfo(23, 2, 3, "Gigi Taring Kiri Atas", "Upper Left Canine", False, False, False),
    24: ToothInfo(24, 2, 4, "Gigi Premolar Pertama Kiri Atas", "Upper Left First Premolar", False, False, False),
    25: ToothInfo(25, 2, 5, "Gigi Premolar Kedua Kiri Atas", "Upper Left Second Premolar", False, False, False),
    26: ToothInfo(26, 2, 6, "Gigi Molar Pertama Kiri Atas (M1)", "Upper Left First Molar", True, False, False),
    27: ToothInfo(27, 2, 7, "Gigi Molar Kedua Kiri Atas (M2)", "Upper Left Second Molar", True, True, False),
    28: ToothInfo(28, 2, 8, "Gigi Molar Ketiga Kiri Atas (M3/Bungsu)", "Upper Left Third Molar", True, False, True),

    # Q3: Lower Left
    31: ToothInfo(31, 3, 1, "Gigi Seri Pertama Kiri Bawah", "Lower Left Central Incisor", False, False, False),
    32: ToothInfo(32, 3, 2, "Gigi Seri Kedua Kiri Bawah", "Lower Left Lateral Incisor", False, False, False),
    33: ToothInfo(33, 3, 3, "Gigi Taring Kiri Bawah", "Lower Left Canine", False, False, False),
    34: ToothInfo(34, 3, 4, "Gigi Premolar Pertama Kiri Bawah", "Lower Left First Premolar", False, False, False),
    35: ToothInfo(35, 3, 5, "Gigi Premolar Kedua Kiri Bawah", "Lower Left Second Premolar", False, False, False),
    36: ToothInfo(36, 3, 6, "Gigi Molar Pertama Kiri Bawah (M1)", "Lower Left First Molar", True, False, False),
    37: ToothInfo(37, 3, 7, "Gigi Molar Kedua Kiri Bawah (M2)", "Lower Left Second Molar", True, True, False),
    38: ToothInfo(38, 3, 8, "Gigi Molar Ketiga Kiri Bawah (M3/Bungsu)", "Lower Left Third Molar", True, False, True),

    # Q4: Lower Right
    41: ToothInfo(41, 4, 1, "Gigi Seri Pertama Kanan Bawah", "Lower Right Central Incisor", False, False, False),
    42: ToothInfo(42, 4, 2, "Gigi Seri Kedua Kanan Bawah", "Lower Right Lateral Incisor", False, False, False),
    43: ToothInfo(43, 4, 3, "Gigi Taring Kanan Bawah", "Lower Right Canine", False, False, False),
    44: ToothInfo(44, 4, 4, "Gigi Premolar Pertama Kanan Bawah", "Lower Right First Premolar", False, False, False),
    45: ToothInfo(45, 4, 5, "Gigi Premolar Kedua Kanan Bawah", "Lower Right Second Premolar", False, False, False),
    46: ToothInfo(46, 4, 6, "Gigi Molar Pertama Kanan Bawah (M1)", "Lower Right First Molar", True, False, False),
    47: ToothInfo(47, 4, 7, "Gigi Molar Kedua Kanan Bawah (M2)", "Lower Right Second Molar", True, True, False),
    48: ToothInfo(48, 4, 8, "Gigi Molar Ketiga Kanan Bawah (M3/Bungsu)", "Lower Right Third Molar", True, False, True),
}

# Supported M2-M3 Research Quadrant Pairs
M2_M3_PAIRS: Dict[str, Tuple[int, int]] = {
    "Upper Right (Q1)": (17, 18),
    "Upper Left (Q2)": (27, 28),
    "Lower Left (Q3)": (37, 38),
    "Lower Right (Q4)": (47, 48),
}


def get_fdi_info(fdi_code: int) -> Optional[ToothInfo]:
    """Retrieve detailed metadata for an FDI tooth code."""
    return FDI_METADATA.get(fdi_code, None)


def get_quadrant_name(quad_id: int) -> str:
    """Return quadrant name by quadrant ID (1..4)."""
    mapping = {
        1: "Kuadran 1 (Maksila Kanan)",
        2: "Kuadran 2 (Maksila Kiri)",
        3: "Kuadran 3 (Mandibula Kiri)",
        4: "Kuadran 4 (Mandibula Kanan)",
    }
    return mapping.get(quad_id, f"Kuadran {quad_id}")
