# CBCT M3–M2 Spatial-Pathological Analysis Workspace (Master Meta-Repository)

Workspace riset dan pengembangan Sistem Pendukung Keputusan Klinis (CDSS) klasifikasi relasi spasial-patologis gigi molar ketiga dan kedua (M3–M2) pada citra CBCT.

Repository ini bertindak sebagai **Parent Meta-Repository** yang mengorkestrasi sub-repositori riset, aplikasi CDSS, dan dokumen akademik menggunakan **Git Submodules**.

---

## Repositori GitHub & Submodules

Repository root ini melacak 4 sub-repositori independen:

| Submodule Path | Remote GitHub Repository | Keterangan |
|:---|:---|:---|
| `documents/` | [`Reza1290/m3-m2-documents`](https://github.com/Reza1290/m3-m2-documents) | Naskah TA, slide presentasi proposal, visualisasi, dan paper acuan |
| `projects/m3-m2-experiments/` | [`Reza1290/m3-m2-experiments`](https://github.com/Reza1290/m3-m2-experiments) | Pipeline riset, Vision Transformer ensemble, KD-Tree geometry, Stage 2 CEJ classifier |
| `projects/m3-m2-cdss-app/` | [`Reza1290/m3-m2-cdss-app`](https://github.com/Reza1290/m3-m2-cdss-app) | Aplikasi Web CDSS interaktif berbasis Streamlit |
| `projects/m3-m2-rebuild/` | [`Reza1290/m3-m2-rebuild`](https://github.com/Reza1290/m3-m2-rebuild) | Independent development pipeline & physical geometry evaluation |

---

## Cara Clone Workspace (Otomatis Men-clone Semua Anak Repo)

### 1. Clone Sekaligus dengan Submodule (Direkomendasikan)
Gunakan flag `--recurse-submodules` agar Git otomatis men-clone parent repo dan **semua sub-repo di dalamnya secara rekursif dalam 1 perintah**:

```bash
git clone --recurse-submodules git@github.com:Reza1290/ta3.git
```

### 2. Jika Terlanjur Melakukan Clone Biasa
Jika Anda atau pengguna lain melakukan `git clone` biasa tanpa flag submodules:
```bash
git clone git@github.com:Reza1290/ta3.git
cd ta3
git submodule update --init --recursive
```
Perintah di atas akan menginisialisasi dan menarik commit terbaru dari semua folder sub-repo.

---

## Arsitektur Modular Data & Weights (Zero-Duplication)

Dataset CBCT (~126.5 GB) dan checkpoint model (~800 MB) ditempatkan di root workspace (`data/` dan `weights/`) dan diabaikan oleh Git (`.gitignore`):

```text
D:\Workspace\TA3/
├── data/                               <-- SHARED DATA LAYER (~126.5 GB - Intact & Untouched)
│   ├── raw/                            (90+ GB CBCT asli NIfTI/DICOM)
│   ├── toothfairy2/                    (36 GB ToothFairy2 dataset)
│   ├── processed/                      (crops_2d 224x224, metadata.csv, split manifest)
│   └── manifests/
│
├── weights/                            <-- SHARED WEIGHTS LAYER (~800 MB model checkpoints)
│   ├── stage2_dl_best.pth
│   ├── swin_small_binary_best.pth
│   ├── swin_s_mha_binary_best.pth
│   ├── vit_base_binary_best.pth
│   └── resnet18_best.pth
│
├── documents/                          <-- SUBMODULE: Reza1290/m3-m2-documents
│
└── projects/
    ├── m3-m2-experiments/              <-- SUBMODULE: Reza1290/m3-m2-experiments
    ├── m3-m2-cdss-app/                 <-- SUBMODULE: Reza1290/m3-m2-cdss-app
    └── m3-m2-rebuild/                  <-- SUBMODULE: Reza1290/m3-m2-rebuild
```

### Setup Link Data & Weights di Lingkungan Baru (Windows)
Untuk panduan detail mengenai pengunduhan dataset mentah, ekstraksi crop 2D, dan pengelolaan data berukuran besar, baca panduan lengkap di:
👉 [**DATA_SETUP.md — Panduan Setup Data Ground-Truth & Volumetrik CBCT**](DATA_SETUP.md)

Jika berpindah perangkat, cukup hubungkan folder dataset fisik ke proyek dengan Windows NTFS Junction:
```powershell
New-Item -ItemType Junction -Path "projects\m3-m2-experiments\data" -Target "D:\Workspace\TA3\data"
New-Item -ItemType Junction -Path "projects\m3-m2-experiments\weights" -Target "D:\Workspace\TA3\weights"
New-Item -ItemType Junction -Path "projects\m3-m2-cdss-app\data" -Target "D:\Workspace\TA3\data"
New-Item -ItemType Junction -Path "projects\m3-m2-cdss-app\weights" -Target "D:\Workspace\TA3\weights"
New-Item -ItemType Junction -Path "projects\m3-m2-cdss-app\src" -Target "D:\Workspace\TA3\projects\m3-m2-experiments\src"
```

---

## Cara Kerja & Workflow Submodules

### Menarik Update dari Sub-repo
```bash
git submodule update --remote --merge
```

### Melakukan Push Perubahan dari Sub-repo
1. Lakukan `commit` dan `push` di dalam folder sub-repo masing-masing (`cd projects/m3-m2-experiments && git push`).
2. Kembali ke root `TA3`, lalu commit perubahan pointer submodule (`git add projects/m3-m2-experiments && git commit -m "chore: update submodule reference"`).
