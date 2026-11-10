# Volumetric & Ground-Truth Data Directory

Direktori ini adalah **Single Source of Truth** untuk seluruh dataset volumetrik CBCT, anotasi FDI gigi, hasil ekstraksi ROI 2D, dan manifest label klinis.

> [!WARNING]
> Folder ini diabaikan oleh Git (`.gitignore`) karena ukurannya yang masif (~126.5 GB).
> Jangan menduplikasi atau menyalin isi folder ini ke dalam folder proyek. Subproyek di `projects/` mengakses folder ini melalui **Windows NTFS Junction** (`mklink /J` atau `New-Item -ItemType Junction`).

## Struktur Folder:
- `raw/`: File CBCT mentah (.nii.gz / DICOM) (~90.4 GB).
- `toothfairy2/`: Dataset benchmark ToothFairy2 (Dataset112_ToothFairy2) (~36.0 GB).
- `processed/`:
  - `crops_2d/`: Gambar crop 2D sagittal 224x224 (~150 MB).
  - `*.csv`: Label klinis dan koordinat CEJ teranotasi.
- `manifests/`:
  - `dataset_4class_manifest.csv`: Tabel master relasi spasial & klasifikasi target.
  - `patient_splits_seed42.json`: Split pembagian pasien (Train 70%, Val 15%, Test 15%).

Untuk panduan lengkap pengunduhan dan penyiapan data, lihat dokumentasi master di:
👉 [`../DATA_SETUP.md`](../DATA_SETUP.md)
