# Panduan Setup Data Ground-Truth & Volumetrik CBCT

Dokumen ini berisi panduan lengkap tata letak, setup, dan penanganan data ground truth untuk riset:
**Sistem Pendukung Keputusan Klinis Klasifikasi Relasi Gigi Molar Ketiga dan Kedua (M3–M2) pada Citra CBCT**.

---

## 1. Menjawab Keamanan Data Labeled ("Yang Labeled Kalo Dimasukin Aman Gak?")

> [!IMPORTANT]
> **Jawaban Singkat:**
> * **Label Tabular / Manifest (CSV & JSON) $\rightarrow$ 100% AMAN dimasukkan ke Git.**
> * **Citra 2D Crops ($224 \times 224$ PNG) $\rightarrow$ AMAN secara privasi, tetapi disarankan tetap di lokal/cloud.**
> * **Volume 3D CBCT Mentah (DICOM / NIfTI) $\rightarrow$ TIDAK BOLEH masuk ke Git.**

Berikut rincian alasannya dari sisi **Etika Medis (HIPAA / De-identification)** dan **Teknis Git**:

| Kategori Data | Contoh File | Privasi Medis | Masuk Git? | Alasan |
|:---|:---|:---:|:---:|:---|
| **Label & Manifests** | `dataset_4class_manifest.csv`, `patient_splits_seed42.json`, `m2_m3_cej_4class_labels.csv` | **Sangat Aman** | **Boleh & Dianjurkan** | Hanya berisi ID kasus teranonimasi (`ToothFairy2P_025`), nomor kuadran FDI (`17_18`), koordinat CEJ numerik, jarak $D_{\min}$ (mm), dan label target (`Contact`, `Independent`, `Resorption`). **Tidak ada nama pasien, rekam medis (MRN), NIK, atau tanggal lahir**. Ukurannya sangat kecil (< 1 MB). |
| **2D Sagittal Crops** | `data/processed/crops_2d/*.png` ($224 \times 224$) | **Aman** | **Opsional (Sebaiknya Lokal/GDrive)** | Secara privasi 100% aman (hanya potongan interdental 2 gigi molar tanpa kontur wajah pasien). Namun ukurannya mencapai **~150 MB** dengan ratusan file gambar terpisah. Memasukkannya ke Git dapat membuat repo menjadi lambat (*bloated*). File ini dapat di-generate ulang secara deterministik kapan saja dari volume asli. |
| **Volume CBCT 3D Asli** | `data/raw/*.nii.gz`, DICOM, `data/toothfairy2/` | **Sensitif** | **DILARANG KERAS** | Ukuran file sangat masif (**~126.5 GB**), jauh melampaui limit GitHub (maks 100 MB per file, maks 2 GB per repo). Selain itu, raw scan mengandung resolusi kraniofasial penuh. |

---

## 2. Struktur Hirarki Folder Data (`D:\Workspace\TA3\data`)

Seluruh data berukuran besar dipusatkan di root workspace (`D:\Workspace\TA3\data`) sebagai **Single Source of Truth** (~126.5 GB) dan dibagi ke dalam sub-folder berikut:

```text
D:\Workspace\TA3\data/
├── raw/                                (~90.4 GB)  CBCT volume mentah (.nii.gz, DICOM series)
├── toothfairy2/                        (~36.0 GB)  Dataset benchmark resmi ToothFairy2 (nnU-Net format)
│   └── Dataset112_ToothFairy2/
│       ├── imagesTr/                   (Citra volume CBCT training .mha)
│       └── labelsTr/                   (Mask multi-label segmentasi 3D FDI 42 gigi)
│
├── processed/                          (~150 MB)   Data hasil pra-pemrosesan
│   ├── crops_2d/                       (Patch 224x224 sagittal ROI per pasangan M3–M2)
│   ├── m2_m3_cej_4class_labels.csv     (Ground-truth klasifikasi CEJ: Above, At, Below, Independent)
│   ├── toothfairy_binary_labels.csv    (Ground-truth klasifikasi biner: Contact vs Non-Contact)
│   └── toothfairy_classified_labels.csv(Ground-truth 3-kelas: Contact, Independent, Resorption)
│
├── manifests/                          (< 1 MB)    Manifest terstandar & data split
│   ├── dataset_4class_manifest.csv     (Tabel master metadata spasial & label lengkap)
│   └── patient_splits_seed42.json      (Pembagian pasien: Train 70%, Val 15%, Test 15%)
│
└── custom/                             (Anotasi tambahan dari klinisi/expert review)
```

---

## 3. Langkah Setup Data dari Nol (*Step-by-Step*)

Jika Anda baru saja meng-clone repositori ini di perangkat baru, ikuti tahapan berikut:

### Langkah 1: Clone Workspace dengan Submodule
```bash
git clone --recurse-submodules git@github.com:Reza1290/ta3.git
cd ta3
```

### Langkah 2: Buat Folder Data Fisik
Pastikan folder `data` tersedia di root workspace:
```powershell
New-Item -ItemType Directory -Force -Path data\raw, data\toothfairy2, data\processed\crops_2d, data\manifests
```

### Langkah 3: Setup Windows NTFS Junction (Nol Duplikasi Disk)
Agar subproyek di `projects/` dapat mengakses dataset dan model checkpoints secara transparan tanpa meng-copy file 126 GB, jalankan perintah Junction berikut di PowerShell:

```powershell
# Link data ke m3-m2-experiments
New-Item -ItemType Junction -Path "projects\m3-m2-experiments\data" -Target "D:\Workspace\TA3\data"
New-Item -ItemType Junction -Path "projects\m3-m2-experiments\weights" -Target "D:\Workspace\TA3\weights"

# Link data dan weights ke m3-m2-cdss-app
New-Item -ItemType Junction -Path "projects\m3-m2-cdss-app\data" -Target "D:\Workspace\TA3\data"
New-Item -ItemType Junction -Path "projects\m3-m2-cdss-app\weights" -Target "D:\Workspace\TA3\weights"
New-Item -ItemType Junction -Path "projects\m3-m2-cdss-app\src" -Target "D:\Workspace\TA3\projects\m3-m2-experiments\src"
```

### Langkah 4: Download Dataset ToothFairy2 (36 GB)
Gunakan script otomatis yang telah disediakan di `projects/m3-m2-experiments/scripts/download_toothfairy.py` untuk mengunduh dataset resmi langsung dari CDN Hugging Face mirror:

```powershell
cd projects/m3-m2-experiments
python scripts/download_toothfairy.py
```
*Script ini mendukung resumable streaming, pengecekan SHA256 checksum, dan ekstraksi otomatis ke `data/toothfairy2/`.*

### Langkah 5: Ekstrak 2D Sagittal ROI Crops ($224 \times 224$)
Setelah volume ToothFairy2 dan mask segmentasi tersedia di `data/toothfairy2/Dataset112_ToothFairy2/`, jalankan script ekstraksi otomatis berbasis KD-Tree $D_{\min}$:

```powershell
python scripts/cache_all_crops.py
```
*Script ini akan memproses 723 pasangan molar M3–M2, menghitung jarak $D_{\min}$, mencari irisan sagittal optimal, dan menyimpan hasil crop $224 \times 224$ ke `data/processed/crops_2d/` dalam waktu ~2-3 menit.*

### Langkah 6: Verifikasi & Split Dataset
Jalankan script pembuatan data split yang bebas kebocoran (*patient-level split*):
```powershell
python scripts/prepare_cej_splits.py
pytest tests -v
```
Jika 32 unit test lulus (`32 passed`), seluruh data dan pipeline geometri telah terhubung 100% dan siap digunakan untuk eksperimen training maupun demo aplikasi CDSS!

---

## 4. Opsi Hosting & Sinkronisasi Data Tambahan

Bila Anda ingin membagikan atau mem-backup dataset 2D crops dan weights ke cloud:
1. **Model Checkpoints (`weights/*.pth` ~800 MB):**
   - Disimpan di **GitHub Releases** pada repo `m3-m2-experiments` atau Hugging Face Model Hub (`Reza1290/cbct-m3-m2-models`).
2. **2D Processed Crops (`crops_2d.zip` ~150 MB):**
   - Disimpan di Google Drive atau Hugging Face Datasets agar rekan peneliti dapat langsung mendownload tanpa harus memproses 36 GB CBCT mentah.
