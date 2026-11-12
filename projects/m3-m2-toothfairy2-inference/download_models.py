"""Model Weight Downloader for ToothFairy2 / FDI Tooth Segmentation.

Provides automated fetching of:
1. ToothSeg (Zenodo DOI: 10.5281/zenodo.14893540) - Official MICCAI ToothFairy2 benchmark.
2. OralSeg (HuggingFace: aiadir/OralSeg) - SwinUNETR 32-tooth dental CBCT segmentation.
"""

import argparse
import json
import struct
import sys
import time
import urllib.request
import zlib
from pathlib import Path

# Paths
DEFAULT_WEIGHTS_DIR = Path(r"D:\Workspace\TA3\weights")


def download_file_with_progress(url: str, dest_path: Path, chunk_size: int = 1024 * 512):
    """Download a file with progress reporting."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})

    print(f"[*] Downloading {dest_path.name} from {url}...")
    t0 = time.time()
    with urllib.request.urlopen(req) as resp, open(dest_path, "wb") as f_out:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            downloaded += len(chunk)
            f_out.write(chunk)
            if total > 0 and downloaded % (5 * 1024 * 1024) < chunk_size:
                pct = (downloaded / total) * 100
                speed = (downloaded / 1024**2) / max(time.time() - t0, 0.01)
                print(f"    Progress: {downloaded/1024**2:.1f}/{total/1024**2:.1f} MB ({pct:.1f}%) @ {speed:.2f} MB/s", end="\r")

    print(f"\n[+] Download complete: {dest_path} ({dest_path.stat().st_size / 1024**2:.1f} MB in {time.time()-t0:.1f}s)")


def download_oralseg_model(dest_dir: Path = DEFAULT_WEIGHTS_DIR / "oralseg") -> Path:
    """Download OralSeg weights from Hugging Face."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / "model_workstation39.pt"
    if dest_path.exists() and dest_path.stat().st_size > 400 * 1024**2:
        print(f"[+] OralSeg model already exists at: {dest_path}")
        return dest_path

    url = "https://huggingface.co/aiadir/OralSeg/resolve/main/model_workstation39.pt"
    download_file_with_progress(url, dest_path)
    return dest_path


def download_toothseg_dataset121(dest_dir: Path = DEFAULT_WEIGHTS_DIR / "toothseg") -> Path:
    """Download ToothSeg Dataset121 (ToothFairy2 32-tooth FDI model) from Zenodo."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    out_pth = dest_dir / "checkpoint_best.pth"
    out_plans = dest_dir / "plans.json"
    out_dataset = dest_dir / "dataset.json"

    if out_pth.exists() and out_pth.stat().st_size > 200 * 1024**2:
        print(f"[+] ToothSeg model already exists at: {out_pth}")
        return out_pth

    url = "https://zenodo.org/records/14893540/files/ToothSeg.zip?download=1"

    # Extract checkpoint_best.pth directly via HTTP Range
    offset = 9180
    comp_size = 230134511
    print(f"[*] Extracting ToothSeg checkpoint_best.pth directly from Zenodo zip ({comp_size/1024**2:.1f} MB)...")

    req = urllib.request.Request(url, headers={"Range": f"bytes={offset}-{offset+100}"})
    with urllib.request.urlopen(req) as resp:
        hdr = resp.read()
    l_fn_len = struct.unpack("<H", hdr[26:28])[0]
    l_extra_len = struct.unpack("<H", hdr[28:30])[0]
    data_start = offset + 30 + l_fn_len + l_extra_len
    data_end = data_start + comp_size - 1

    t0 = time.time()
    decompressor = zlib.decompressobj(-15)
    downloaded = 0
    req = urllib.request.Request(url, headers={"Range": f"bytes={data_start}-{data_end}"})
    with urllib.request.urlopen(req) as resp, open(out_pth, "wb") as f_out:
        while True:
            chunk = resp.read(1024 * 512)
            if not chunk:
                break
            downloaded += len(chunk)
            f_out.write(decompressor.decompress(chunk))
            if downloaded % (5 * 1024 * 1024) < 1024 * 512:
                pct = (downloaded / comp_size) * 100
                speed = (downloaded / 1024**2) / max(time.time() - t0, 0.01)
                print(f"    Progress: {downloaded/1024**2:.1f}/{comp_size/1024**2:.1f} MB ({pct:.1f}%) @ {speed:.2f} MB/s", end="\r")
        f_out.write(decompressor.flush())

    print(f"\n[+] Extracted ToothSeg checkpoint ({out_pth.stat().st_size / 1024**2:.1f} MB in {time.time()-t0:.1f}s)")
    return out_pth


def main():
    parser = argparse.ArgumentParser(description="Download pre-trained ToothFairy2 / FDI models")
    parser.add_argument("--model", choices=["toothseg", "oralseg", "all"], default="all")
    parser.add_argument("--weights_dir", type=str, default=str(DEFAULT_WEIGHTS_DIR))
    args = parser.parse_args()

    w_dir = Path(args.weights_dir)
    if args.model in ["toothseg", "all"]:
        download_toothseg_dataset121(w_dir / "toothseg")
    if args.model in ["oralseg", "all"]:
        download_oralseg_model(w_dir / "oralseg")


if __name__ == "__main__":
    main()
