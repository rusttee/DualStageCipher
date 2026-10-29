"""First-run medical dataset bootstrap for the paper-based extensions.

First run only:
  * downloads full-length public echocardiography sample videos;
  * downloads one real 3-D T1-weighted brain MRI from OpenNeuro/AOMIC;
  * prepares a documented uint8 brain volume because the paper cipher operates on bytes.

After OFFLINE_READY.json is written, the normal menu never contacts the network.
"""
from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from common import ROOT

DATA_ROOT = ROOT / "data" / "extended_datasets"
VIDEO_DIR = DATA_ROOT / "medical_video"
VOLUME_DIR = DATA_ROOT / "brain_mri_3d"
RAW_DIR = VOLUME_DIR / "raw"
PREPARED_DIR = VOLUME_DIR / "prepared"
MANIFEST = DATA_ROOT / "offline_dataset_manifest.json"
READY = DATA_ROOT / "OFFLINE_READY.json"
FORMAT = "nature-cipher-medical-datasets-v3"

# Full sample echocardiography clips published by the University of Tokyo
# Hospital / University of Tokyo Video-CLIP repository.  We use the complete
# downloaded clip; no 4-second truncation or frame subsampling is performed.
ECHO_VIDEOS = {
    "echo_example1.mp4": {
        "urls": [
            "https://raw.githubusercontent.com/UTcardiology/video-echo-clip/main/example1.mp4",
            "https://github.com/UTcardiology/video-echo-clip/raw/refs/heads/main/example1.mp4",
        ],
        "description": "Sample multi-view echocardiography video (example1), full original clip",
        "source_page": "https://github.com/UTcardiology/video-echo-clip",
    },
    "echo_example2.mp4": {
        "urls": [
            "https://raw.githubusercontent.com/UTcardiology/video-echo-clip/main/example2.mp4",
            "https://github.com/UTcardiology/video-echo-clip/raw/refs/heads/main/example2.mp4",
        ],
        "description": "Sample multi-view echocardiography video (example2), full original clip",
        "source_page": "https://github.com/UTcardiology/video-echo-clip",
    },
}

# A real T1-weighted brain MRI from the Amsterdam Open MRI Collection (AOMIC),
# hosted publicly by OpenNeuro.  The URL form is documented by OpenNeuro/AOMIC
# download examples and downloads only this one anatomical NIfTI file.
BRAIN = {
    "name": "aomic_sub0001_t1w",
    "url": "https://s3.amazonaws.com/openneuro.org/ds003097/sub-0001/anat/sub-0001_run-1_T1w.nii.gz",
    "source_page": "https://openneuro.org/datasets/ds003097",
    "description": "AOMIC ID1000 subject 0001 T1-weighted 3-D structural brain MRI",
    "license_note": "OpenNeuro published datasets are distributed under CC0; retain dataset citation/provenance.",
}


def _hash(path: Path, algo: str = "sha256") -> str:
    h = hashlib.new(algo)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(urls: list[str], dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    errors = []
    for url in urls:
        try:
            print(f"Downloading: {url}")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 NatureCipherMedicalDataset/3.0"})
            with urllib.request.urlopen(req, timeout=120) as r, tmp.open("wb") as f:
                total = int(r.headers.get("Content-Length", "0") or 0)
                done = 0
                while True:
                    block = r.read(1024 * 1024)
                    if not block:
                        break
                    f.write(block); done += len(block)
                    if total:
                        print(f"  {done/1024/1024:7.1f}/{total/1024/1024:7.1f} MiB ({100*done/total:5.1f}%)", end="\r")
                    else:
                        print(f"  {done/1024/1024:7.1f} MiB", end="\r")
            print()
            tmp.replace(dst)
            return
        except Exception as exc:
            errors.append(f"{url}: {exc}")
            tmp.unlink(missing_ok=True)
    raise RuntimeError("Dataset download failed. Internet is required only for first-run preparation.\n" + "\n".join(errors))


def _video_info(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Downloaded medical video cannot be decoded by OpenCV: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if frames <= 0 or width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid video metadata for {path}")
    return {
        "frames": frames,
        "fps": fps,
        "duration_seconds": (frames / fps) if fps > 0 else None,
        "width": width,
        "height": height,
    }


def _ensure_medical_videos() -> list[dict]:
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    for filename, spec in ECHO_VIDEOS.items():
        dst = VIDEO_DIR / filename
        if not dst.exists() or dst.stat().st_size < 1024:
            _download(spec["urls"], dst)
        info = _video_info(dst)
        records.append({
            "name": filename,
            "local": str(dst.relative_to(ROOT)),
            "sha256": _hash(dst),
            "source_page": spec["source_page"],
            "description": spec["description"],
            "processing_policy": "Use ALL decoded frames at the source FPS; no short demo derivative is created.",
            **info,
        })
    return records


def _prepare_brain(raw_path: Path) -> tuple[Path, dict]:
    try:
        import nibabel as nib
    except ImportError as exc:
        raise RuntimeError("nibabel is required for the brain MRI dataset. Run: pip install -r requirements.txt") from exc

    PREPARED_DIR.mkdir(parents=True, exist_ok=True)
    out = PREPARED_DIR / "aomic_sub0001_t1w_uint8.npy"
    meta_path = PREPARED_DIR / "aomic_sub0001_t1w_preprocessing.json"

    if out.exists() and meta_path.exists():
        return out, json.loads(meta_path.read_text(encoding="utf-8"))

    img = nib.load(str(raw_path))
    data = np.asanyarray(img.dataobj).astype(np.float32)
    if data.ndim != 3:
        raise ValueError(f"Expected a 3-D T1w brain MRI, got shape {data.shape}")

    finite = data[np.isfinite(data)]
    if finite.size == 0:
        raise ValueError("Brain MRI contains no finite voxel values.")
    # Robust, deterministic MRI->uint8 adapter.  This is preprocessing only;
    # the encryption stages themselves remain the paper's uint8 algorithm.
    lo, hi = np.percentile(finite, [0.5, 99.5])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(finite.min()), float(finite.max())
    scaled = np.clip((data - lo) / max(hi - lo, 1e-12), 0.0, 1.0)
    u8 = np.rint(scaled * 255.0).astype(np.uint8)

    # NIfTI convention here is (X,Y,Z); cipher code expects (D,H,W), so Z is
    # moved to the leading axis. Pad only when required by the paper's 4-quadrant
    # diffusion step, which needs even H/W.
    volume = np.moveaxis(u8, 2, 0)
    pad_h = volume.shape[1] % 2
    pad_w = volume.shape[2] % 2
    if pad_h or pad_w:
        volume = np.pad(volume, ((0, 0), (0, pad_h), (0, pad_w)), mode="constant")

    np.save(out, volume, allow_pickle=False)
    meta = {
        "source_nifti": str(raw_path.relative_to(ROOT)),
        "source_shape_xyz": list(data.shape),
        "prepared_shape_dhw": list(volume.shape),
        "dtype": "uint8",
        "normalization": "global 0.5th-99.5th percentile clip, linear scale to 0..255, round to uint8",
        "axis_mapping": "NIfTI (X,Y,Z) -> cipher volume (Z,X,Y)",
        "even_dimension_padding": {"bottom_rows": int(pad_h), "right_columns": int(pad_w)},
        "note": "Only this documented adapter is new; every 2-D slice then uses the paper's exact four stage structure.",
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out, meta


def _ensure_brain() -> dict:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    raw = RAW_DIR / "sub-0001_run-1_T1w.nii.gz"
    if not raw.exists() or raw.stat().st_size < 1024:
        _download([BRAIN["url"]], raw)
    prepared, prep_meta = _prepare_brain(raw)
    return {
        "name": BRAIN["name"],
        "description": BRAIN["description"],
        "source": BRAIN["source_page"],
        "download_url": BRAIN["url"],
        "license_note": BRAIN["license_note"],
        "raw_local": str(raw.relative_to(ROOT)),
        "raw_sha256": _hash(raw),
        "prepared_local": str(prepared.relative_to(ROOT)),
        "prepared_sha256": _hash(prepared),
        "preprocessing": prep_meta,
    }


def all_local_files_present() -> bool:
    try:
        if not READY.exists() or not MANIFEST.exists():
            return False
        state = json.loads(READY.read_text(encoding="utf-8"))
        if state.get("format") != FORMAT:
            return False
        for rel in state.get("required_local_files", []):
            if not (ROOT / rel).exists():
                return False
        return True
    except Exception:
        return False


def ensure_offline_datasets() -> dict:
    """Download/prepare once. All later menu operations are filesystem-only."""
    if all_local_files_present():
        print("Medical dataset cache: READY (fully offline; no network access needed).")
        return json.loads(MANIFEST.read_text(encoding="utf-8"))

    print("=" * 78)
    print("FIRST-RUN MEDICAL DATASET PREPARATION")
    print("Downloading full echocardiography clips + a real 3-D brain MRI.")
    print("After this finishes, all encryption/decryption runs use local files only.")
    print("=" * 78)
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    videos = _ensure_medical_videos()
    brain = _ensure_brain()
    manifest = {
        "format": FORMAT,
        "video": videos,
        "brain_volume": brain,
        "network_policy": "Network is used only if a required cache file is absent. No network access occurs after the ready marker validates.",
        "paper_extension_policy": (
            "Every video frame and every 3-D brain slice uses the same published 2-level confusion + 2-level diffusion structure. "
            "Media decoding, slice sequencing and the documented MRI uint8 adapter are extension wrappers, not replacement ciphers."
        ),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    required = [v["local"] for v in videos] + [brain["raw_local"], brain["prepared_local"]]
    READY.write_text(json.dumps({"format": FORMAT, "ready": True, "required_local_files": required}, indent=2), encoding="utf-8")
    print("Medical dataset preparation complete. Future runs are fully offline.")
    return manifest


def medical_video_paths() -> list[Path]:
    return [VIDEO_DIR / name for name in ECHO_VIDEOS]


def default_video_path() -> Path:
    return medical_video_paths()[0]


def full_video_path() -> Path:
    # Compatibility with older launcher name. This is already the full medical clip.
    return default_video_path()


def prepared_volume_paths() -> list[Path]:
    p = PREPARED_DIR / "aomic_sub0001_t1w_uint8.npy"
    return [p] if p.exists() else []


if __name__ == "__main__":
    ensure_offline_datasets()
