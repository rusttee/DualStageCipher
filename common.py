from __future__ import annotations

import math
import urllib.parse
import urllib.request
import json
import shutil
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
PAPER_IMAGES = DATA / "paper_images"
MEDICAL = DATA / "medical"
OUTPUT = ROOT / "output"
OFFLINE_FALLBACK = DATA / "offline_fallback"
ASSET_MANIFEST = DATA / "assets_manifest.json"

PAPER_IMAGES.mkdir(parents=True, exist_ok=True)
MEDICAL.mkdir(parents=True, exist_ok=True)
OUTPUT.mkdir(parents=True, exist_ok=True)
OFFLINE_FALLBACK.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class CipherConfig:
    # Paper parameters
    y01: float = 0.55427
    mu01: float = 20.389429
    y02: float = 0.54324
    mu02: float = 20.385437
    y03: float = 0.56543
    mu03: float = 20.389429
    k: float = 20.0

    # The paper says to discard 100 values.  For study, the user can disable it.
    discard_first_100: bool = False

    # Eq.(5) is under-specified in the paper.  This is the existing study-project
    # reconstruction and is kept explicit instead of being presented as author-confirmed.
    quantization_mode: str = "abs_scale"
    quantization_scale: float = 1e15

    # For color source images the official USC-SIPI files are 512x512, while the
    # paper tables label color tests as 256x256.  The paper does not publish the
    # exact resize/crop rule.  "official" keeps 512x512; "paper256" resizes to 256.
    color_preprocess: str = "official"

    @property
    def transient(self) -> int:
        return 100 if self.discard_first_100 else 0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["transient"] = self.transient
        return d


# Exact publicly identifiable USC-SIPI images named in the paper / attached project.
SIPI_CATALOG = {
    "5.1.09": {"name": "Moon", "mode": "L", "size": (256, 256)},
    "5.1.10": {"name": "Aerial", "mode": "L", "size": (256, 256)},
    "5.1.11": {"name": "Airplane (gray)", "mode": "L", "size": (256, 256)},
    "5.1.12": {"name": "Clock", "mode": "L", "size": (256, 256)},
    "5.1.13": {"name": "Ruler / resolution chart", "mode": "L", "size": (256, 256)},
    "5.1.14": {"name": "Plant", "mode": "L", "size": (256, 256)},
    "4.2.03": {"name": "Mandril", "mode": "RGB", "size": (512, 512)},
    "4.2.07": {"name": "Peppers", "mode": "RGB", "size": (512, 512)},
    "4.2.06": {"name": "Lake / sailboat", "mode": "RGB", "size": (512, 512)},
    "4.2.05": {"name": "Airplane (color)", "mode": "RGB", "size": (512, 512)},
}


def matlab_mod(x: float, m: float) -> float:
    """MATLAB-compatible mod(x,m) for positive m."""
    return x - m * math.floor(x / m)


def cosine_map_next(y: float, mu: float, k: float) -> float:
    """Paper Eq.(3), evaluated as MATLAB cos(): radians, double precision."""
    if y == 0.0:
        raise ZeroDivisionError("Eq.(3) reached y=0; expression contains mu/y.")
    inner = mu * (1.0 - mu / y) * (1.0 - mu * mu)
    frac = matlab_mod(inner, 1.0)
    return math.cos(2.0 ** (k + frac))


def generate_sequence(y0: float, mu: float, k: float, count: int, transient: int = 0,
                      capture_math: bool = False):
    total = transient + count
    seq = np.empty(total, dtype=np.float64)
    rows = [] if capture_math else None
    y = float(y0)
    for i in range(total):
        y_in = y
        if y_in == 0.0:
            raise ZeroDivisionError("Eq.(3) reached y=0.")
        mu_div_y = mu / y_in
        one_minus = 1.0 - mu_div_y
        mu_sq = mu * mu
        one_minus_mu_sq = 1.0 - mu_sq
        inner = mu * one_minus * one_minus_mu_sq
        frac = matlab_mod(inner, 1.0)
        exponent = k + frac
        power = 2.0 ** exponent
        y = math.cos(power)
        seq[i] = y
        if capture_math:
            rows.append((i + 1, y_in, mu_div_y, one_minus, mu_sq,
                         one_minus_mu_sq, inner, frac, exponent, power, y))
    usable = seq[transient:].copy()
    if capture_math:
        return usable, np.asarray(rows, dtype=np.float64)
    return usable


def quantize_sequence(seq: np.ndarray, mode: str = "abs_scale", scale: float = 1e15) -> np.ndarray:
    """Explicit reconstruction of the paper's missing float->byte conversion."""
    seq = np.asarray(seq, dtype=np.float64)
    if mode == "abs_scale":
        q = np.floor(np.abs(seq) * float(scale))
    elif mode == "scale":
        q = np.floor(seq * float(scale))
    elif mode == "frac":
        q = np.floor(np.mod(seq, 1.0) * 256.0)
    elif mode == "literal":
        raise ValueError(
            "Eq.(5) as printed leaves floating values. Select a reconstruction "
            "quantization mode before the bitwise stage."
        )
    else:
        raise ValueError(f"Unknown quantization mode: {mode}")
    return np.mod(q.astype(np.int64), 256).astype(np.uint8)


def ensure_even_image(a: np.ndarray) -> np.ndarray:
    h, w = a.shape[:2]
    if h % 2 or w % 2:
        raise ValueError(
            f"Image size {w}x{h} has an odd dimension. Diffusion-2 divides the "
            "image into four equal quadrants, so both dimensions must be even."
        )
    return a


def _validate_sipi(path: Path, image_id: str) -> None:
    spec = SIPI_CATALOG[image_id]
    with Image.open(path) as im:
        im.load()
        if im.mode != spec["mode"] or im.size != spec["size"]:
            raise ValueError(
                f"{image_id}: expected {spec['mode']} {spec['size']}, got {im.mode} {im.size}"
            )


def asset_manifest() -> dict:
    try:
        return json.loads(ASSET_MANIFEST.read_text(encoding="utf-8"))
    except Exception:
        return {}


def exact_sipi_path(image_id: str) -> Path:
    return PAPER_IMAGES / f"{image_id}.tiff"


def fallback_sipi_path(image_id: str) -> Path:
    # The package ships a presentation fallback for every paper image except
    # that 5.1.14 is intentionally a labeled placeholder until first-run setup.
    p1 = PAPER_IMAGES / f"{image_id}.png"
    if p1.exists():
        return p1
    return OFFLINE_FALLBACK / f"{image_id}.png"


def asset_quality(image_id: str, path: Path | None = None) -> str:
    if path is None:
        path = exact_sipi_path(image_id) if exact_sipi_path(image_id).exists() else fallback_sipi_path(image_id)
    if path.suffix.lower() in (".tif", ".tiff"):
        return "exact_cached_sipi"
    info = asset_manifest().get("standard_images", {}).get(image_id, {})
    return info.get("bundled", "bundled_fallback")


def download_sipi_exact(image_id: str, timeout: int = 60) -> Path:
    """Download and cache the official USC-SIPI TIFF. Used by FIRST_RUN_SETUP."""
    if image_id not in SIPI_CATALOG:
        raise KeyError(image_id)
    dest = exact_sipi_path(image_id)
    if dest.exists():
        try:
            _validate_sipi(dest, image_id)
            return dest
        except Exception:
            dest.unlink(missing_ok=True)

    params = urllib.parse.urlencode({"img": image_id, "vol": "misc"})
    urls = [
        "https://sipi.usc.edu/database/download.php?" + params,
        "https://sipi.usc.edu/database/download.php?vol=misc&img=" + urllib.parse.quote(image_id),
    ]
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124 Safari/537.36",
        "Referer": "https://sipi.usc.edu/database/database.php?volume=misc",
        "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
    }
    errors=[]
    for url in urls:
        try:
            req=urllib.request.Request(url,headers=headers)
            with urllib.request.urlopen(req,timeout=timeout) as r:
                data=r.read()
            dest.write_bytes(data)
            _validate_sipi(dest,image_id)
            return dest
        except Exception as exc:
            errors.append(f"{url}: {exc}")
            dest.unlink(missing_ok=True)

    # Secondary public mirrors are only used for grayscale images whose content
    # is known to correspond to the same USC-SIPI ID. Color images are NOT taken
    # from the C4L mirror because that repository converts them to grayscale.
    mirrors = {
        "5.1.09":"https://raw.githubusercontent.com/coderforlife/c4l-image-dataset/master/256x256/moon.png",
        "5.1.12":"https://raw.githubusercontent.com/coderforlife/c4l-image-dataset/master/256x256/clock.png",
        "5.1.14":"https://raw.githubusercontent.com/coderforlife/c4l-image-dataset/master/256x256/chemical.png",
    }
    if image_id in mirrors:
        png=PAPER_IMAGES/f"{image_id}.exact.png"
        try:
            req=urllib.request.Request(mirrors[image_id],headers=headers)
            with urllib.request.urlopen(req,timeout=timeout) as r:
                png.write_bytes(r.read())
            with Image.open(png) as im:
                im.load()
                if im.size != SIPI_CATALOG[image_id]["size"]:
                    raise ValueError(f"mirror size {im.size}")
            return png
        except Exception as exc:
            errors.append(f"mirror: {exc}")
            png.unlink(missing_ok=True)
    raise RuntimeError("Could not cache exact USC-SIPI image.\n" + "\n".join(errors))


def download_sipi(image_id: str) -> Path:
    """OFFLINE-FIRST image resolver used by normal encryption/simulation runs.

    Order:
      1) exact TIFF cached by FIRST_RUN_SETUP
      2) exact mirror PNG cached by FIRST_RUN_SETUP
      3) bundled paper-figure fallback PNG

    Normal presentation runs never require the Internet.
    """
    if image_id not in SIPI_CATALOG:
        raise KeyError(image_id)
    exact=exact_sipi_path(image_id)
    if exact.exists():
        try:
            _validate_sipi(exact,image_id)
            return exact
        except Exception:
            pass
    mirror=PAPER_IMAGES/f"{image_id}.exact.png"
    if mirror.exists():
        return mirror
    fallback=fallback_sipi_path(image_id)
    if fallback.exists():
        return fallback
    raise RuntimeError(
        f"No cached or bundled asset for {image_id}. Run FIRST_RUN_SETUP.bat while online."
    )

def load_image(path: Path, config: CipherConfig) -> np.ndarray:
    with Image.open(path) as im:
        im.load()
        if im.mode not in ("L", "RGB"):
            im = im.convert("RGB")
        if im.mode == "RGB" and config.color_preprocess == "paper256" and im.size != (256, 256):
            # The paper does not disclose which 512->256 preprocessing rule it used.
            # LANCZOS is therefore explicitly a study choice, not claimed author behavior.
            im = im.resize((256, 256), Image.Resampling.LANCZOS)
        a = np.asarray(im, dtype=np.uint8).copy()
    return ensure_even_image(a)


def build_black_white(which: str) -> tuple[np.ndarray, str]:
    if which == "black":
        return np.zeros((256, 256), dtype=np.uint8), "black_256x256"
    if which == "white":
        return np.full((256, 256), 255, dtype=np.uint8), "white_256x256"
    raise ValueError(which)


def save_image(a: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(a, dtype=np.uint8)).save(path)


def image_change_stats(before: np.ndarray, after: np.ndarray) -> dict[str, float | int]:
    a = np.asarray(before)
    b = np.asarray(after)
    if a.shape != b.shape:
        return {"shape_changed": 1, "before_shape": list(a.shape), "after_shape": list(b.shape)}
    if a.ndim == 2:
        changed_mask = a != b
        changed_components = int(np.count_nonzero(changed_mask))
        changed_pixels = changed_components
        total_pixels = a.size
        total_components = a.size
    else:
        comp = a != b
        changed_components = int(np.count_nonzero(comp))
        changed_mask = np.any(comp[..., :3], axis=2)
        changed_pixels = int(np.count_nonzero(changed_mask))
        total_pixels = a.shape[0] * a.shape[1]
        total_components = a[..., :3].size
    return {
        "shape_changed": 0,
        "changed_pixels": changed_pixels,
        "total_pixels": int(total_pixels),
        "changed_pixels_percent": changed_pixels / total_pixels * 100.0,
        "changed_components": changed_components,
        "total_components": int(total_components),
        "changed_components_percent": changed_components / total_components * 100.0,
    }


def difference_image(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    a = np.asarray(before, dtype=np.int16)
    b = np.asarray(after, dtype=np.int16)
    if a.shape != b.shape:
        # Diffusion-1 transposes rectangular images; resize is intentionally avoided.
        return np.zeros((64, 64), dtype=np.uint8)
    d = np.abs(a - b)
    if d.ndim == 3:
        d = np.max(d[..., :3], axis=2)
    return np.clip(d, 0, 255).astype(np.uint8)


def config_summary(config: CipherConfig) -> str:
    return (
        f"Eq.(3): radians / Python double (MATLAB cos-compatible)\n"
        f"Transient handling: {'discard first 100' if config.discard_first_100 else 'NO discard (study mode)'}\n"
        f"Diffusion Eq.(5) reconstruction: {config.quantization_mode}, scale={config.quantization_scale:g}\n"
        f"Color preprocessing: {config.color_preprocess}"
    )
