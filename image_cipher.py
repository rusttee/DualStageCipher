"""Reversible image encryption/decryption with full stage and bit tracing."""
from __future__ import annotations
import hashlib
import json
from dataclasses import fields
from pathlib import Path
import numpy as np
from PIL import Image

from common import CipherConfig, load_image, save_image
from cipher_core import encrypt_array_with_stages, decrypt_array_with_stages
from stage_trace import export_stage_comparisons, show_matlab_stage_windows

ENC_ORDER = ["original", "confusion1", "confusion2", "diffusion1", "diffusion2"]
DEC_ORDER = ["cipher", "inverse_diffusion2", "inverse_diffusion1", "inverse_confusion2", "inverse_confusion1"]


def _config_from_dict(d: dict) -> CipherConfig:
    allowed = {f.name for f in fields(CipherConfig)}
    return CipherConfig(**{k: v for k, v in d.items() if k in allowed})


def _sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(a, dtype=np.uint8).tobytes(order="C")).hexdigest()


def _save_stages(stages: dict[str, np.ndarray], out: Path, order: list[str]) -> None:
    for i, key in enumerate(order, 1):
        save_image(stages[key], out / f"{i:02d}_{key}.png")


def encrypt_image(src: Path, out_dir: Path, config: CipherConfig, show_steps: bool = True) -> Path:
    image = load_image(src, config)
    cipher, stages, trace = encrypt_array_with_stages(image, config)
    out_dir.mkdir(parents=True, exist_ok=True)
    _save_stages(stages, out_dir, ENC_ORDER)
    reports = export_stage_comparisons(stages, ENC_ORDER, out_dir / "bit_value_comparisons")
    show_matlab_stage_windows(stages, ENC_ORDER, out_dir / "web_dashboard",
                              "IMAGE ENCRYPTION — forward stages", show=False, config=config, trace=trace)

    meta = {
        "format": "nature-cipher-image-v2",
        "source_name": src.name,
        "source_path_at_encryption": str(src.resolve()),
        "original_shape": list(image.shape),
        "cipher_shape": list(cipher.shape),
        "config": config.to_dict(),
        "plain_sha256": _sha(image),
        "cipher_sha256": _sha(cipher),
        "stage_bit_reports": reports,
    }
    package = out_dir / "encrypted_image.npz"
    np.savez_compressed(package, cipher=cipher, metadata=np.array(json.dumps(meta)))
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("\nIMAGE ENCRYPTION COMPLETE")
    print("Cipher package :", package)
    print("Cipher image   :", out_dir / "05_diffusion2.png")
    print("Full bit trace :", out_dir / "bit_value_comparisons")
    return package


def decrypt_image(src_package: Path, out_dir: Path, show_steps: bool = True) -> bool:
    with np.load(src_package, allow_pickle=False) as z:
        cipher = np.asarray(z["cipher"], dtype=np.uint8)
        meta = json.loads(str(z["metadata"].item()))
    if meta.get("format") != "nature-cipher-image-v2":
        raise ValueError("Unsupported image cipher package.")
    config = _config_from_dict(meta["config"])
    oh, ow = [int(x) for x in meta["original_shape"][:2]]
    plain, stages = decrypt_array_with_stages(cipher, config, (oh, ow))
    out_dir.mkdir(parents=True, exist_ok=True)
    _save_stages(stages, out_dir, DEC_ORDER)
    save_image(plain, out_dir / "06_recovered_image.png")
    reports = export_stage_comparisons(stages, DEC_ORDER, out_dir / "bit_value_comparisons")
    show_matlab_stage_windows(stages, DEC_ORDER, out_dir / "web_dashboard",
                              "IMAGE DECRYPTION — inverse stages", show=False, config=config)

    actual = _sha(plain); expected = meta.get("plain_sha256")
    exact = actual == expected
    verification = {
        "exact_recovery": exact,
        "expected_plain_sha256": expected,
        "recovered_plain_sha256": actual,
        "recovered_shape": list(plain.shape),
        "stage_bit_reports": reports,
    }
    original_path = Path(meta.get("source_path_at_encryption", ""))
    if original_path.exists():
        try:
            original = load_image(original_path, config)
            if original.shape == plain.shape:
                mismatch = original != plain
                verification["mismatched_byte_components"] = int(np.count_nonzero(mismatch))
                verification["matched_byte_components"] = int(original.size - np.count_nonzero(mismatch))
                verification["total_byte_components"] = int(original.size)
        except Exception:
            pass
    (out_dir / "decryption_verification.json").write_text(json.dumps(verification, indent=2), encoding="utf-8")
    print("\nIMAGE DECRYPTION COMPLETE")
    print("Recovered image :", out_dir / "06_recovered_image.png")
    print("Exact recovery  :", "YES" if exact else "NO")
    if "mismatched_byte_components" in verification:
        print("Mismatched bytes:", verification["mismatched_byte_components"], "/", verification["total_byte_components"])
    print("Full bit trace  :", out_dir / "bit_value_comparisons")
    return exact
