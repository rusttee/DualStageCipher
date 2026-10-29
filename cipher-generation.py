"""Final cipher generation / export.

In the paper, the output of Diffusion Level 2 is C, the cipher image.  This file
turns that matrix into useful ciphertext representations: image, raw bytes,
hexadecimal text, and CSV matrices.
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
from common import save_image


def run(diffusion2_output: np.ndarray, output_dir: Path):
    c = np.asarray(diffusion2_output, dtype=np.uint8)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_image(c, output_dir / "06_cipher_image.png")

    raw = c.tobytes(order="C")
    (output_dir / "cipher_bytes.bin").write_bytes(raw)
    (output_dir / "cipher_hex.txt").write_text(raw.hex(), encoding="ascii")

    if c.ndim == 2:
        np.savetxt(output_dir / "cipher_matrix.csv", c, delimiter=",", fmt="%d")
    else:
        for i, name in enumerate(("R", "G", "B")):
            np.savetxt(output_dir / f"cipher_matrix_{name}.csv", c[...,i], delimiter=",", fmt="%d")

    meta = {
        "shape": list(c.shape),
        "dtype": str(c.dtype),
        "cipher_byte_count": len(raw),
        "hex_character_count": len(raw) * 2,
        "note": "For image encryption, C is the ciphertext matrix/image; cipher_hex.txt is a textual hexadecimal representation of the same bytes.",
    }
    (output_dir / "cipher_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return c, meta
