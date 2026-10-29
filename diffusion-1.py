"""Stage 3 — Level 1 diffusion: quantized yn3 matrix + mod 256 + transpose."""
from __future__ import annotations
import numpy as np
from common import CipherConfig, generate_sequence, quantize_sequence


def run(image: np.ndarray, config: CipherConfig):
    a = np.asarray(image, dtype=np.uint8)
    h, w = a.shape[:2]
    seq3, math_rows = generate_sequence(
        config.y03, config.mu03, config.k, h*w,
        transient=config.transient, capture_math=True,
    )
    q3 = quantize_sequence(seq3, config.quantization_mode, config.quantization_scale)
    R = q3.reshape((h, w), order="F")

    def one(ch):
        s = (ch.astype(np.uint16) + R.astype(np.uint16)) % 256
        return s.astype(np.uint8).T.copy()

    if a.ndim == 2:
        out = one(a)
    else:
        out = np.stack([one(a[..., c]) for c in range(3)], axis=-1)

    return out.astype(np.uint8), {
        "seq3": seq3,
        "seq3_math": math_rows,
        "q3_bytes": q3,
        "R": R,
        "note": (
            "Eq.(5) in the paper does not publish the float-to-byte conversion needed before bitwise diffusion-2. "
            f"This run uses {config.quantization_mode} with scale {config.quantization_scale:g}."
        ),
    }
