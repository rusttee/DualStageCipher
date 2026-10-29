"""Stage 1 — Level 1 confusion: chaotic sorting + pixel permutation."""
from __future__ import annotations
import numpy as np
from common import CipherConfig, generate_sequence


def run(image: np.ndarray, config: CipherConfig):
    a = np.asarray(image, dtype=np.uint8)
    h, w = a.shape[:2]
    seq1, math_rows = generate_sequence(
        config.y01, config.mu01, config.k, h*w,
        transient=config.transient, capture_math=True,
    )
    idx = np.argsort(seq1, kind="stable")

    def one(channel):
        flat = np.asarray(channel, dtype=np.uint8).ravel(order="F")
        return flat[idx].reshape((h, w), order="F")

    if a.ndim == 2:
        out = one(a)
    else:
        out = np.stack([one(a[..., c]) for c in range(3)], axis=-1)

    return out.astype(np.uint8), {
        "seq1": seq1,
        "seq1_math": math_rows,
        "perm_idx_python": idx,
        "perm_idx_matlab": idx + 1,
        "sorted_seq1": seq1[idx],
    }
