"""Stage 2 — Level 2 confusion: compare yn1/yn2 and reverse rows/columns."""
from __future__ import annotations
import numpy as np
from common import CipherConfig, generate_sequence


def _one(channel, seq1_head, seq2):
    out = np.array(channel, dtype=np.uint8, copy=True)
    h, w = out.shape
    steps = min(h, w, len(seq1_head), len(seq2))
    actions = []
    for i in range(steps):
        if seq1_head[i] > seq2[i]:
            out[i, :] = out[i, ::-1]
            actions.append("row-left-right")
        else:
            out[:, i] = out[::-1, i]
            actions.append("column-up-down")
    return out, actions


def run(image: np.ndarray, seq1: np.ndarray, config: CipherConfig):
    a = np.asarray(image, dtype=np.uint8)
    h, w = a.shape[:2]
    steps = min(h, w)
    seq2, math_rows = generate_sequence(
        config.y02, config.mu02, config.k, steps,
        transient=config.transient, capture_math=True,
    )
    head = np.asarray(seq1[:steps], dtype=np.float64)

    if a.ndim == 2:
        out, actions = _one(a, head, seq2)
    else:
        channels = []
        actions = None
        for c in range(3):
            ch, act = _one(a[..., c], head, seq2)
            channels.append(ch)
            if actions is None:
                actions = act
        out = np.stack(channels, axis=-1)

    return out.astype(np.uint8), {
        "seq2": seq2,
        "seq2_math": math_rows,
        "seq1_head": head,
        "actions": actions,
        "note": (
            "The paper writes fliplr[i]/flipud[i] but does not fully define the indexed MATLAB operation. "
            "This project reconstructs true -> reverse row i, false -> reverse column i."
        ),
    }
