"""Stage 4 — Level 2 diffusion: high/low nibble cross-quadrant exchange."""
from __future__ import annotations
import numpy as np


def _quadrants(a):
    h, w = a.shape
    if h % 2 or w % 2:
        raise ValueError("Diffusion-2 needs even dimensions.")
    hh, hw = h//2, w//2
    return a[:hh,:hw], a[:hh,hw:], a[hh:,:hw], a[hh:,hw:]


def _one(ch):
    a = np.asarray(ch, dtype=np.uint8)
    msb = np.bitwise_and(a, np.uint8(240))
    lsb = np.bitwise_and(a, np.uint8(15))
    m1,m2,m3,m4 = _quadrants(msb)
    l1,l2,l3,l4 = _quadrants(lsb)
    q1 = np.bitwise_or(m1,l4)
    q2 = np.bitwise_or(m2,l3)
    q3 = np.bitwise_or(m3,l2)
    q4 = np.bitwise_or(m4,l1)
    hh, hw = a.shape[0]//2, a.shape[1]//2
    out = np.empty_like(a)
    out[:hh,:hw] = q1
    out[:hh,hw:] = q2
    out[hh:,:hw] = q3
    out[hh:,hw:] = q4
    return out, msb, lsb


def run(image: np.ndarray):
    a = np.asarray(image, dtype=np.uint8)
    if a.ndim == 2:
        out, msb, lsb = _one(a)
    else:
        results = [_one(a[...,c]) for c in range(3)]
        out = np.stack([r[0] for r in results], axis=-1)
        msb = np.stack([r[1] for r in results], axis=-1)
        lsb = np.stack([r[2] for r in results], axis=-1)
    return out.astype(np.uint8), {"IMSB": msb, "ILSB": lsb}
