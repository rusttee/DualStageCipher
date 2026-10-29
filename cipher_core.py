"""Reusable reversible core for the paper-reproduction cipher.

The original project only implemented the forward image-encryption direction.
This module keeps the same four forward stages and adds their exact inverse
operations so the same cipher can be reused for images, video frames and 3-D
volume slices.

Important: this reproduces the *project's reconstruction* of the paper,
including the published ambiguities documented in README.md/common.py.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np

from common import CipherConfig, ROOT, generate_sequence, quantize_sequence, ensure_even_image


def _load_hyphen_module(filename: str, module_name: str):
    path = ROOT / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(module_name, module)
    spec.loader.exec_module(module)
    return module


C1 = _load_hyphen_module("confusion-1.py", "core_confusion_1")
C2 = _load_hyphen_module("confusion-2.py", "core_confusion_2")
D1 = _load_hyphen_module("diffusion-1.py", "core_diffusion_1")
D2 = _load_hyphen_module("diffusion-2.py", "core_diffusion_2")


def encrypt_array(image: np.ndarray, config: CipherConfig) -> tuple[np.ndarray, dict[str, Any]]:
    """Run the unchanged four forward stages on one uint8 image/frame/slice."""
    a = ensure_even_image(np.asarray(image, dtype=np.uint8))
    i1, t1 = C1.run(a, config)
    i2, t2 = C2.run(i1, t1["seq1"], config)
    i3, t3 = D1.run(i2, config)
    i4, t4 = D2.run(i3)
    return i4.astype(np.uint8), {
        "original_shape": list(a.shape),
        "cipher_shape": list(i4.shape),
        "trace": {"confusion1": t1, "confusion2": t2, "diffusion1": t3, "diffusion2": t4},
    }


def _inverse_diffusion2(image: np.ndarray) -> np.ndarray:
    # The nibble map is an involution: Q1<->Q4 and Q2<->Q3 low nibbles.
    return D2.run(np.asarray(image, dtype=np.uint8))[0]


def _inverse_diffusion1(image: np.ndarray, config: CipherConfig, original_hw: tuple[int, int]) -> np.ndarray:
    h, w = map(int, original_hw)
    a = np.asarray(image, dtype=np.uint8)
    expected = (w, h)
    if a.shape[:2] != expected:
        raise ValueError(
            f"Cipher shape {a.shape[:2]} is incompatible with original HxW={h}x{w}; "
            f"expected encrypted HxW={expected[0]}x{expected[1]}."
        )

    seq3 = generate_sequence(
        config.y03, config.mu03, config.k, h * w,
        transient=config.transient, capture_math=False,
    )
    q3 = quantize_sequence(seq3, config.quantization_mode, config.quantization_scale)
    R = q3.reshape((h, w), order="F")

    def one(ch: np.ndarray) -> np.ndarray:
        pre_transpose = np.asarray(ch, dtype=np.uint8).T
        return ((pre_transpose.astype(np.int16) - R.astype(np.int16)) % 256).astype(np.uint8)

    if a.ndim == 2:
        return one(a)
    if a.ndim == 3 and a.shape[2] == 3:
        return np.stack([one(a[..., c]) for c in range(3)], axis=-1)
    raise ValueError(f"Expected grayscale or RGB-like uint8 array, got shape {a.shape}.")


def _inverse_confusion2(image: np.ndarray, config: CipherConfig) -> np.ndarray:
    a = np.asarray(image, dtype=np.uint8)
    h, w = a.shape[:2]
    steps = min(h, w)
    seq1 = generate_sequence(
        config.y01, config.mu01, config.k, h * w,
        transient=config.transient, capture_math=False,
    )
    seq2 = generate_sequence(
        config.y02, config.mu02, config.k, steps,
        transient=config.transient, capture_math=False,
    )
    row_action = np.asarray(seq1[:steps]) > np.asarray(seq2)

    def one(ch: np.ndarray) -> np.ndarray:
        out = np.array(ch, dtype=np.uint8, copy=True)
        # Forward stage applies operations i=0..steps-1.  Each operation is its
        # own inverse, so exact inversion applies the same operations backwards.
        for i in range(steps - 1, -1, -1):
            if row_action[i]:
                out[i, :] = out[i, ::-1]
            else:
                out[:, i] = out[::-1, i]
        return out

    if a.ndim == 2:
        return one(a)
    return np.stack([one(a[..., c]) for c in range(3)], axis=-1)


def _inverse_confusion1(image: np.ndarray, config: CipherConfig) -> np.ndarray:
    a = np.asarray(image, dtype=np.uint8)
    h, w = a.shape[:2]
    seq1 = generate_sequence(
        config.y01, config.mu01, config.k, h * w,
        transient=config.transient, capture_math=False,
    )
    idx = np.argsort(seq1, kind="stable")

    def one(ch: np.ndarray) -> np.ndarray:
        enc_flat = np.asarray(ch, dtype=np.uint8).ravel(order="F")
        plain_flat = np.empty_like(enc_flat)
        plain_flat[idx] = enc_flat
        return plain_flat.reshape((h, w), order="F")

    if a.ndim == 2:
        return one(a)
    return np.stack([one(a[..., c]) for c in range(3)], axis=-1)


def decrypt_array(cipher: np.ndarray, config: CipherConfig, original_hw: tuple[int, int]) -> np.ndarray:
    """Invert diffusion-2 -> diffusion-1 -> confusion-2 -> confusion-1."""
    c = ensure_even_image(np.asarray(cipher, dtype=np.uint8))
    i3 = _inverse_diffusion2(c)
    i2 = _inverse_diffusion1(i3, config, original_hw)
    i1 = _inverse_confusion2(i2, config)
    i0 = _inverse_confusion1(i1, config)
    return i0.astype(np.uint8)


def roundtrip_ok(image: np.ndarray, config: CipherConfig) -> bool:
    a = ensure_even_image(np.asarray(image, dtype=np.uint8))
    c, _ = encrypt_array(a, config)
    d = decrypt_array(c, config, a.shape[:2])
    return bool(np.array_equal(a, d))

# ---------------------------------------------------------------------------
# Precomputed plan API for video / volume workloads
# ---------------------------------------------------------------------------
from dataclasses import dataclass


@dataclass
class CipherPlan:
    config: CipherConfig
    h: int
    w: int
    perm_idx: np.ndarray
    row_action: np.ndarray
    R: np.ndarray


def build_plan(config: CipherConfig, h: int, w: int) -> CipherPlan:
    """Precompute all size/key-dependent material for repeated equal-sized arrays."""
    h, w = int(h), int(w)
    if h % 2 or w % 2:
        raise ValueError(f"Both dimensions must be even; got HxW={h}x{w}.")
    seq1 = generate_sequence(
        config.y01, config.mu01, config.k, h * w,
        transient=config.transient, capture_math=False,
    )
    perm_idx = np.argsort(seq1, kind="stable")
    steps = min(h, w)
    seq2 = generate_sequence(
        config.y02, config.mu02, config.k, steps,
        transient=config.transient, capture_math=False,
    )
    row_action = np.asarray(seq1[:steps]) > np.asarray(seq2)
    seq3 = generate_sequence(
        config.y03, config.mu03, config.k, h * w,
        transient=config.transient, capture_math=False,
    )
    q3 = quantize_sequence(seq3, config.quantization_mode, config.quantization_scale)
    R = q3.reshape((h, w), order="F")
    return CipherPlan(config=config, h=h, w=w, perm_idx=perm_idx, row_action=row_action, R=R)


def _plan_confusion1_forward(a: np.ndarray, plan: CipherPlan) -> np.ndarray:
    def one(ch):
        flat = np.asarray(ch, dtype=np.uint8).ravel(order="F")
        return flat[plan.perm_idx].reshape((plan.h, plan.w), order="F")
    if a.ndim == 2:
        return one(a)
    return np.stack([one(a[..., c]) for c in range(3)], axis=-1)


def _plan_confusion1_inverse(a: np.ndarray, plan: CipherPlan) -> np.ndarray:
    def one(ch):
        enc_flat = np.asarray(ch, dtype=np.uint8).ravel(order="F")
        plain_flat = np.empty_like(enc_flat)
        plain_flat[plan.perm_idx] = enc_flat
        return plain_flat.reshape((plan.h, plan.w), order="F")
    if a.ndim == 2:
        return one(a)
    return np.stack([one(a[..., c]) for c in range(3)], axis=-1)


def _plan_confusion2_forward(a: np.ndarray, plan: CipherPlan) -> np.ndarray:
    def one(ch):
        out = np.array(ch, dtype=np.uint8, copy=True)
        for i, is_row in enumerate(plan.row_action):
            if is_row:
                out[i, :] = out[i, ::-1]
            else:
                out[:, i] = out[::-1, i]
        return out
    if a.ndim == 2:
        return one(a)
    return np.stack([one(a[..., c]) for c in range(3)], axis=-1)


def _plan_confusion2_inverse(a: np.ndarray, plan: CipherPlan) -> np.ndarray:
    def one(ch):
        out = np.array(ch, dtype=np.uint8, copy=True)
        for i in range(len(plan.row_action) - 1, -1, -1):
            if plan.row_action[i]:
                out[i, :] = out[i, ::-1]
            else:
                out[:, i] = out[::-1, i]
        return out
    if a.ndim == 2:
        return one(a)
    return np.stack([one(a[..., c]) for c in range(3)], axis=-1)


def _plan_diffusion1_forward(a: np.ndarray, plan: CipherPlan) -> np.ndarray:
    def one(ch):
        s = (np.asarray(ch, dtype=np.uint16) + plan.R.astype(np.uint16)) % 256
        return s.astype(np.uint8).T.copy()
    if a.ndim == 2:
        return one(a)
    return np.stack([one(a[..., c]) for c in range(3)], axis=-1)


def _plan_diffusion1_inverse(a: np.ndarray, plan: CipherPlan) -> np.ndarray:
    def one(ch):
        pre = np.asarray(ch, dtype=np.uint8).T
        return ((pre.astype(np.int16) - plan.R.astype(np.int16)) % 256).astype(np.uint8)
    if a.ndim == 2:
        return one(a)
    return np.stack([one(a[..., c]) for c in range(3)], axis=-1)


def encrypt_with_plan(image: np.ndarray, plan: CipherPlan) -> np.ndarray:
    a = ensure_even_image(np.asarray(image, dtype=np.uint8))
    if a.shape[:2] != (plan.h, plan.w):
        raise ValueError(f"Plan expects {(plan.h, plan.w)}, got {a.shape[:2]}.")
    i1 = _plan_confusion1_forward(a, plan)
    i2 = _plan_confusion2_forward(i1, plan)
    i3 = _plan_diffusion1_forward(i2, plan)
    return _inverse_diffusion2(i3).astype(np.uint8)  # same transform forward/inverse


def decrypt_with_plan(cipher: np.ndarray, plan: CipherPlan) -> np.ndarray:
    c = ensure_even_image(np.asarray(cipher, dtype=np.uint8))
    if c.shape[:2] != (plan.w, plan.h):
        raise ValueError(f"Plan expects cipher shape {(plan.w, plan.h)}, got {c.shape[:2]}.")
    i3 = _inverse_diffusion2(c)
    i2 = _plan_diffusion1_inverse(i3, plan)
    i1 = _plan_confusion2_inverse(i2, plan)
    return _plan_confusion1_inverse(i1, plan).astype(np.uint8)

# ---------------------------------------------------------------------------
# Stage-preserving APIs used by the offline visual/bit-trace demonstrations
# ---------------------------------------------------------------------------

def encrypt_array_with_stages(image: np.ndarray, config: CipherConfig):
    """Encrypt one array and return every visible intermediate stage."""
    a = ensure_even_image(np.asarray(image, dtype=np.uint8))
    i1, t1 = C1.run(a, config)
    i2, t2 = C2.run(i1, t1["seq1"], config)
    i3, t3 = D1.run(i2, config)
    i4, t4 = D2.run(i3)
    stages = {
        "original": a,
        "confusion1": i1.astype(np.uint8),
        "confusion2": i2.astype(np.uint8),
        "diffusion1": i3.astype(np.uint8),
        "diffusion2": i4.astype(np.uint8),
        "cipher": i4.astype(np.uint8),
    }
    trace = {**t1, **t2, **t3, **t4}
    return i4.astype(np.uint8), stages, trace


def decrypt_array_with_stages(cipher: np.ndarray, config: CipherConfig, original_hw: tuple[int, int]):
    """Decrypt one array and return each inverse stage for presentation."""
    c = ensure_even_image(np.asarray(cipher, dtype=np.uint8))
    d2 = _inverse_diffusion2(c)
    d1 = _inverse_diffusion1(d2, config, original_hw)
    c2 = _inverse_confusion2(d1, config)
    plain = _inverse_confusion1(c2, config)
    stages = {
        "cipher": c,
        "inverse_diffusion2": d2.astype(np.uint8),
        "inverse_diffusion1": d1.astype(np.uint8),
        "inverse_confusion2": c2.astype(np.uint8),
        "inverse_confusion1": plain.astype(np.uint8),
        "recovered": plain.astype(np.uint8),
    }
    return plain.astype(np.uint8), stages
