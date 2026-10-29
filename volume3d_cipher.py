"""3-D volumetric image encryption/decryption using the paper image cipher.

Definition used here: a "3-D image" is a voxel/volume stack, e.g. CT/MRI or a
stack of grayscale/RGB slices.  Canonical arrays are:
    grayscale: (D, H, W)
    color:     (D, H, W, 3)

The existing 2-D cipher is applied to each slice.  An optional extra 3-D
confusion step permutes the slice/depth order with a chaotic sequence.  Both
operations are exactly reversible.
"""

from __future__ import annotations

import argparse
import json
import hashlib
from dataclasses import fields
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageSequence

from common import CipherConfig, generate_sequence
from cipher_core import (build_plan, encrypt_with_plan, decrypt_with_plan,
                         encrypt_array_with_stages, decrypt_array_with_stages)
from stage_trace import export_stage_comparisons, show_matlab_stage_windows


def _config_from_dict(d: dict) -> CipherConfig:
    allowed = {f.name for f in fields(CipherConfig)}
    return CipherConfig(**{k: v for k, v in d.items() if k in allowed})


def load_volume(path: Path) -> np.ndarray:
    low = path.name.lower()
    if low.endswith(".npy"):
        a = np.load(path, allow_pickle=False)
    elif low.endswith(".npz"):
        with np.load(path, allow_pickle=False) as z:
            if "volume" in z:
                a = z["volume"]
            elif "cipher" in z:
                a = z["cipher"]
            else:
                keys = list(z.keys())
                if len(keys) != 1:
                    raise ValueError(f"NPZ must contain 'volume' or one array; found {keys}")
                a = z[keys[0]]
    elif low.endswith((".tif", ".tiff")):
        frames = []
        with Image.open(path) as im:
            for frame in ImageSequence.Iterator(im):
                f = frame.copy()
                if f.mode not in ("L", "RGB"):
                    f = f.convert("L")
                frames.append(np.asarray(f, dtype=np.uint8))
        if not frames:
            raise ValueError("TIFF contained no frames.")
        a = np.stack(frames, axis=0)
    else:
        raise ValueError("Supported 3-D inputs: .npy, .npz, multi-page .tif/.tiff")

    if a.dtype != np.uint8:
        raise TypeError(
            f"Volume dtype is {a.dtype}; this cipher works on uint8 values. "
            "Convert/normalize the volume to uint8 explicitly so the preprocessing is scientifically documented."
        )
    if a.ndim == 3:
        pass
    elif a.ndim == 4 and a.shape[-1] == 3:
        pass
    else:
        raise ValueError(f"Expected (D,H,W) or (D,H,W,3), got {a.shape}")
    if a.shape[1] % 2 or a.shape[2] % 2:
        raise ValueError(f"Slice H and W must be even; got HxW={a.shape[1]}x{a.shape[2]}")
    return np.asarray(a, dtype=np.uint8)


def save_plain_volume(a: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    low = path.name.lower()
    if low.endswith(".npy"):
        np.save(path, np.asarray(a, dtype=np.uint8), allow_pickle=False)
        return
    if low.endswith((".tif", ".tiff")):
        arr = np.asarray(a, dtype=np.uint8)
        images = [Image.fromarray(arr[i]) for i in range(arr.shape[0])]
        images[0].save(path, save_all=True, append_images=images[1:], compression="tiff_lzw")
        return
    raise ValueError("Decrypted output must be .npy or .tif/.tiff")


def _depth_permutation(depth: int, config: CipherConfig) -> np.ndarray:
    seq = generate_sequence(
        config.y01, config.mu01, config.k, depth,
        transient=config.transient, capture_math=False,
    )
    return np.argsort(seq, kind="stable")



def _save_depth_permutation_trace(perm: np.ndarray, out_dir: Path, show: bool, inverse: bool = False) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    perm = np.asarray(perm, dtype=int)
    if not inverse:
        # Forward: cipher_depth[k] = pre_depth[perm[k]]
        rows = np.column_stack([np.arange(len(perm)), perm])
        header = "cipher_output_z,source_pre_permutation_z"
        title = "3-D depth permutation: cipher Z <- source Z"
        name = "depth_permutation.csv"
        y = perm
    else:
        # Inverse: restored_depth[perm[k]] = cipher_depth[k]
        rows = np.column_stack([np.arange(len(perm)), perm])
        header = "cipher_input_z,recovered_plain_z"
        title = "3-D inverse depth permutation: recovered Z <- cipher Z"
        name = "inverse_depth_permutation.csv"
        y = perm
    np.savetxt(out_dir / name, rows, fmt="%d", delimiter=",", header=header, comments="")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(np.arange(len(perm)), y, marker="o")
    ax.set_xlabel("Cipher Z index")
    ax.set_ylabel("Plain/pre-permutation Z index")
    ax.set_title(title)
    ax.grid(True, alpha=.3)
    fig.tight_layout(); fig.savefig(out_dir / ("inverse_depth_permutation.png" if inverse else "depth_permutation.png"), dpi=160)
    if show: plt.show()
    plt.close(fig)

def encrypt_volume(src: Path, dst: Path, config: CipherConfig, permute_depth: bool,
                   trace_slice: int = 0, trace_dir: Path | None = None, show_steps: bool = False) -> None:
    if dst.suffix.lower() != ".npz":
        raise ValueError("Encrypted 3-D volume output should use .npz so cipher data and metadata stay together.")
    volume = load_volume(src)
    d, h, w = volume.shape[:3]
    enc_slices = []
    static_plan = build_plan(config, h, w)
    trace_stages = None
    trace_math = None
    trace_td = None
    for z in range(d):
        c = encrypt_with_plan(volume[z], static_plan)
        if z == trace_slice:
            c_trace, stages, stage_trace = encrypt_array_with_stages(volume[z], config)
            if not np.array_equal(c_trace, c):
                raise RuntimeError("Trace-stage cipher differs from 3-D slice encryption result.")
            trace_stages = stages
            trace_math = stage_trace
            trace_td = trace_dir or (dst.parent / f"{dst.stem}_slice_{z:03d}_encryption_trace")
            order = ["original", "confusion1", "confusion2", "diffusion1", "diffusion2"]
            export_stage_comparisons(stages, order, trace_td / "bit_value_comparisons")
        enc_slices.append(c)
        if z == 0 or (z + 1) % 10 == 0 or z + 1 == d:
            print(f"Encrypted slices: {z+1}/{d}", flush=True)
    cipher = np.stack(enc_slices, axis=0)

    perm = None
    if permute_depth:
        perm = _depth_permutation(d, config)
        cipher = cipher[perm]
        td = trace_dir or (dst.parent / f"{dst.stem}_depth_trace")
        _save_depth_permutation_trace(perm, td / "depth_permutation", False, inverse=False)

    meta = {
        "format": "nature-cipher-volume3d-v1",
        "source_name": src.name,
        "original_shape": list(volume.shape),
        "cipher_shape": list(cipher.shape),
        "dtype": "uint8",
        "slice_axis": 0,
        "paper_key_policy": "unchanged published parameters reused for every Z slice",
        "depth_permutation": bool(permute_depth),
        "config": config.to_dict(),
        "plain_sha256": hashlib.sha256(volume.tobytes(order="C")).hexdigest(),
        "trace_slice": int(trace_slice),
        "definition": "3-D image = voxel/volume stack; each Z slice is encrypted with the 2-D paper cipher.",
    }
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez(dst, cipher=cipher, metadata=np.array(json.dumps(meta)))
    if trace_stages is not None and trace_td is not None:
        order = ["original", "confusion1", "confusion2", "diffusion1", "diffusion2"]
        show_matlab_stage_windows(trace_stages, order, trace_td / "web_dashboard",
                                  f"3-D ENCRYPTION — Z slice {trace_slice}", show=False, sample_count=30,
                                  config=config, trace=trace_math)
    elif show_steps:
        print(f"Requested trace slice {trace_slice} was not present; full 3-D volume was still processed.")
    print(f"\nEncrypted 3-D volume: {dst}")
    print(f"Original shape       : {volume.shape}")
    print(f"Cipher shape         : {cipher.shape}")
    print(f"Depth permutation    : {'ON' if permute_depth else 'OFF'}")


def decrypt_volume(src: Path, dst: Path, trace_slice: int = 0, trace_dir: Path | None = None,
                   show_steps: bool = False) -> bool:
    with np.load(src, allow_pickle=False) as z:
        if "cipher" not in z or "metadata" not in z:
            raise ValueError("Encrypted NPZ must contain cipher and metadata arrays.")
        cipher = np.asarray(z["cipher"], dtype=np.uint8)
        meta = json.loads(str(z["metadata"].item()))
    if meta.get("format") != "nature-cipher-volume3d-v1":
        raise ValueError("Unsupported encrypted 3-D volume format.")

    config = _config_from_dict(meta["config"])
    original_shape = tuple(int(x) for x in meta["original_shape"])
    d, h, w = original_shape[:3]
    work = cipher

    if meta.get("depth_permutation"):
        perm = _depth_permutation(d, config)
        td = trace_dir or (dst.parent / f"{dst.stem}_depth_trace")
        _save_depth_permutation_trace(perm, td / "depth_permutation", False, inverse=True)
        restored_order = np.empty_like(work)
        restored_order[perm] = work
        work = restored_order

    plain_slices = []
    static_plan = build_plan(config, h, w)
    trace_stages = None
    trace_td = None
    for z in range(d):
        p = decrypt_with_plan(work[z], static_plan)
        if z == trace_slice:
            p_trace, stages = decrypt_array_with_stages(work[z], config, (h, w))
            if not np.array_equal(p_trace, p):
                raise RuntimeError("Trace-stage plaintext differs from 3-D slice decryption result.")
            trace_stages = stages
            trace_td = trace_dir or (dst.parent / f"{dst.stem}_slice_{z:03d}_decryption_trace")
            order = ["cipher", "inverse_diffusion2", "inverse_diffusion1", "inverse_confusion2", "inverse_confusion1"]
            export_stage_comparisons(stages, order, trace_td / "bit_value_comparisons")
        plain_slices.append(p)
        if z == 0 or (z + 1) % 10 == 0 or z + 1 == d:
            print(f"Decrypted slices: {z+1}/{d}", flush=True)
    plain = np.stack(plain_slices, axis=0)
    if trace_stages is not None and trace_td is not None:
        order = ["cipher", "inverse_diffusion2", "inverse_diffusion1", "inverse_confusion2", "inverse_confusion1"]
        show_matlab_stage_windows(trace_stages, order, trace_td / "web_dashboard",
                                  f"3-D DECRYPTION — Z slice {trace_slice}", show=False, sample_count=30,
                                  config=config)
    elif show_steps:
        print(f"Requested trace slice {trace_slice} was not present; full encrypted volume was still decrypted.")
    exact_shape = plain.shape == original_shape
    actual_sha = hashlib.sha256(plain.tobytes(order="C")).hexdigest()
    expected_sha = meta.get("plain_sha256")
    exact_bytes = (actual_sha == expected_sha) if expected_sha else exact_shape
    save_plain_volume(plain, dst)
    print(f"\nDecrypted 3-D volume: {dst}")
    print(f"Recovered shape      : {plain.shape}")
    print(f"Shape verification   : {'YES' if exact_shape else 'NO'}")
    print(f"Exact byte recovery  : {'YES' if exact_bytes else 'NO'}")
    print(f"Recovered SHA256     : {actual_sha}")
    if expected_sha:
        print(f"Expected SHA256      : {expected_sha}")
    return bool(exact_shape and exact_bytes)


def main() -> None:
    p = argparse.ArgumentParser(description="Encrypt/decrypt 3-D uint8 voxel/volume images.")
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("encrypt")
    e.add_argument("input", type=Path)
    e.add_argument("output", type=Path)
    e.add_argument("--permute-depth", action="store_true",
                   help="Add a reversible chaotic permutation across the Z/depth dimension.")
    e.add_argument("--trace-slice", type=int, default=0)
    e.add_argument("--show-steps", action="store_true")

    d = sub.add_parser("decrypt")
    d.add_argument("input", type=Path)
    d.add_argument("output", type=Path)
    d.add_argument("--trace-slice", type=int, default=0)
    d.add_argument("--show-steps", action="store_true")

    args = p.parse_args()
    if args.cmd == "encrypt":
        cfg = CipherConfig(discard_first_100=True)
        encrypt_volume(args.input, args.output, cfg, args.permute_depth, args.trace_slice, show_steps=args.show_steps)
    else:
        ok = decrypt_volume(args.input, args.output, args.trace_slice, show_steps=args.show_steps)
        raise SystemExit(0 if ok else 2)


if __name__ == "__main__":
    main()
