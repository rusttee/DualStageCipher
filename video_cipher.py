"""Frame-by-frame video encryption/decryption using the paper image cipher.

Decryptable ciphertext video is stored with FFV1 in Matroska (.mkv), because a
lossy codec would alter ciphertext pixels and make exact decryption impossible.
The current extension encrypts the VISUAL stream only; audio is intentionally
not copied into the encrypted file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import fields
from pathlib import Path

import cv2
import numpy as np

from common import CipherConfig
from cipher_core import (build_plan, encrypt_with_plan, decrypt_with_plan,
                         encrypt_array_with_stages, decrypt_array_with_stages)
from stage_trace import export_stage_comparisons, show_matlab_stage_windows


def _config_from_dict(d: dict) -> CipherConfig:
    allowed = {f.name for f in fields(CipherConfig)}
    return CipherConfig(**{k: v for k, v in d.items() if k in allowed})


def _metadata_path(cipher_path: Path) -> Path:
    return cipher_path.with_suffix(cipher_path.suffix + ".json")


def _open_ffv1_writer(path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"FFV1")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (int(width), int(height)), True)
    if not writer.isOpened():
        raise RuntimeError(
            "OpenCV could not create an FFV1 video. Use a .mkv output and make sure "
            "your OpenCV build has FFmpeg/FFV1 support."
        )
    return writer


def encrypt_video(src: Path, dst: Path, config: CipherConfig,
                  trace_frame: int = 0, trace_dir: Path | None = None, show_steps: bool = False) -> Path:
    if dst.suffix.lower() != ".mkv":
        raise ValueError("Use a .mkv destination for the lossless FFV1 ciphertext video.")

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {src}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    declared_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if width % 2 or height % 2:
        cap.release()
        raise ValueError(f"Video dimensions must both be even; got {width}x{height}.")

    # Diffusion-1 transposes each image, so encrypted frame WxH becomes HxW.
    writer = _open_ffv1_writer(dst, fps, height, width)
    sha_plain = hashlib.sha256()
    sha_cipher = hashlib.sha256()
    count = 0
    started = time.perf_counter()
    static_plan = build_plan(config, height, width)
    trace_stages = None
    trace_math = None
    trace_td = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame.shape[:2] != (height, width):
                raise RuntimeError(f"Frame {count} changed size unexpectedly: {frame.shape[:2]}")
            cipher = encrypt_with_plan(frame, static_plan)
            if count == trace_frame:
                c_trace, stages, stage_trace = encrypt_array_with_stages(frame, config)
                if not np.array_equal(c_trace, cipher):
                    raise RuntimeError("Trace-stage cipher differs from the video encryption result.")
                trace_stages = stages
                trace_math = stage_trace
                trace_td = trace_dir or (dst.parent / f"{dst.stem}_frame_{count:06d}_encryption_trace")
                order = ["original", "confusion1", "confusion2", "diffusion1", "diffusion2"]
                export_stage_comparisons(stages, order, trace_td / "bit_value_comparisons", bgr=True)
            sha_plain.update(frame.tobytes(order="C"))
            sha_cipher.update(cipher.tobytes(order="C"))
            writer.write(cipher)
            count += 1
            if count == 1 or count % 10 == 0:
                elapsed = max(time.perf_counter() - started, 1e-9)
                total = f"/{declared_frames}" if declared_frames > 0 else ""
                print(f"Encrypted frames: {count}{total}  ({count/elapsed:.2f} fps processing)", flush=True)
    finally:
        cap.release()
        writer.release()

    meta = {
        "format": "nature-cipher-video-v1",
        "source_name": src.name,
        "cipher_video": dst.name,
        "codec": "FFV1",
        "container": "Matroska",
        "audio": "not included; visual frames only",
        "fps": fps,
        "frame_count": count,
        "original_width": width,
        "original_height": height,
        "cipher_width": height,
        "cipher_height": width,
        "paper_key_policy": "unchanged published parameters reused for every decoded frame",
        "config": config.to_dict(),
        "plain_decoded_frame_sha256": sha_plain.hexdigest(),
        "cipher_frame_sha256_before_video_write": sha_cipher.hexdigest(),
        "trace_frame": int(trace_frame),
        "note": (
            "The source video is decoded to uint8 BGR frames. Those decoded frames are the plaintext domain. "
            "Exact decryption is verified against their SHA-256, not against the original compressed file bytes."
        ),
    }
    mp = _metadata_path(dst)
    mp.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if trace_stages is not None and trace_td is not None:
        order = ["original", "confusion1", "confusion2", "diffusion1", "diffusion2"]
        show_matlab_stage_windows(trace_stages, order, trace_td / "web_dashboard",
                                  f"VIDEO ENCRYPTION — frame {trace_frame}", show=False, bgr=True, sample_count=30,
                                  config=config, trace=trace_math)
    elif show_steps:
        print(f"Requested trace frame {trace_frame} was not present; full video was still processed.")
    print(f"\nEncrypted video : {dst}")
    print(f"Metadata        : {mp}")
    print(f"Frames          : {count}")
    print("Audio           : removed (visual-stream encryption experiment)")
    return mp


def decrypt_video(src: Path, dst: Path, metadata: Path | None = None, trace_frame: int = 0,
                  trace_dir: Path | None = None, show_steps: bool = False) -> bool:
    mp = metadata or _metadata_path(src)
    if not mp.exists():
        raise FileNotFoundError(f"Metadata sidecar not found: {mp}")
    meta = json.loads(mp.read_text(encoding="utf-8"))
    if meta.get("format") != "nature-cipher-video-v1":
        raise ValueError("Unsupported or invalid video cipher metadata file.")

    config = _config_from_dict(meta["config"])
    width = int(meta["original_width"])
    height = int(meta["original_height"])
    fps = float(meta["fps"])

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open encrypted video: {src}")
    writer = _open_ffv1_writer(dst, fps, width, height)
    sha_plain = hashlib.sha256()
    count = 0
    started = time.perf_counter()
    static_plan = build_plan(config, height, width)
    trace_stages = None
    trace_td = None

    try:
        while True:
            ok, cipher = cap.read()
            if not ok:
                break
            plain = decrypt_with_plan(cipher, static_plan)
            if count == trace_frame:
                p_trace, stages = decrypt_array_with_stages(cipher, config, (height, width))
                if not np.array_equal(p_trace, plain):
                    raise RuntimeError("Trace-stage plaintext differs from the video decryption result.")
                trace_stages = stages
                trace_td = trace_dir or (dst.parent / f"{dst.stem}_frame_{count:06d}_decryption_trace")
                order = ["cipher", "inverse_diffusion2", "inverse_diffusion1", "inverse_confusion2", "inverse_confusion1"]
                export_stage_comparisons(stages, order, trace_td / "bit_value_comparisons", bgr=True)
            sha_plain.update(plain.tobytes(order="C"))
            writer.write(plain)
            count += 1
            if count == 1 or count % 10 == 0:
                elapsed = max(time.perf_counter() - started, 1e-9)
                print(f"Decrypted frames: {count}/{meta.get('frame_count', '?')}  ({count/elapsed:.2f} fps processing)", flush=True)
    finally:
        cap.release()
        writer.release()

    if trace_stages is not None and trace_td is not None:
        order = ["cipher", "inverse_diffusion2", "inverse_diffusion1", "inverse_confusion2", "inverse_confusion1"]
        show_matlab_stage_windows(trace_stages, order, trace_td / "web_dashboard",
                                  f"VIDEO DECRYPTION — frame {trace_frame}", show=False, bgr=True, sample_count=30,
                                  config=config)
    elif show_steps:
        print(f"Requested trace frame {trace_frame} was not present; full encrypted video was still decrypted.")

    actual = sha_plain.hexdigest()
    expected = meta.get("plain_decoded_frame_sha256")
    exact = actual == expected and count == int(meta.get("frame_count", count))
    print(f"\nDecrypted video : {dst}")
    print(f"Frames          : {count}")
    print(f"Exact recovery  : {'YES' if exact else 'NO'}")
    print(f"Recovered SHA256: {actual}")
    if expected:
        print(f"Expected SHA256 : {expected}")
    return exact


def _build_config(args=None) -> CipherConfig:
    return CipherConfig(discard_first_100=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Encrypt/decrypt video frames with the paper-reproduction cipher.")
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("encrypt", help="Encrypt every decoded video frame into a lossless FFV1/MKV stream.")
    e.add_argument("input", type=Path)
    e.add_argument("output", type=Path)
    e.add_argument("--trace-frame", type=int, default=0)
    e.add_argument("--show-steps", action="store_true")

    d = sub.add_parser("decrypt", help="Decrypt an FFV1/MKV ciphertext stream.")
    d.add_argument("input", type=Path)
    d.add_argument("output", type=Path)
    d.add_argument("--metadata", type=Path, default=None)
    d.add_argument("--trace-frame", type=int, default=0)
    d.add_argument("--show-steps", action="store_true")

    args = p.parse_args()
    if args.cmd == "encrypt":
        encrypt_video(args.input, args.output, _build_config(), args.trace_frame, show_steps=args.show_steps)
    else:
        ok = decrypt_video(args.input, args.output, args.metadata, args.trace_frame, show_steps=args.show_steps)
        raise SystemExit(0 if ok else 2)


if __name__ == "__main__":
    main()
