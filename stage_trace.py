"""Offline web trace generation for the paper-locked chaotic image cipher.

No Matplotlib/Tk windows are opened.  Every detailed presentation is serialized
as JSON + local PNG assets and displayed by the local HTML GUI (gui_server.py).

Scientific notes:
- Confusion-1 and Confusion-2 only rearrange pixels.  Their histograms must be
  identical to their inputs; the GUI therefore visualizes mappings and matrix
  flips rather than pretending the histogram should change.
- Paper Eq.(7) uses AND + bit OR for the MSB/LSB cross-quadrant diffusion.
  The GUI also reports before XOR after as a diagnostic bit-change mask only.
"""
from __future__ import annotations

import csv
import gzip
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from common import CipherConfig, generate_sequence, quantize_sequence


def _bits(v: int) -> str:
    return format(int(v) & 0xFF, "08b")


def _channel_names(a: np.ndarray, bgr: bool):
    if a.ndim == 2:
        return ["Gray"]
    return ["B", "G", "R"] if bgr else ["R", "G", "B"]


def _display_rgb(a: np.ndarray, bgr: bool = False) -> np.ndarray:
    x = np.asarray(a, dtype=np.uint8)
    if x.ndim == 3 and bgr:
        return x[..., ::-1]
    return x


def _pretty(name: str) -> str:
    labels = {
        "original": "Original",
        "confusion1": "Confusion 1",
        "confusion2": "Confusion 2",
        "diffusion1": "Diffusion 1",
        "diffusion2": "Diffusion 2",
        "encrypted": "Encrypted output",
        "cipher": "Cipher input",
        "inverse_diffusion2": "Inverse Diffusion 2",
        "inverse_diffusion1": "Inverse Diffusion 1",
        "inverse_confusion2": "Inverse Confusion 2",
        "inverse_confusion1": "Inverse Confusion 1",
        "recovered": "Recovered output",
    }
    return labels.get(name, name.replace("_", " ").title())


def _presentation_stages(stages: dict[str, np.ndarray], order: Iterable[str]):
    keys = list(order)
    items = [(k, stages[k]) for k in keys]
    if keys and keys[0] == "original":
        items.append(("encrypted", stages[keys[-1]]))
    elif keys and keys[0] == "cipher":
        items.append(("recovered", stages[keys[-1]]))
    return items


def write_pair_bit_csv(before: np.ndarray, after: np.ndarray, stage_name: str,
                       path: Path, bgr: bool = False) -> dict:
    """Write every same-buffer-slot transition and a diagnostic XOR change mask."""
    a = np.asarray(before, dtype=np.uint8)
    b = np.asarray(after, dtype=np.uint8)
    if a.size != b.size:
        raise ValueError(f"{stage_name}: byte counts differ: {a.size} vs {b.size}")
    path.parent.mkdir(parents=True, exist_ok=True)
    channels = _channel_names(a, bgr)
    af = a.reshape(-1)
    bf = b.reshape(-1)
    changed_bytes = int(np.count_nonzero(af != bf))
    changed_bits = 0
    with gzip.open(path, "wt", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "stage", "linear_byte_slot", "before_row", "before_col", "after_row", "after_col",
            "channel", "before_dec", "after_dec", "before_bits", "after_bits",
            "diagnostic_xor_bits", "changed_bit_count",
        ])
        ach = 1 if a.ndim == 2 else a.shape[2]
        bch = 1 if b.ndim == 2 else b.shape[2]
        for i, (x, y) in enumerate(zip(af, bf)):
            apix, ac = divmod(i, ach) if ach > 1 else (i, 0)
            bpix, bc = divmod(i, bch) if bch > 1 else (i, 0)
            ar, acol = divmod(apix, a.shape[1])
            br, bcol = divmod(bpix, b.shape[1])
            xv, yv = int(x), int(y)
            xo = xv ^ yv
            nbits = xo.bit_count()
            changed_bits += nbits
            w.writerow([
                stage_name, i, ar, acol, br, bcol,
                channels[min(ac, len(channels) - 1)], xv, yv,
                _bits(xv), _bits(yv), _bits(xo), nbits,
            ])
    return {
        "stage": stage_name,
        "total_bytes": int(af.size),
        "changed_bytes": changed_bytes,
        "changed_bytes_percent": 100.0 * changed_bytes / max(1, af.size),
        "total_bits": int(af.size * 8),
        "changed_bits": int(changed_bits),
        "changed_bits_percent": 100.0 * changed_bits / max(1, af.size * 8),
        "csv_gz": str(path),
    }


def export_stage_comparisons(stages: dict[str, np.ndarray], order: Iterable[str], output_dir: Path,
                             bgr: bool = False) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    keys = list(order)
    reports = []
    for n, (k0, k1) in enumerate(zip(keys, keys[1:]), 1):
        reports.append(write_pair_bit_csv(
            stages[k0], stages[k1], f"{k0} -> {k1}",
            output_dir / f"{n:02d}_{k0}_to_{k1}_all_bytes_bits.csv.gz", bgr=bgr,
        ))
    with (output_dir / "bit_change_summary.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(reports[0].keys()) if reports else ["stage"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader(); w.writerows(reports)
    return reports


def _entropy(hist: np.ndarray) -> float:
    h = np.asarray(hist, dtype=np.float64)
    if h.sum() <= 0:
        return 0.0
    p = h[h > 0] / h.sum()
    return float(-(p * np.log2(p)).sum())


def _hist_payload(a: np.ndarray, bgr: bool) -> dict:
    x = _display_rgb(a, bgr=bgr)
    if x.ndim == 2:
        gray = x.reshape(-1)
        rgb = None
        channel_stats = {"Gray": _basic_stats(gray)}
    else:
        # x is RGB here.
        rgb = {}
        channel_stats = {}
        for c, label in enumerate(("R", "G", "B")):
            vals = x[..., c].reshape(-1)
            rgb[label] = np.bincount(vals, minlength=256).astype(int).tolist()
            channel_stats[label] = _basic_stats(vals)
        # Luma used only for histogram analysis; encryption still uses raw channels.
        gray = np.rint(0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]).astype(np.uint8).reshape(-1)
    hist = np.bincount(gray, minlength=256).astype(np.int64)
    cum = np.cumsum(hist)
    expected = gray.size / 256.0 if gray.size else 0.0
    chi = float(np.sum((hist - expected) ** 2 / expected)) if expected > 0 else 0.0
    return {
        "intensity": hist.astype(int).tolist(),
        "cumulative": cum.astype(int).tolist(),
        "rgb": rgb,
        "stats": {
            **_basic_stats(gray),
            "entropy_bits": _entropy(hist),
            "unique_intensities": int(np.count_nonzero(hist)),
            "clipped_0_percent": float(100.0 * hist[0] / max(1, gray.size)),
            "clipped_255_percent": float(100.0 * hist[255] / max(1, gray.size)),
            "chi_square_vs_uniform": chi,
            "pixel_components": int(np.asarray(a).size),
            "channel_stats": channel_stats,
        },
    }


def _basic_stats(vals: np.ndarray) -> dict:
    v = np.asarray(vals, dtype=np.float64).reshape(-1)
    if not v.size:
        return {"min": 0, "max": 0, "mean": 0, "std": 0, "median": 0, "dynamic_range": 0}
    mn, mx = float(v.min()), float(v.max())
    return {
        "min": mn, "max": mx, "mean": float(v.mean()), "std": float(v.std()),
        "median": float(np.median(v)), "dynamic_range": mx - mn,
    }


def _save_stage_image(arr: np.ndarray, path: Path, bgr: bool) -> None:
    x = _display_rgb(arr, bgr=bgr)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(x).save(path)


def _coord_for_linear(arr: np.ndarray, linear: int) -> tuple[int, int, int, str]:
    a = np.asarray(arr)
    chn = 1 if a.ndim == 2 else a.shape[2]
    pixel, ch = divmod(int(linear), chn) if chn > 1 else (int(linear), 0)
    row, col = divmod(pixel, a.shape[1])
    names = _channel_names(a, False)
    return int(row), int(col), int(ch), names[min(ch, len(names) - 1)]


def _informative_indices(first: np.ndarray, count: int = 30) -> list[int]:
    """Choose deterministic non-background byte slots instead of blindly starting at zero."""
    flat = np.asarray(first, dtype=np.uint8).reshape(-1)
    if flat.size == 0:
        return []
    primary = np.flatnonzero((flat > 0) & (flat < 255))
    if primary.size < count:
        primary = np.flatnonzero(flat != 0)
    if primary.size < count:
        primary = np.arange(flat.size)
    if primary.size <= count:
        return primary.astype(int).tolist()
    # Spread samples through the complete image/frame/slice, avoiding a corner-only view.
    pick = np.linspace(0, primary.size - 1, count, dtype=int)
    return primary[pick].astype(int).tolist()


def _sample_progression(stages: dict[str, np.ndarray], order: Iterable[str], bgr: bool,
                        count: int = 30) -> list[dict]:
    items = _presentation_stages(stages, order)
    base = stages[list(order)[0]]
    indices = _informative_indices(base, count=count)
    rows = []
    for n, idx in enumerate(indices, 1):
        vals = []
        for name, arr in items:
            flat = np.asarray(arr, dtype=np.uint8).reshape(-1)
            if idx >= flat.size:
                continue
            row, col, ch, _ = _coord_for_linear(arr, idx)
            channel = _channel_names(np.asarray(arr), bgr)[min(ch, len(_channel_names(np.asarray(arr), bgr))-1)]
            v = int(flat[idx])
            vals.append({
                "stage": name, "label": _pretty(name), "value": v, "bits": _bits(v),
                "row": row, "col": col, "channel": channel,
            })
        transitions = []
        for a, b in zip(vals, vals[1:]):
            xo = int(a["value"]) ^ int(b["value"])
            transitions.append({
                "from": a["stage"], "to": b["stage"],
                "before": a["value"], "after": b["value"],
                "before_bits": a["bits"], "after_bits": b["bits"],
                "diagnostic_xor": xo, "diagnostic_xor_bits": _bits(xo),
                "changed_bits": int(xo.bit_count()),
            })
        rows.append({"sample": n, "linear_byte_slot": int(idx), "values": vals, "transitions": transitions})
    return rows


def _matrix_window(a: np.ndarray, row: int, col: int, channel: int = 0, radius: int = 3) -> dict:
    x = np.asarray(a, dtype=np.uint8)
    if x.ndim == 3:
        channel = min(channel, x.shape[2] - 1)
        x = x[..., channel]
    r0, r1 = max(0, row-radius), min(x.shape[0], row+radius+1)
    c0, c1 = max(0, col-radius), min(x.shape[1], col+radius+1)
    return {
        "row0": r0, "col0": c0,
        "values": x[r0:r1, c0:c1].astype(int).tolist(),
        "focus": [int(row-r0), int(col-c0)],
    }


def _confusion1_details(stages: dict[str, np.ndarray], trace: dict | None,
                        config: CipherConfig | None, samples: list[dict], bgr: bool) -> dict | None:
    if "original" not in stages or "confusion1" not in stages:
        return None
    a = np.asarray(stages["original"], dtype=np.uint8)
    h, w = a.shape[:2]
    if trace and "perm_idx_python" in trace:
        perm = np.asarray(trace["perm_idx_python"], dtype=int)
    elif config:
        seq = generate_sequence(config.y01, config.mu01, config.k, h*w, transient=config.transient)
        perm = np.argsort(seq, kind="stable")
    else:
        return None
    mappings = []
    for s in samples:
        idx = int(s["linear_byte_slot"])
        row, col, ch, _ = _coord_for_linear(a, idx)
        dest_f = row + col*h
        source_f = int(perm[dest_f])
        sr, sc = source_f % h, source_f // h
        srcv = int(a[sr, sc] if a.ndim == 2 else a[sr, sc, ch])
        dst_arr = np.asarray(stages["confusion1"], dtype=np.uint8)
        dstv = int(dst_arr[row, col] if dst_arr.ndim == 2 else dst_arr[row, col, ch])
        mappings.append({
            "linear_byte_slot": idx, "channel": _channel_names(a, bgr)[min(ch, len(_channel_names(a,bgr))-1)],
            "source": [int(sr), int(sc)], "destination": [int(row), int(col)],
            "source_value": srcv, "destination_value": dstv,
            "source_bits": _bits(srcv), "destination_bits": _bits(dstv),
            "same_value_preserved": bool(srcv == dstv),
            "source_window": _matrix_window(stages["original"], sr, sc, ch),
            "destination_window": _matrix_window(stages["confusion1"], row, col, ch),
        })
    return {
        "explanation": "Confusion 1 sorts the chaotic sequence and permutes pixel positions. Pixel values are moved, not numerically modified; therefore its histogram is unchanged.",
        "mappings": mappings,
    }


def _apply_confusion2_one(work: np.ndarray, i: int, is_row: bool) -> None:
    if is_row:
        work[i, :] = work[i, ::-1]
    else:
        work[:, i] = work[::-1, i]


def _confusion2_iterations(stages: dict[str, np.ndarray], trace: dict | None,
                           config: CipherConfig | None, bgr: bool, limit: int = 30) -> dict | None:
    if "confusion1" not in stages or "confusion2" not in stages:
        return None
    inp = np.asarray(stages["confusion1"], dtype=np.uint8)
    h, w = inp.shape[:2]
    steps = min(h, w)
    if trace and all(k in trace for k in ("seq1_head", "seq2", "actions")):
        s1 = np.asarray(trace["seq1_head"], dtype=float)
        s2 = np.asarray(trace["seq2"], dtype=float)
        actions = list(trace["actions"])
    elif config:
        seq1 = generate_sequence(config.y01, config.mu01, config.k, h*w, transient=config.transient)
        s1 = np.asarray(seq1[:steps])
        s2 = np.asarray(generate_sequence(config.y02, config.mu02, config.k, steps, transient=config.transient))
        actions = ["row-left-right" if s1[i] > s2[i] else "column-up-down" for i in range(steps)]
    else:
        return None
    work = inp.copy()
    records = []
    for i in range(min(steps, limit)):
        is_row = bool(s1[i] > s2[i])
        center_r = i if is_row else h//2
        center_c = w//2 if is_row else i
        before = _matrix_window(work, center_r, center_c, 0)
        _apply_confusion2_one(work, i, is_row)
        after = _matrix_window(work, center_r, center_c, 0)
        records.append({
            "step": i+1,
            "chaotic_generated_iteration": (config.transient if config else 100) + i + 1,
            "yn1": float(s1[i]), "yn2": float(s2[i]),
            "comparison": ">" if is_row else "<=",
            "action": actions[i], "affected_index": i,
            "before": before, "after": after,
            "channel": _channel_names(inp, bgr)[0],
        })
    return {
        "explanation": "Confusion 2 compares usable yn1 and yn2 values and reverses one row or one column per iteration. It rearranges values only, so the histogram remains unchanged.",
        "iterations": records,
    }


def _diffusion1_details(stages: dict[str, np.ndarray], trace: dict | None,
                        config: CipherConfig | None, samples: list[dict], bgr: bool) -> dict | None:
    if "confusion2" not in stages or "diffusion1" not in stages:
        return None
    inp = np.asarray(stages["confusion2"], dtype=np.uint8)
    out = np.asarray(stages["diffusion1"], dtype=np.uint8)
    h, w = inp.shape[:2]
    if trace and "R" in trace:
        R = np.asarray(trace["R"], dtype=np.uint8)
    elif config:
        seq3 = generate_sequence(config.y03, config.mu03, config.k, h*w, transient=config.transient)
        R = quantize_sequence(seq3, config.quantization_mode, config.quantization_scale).reshape((h,w), order="F")
    else:
        return None
    records = []
    for s in samples:
        idx = int(s["linear_byte_slot"])
        if idx >= out.size:
            continue
        orow, ocol, ch, _ = _coord_for_linear(out, idx)
        # I3 = transpose(mod(I2 + R, 256))
        sr, sc = ocol, orow
        if sr >= h or sc >= w:
            continue
        iv = int(inp[sr, sc] if inp.ndim == 2 else inp[sr, sc, ch])
        rv = int(R[sr, sc])
        sm = iv + rv
        mv = sm % 256
        ov = int(out[orow, ocol] if out.ndim == 2 else out[orow, ocol, ch])
        xo = iv ^ ov
        records.append({
            "linear_byte_slot": idx,
            "channel": _channel_names(inp, bgr)[min(ch, len(_channel_names(inp,bgr))-1)],
            "input_coord": [int(sr), int(sc)], "output_coord_after_transpose": [int(orow), int(ocol)],
            "I2": iv, "R": rv, "sum": sm, "mod256": mv, "I3": ov,
            "I2_bits": _bits(iv), "R_bits": _bits(rv), "I3_bits": _bits(ov),
            "diagnostic_xor_bits": _bits(xo),
            "formula": f"({iv} + {rv}) mod 256 = {mv}; transpose -> {ov}",
            "before_window": _matrix_window(inp, sr, sc, ch),
            "after_window": _matrix_window(out, orow, ocol, ch),
        })
    return {
        "explanation": "Paper Eq.(6): R1=I2+R, S=mod(R1,256), I3=transpose(S). This stage changes pixel values and therefore changes the histogram.",
        "records": records,
    }


def _d2_partner(row: int, col: int, h: int, w: int) -> tuple[int,int,str,str]:
    hh, hw = h//2, w//2
    if row < hh and col < hw:
        return row+hh, col+hw, "q1", "q4"
    if row < hh and col >= hw:
        return row+hh, col-hw, "q2", "q3"
    if row >= hh and col < hw:
        return row-hh, col+hw, "q3", "q2"
    return row-hh, col-hw, "q4", "q1"


def _diffusion2_details(stages: dict[str, np.ndarray], samples: list[dict], bgr: bool) -> dict | None:
    if "diffusion1" not in stages or "diffusion2" not in stages:
        return None
    inp = np.asarray(stages["diffusion1"], dtype=np.uint8)
    out = np.asarray(stages["diffusion2"], dtype=np.uint8)
    h, w = inp.shape[:2]
    records = []
    for s in samples:
        idx = int(s["linear_byte_slot"])
        if idx >= out.size:
            continue
        row, col, ch, _ = _coord_for_linear(out, idx)
        pr, pc, q, pq = _d2_partner(row, col, h, w)
        v = int(inp[row,col] if inp.ndim == 2 else inp[row,col,ch])
        partner = int(inp[pr,pc] if inp.ndim == 2 else inp[pr,pc,ch])
        msb = v & 0xF0
        lsb_partner = partner & 0x0F
        result = msb | lsb_partner
        actual = int(out[row,col] if out.ndim == 2 else out[row,col,ch])
        delta = v ^ actual
        records.append({
            "linear_byte_slot": idx,
            "channel": _channel_names(inp,bgr)[min(ch, len(_channel_names(inp,bgr))-1)],
            "coord": [row,col], "quadrant": q,
            "partner_coord": [pr,pc], "partner_quadrant": pq,
            "I3": v, "I3_bits": _bits(v),
            "MSB": msb, "MSB_bits": _bits(msb),
            "partner_I3": partner, "partner_bits": _bits(partner),
            "partner_LSB": lsb_partner, "partner_LSB_bits": _bits(lsb_partner),
            "bit_or_result": result, "bit_or_result_bits": _bits(result),
            "cipher": actual, "cipher_bits": _bits(actual),
            "diagnostic_xor": delta, "diagnostic_xor_bits": _bits(delta),
            "formula": f"({msb:08b}) OR ({lsb_partner:08b}) = {actual:08b}",
            "input_window": _matrix_window(inp, row, col, ch),
            "output_window": _matrix_window(out, row, col, ch),
        })
    return {
        "paper_note": "The paper's detailed Eq.(7) specifies bit OR after AND-based MSB/LSB extraction. Fig. 5 uses the word XOR, which conflicts with Eq.(7). This implementation follows Eq.(7). The XOR shown below is only before XOR after, a diagnostic changed-bit mask.",
        "records": records,
    }


def _inverse_details(stages: dict[str,np.ndarray], config: CipherConfig | None,
                     samples: list[dict], bgr: bool) -> dict:
    """Detailed inverse tables matching the exact reverse stage order."""
    result = {}
    if not config:
        return result

    # Inverse Diffusion 2: the cross-quadrant nibble exchange is self-inverse.
    if "cipher" in stages and "inverse_diffusion2" in stages:
        inp=np.asarray(stages["cipher"],dtype=np.uint8)
        out=np.asarray(stages["inverse_diffusion2"],dtype=np.uint8)
        h,w=inp.shape[:2]; rows=[]
        for sm in samples:
            idx=int(sm["linear_byte_slot"])
            if idx>=out.size: continue
            r,c,ch,_=_coord_for_linear(out,idx)
            pr,pc,q,pq=_d2_partner(r,c,h,w)
            v=int(inp[r,c] if inp.ndim==2 else inp[r,c,ch])
            partner=int(inp[pr,pc] if inp.ndim==2 else inp[pr,pc,ch])
            msb=v&0xF0; lsb=partner&0x0F; recovered=msb|lsb
            actual=int(out[r,c] if out.ndim==2 else out[r,c,ch])
            delta=v^actual
            rows.append({
                "linear_byte_slot":idx,"coord":[r,c],"quadrant":q,"partner_coord":[pr,pc],"partner_quadrant":pq,
                "cipher":v,"cipher_bits":_bits(v),"MSB":msb,"MSB_bits":_bits(msb),
                "partner_cipher":partner,"partner_bits":_bits(partner),"partner_LSB":lsb,"partner_LSB_bits":_bits(lsb),
                "recovered_I3":actual,"recovered_bits":_bits(actual),"diagnostic_xor_bits":_bits(delta),
                "formula":f"({msb:08b}) OR ({lsb:08b}) = {actual:08b}",
                "input_window":_matrix_window(inp,r,c,ch),"output_window":_matrix_window(out,r,c,ch),
            })
        result["inverse_diffusion2"]={
            "paper_note":"Inverse Diffusion 2 applies the same opposite-quadrant nibble exchange because the Eq.(7) transformation is self-inverse.",
            "records":rows,
        }

    # Inverse Diffusion 1.
    if "inverse_diffusion2" in stages and "inverse_diffusion1" in stages:
        inp = np.asarray(stages["inverse_diffusion2"], dtype=np.uint8)
        out = np.asarray(stages["inverse_diffusion1"], dtype=np.uint8)
        h, w = out.shape[:2]
        seq3 = generate_sequence(config.y03, config.mu03, config.k, h*w, transient=config.transient)
        R = quantize_sequence(seq3, config.quantization_mode, config.quantization_scale).reshape((h,w), order="F")
        rows=[]
        for sm in samples:
            idx=int(sm["linear_byte_slot"])
            if idx>=out.size: continue
            r,c,ch,_=_coord_for_linear(out,idx)
            pre_r, pre_c = c, r
            if pre_r>=inp.shape[0] or pre_c>=inp.shape[1]: continue
            pv=int(inp[pre_r,pre_c] if inp.ndim==2 else inp[pre_r,pre_c,ch])
            rv=int(R[r,c]); ov=int(out[r,c] if out.ndim==2 else out[r,c,ch])
            delta=pv^ov
            rows.append({"linear_byte_slot":idx,"output_coord":[r,c],"pretranspose_coord":[pre_r,pre_c],
                         "pre_value":pv,"R":rv,"recovered_I2":ov,
                         "formula":f"({pv} - {rv}) mod 256 = {ov}","pre_bits":_bits(pv),"out_bits":_bits(ov),
                         "diagnostic_xor_bits":_bits(delta),
                         "before_window":_matrix_window(inp,pre_r,pre_c,ch),"after_window":_matrix_window(out,r,c,ch)})
        result["inverse_diffusion1"]={"records":rows,
            "explanation":"Undo transpose, then subtract the same chaotic R matrix modulo 256."}

    # Inverse Confusion 2: apply the same row/column reversals in reverse order.
    if "inverse_diffusion1" in stages and "inverse_confusion2" in stages:
        inp=np.asarray(stages["inverse_diffusion1"],dtype=np.uint8)
        h,w=inp.shape[:2]; steps=min(h,w)
        seq1=generate_sequence(config.y01,config.mu01,config.k,h*w,transient=config.transient)
        seq2=generate_sequence(config.y02,config.mu02,config.k,steps,transient=config.transient)
        row_action=np.asarray(seq1[:steps])>np.asarray(seq2)
        work=inp.copy(); records=[]
        reverse_indices=list(range(steps-1,-1,-1))[:30]
        for n,i in enumerate(reverse_indices,1):
            is_row=bool(row_action[i]); center_r=i if is_row else h//2; center_c=w//2 if is_row else i
            before=_matrix_window(work,center_r,center_c,0)
            _apply_confusion2_one(work,i,is_row)
            after=_matrix_window(work,center_r,center_c,0)
            records.append({"inverse_step":n,"forward_step":i+1,
                            "chaotic_generated_iteration":config.transient+i+1,
                            "yn1":float(seq1[i]),"yn2":float(seq2[i]),
                            "action":"reverse row" if is_row else "reverse column","affected_index":i,
                            "before":before,"after":after,"channel":_channel_names(inp,bgr)[0]})
        result["inverse_confusion2"]={
            "explanation":"The row/column reversal operations are self-inverse, but they must be replayed in reverse forward-iteration order.",
            "iterations":records,
        }

    # Inverse Confusion 1: scatter I1 positions back to their original destinations.
    if "inverse_confusion2" in stages and "inverse_confusion1" in stages:
        inp=np.asarray(stages["inverse_confusion2"],dtype=np.uint8) # I1
        out=np.asarray(stages["inverse_confusion1"],dtype=np.uint8) # I0
        h,w=out.shape[:2]
        seq1=generate_sequence(config.y01,config.mu01,config.k,h*w,transient=config.transient)
        perm=np.argsort(seq1,kind="stable")
        inv=np.empty_like(perm); inv[perm]=np.arange(perm.size)
        rows=[]
        for sm in samples:
            idx=int(sm["linear_byte_slot"])
            if idx>=out.size: continue
            dr,dc,ch,_=_coord_for_linear(out,idx)
            dest_f=dr+dc*h
            src_f=int(inv[dest_f]); sr,sc=src_f%h,src_f//h
            sv=int(inp[sr,sc] if inp.ndim==2 else inp[sr,sc,ch]); dv=int(out[dr,dc] if out.ndim==2 else out[dr,dc,ch])
            rows.append({"linear_byte_slot":idx,"source_I1":[sr,sc],"destination_I0":[dr,dc],
                         "source_value":sv,"recovered_value":dv,"source_bits":_bits(sv),"recovered_bits":_bits(dv),
                         "source_window":_matrix_window(inp,sr,sc,ch),"destination_window":_matrix_window(out,dr,dc,ch)})
        result["inverse_confusion1"]={
            "explanation":"Inverse Confusion 1 scatters each permuted I1 value back to the original location defined by the inverse sorted-index permutation.",
            "mappings":rows,
        }
    return result

def _chaotic_payload(config: CipherConfig | None, trace: dict | None, limit: int = 30) -> dict | None:
    if not config:
        return None
    transient = int(config.transient)
    data = {"discarded_values": transient, "first_usable_generated_iteration": transient + 1, "sequences": {}}
    specs = {
        "yn1": (config.y01, config.mu01, None),
        "yn2": (config.y02, config.mu02, None),
        "yn3": (config.y03, config.mu03, None),
    }
    for name,(y0,mu,_) in specs.items():
        key = {"yn1":"seq1","yn2":"seq2","yn3":"seq3"}[name]
        if trace and key in trace:
            seq=np.asarray(trace[key],dtype=float)
        else:
            # 30 is enough for presentation and honors the exact transient discard.
            seq=np.asarray(generate_sequence(y0,mu,config.k,limit,transient=transient),dtype=float)
        rows=[]
        for i,v in enumerate(seq[:limit]):
            rows.append({"usable_index":i+1,"generated_iteration":transient+i+1,"value":float(v)})
        data["sequences"][name]={"y0":y0,"mu":mu,"k":config.k,"rows":rows}
    return data


def build_web_trace(stages: dict[str, np.ndarray], order: Iterable[str], output_dir: Path,
                    title: str, config: CipherConfig | None = None, trace: dict | None = None,
                    bgr: bool = False, sample_count: int = 30) -> Path:
    """Create a complete local web-dashboard payload and PNG assets."""
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    items = _presentation_stages(stages, order)
    stage_payload=[]
    for i,(name,arr) in enumerate(items,1):
        img_path=images_dir/f"{i:02d}_{name}.png"
        _save_stage_image(arr,img_path,bgr=bgr)
        a=np.asarray(arr,dtype=np.uint8)
        stage_payload.append({
            "name":name,"label":_pretty(name),"shape":list(a.shape),
            "image_file":str(img_path.resolve()),"histogram":_hist_payload(a,bgr),
        })
    samples=_sample_progression(stages,order,bgr,count=sample_count)
    forward = list(order)[0] == "original"
    operations={}
    if forward:
        operations["confusion1"]=_confusion1_details(stages,trace,config,samples,bgr)
        operations["confusion2"]=_confusion2_iterations(stages,trace,config,bgr,limit=30)
        operations["diffusion1"]=_diffusion1_details(stages,trace,config,samples,bgr)
        operations["diffusion2"]=_diffusion2_details(stages,samples,bgr)
    else:
        operations.update(_inverse_details(stages,config,samples,bgr))
    payload={
        "format":"nature-cipher-web-trace-v4",
        "title":title,
        "direction":"encryption" if forward else "decryption",
        "paper_pipeline":"Confusion 1 -> Confusion 2 -> Diffusion 1 -> Diffusion 2" if forward else "Inverse Diffusion 2 -> Inverse Diffusion 1 -> Inverse Confusion 2 -> Inverse Confusion 1",
        "paper_notes":[
            "The first 100 generated chaotic values are discarded. The first displayed usable value is generated iteration 101.",
            "Confusion stages rearrange values and therefore preserve the histogram exactly; movement is demonstrated with mapping/matrix tables.",
            "Diffusion 1 follows Eq.(6): addition, mod 256, then transpose.",
            "Diffusion 2 follows detailed Eq.(7): MSB/LSB extraction with AND and cross-quadrant bit OR. Diagnostic XOR masks are shown separately.",
        ],
        "stages":stage_payload,
        "samples":samples,
        "chaotic":_chaotic_payload(config,trace),
        "operations":operations,
        "bgr_input":bool(bgr),
        "sample_policy":"30 deterministic non-background byte slots spread across the selected image/frame/slice; not simply the first 30 bytes.",
    }
    out=output_dir/"trace_dashboard.json"
    out.write_text(json.dumps(payload,indent=2),encoding="utf-8")
    return out


# Backward-compatible name used by v3 call sites.  It now creates web data only.
def show_matlab_stage_windows(stages: dict[str, np.ndarray], order: Iterable[str], output_dir: Path,
                              title: str, show: bool = True, bgr: bool = False,
                              sample_count: int = 30, config: CipherConfig | None = None,
                              trace: dict | None = None):
    return build_web_trace(stages, order, output_dir, title, config=config, trace=trace,
                           bgr=bgr, sample_count=sample_count)


def close_all_windows():
    return None
