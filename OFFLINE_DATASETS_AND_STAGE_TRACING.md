# Offline datasets and browser stage tracing - V4

Run `python extended_main.py`. A local browser page opens and performs first-run medical dataset preparation in the background. Once `OFFLINE_READY.json` validates, subsequent normal operations are local-only.

## Browser workflow

The old Matplotlib/Tk sequence has been removed from the extended workflow. All six operations, progress output, stage images, 30 representative value paths, chaotic sequence values, matrix operation tables and histogram analysis are displayed in one local HTML GUI.

## First usable chaotic value

The paper instructs that the first 100 generated values are neglected. The first usable value shown by V4 is therefore labeled **generated iteration 101**; usable sequence index 1 corresponds to generated iteration 101.

## Confusion visualization

Confusion changes positions, not the multiset of pixel values. Therefore its histogram is unchanged by design.

V4 uses mapping and matrix tables:

- Confusion 1: highlighted source and destination cells for representative mappings.
- Confusion 2: first 30 usable iterations, `yn1`/`yn2` comparison, row/column action, and highlighted before/after local matrices.

## Diffusion visualization

Diffusion 1 shows Eq. (6) arithmetic (`I2 + R`, modulo 256, transpose) with per-cell values and highlighted matrices.

Diffusion 2 shows AND-based MSB/LSB extraction and detailed Eq. (7) bit OR. `before XOR after` is included only as a changed-bit diagnostic.

## Histograms

The HTML dashboard includes intensity, RGB, cumulative and input-vs-final views, plus entropy, chi-square-to-uniform, mean/std/median/range/unique-level/clipping metrics.

## Video and 3-D scope

The full source video and full brain volume are processed. Only the selected frame or slice receives the large explanatory trace package.
