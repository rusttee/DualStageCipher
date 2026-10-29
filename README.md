# Chaotic Cipher + Medical Video / 3-D Brain Extension (Web GUI)

Run:

```bash
python extended_main.py
```

A local browser GUI opens automatically. The server binds only to `127.0.0.1` and the GUI uses local HTML/CSS/JavaScript only; it does not require a cloud service or CDN.

On the first launch, the GUI prepares the required medical datasets. Once `data/extended_datasets/OFFLINE_READY.json` validates, normal encryption/decryption runs use local files only.

## GUI operations 0-5

0. Image encryption
1. Image decryption
2. Full medical video encryption
3. Full medical video decryption
4. Full 3-D brain MRI encryption
5. Full 3-D brain MRI decryption

No Matplotlib/Tk stage popup workflow.

## core

The extension keeps the supplied paper-reproduction core and stage order unchanged:

`I0 -> Confusion-1 -> I1 -> Confusion-2 -> I2 -> Diffusion-1 -> I3 -> Diffusion-2 -> C`

Published parameters used by the launcher:

- `y01 = 0.55427`, `mu01 = 20.389429`
- `y02 = 0.54324`, `mu02 = 20.385437`
- `y03 = 0.56543`, `mu03 = 20.389429`
- `k = 20`
- first 100 generated chaotic values discarded

The GUI therefore labels the first usable sequence value as **generated iteration 101**.

Video does not substitute another cipher: every decoded frame runs through the same four 2-D stages. The 3-D MRI wrapper does the same for every Z slice. Optional Z/depth permutation is clearly marked as an extension and defaults OFF.

### Reproducibility note

The paper's Eq. (5) does not fully specify the floating-chaotic-value-to-byte conversion required before later byte/bit operations. This package preserves the explicit reconstruction from the supplied complete implementation instead of silently changing that core.

## Web stage viewer

The browser stacks all stages **top to bottom**:

- Original
- Confusion 1
- Confusion 2
- Diffusion 1
- Diffusion 2
- Encrypted output

For decryption it stacks:

- Cipher input
- Inverse Diffusion 2
- Inverse Diffusion 1
- Inverse Confusion 2
- Inverse Confusion 1
- Recovered output

The dashboard automatically chooses 30 representative non-background byte slots spread across the selected image/frame/slice, avoiding the misleading all-zero top-left/background samples that can occur in medical media.

Each sample is shown as a direct path such as:

`Original=144 -> Confusion1=132 -> Confusion2=201 -> Diffusion1=63 -> Diffusion2=79 -> Encrypted=79`

Each transition also shows binary values and `before XOR after` as a **diagnostic changed-bit mask**.

## Confusion tables

A histogram should not change during Confusion 1 or Confusion 2 because those stages move/reverse pixels without changing their values. The GUI therefore demonstrates confusion using matrix/mapping tables instead:

- Confusion 1: source coordinate -> destination coordinate, moved value, binary value, and highlighted source/destination matrix windows.
- Confusion 2: first 30 usable iterations, starting at generated chaotic iteration 101, with `yn1` vs `yn2`, selected row/column flip, and highlighted before/after matrix windows.

## Diffusion tables

Diffusion 1 follows the supplied paper implementation of Eq. (6) and displays, per selected sample:

`I2 + R -> sum -> mod 256 -> transpose -> I3`

with highlighted before/after matrix windows.

Diffusion 2 displays the actual bit path from the detailed Eq. (7):

- `MSB = I3 AND 240`
- opposite-quadrant `LSB = partner AND 15`
- `cipher = MSB OR partner_LSB`

The paper's high-level Fig. 5 uses the word XOR while detailed Eq. (7) specifies bit OR. V4 follows Eq. (7) and labels XOR only as a diagnostic difference mask.

## Histogram analysis

The local GUI provides:

- grayscale/intensity histogram
- RGB histogram for color/video frames
- cumulative histogram
- original/input vs final overlay
- min, max, mean, median, standard deviation and dynamic range
- unique intensity levels
- clipping percentage at 0 and 255
- information entropy
- paper-style chi-square distance from a uniform histogram
- summary table across every stage

Confusion-stage histogram equality is explicitly shown as expected behavior; diffusion-stage histograms normally change.

## Full medical video

The first-run downloader caches the complete echocardiography samples. Encryption processes every decoded frame at the source FPS/frame count; the selected frame is only the one used for the detailed GUI trace.

Decryptable ciphertext video remains lossless FFV1/Matroska (`.mkv`) so stored cipher pixels are not altered by lossy compression. Audio is outside the current visual-stream experiment.

## Real 3-D brain MRI

The first-run downloader caches a real AOMIC/OpenNeuro T1-weighted structural MRI. The original NIfTI is retained and a documented uint8 volume is prepared for the byte-based paper cipher. The entire volume is encrypted/decrypted; the selected Z slice is only for detailed GUI tracing.

## Offline behavior

External extension data is stored under:

```text
data/extended_datasets/
  medical_video/
  brain_mri_3d/raw/
  brain_mri_3d/prepared/
  offline_dataset_manifest.json
  OFFLINE_READY.json
```

The GUI itself is under:

```text
web/
  index.html
  style.css
  app.js
```

Trace outputs contain:

```text
web_dashboard/
  trace_dashboard.json
  images/
bit_value_comparisons/
  *_all_bytes_bits.csv.gz
  bit_change_summary.csv
```

## Install dependencies

```bash
pip install -r requirements.txt
```

No Flask or other web framework is required; the browser GUI uses Python's standard-library local HTTP server.
