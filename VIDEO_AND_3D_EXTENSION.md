# Video and 3-D extension - paper-locked V4

The new media handling does not replace the paper cipher.

## Video

Every decoded BGR frame from the complete medical echocardiography clip is passed through the same 2-D paper-reproduction stages:

`Confusion 1 -> Confusion 2 -> Diffusion 1 -> Diffusion 2`

The complete video is processed. One selected frame receives the large explanatory browser trace. Cipher frames are written losslessly with FFV1 in Matroska so exact decryption remains possible.

## 3-D brain MRI

The raw AOMIC/OpenNeuro T1-weighted NIfTI is retained. A documented uint8 `(D,H,W)` adapter is prepared because the paper cipher is byte based. Every Z slice is processed with the same four 2-D stages. One selected Z slice receives the explanatory trace.

The optional Z/depth permutation is a separate extension and is OFF by default.

## Browser trace

For the selected frame/slice the local HTML GUI shows:

- top-to-bottom stage images;
- 30 representative non-background value paths;
- usable chaotic values beginning at generated iteration 101;
- Confusion 1 source/destination mappings;
- Confusion 2 row/column reversal matrices per iteration;
- Diffusion 1 modulo-addition table;
- Diffusion 2 AND/OR MSB-LSB bit construction and diagnostic XOR change mask;
- intensity/RGB/cumulative histograms and statistics.
