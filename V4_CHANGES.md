# V4 changes requested

1. Replaced the extended Matplotlib/Tk stage viewer with a local HTML GUI (`web/index.html`).
2. Stages are stacked top-to-bottom.
3. Added Pixlane-style local histogram analysis: intensity, RGB, cumulative, original-vs-final, statistics, entropy and chi-square.
4. Chaotic sequence presentation now explicitly starts with generated iteration 101 after discarding iterations 1-100.
5. Default 30 trace samples are deterministic representative non-background byte slots, avoiding misleading all-zero corner/background samples.
6. Added direct 30-value paths across Original -> C1 -> C2 -> D1 -> D2 -> Cipher plus binary and diagnostic XOR changed-bit masks.
7. Confusion 1 now has source/destination mapping tables and highlighted matrix windows.
8. Confusion 2 now has per-iteration yn1/yn2 comparisons and highlighted before/after matrix windows.
9. Diffusion 1 now has Eq.(6) arithmetic tables and highlighted before/after matrix windows.
10. Diffusion 2 now has AND/OR MSB-LSB bit construction, partner quadrant coordinates, highlighted matrix windows, and separately labeled diagnostic XOR masks.
11. Decryption receives corresponding inverse operation tables.
12. Full medical videos and the full 3-D brain MRI remain fully processed; selected frame/slice is for detailed tracing only.
13. GUI and analysis assets use no CDN and no web framework; normal runs remain offline after dataset preparation.
