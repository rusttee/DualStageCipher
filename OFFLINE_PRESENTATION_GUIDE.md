# Offline presentation guide - V4 local HTML GUI

Start with:

```bash
python extended_main.py
```

Your browser opens the project locally on `127.0.0.1`. No cloud UI or CDN is used.

For a presentation:

1. Wait for the header badge to show **OFFLINE READY**.
2. Choose operation 0-5.
3. For video or 3-D, choose the full medical source plus one frame/Z-slice for the explanatory trace.
4. Click **Start full operation**. The processing log remains in the page.
5. Click **Open full GUI analysis** when complete.
6. Scroll vertically through Original -> Confusion 1 -> Confusion 2 -> Diffusion 1 -> Diffusion 2 -> Encrypted.
7. Use the 30 selected value paths to show decimal/binary changes.
8. Show the chaotic table: first usable generated iteration is 101 because iterations 1-100 were discarded.
9. Open the Confusion 1 mapping tables and Confusion 2 iteration matrices to demonstrate movement even though the histogram remains identical.
10. Use Diffusion 1 formula tables and Diffusion 2 MSB/LSB bit tables to demonstrate actual value changes.
11. Finish with the local histogram-analysis section: intensity, RGB, cumulative, original-vs-final, entropy and chi-square.

Important: Confusion histogram equality is correct behavior. It is a permutation stage. The matrix tables are the appropriate evidence of the confusion operation.
