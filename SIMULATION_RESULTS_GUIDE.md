# Simulation Results & Discussion — full reproduction suite

Run:

```bash
python simulation-main.py
```

The suite reproduces the paper's simulation/results section test-by-test and writes our value beside the paper value.

## Tests

1. `00_preprocessing_audit.py` — checks plausible 512→256 color preprocessing against the paper's **plain-image entropy** before blaming encryption.
2. `01_histogram_analysis.py` — plain/cipher images and histograms (Figs. 7–9, 13 style).
3. `02_chi_square_test.py` — Eq. (8), Tables 1–4 and Table 20.
4. `03_correlation_analysis.py` — Eq. (9), Tables 5–7, correlation scatter plots like Figs. 10–12.
5. `04_global_entropy.py` — Eq. (10), Tables 8–11.
6. `05_local_entropy.py` — Eq. (11), Tables 12–13 and Table 20.
7. `06_mse_psnr.py` — Eqs. (12)–(13), Tables 14–15 and Table 20.
8. `07_npcr_uaci.py` — Eqs. (14)–(15), Tables 16–17 and Table 20, using a standard one-pixel differential experiment.
9. `08_chosen_plaintext.py` — Black/White 256×256, Fig. 13 and Table 20.
10. `09_keyspace_runtime.py` — Tables 18–19.
11. `10_full_paper_comparison.py` — runs everything and creates CSV + HTML report.

## Output

Every full run creates:

```text
simulation_output/paper_comparison_TIMESTAMP/
    comparison_summary.csv
    comparison_summary.html
    SUMMARY.txt
    RUN_NOTES.txt
    00_preprocessing_audit/
    grayscale/
    color/
    medical/
    chosen_plaintext/
    keyspace_runtime/
```

## Important reproducibility rules

Do not force values to match. Diagnose in this order:

1. **Plain-image entropy/correlation mismatch** → wrong input or preprocessing.
2. **Plain metrics match but cipher metrics differ** → encryption convention/implementation differs.
3. **NPCR/UACI differ strongly** → report it. The suite uses the standard same-key, one-pixel-plaintext-change experiment.

The paper leaves several details incomplete: Eq. (5) byte quantization, indexed `fliplr/flipud`, color 512→256 preprocessing, local-entropy `k`/seed, and correlation sampling convention. The suite writes these assumptions into each comparison instead of hiding them.

## Paper-style Tables 1–20

Every full comparison run now also creates:

```text
paper_style_tables.html
```

This report mirrors the organization of the paper's Tables 1–20. Each section has two panels:

- **Published paper** — the values/structure printed in the supplied paper.
- **Our reproduction** — our calculated values in the same row/column organization.

Where a paper table contains comparison values from other authors, those columns are kept unchanged on both sides so that the only replaced column is the paper's **Proposed** result versus **Our reproduction**.

The report colors our cells to make verification immediate:

- green: matches to 4 decimal places
- yellow: close
- red: different
- gray: unavailable input/result
- blue: runtime/environment-dependent

For exact side-by-side paper-style verification, open `paper_style_tables.html`, not only `comparison_summary.html`.
