# Noise Accumulation Experiments

This directory contains the noise-accumulation diagnostic used in the
revision.

- `noise_accumulation.py`: the single self-contained script for data
  loading, streaming OLS, histogram construction, plotting, CSV output, and
  optional contact-sheet generation. Its `--analysis START END` argument is a
  zero-based, half-open rank range; bounds may use decimal `K`, `M`, or `B`
  suffixes.

## Reproducing

Run from the FH-index repository root with Python packages installed as
described in [experiments/README.md](../README.md). For example, the following command analyzes the two-million-key range
centered at rank 100M:

```bash
.venv/bin/python experiments/noise_accumulation/noise_accumulation.py \
  --data-dir /path/to/data --analysis 99M 101M \
  --out-dir results/noise/median_2m_20260626 \
  --bins 100 --save-pdf \
  --contact-sheet median_2m_contact_sheet.png
```

The same file can regenerate the full-range figures and their existing naming
style by selecting the entire 200M-key rank range:

```bash
.venv/bin/python experiments/noise_accumulation/noise_accumulation.py \
  --data-dir /path/to/data --analysis 0 200M \
  --out-dir results/noise/full_200m_20260626 \
  --bins 100 --save-pdf \
  --contact-sheet full_200m_contact_sheet.png
```
